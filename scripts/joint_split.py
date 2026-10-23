"""Build the joint scaffold split that the released models were trained on.

`SMILESDataModule` splits each dataset on its own, so a scaffold in the natural
training set can reappear in the synthetic test set. This script pools COCONUT,
ZINC 22 and ChEMBL before clustering instead, and assigns every cluster to exactly
one split, so no scaffold can sit in the training data of one dataset and the
evaluation data of another.

Steps
    1. Read ``{natural,synthetic,general}_all.smi`` from ``--data-dir``, as written
       by ``SMILESDataModule.prepare_data``. Unparsable SMILES are dropped.
    2. Remove exact duplicates within each dataset, and remove the structures that
       natural and synthetic share on the stereo-free canonical SMILES (a skeleton
       in both cannot be both a positive and a negative). ChEMBL is left alone: the
       general model is meant to overlap both classes.
    3. Take the Bemis-Murcko scaffold of every molecule (chirality removed; acyclic
       molecules share the empty scaffold), and merge scaffolds whose ECFP4
       Tanimoto similarity is >= 0.6, transitively.
    4. Split the clusters 80/10/10 with ``GroupShuffleSplit`` (seed 42). The largest
       cluster holds about two thirds of the molecules and goes to train as a whole,
       so the realised ratios differ from the nominal ones.

The output directory holds ``{dataset}_{all,train,val,test}.smi`` and a
``manifest.json`` with the counts at every step. Pass it to training with
``scripts/train.py --data_dir <out-dir>``; the data module picks up the split files.

The pairwise scan dominates the cost. The full corpora take several hours on 16
processes.

Usage
    uv run python scripts/joint_split.py --data-dir data --out-dir data/joint --workers 16
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

# Keep every worker single-threaded so the process count stays at --workers.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATASETS = ("natural", "synthetic", "general")
CLASS_DATASETS = ("natural", "synthetic")

# Populated in the parent before forking workers, which inherit it copy-on-write.
_FPS: list = []


class UnionFind:
    """Iterative union-find; ``clamnp.utils.UnionFind`` recurses too deep at this scale."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def roots(self) -> np.ndarray:
        return np.fromiter(
            (self.find(i) for i in range(len(self.parent))), dtype=np.int32
        )

    def labels(self) -> np.ndarray:
        """Cluster ids numbered by each cluster's first member.

        Which element ends up as a root depends on the order the unions arrive in,
        and that varies between parallel runs. Numbering by first member does not,
        so the split is reproducible for any number of workers.
        """
        _, first, inverse = np.unique(
            self.roots(), return_index=True, return_inverse=True
        )
        rank = np.empty(len(first), dtype=np.int32)
        rank[np.argsort(first)] = np.arange(len(first), dtype=np.int32)
        return rank[inverse]


def keys_of(smiles: str) -> tuple[str, str] | None:
    """Stereo-free canonical SMILES and chirality-free Murcko scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    flat = Chem.MolToSmiles(mol, isomericSmiles=False)
    try:
        scaffold = Chem.MolToSmiles(
            MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False
        )
    except Exception:
        scaffold = ""
    return flat, scaffold


def load_dataset(data_dir: Path, dataset: str, workers: int) -> pd.DataFrame:
    path = data_dir / f"{dataset}_all.smi"
    smiles = [line.strip() for line in path.open()]
    print(f"  {dataset}: {len(smiles):,} lines from {path}", flush=True)
    with mp.Pool(workers) as pool:
        keys = pool.map(keys_of, smiles, chunksize=500)
    rows = [
        (idx, s, k[0], k[1])
        for idx, (s, k) in enumerate(zip(smiles, keys))
        if s and k is not None
    ]
    df = pd.DataFrame(rows, columns=["idx", "smiles", "flat", "scaffold"])
    df["dataset"] = dataset
    return df


def _scan(args: tuple[int, int, int, float]) -> np.ndarray:
    """Union every pair at or above the threshold, for a strided slice of rows."""
    start, stride, n, threshold = args
    uf = UnionFind(n)
    for i in range(start, n, stride):
        tail = _FPS[i + 1 :]
        if not tail:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(_FPS[i], tail)
        for offset, sim in enumerate(sims):
            if sim >= threshold:
                uf.union(i, i + 1 + offset)
    return uf.roots()


def cluster_scaffolds(
    scaffolds: list[str], threshold: float, workers: int
) -> np.ndarray:
    """Cluster scaffolds by ECFP4 Tanimoto >= threshold, merged transitively."""
    global _FPS
    mfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    print(f"  fingerprinting {len(scaffolds):,} scaffolds", flush=True)
    _FPS = []
    for scaffold in scaffolds:
        mol = Chem.MolFromSmiles(scaffold) if scaffold else None
        _FPS.append(
            mfp.GetFingerprint(mol)
            if mol is not None
            else DataStructs.ExplicitBitVect(2048)
        )

    n = len(scaffolds)
    started = time.time()
    print(f"  pairwise scan with {workers} processes", flush=True)
    uf = UnionFind(n)
    with mp.Pool(workers) as pool:
        for roots in pool.imap_unordered(
            _scan, [(w, workers, n, threshold) for w in range(workers)]
        ):
            # Merging each worker's partition reproduces the transitive closure
            # over the union of the edges they each saw.
            for i, r in enumerate(roots):
                if r != i:
                    uf.union(i, int(r))
    print(f"  clustered in {time.time() - started:.0f}s", flush=True)
    return uf.labels()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory holding {natural,synthetic,general}_all.smi",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/joint"))
    ap.add_argument("--threshold", type=float, default=0.6, help="Tanimoto cutoff")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    manifest: dict = {"parameters": {k: str(v) for k, v in vars(args).items()}}

    print("=== loading ===")
    df = pd.concat(
        [load_dataset(args.data_dir, d, args.workers) for d in DATASETS],
        ignore_index=True,
    )
    manifest["parsed"] = {d: int((df["dataset"] == d).sum()) for d in DATASETS}

    print("\n=== deduplicating ===")
    before = len(df)
    df = df.drop_duplicates(subset=["dataset", "smiles"], keep="first")
    manifest["within_dataset_duplicates_removed"] = before - len(df)
    print(f"  within-dataset exact duplicates removed: {before - len(df):,}")

    # Count distinct datasets per skeleton. Counting duplicated rows instead would
    # flag a dataset's own stereoisomers, which share a stereo-free key.
    sub = df[df["dataset"].isin(CLASS_DATASETS)]
    n_datasets = sub.groupby("flat")["dataset"].nunique()
    shared = n_datasets.index[n_datasets > 1]
    df = df[~(df["dataset"].isin(CLASS_DATASETS) & df["flat"].isin(shared))]
    manifest["cross_class_structures_removed"] = int(len(shared))
    print(f"  structures shared by natural and synthetic removed: {len(shared):,}")
    manifest["after_dedup"] = {d: int((df["dataset"] == d).sum()) for d in DATASETS}

    print("\n=== clustering ===")
    scaffolds = df["scaffold"].unique().tolist()
    labels = cluster_scaffolds(scaffolds, args.threshold, args.workers)
    label_of = dict(zip(scaffolds, labels.tolist()))
    df = df.assign(cluster=df["scaffold"].map(label_of))
    sizes = np.array(sorted(Counter(df["cluster"]).values(), reverse=True))
    manifest["clusters"] = {
        "n_scaffolds": len(scaffolds),
        "n_clusters": len(sizes),
        "largest_cluster_pct_of_molecules": round(100 * sizes[0] / sizes.sum(), 2),
    }
    print(f"  {manifest['clusters']}")

    print("\n=== splitting ===")
    holdout = args.val_split + args.test_split
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=args.seed)
    train_idx, rest_idx = next(
        gss.split(np.arange(len(df)), groups=df["cluster"].to_numpy())
    )
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=1 - args.val_split / holdout, random_state=args.seed
    )
    val_local, test_local = next(
        gss2.split(np.arange(len(rest)), groups=rest["cluster"].to_numpy())
    )
    split = np.full(len(df), "train", dtype=object)
    split[rest_idx[val_local]] = "val"
    split[rest_idx[test_local]] = "test"
    df = df.assign(split=split)

    leaks = int((df.groupby("cluster")["split"].nunique() > 1).sum())
    assert leaks == 0, f"{leaks} clusters span more than one split"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest["split_sizes"] = {}
    for d in DATASETS:
        sub = df[df["dataset"] == d]
        counts = sub["split"].value_counts()
        manifest["split_sizes"][d] = {
            k: int(counts.get(k, 0)) for k in ("train", "val", "test")
        }
        print(f"  {d:10s} {manifest['split_sizes'][d]}")
        for name, frame in (
            ("all", sub),
            ("train", sub[sub["split"] == "train"]),
            ("val", sub[sub["split"] == "val"]),
            ("test", sub[sub["split"] == "test"]),
        ):
            path = args.out_dir / f"{d}_{name}.smi"
            path.write_text("".join(f"{s}\n" for s in frame["smiles"]))

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote the split and manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()

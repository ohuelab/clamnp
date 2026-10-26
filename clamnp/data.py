import csv
import json
import multiprocessing as mp
import zipfile
from pathlib import Path

import lightning as L
import molvs
import numpy as np
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from .utils import UnionFind


class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        max_length: int = 512,
        randomize: bool = False,
        canonical: bool = False,
        scaffold_only: bool = False,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.randomize = randomize
        self.canonical = canonical
        self.scaffold_only = scaffold_only

    def __len__(self):
        return len(self.smiles_list)

    def _process_smiles(self, smiles: str) -> str:
        """Process SMILES with optional scaffold extraction, canonicalization and randomization."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles

            # Extract scaffold if scaffold_only mode is enabled
            if self.scaffold_only:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    mol = scaffold  # Use scaffold for further processing
                except Exception:
                    # Keep original molecule if scaffold extraction fails
                    pass

            if self.canonical:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            elif self.randomize:
                smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            else:
                # If scaffold was extracted but no canonical/randomize, convert to SMILES
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

            return smiles
        except Exception:
            return smiles

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        # Process SMILES if needed
        smiles = self._process_smiles(smiles)

        # Tokenize and return the full encoding dict (DataCollator needs this format)
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
        }


DATASET_TYPES = ("natural", "synthetic", "general")
# A structure in both class datasets cannot be both a positive and a negative.
# ChEMBL is left alone: the general model is meant to overlap both classes.
CLASS_DATASETS = ("natural", "synthetic")
SPLITS = ("train", "val", "test")

# Populated in the parent before forking scan workers, which inherit it copy-on-write.
_FPS: list = []


def _keys(smiles: str) -> tuple[str, str] | None:
    """Stereo-free canonical SMILES and chirality-free Bemis-Murcko scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = Chem.MolToSmiles(
            MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False
        )
    except Exception:
        scaffold = ""
    return Chem.MolToSmiles(mol, isomericSmiles=False), scaffold


def _scan(args: tuple[int, int, int, float]) -> list[int]:
    """Union every scaffold pair at or above the threshold, for a strided slice."""
    start, stride, n, threshold = args
    uf = UnionFind(n)
    for i in range(start, n, stride):
        tail = _FPS[i + 1 :]
        if not tail:
            continue
        for offset, sim in enumerate(DataStructs.BulkTanimotoSimilarity(_FPS[i], tail)):
            if sim >= threshold:
                uf.union(i, i + 1 + offset)
    return [uf.find(i) for i in range(n)]


def cluster_scaffolds(
    scaffolds: list[str], threshold: float, workers: int
) -> list[int]:
    """Cluster scaffolds by ECFP4 Tanimoto >= threshold, merged transitively."""
    global _FPS
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _FPS = []
    for scaffold in tqdm(scaffolds, desc="Fingerprinting scaffolds"):
        mol = Chem.MolFromSmiles(scaffold) if scaffold else None
        _FPS.append(
            mfpgen.GetFingerprint(mol) if mol else DataStructs.ExplicitBitVect(2048)
        )
    n = len(scaffolds)
    uf = UnionFind(n)
    with mp.Pool(workers) as pool:
        for roots in tqdm(
            pool.imap_unordered(
                _scan, [(w, workers, n, threshold) for w in range(workers)]
            ),
            total=workers,
            desc="Comparing scaffolds",
        ):
            # Merging each worker's partition gives the transitive closure over the
            # union of the edges they each saw.
            for i, r in enumerate(roots):
                if r != i:
                    uf.union(i, r)
    # Numbered by first member, so the labels do not depend on worker order.
    return uf.get_clusters()


class SplitManager:
    """Prepares the three datasets and splits them jointly by scaffold cluster.

    Run :meth:`prepare` once before training (``scripts/prepare_data.py``).
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        max_samples: int | None = None,
        threshold: float = 0.6,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.threshold = threshold
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.workers = workers

    def prepare(self) -> None:
        """Write each dataset's SMILES and the joint split, unless they exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for dataset_type in DATASET_TYPES:
            self._prepare_dataset(dataset_type)
        if all(self._split_file(d, p).exists() for d in DATASET_TYPES for p in SPLITS):
            return

        smiles_by_dataset = {}
        for dataset_type in DATASET_TYPES:
            with open(self.data_dir / f"{dataset_type}_all.smi") as f:
                smiles_by_dataset[dataset_type] = [line.strip() for line in f]
        splits, summary = self._split(smiles_by_dataset)
        for dataset_type, parts in splits.items():
            for part, smiles_list in parts.items():
                with open(self._split_file(dataset_type, part), "w") as f:
                    f.writelines(s + "\n" for s in smiles_list)
        (self.data_dir / "joint_split.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(f"Saved joint splits: {summary['split_sizes']}")

    def load(self, dataset_type: str) -> dict[str, list[str]]:
        """Train, val and test SMILES of one dataset."""
        if not all(self._split_file(dataset_type, p).exists() for p in SPLITS):
            self.prepare()
        smiles = {}
        for part in SPLITS:
            with open(self._split_file(dataset_type, part)) as f:
                smiles[part] = [line.strip() for line in f]
        return smiles

    def _split_file(self, dataset_type: str, part: str) -> Path:
        return self.data_dir / f"{dataset_type}_{part}.smi"

    def _split(
        self, smiles_by_dataset: dict[str, list[str]]
    ) -> tuple[dict[str, dict[str, list[str]]], dict]:
        """Split all datasets jointly; returns the splits and a summary."""
        datasets, smiles, flats, scaffolds = [], [], [], []
        with mp.Pool(self.workers) as pool:
            for name, items in smiles_by_dataset.items():
                seen: set[str] = set()
                keys = pool.map(_keys, items, chunksize=500)
                for s, k in zip(items, keys):
                    if k is None or s in seen:
                        continue
                    seen.add(s)
                    datasets.append(name)
                    smiles.append(s)
                    flats.append(k[0])
                    scaffolds.append(k[1])

        in_classes: dict[str, set[str]] = {}
        for d, f in zip(datasets, flats):
            if d in CLASS_DATASETS:
                in_classes.setdefault(f, set()).add(d)
        shared = {f for f, ds in in_classes.items() if len(ds) > 1}
        keep = [
            i
            for i, (d, f) in enumerate(zip(datasets, flats))
            if not (d in CLASS_DATASETS and f in shared)
        ]
        datasets = [datasets[i] for i in keep]
        smiles = [smiles[i] for i in keep]
        scaffolds = [scaffolds[i] for i in keep]

        unique = list(dict.fromkeys(scaffolds))
        label_of = dict(
            zip(unique, cluster_scaffolds(unique, self.threshold, self.workers))
        )
        groups = np.array([label_of[s] for s in scaffolds])

        holdout = self.val_split + self.test_split
        gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=self.seed)
        _, rest = next(gss.split(np.arange(len(smiles)), groups=groups))
        gss = GroupShuffleSplit(
            n_splits=1, test_size=1 - self.val_split / holdout, random_state=self.seed
        )
        val_local, test_local = next(gss.split(rest, groups=groups[rest]))
        part = np.full(len(smiles), "train", dtype=object)
        part[rest[val_local]] = "val"
        part[rest[test_local]] = "test"

        splits: dict[str, dict[str, list[str]]] = {
            d: {p: [] for p in SPLITS} for d in smiles_by_dataset
        }
        for d, s, p in zip(datasets, smiles, part):
            splits[d][p].append(s)
        summary = {
            "threshold": self.threshold,
            "seed": self.seed,
            "structures_shared_by_classes_removed": len(shared),
            "n_scaffolds": len(unique),
            "n_clusters": len(set(label_of.values())),
            "split_sizes": {
                d: {p: len(v) for p, v in parts.items()} for d, parts in splits.items()
            },
        }
        return splits, summary

    @staticmethod
    def _standardize_smiles(smiles: str) -> str:
        try:
            return molvs.standardize_smiles(smiles)
        except Exception:
            return smiles

    def _download_coconut(self) -> Path:
        coconut_path = self.data_dir / "coconut.csv"

        if coconut_path.exists():
            return coconut_path

        print("Downloading COCONUT dataset...")
        url = "https://coconut.s3.uni-jena.de/prod/downloads/2025-08/coconut_csv-08-2025.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        zip_path = self.data_dir / "coconut.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        zip_path.unlink()
        extracted_csv = self.data_dir / "coconut_csv-08-2025.csv"
        extracted_csv.rename(coconut_path)

        return coconut_path

    def _prepare_dataset(self, dataset_type: str) -> None:
        """Write ``{dataset_type}_all.smi`` (standardized SMILES)."""
        smiles_file = self.data_dir / f"{dataset_type}_all.smi"
        if smiles_file.exists():
            return

        if dataset_type == "natural":
            filepath = self._download_coconut()

            # Extract SMILES from CSV
            csv.field_size_limit(csv.field_size_limit() * 10)
            smiles_list = []
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "canonical_smiles" in row and row["canonical_smiles"]:
                        smiles = row["canonical_smiles"].strip()
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        elif dataset_type == "synthetic":
            filepath = self.data_dir / "zinc22.txt"

            # Read SMILES from file
            smiles_list = []
            with open(filepath, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line_num == 0:  # Skip empty lines and header
                        continue

                    parts = line.split()
                    if len(parts) >= 3:  # Ensure we have tranche, zincid, and SMILES
                        smiles = parts[2]  # Third column is SMILES
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        elif dataset_type == "general":
            filepath = self.data_dir / "chembl35.txt"

            if not filepath.exists():
                raise FileNotFoundError(
                    f"ChEMBL data file not found at {filepath}. "
                    "Please ensure chembl35.txt is in the data directory."
                )

            # Read SMILES from ChEMBL file (tab-separated: SMILES\tCHEMBL_ID)
            smiles_list = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 1:  # At least SMILES column
                        smiles = parts[0]  # First column is SMILES
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        else:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Must be 'natural', 'synthetic', or 'general'"
            )

        if self.max_samples:
            smiles_list = smiles_list[: self.max_samples]

        with open(smiles_file, "w") as f:
            for smiles in tqdm(smiles_list, desc="Standardizing and saving SMILES"):
                smiles = self._standardize_smiles(smiles)
                f.write(smiles + "\n")

        print(f"Prepared {len(smiles_list)} SMILES for {dataset_type} dataset")


class SMILESDataModule(L.LightningDataModule):
    """Training data for one dataset, taken from a :class:`SplitManager`."""

    def __init__(
        self,
        split_manager: SplitManager,
        dataset_type: str = "natural",  # "natural", "synthetic" or "general"
        tokenizer_name: str = "kohbanye/SmilesTokenizer_PubChem_1M",
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        randomize: bool = False,
        canonical: bool = False,
        scaffold_only: bool = False,
    ):
        super().__init__()
        self.split_manager = split_manager
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.randomize = randomize
        self.canonical = canonical
        self.scaffold_only = scaffold_only

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def prepare_data(self):
        self.split_manager.prepare()

    def setup(self, stage: str | None = None):
        smiles = self.split_manager.load(self.dataset_type)
        print(
            f"Loaded splits for {self.dataset_type}: "
            + ", ".join(f"{p}={len(v)}" for p, v in smiles.items())
        )
        self.train_dataset = SMILESDataset(
            smiles["train"],
            self.tokenizer,
            self.max_length,
            randomize=self.randomize,
            canonical=self.canonical,
            scaffold_only=self.scaffold_only,
        )
        self.val_dataset = SMILESDataset(
            smiles["val"],
            self.tokenizer,
            self.max_length,
            randomize=False,  # Don't randomize validation
            canonical=self.canonical,
            scaffold_only=self.scaffold_only,
        )
        self.test_dataset = SMILESDataset(
            smiles["test"],
            self.tokenizer,
            self.max_length,
            randomize=False,  # Don't randomize test
            canonical=self.canonical,
            scaffold_only=self.scaffold_only,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

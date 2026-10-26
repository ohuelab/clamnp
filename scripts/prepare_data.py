"""Prepare the three datasets and the joint scaffold split they are trained on.

Run once before training, so that parallel training jobs find the split ready
instead of each building it::

    uv run python scripts/prepare_data.py --data_dir data --max_samples 700000
    uv run python scripts/train.py --dataset_type natural --data_dir data --randomize

ChEMBL is read from ``data/chembl35.txt`` and ZINC 22 from ``data/zinc22.txt``;
COCONUT is downloaded. ``--max_samples 700000`` reproduces the released models'
data. The pairwise scaffold comparison takes several hours on 16 processes.
"""

import argparse
import sys
from pathlib import Path

# Allow running this script directly (before `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clamnp.data import SplitManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum SMILES per dataset"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="ECFP4 Tanimoto cutoff"
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    SplitManager(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        threshold=args.threshold,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        workers=args.workers,
    ).prepare()


if __name__ == "__main__":
    main()

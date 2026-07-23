"""Convert local Lightning ``.ckpt`` checkpoints to native Hugging Face models.

Each checkpoint is exported as a standalone ``transformers`` model (config +
safetensors + tokenizer) under a subfolder of the output directory, matching the
layout expected by ``CLaMNPScorer.from_pretrained``::

    <output>/
    ├── natural/
    ├── synthetic/
    └── general/       (optional)

Usage (export locally)::

    uv run python scripts/convert_to_hf.py \
        --natural-checkpoint  checkpoints/.../gpt2_natural/last.ckpt \
        --synthetic-checkpoint checkpoints/.../gpt2_synthetic/last.ckpt \
        --general-checkpoint  checkpoints/.../gpt2_general/last.ckpt \
        --output hf_export

Then push to the Hub (requires `hf auth login` or HF_TOKEN)::

    uv run python scripts/convert_to_hf.py ... --push --repo-id kohbanye/clamnp
"""

import argparse
import sys
from pathlib import Path

# Allow running this script directly (before `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clamnp.loading import load_checkpoint

MODEL_CARD = """---
license: mit
library_name: transformers
tags:
  - chemistry
  - cheminformatics
  - natural-products
  - smiles
---

# CLaM-NP

SMILES language models for **natural-product-likeness** scoring (CLaM-NP Score).

This repository hosts up to three causal language models, one per subfolder:

| Subfolder    | Training data | Role                          |
|--------------|---------------|-------------------------------|
| `natural`    | COCONUT       | P(x \\| natural)               |
| `synthetic`  | ZINC22        | P(x \\| synthetic)             |
| `general`    | ChEMBL        | P(x \\| general) — stabilizer  |

Tokenizer: [`kohbanye/SmilesTokenizer_PubChem_1M`](https://huggingface.co/kohbanye/SmilesTokenizer_PubChem_1M).

## Usage

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained()          # this repo
print(scorer.score("CC(=O)Oc1ccccc1C(=O)O"))         # CLaM-NP Score
```

Or load a single model directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{repo_id}", subfolder="natural")
```
"""


def export_one(checkpoint_path: str, out_dir: Path) -> None:
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    # `model.model` is the underlying transformers model; `model.tokenizer` its tokenizer.
    model.model.save_pretrained(out_dir)
    model.tokenizer.save_pretrained(out_dir)
    print(f"  -> exported to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .ckpt checkpoints to HF format"
    )
    parser.add_argument("--natural-checkpoint", required=True)
    parser.add_argument("--synthetic-checkpoint", required=True)
    parser.add_argument("--general-checkpoint", default=None)
    parser.add_argument(
        "--output",
        default="hf_export",
        help="Local output directory (default: hf_export).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the export to the Hugging Face Hub (needs auth + --repo-id).",
    )
    parser.add_argument(
        "--repo-id",
        default="kohbanye/clamnp",
        help="Target Hub repo id when --push is set.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private (only with --push).",
    )
    args = parser.parse_args()

    out = Path(args.output)
    export_one(args.natural_checkpoint, out / "natural")
    export_one(args.synthetic_checkpoint, out / "synthetic")
    if args.general_checkpoint:
        export_one(args.general_checkpoint, out / "general")

    # Write a model card at the repo root.
    (out / "README.md").write_text(MODEL_CARD.format(repo_id=args.repo_id))

    print(f"\nExport complete: {out.resolve()}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        print(f"\nCreating/using repo: {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print("Uploading folder to the Hub...")
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    else:
        print(
            "\nTo publish, log in first (`hf auth login`) then re-run with "
            f"--push --repo-id {args.repo_id}"
        )


if __name__ == "__main__":
    main()

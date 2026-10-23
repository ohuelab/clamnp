# CLaM-NP Score

**C**hemical **La**nguage **M**odels for **N**atural **P**roduct-Likeness Score.

CLaM-NP scores how "natural-product-like" a molecule is by comparing the
likelihood of its SMILES string under decoder-only language models (GPT-2)
trained on different chemical spaces. A molecule that a natural-product model
finds likely but a synthetic model finds unlikely gets a high score.

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained()          # downloads models from the HF Hub
scorer.score(["COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",            # gefitinib
              "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"])  # morphine
# -> [0.14, 0.67]   (synthetic-like, natural-like)
```

## Installation

Inference only (lightweight):

```bash
pip install clamnp
pip install git+https://github.com/ohuelab/clamnp
```

With [uv](https://docs.astral.sh/uv/), no install needed:

```bash
uvx --from clamnp clamnp "CC(=O)Oc1ccccc1C(=O)O"
```

Extras:

| Extra   | For                                  | Install                     |
|---------|--------------------------------------|-----------------------------|
| `train` | Training models, dataset preparation | `pip install clamnp[train]` |
| `viz`   | Notebooks & paper figures            | `pip install clamnp[viz]`   |

For local development on this repo, `uv sync` installs everything (the `dev`
dependency group bundles all extras).

## Usage

### Command line

```bash
# Single molecule (models auto-downloaded and cached on first run)
clamnp "CC(=O)Oc1ccccc1C(=O)O"

# Several molecules
clamnp "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "O=C(O)c1ccccc1O"

# A file (one SMILES per line) -> CSV
clamnp --input molecules.smi --output scores.csv

# Raw log-likelihood ratio instead of the normalized [0, 1] score
clamnp --raw-score "CC(=O)Oc1ccccc1C(=O)O"

# Unstabilized (2-model) score, on CPU
clamnp --mode unstabilized --device cpu "CC(=O)Oc1ccccc1C(=O)O"

# Average over 32 randomized SMILES instead of scoring the canonical form once
clamnp --n-enum 32 "CC(=O)Oc1ccccc1C(=O)O"
```

### Python

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained(scoring_mode="stabilized")

scorer.score(["CCO", "c1ccccc1"])                # batch -> list of normalized [0, 1] scores
scorer.score("CN1CCC[C@H]1c1cccnc1")             # single SMILES -> one score
scorer.score(["CCO", "c1ccccc1"], raw=True)      # raw log-likelihood ratios
scorer.score_with_details("CCO")                 # full breakdown (log P, perplexity, α, ...)
```

### Averaging over randomized SMILES

A language model reads a SMILES string in one particular atom order, so the score
of a single molecule moves a little with how it is written. Pass `n_enum` to
average over that many randomized SMILES instead of scoring the canonical form
once. This is worth doing when individual values are compared; for ranking a
large set the canonical score is usually enough. The cost grows linearly with
`n_enum`, and `seed` makes the enumeration reproducible.

```python
scorer.score(["CCO", "c1ccccc1"], n_enum=32)     # mean over 32 randomized SMILES
```

## Models

Models are hosted on the Hugging Face Hub at
[`kohbanye/clamnp`](https://huggingface.co/kohbanye/clamnp), with one subfolder
per model (`natural`, `synthetic`, `general`). The shared SMILES tokenizer is
[`kohbanye/SmilesTokenizer_PubChem_1M`](https://huggingface.co/kohbanye/SmilesTokenizer_PubChem_1M).

## Reproducing the models

The released models are trained on a joint scaffold split: the scaffolds of all
three datasets are clustered together, and each cluster goes to a single split,
so no scaffold is shared between the training data of one dataset and the
evaluation data of another. Training needs the `train` extra (or `uv sync`).

1. Put one standardized SMILES per line in `data/{natural,synthetic,general}_all.smi`.

   ```bash
   # natural: COCONUT, August 2025 release (downloaded automatically)
   uv run python -c "from clamnp.data import SMILESDataModule as D; D(dataset_type='natural').prepare_data()"

   # synthetic: the ZINC 22 sample the released models were trained on
   hf download kohbanye/clamnp data/zinc22_sample.smi --local-dir .
   mv data/zinc22_sample.smi data/synthetic_all.smi

   # general: the first 700,000 entries of ChEMBL 35, from data/chembl35.txt (SMILES<TAB>ChEMBL ID)
   uv run python -c "from clamnp.data import SMILESDataModule as D; D(dataset_type='general', max_samples=700_000).prepare_data()"
   ```

2. Build the joint split. The pairwise similarity scan takes several hours.

   ```bash
   uv run python scripts/joint_split.py --data-dir data --out-dir data/joint --workers 16
   ```

3. Train each model on it.

   ```bash
   for t in natural synthetic general; do
     uv run python scripts/train.py --dataset_type $t --data_dir data/joint --model_type gpt2 --randomize
   done
   ```

`scripts/convert_to_hf.py` turns the resulting `.ckpt` files into the Hub layout.

## License

MIT — see [LICENSE](LICENSE).

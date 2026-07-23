# CLaM-NP Score

**C**hemical **La**nguage **M**odels for **N**atural **P**roduct-Likeness Score.

CLaM-NP scores how "natural-product-like" a molecule is by comparing the
likelihood of its SMILES string under decoder-only (GPT-2) language models
trained on different chemical spaces. The score is in `[0, 1]` (0.5 = neutral);
higher means more natural-product-like. Under the hood it is a sigmoid-squashed
log-likelihood ratio (see `clamnp/scorer.py`; pass `raw=True` for the raw ratio).

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained()          # downloads models from the HF Hub

# Pass a list to score in one fast batched pass
scorer.score(["CC(=O)Oc1ccccc1C(=O)O",           # aspirin  -> 0.44 (synthetic-like)
              "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"])    # caffeine -> 0.59 (natural-like)
# -> [0.44, 0.59]
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
```

By default `clamnp` prints the normalized `[0, 1]` score; pass `--raw-score` for
the raw log-likelihood ratio.

### Python

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained(scoring_mode="stabilized")

scorer.score(["CCO", "c1ccccc1"])                # batch -> list of normalized [0, 1] scores
scorer.score("CN1CCC[C@H]1c1cccnc1")             # single SMILES -> one score
scorer.score(["CCO", "c1ccccc1"], raw=True)      # raw log-likelihood ratios
scorer.score_with_details("CCO")                 # full breakdown (log P, perplexity, α, ...)
```

## Models

Models are hosted on the Hugging Face Hub at
[`kohbanye/clamnp`](https://huggingface.co/kohbanye/clamnp), with one subfolder
per model (`natural`, `synthetic`, `general`). The shared SMILES tokenizer is
[`kohbanye/SmilesTokenizer_PubChem_1M`](https://huggingface.co/kohbanye/SmilesTokenizer_PubChem_1M).

## License

MIT — see [LICENSE](LICENSE).

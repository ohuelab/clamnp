# CLaM-NP Score

**C**hemical **La**nguage **M**odels for **N**atural **P**roduct-Likeness Score.

CLaM-NP scores how "natural-product-like" a molecule is by comparing the
likelihood of its SMILES string under decoder-only language models trained on
different chemical spaces. A molecule that a natural-product model finds likely
but a synthetic model finds unlikely gets a high score.

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained()      # downloads models from the HF Hub
scorer.score("CC(=O)Oc1ccccc1C(=O)O")            # aspirin  -> low (synthetic-like)
scorer.score("CN1CCC[C@H]1c1cccnc1")             # nicotine -> high (natural-like)
```

## How it works

Each model is a small GPT-2 causal LM trained on SMILES. The perplexity
of a SMILES under a model measures how well that molecule fits the model's
training distribution. The **CLaM-NP Score** is a log-likelihood ratio:

- **`stabilized`** (default, "CLaM-NP Score"):

  ```
  score = log P(x | natural) − log( α·P(x | synthetic) + (1−α)·P(x | general) )
  ```

  where `α` is a per-molecule gating weight and `general` is a model trained on
  a broad bioactive set that stabilizes the denominator.

- **`unstabilized`** ("CLaM-NP Score (unstabilized)"):

  ```
  score = log P(x | natural) − log P(x | synthetic)
  ```

Higher = more natural-product-like. See `clamnp/scorer.py` for the exact math.

## Installation

Inference only (lightweight):

```bash
pip install clamnp                 # from PyPI (once published)
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

# A file (one SMILES per line) -> CSV, with normalized [0,1] scores
clamnp --input molecules.smi --output scores.csv --normalized

# Unstabilized (2-model) score, on CPU
clamnp --mode unstabilized --device cpu "CC(=O)Oc1ccccc1C(=O)O"
```

### Python

```python
from clamnp import CLaMNPScorer

scorer = CLaMNPScorer.from_pretrained(scoring_mode="stabilized")

scorer.score("CN1CCC[C@H]1c1cccnc1")             # raw score
scorer.score_normalized("CN1CCC[C@H]1c1cccnc1")  # sigmoid-normalized to [0, 1]
scorer.batch_score(["CCO", "c1ccccc1"])          # list of scores
scorer.score_with_details("CCO")                 # full breakdown (log P, perplexity, α, ...)
```

## Models

Models are hosted on the Hugging Face Hub at
[`kohbanye/clamnp`](https://huggingface.co/kohbanye/clamnp), with one subfolder
per model (`natural`, `synthetic`, `general`). The shared SMILES tokenizer is
[`kohbanye/SmilesTokenizer_PubChem_1M`](https://huggingface.co/kohbanye/SmilesTokenizer_PubChem_1M).

## License

MIT — see [LICENSE](LICENSE).

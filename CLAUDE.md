# CLAUDE.md

This project builds a model that predicts natural-product-likeness.

## Dataset

The dataset is a list of SMILES strings for natural products and synthetic compounds. The goal is to predict a compound as natural-product-like when it resembles natural products but not synthetic compounds.

The natural-product dataset can be obtained from COCONUT as follows. Unzipping the archive yields a CSV file. The CSV has a `canonical_smiles` column; extract it to build the dataset.

```shell
curl 'https://coconut.s3.uni-jena.de/prod/downloads/2025-08/coconut_csv-08-2025.zip'
```

The synthetic-compound dataset can be obtained from ZINC22 as follows, specifying `count` and `subset` (`none` is fine). The resulting txt file contains SMILES and compound IDs, so extract only the SMILES to build the dataset.

```shell
curl 'https://cartblanche.docking.org/substance/random.txt' \
  -H 'Accept: application/json, text/plain, */*' \
  -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundary0cFylzxojzDWwpB0' \
  --data-raw $'------WebKitFormBoundary0cFylzxojzDWwpB0\r\nContent-Disposition: form-data; name="count"\r\n\r\n1000000\r\n------WebKitFormBoundary0cFylzxojzDWwpB0\r\nContent-Disposition: form-data; name="subset"\r\n\r\nnone\r\n------WebKitFormBoundary0cFylzxojzDWwpB0--\r\n'
```

Save datasets under the `data/` directory.

## Model

The model is a GPT-style decoder-only Transformer. When a SMILES is fed to a language model trained on a given dataset, its perplexity indicates how likely that compound is to belong to the training dataset. By comparing a model trained on the natural-product dataset against one trained on the synthetic-compound dataset, we predict natural-product-likeness.

## Score

The natural-product-likeness score is computed from a log-likelihood ratio, specifically as follows:

$$
\text{score} = \log \frac{P(\text{natural} | x)}{P(\text{synth} | x)} \\
= \log \frac{P(x | \text{natural})}{P(x | \text{synth})} + \log \frac{P(\text{natural})}{P(\text{synth})} \\
= \log P(x | \text{natural}) - \log P(x | \text{synth}) + \log \frac{P(\text{natural})}{P(\text{synth})}
$$

$\frac{P(\text{natural})}{P(\text{synth})}$ is the prior and depends on the dataset balance. Here we assume the natural-product and synthetic-compound datasets are the same size, so the prior is 1. The released CLaM-NP Score uses a denominator in which the synthetic model's probability is stabilized with a general model (ChEMBL).

## Implementation

The model is a GPT-2-style decoder-only Transformer; training uses transformers and pytorch-lightning (inference is Lightning-free). Libraries are installed with uv; lint and format use ruff, and type checking uses ty. Source code lives in the `clamnp/` package, and training scripts and the like live in the `scripts/` directory. Trained models are published on the Hugging Face Hub (`kohbanye/clamnp`) and can be used for inference via the `clamnp` command.

"""Shared constants for the clamnp package."""

# SMILES tokenizer hosted on the Hugging Face Hub.
DEFAULT_TOKENIZER = "kohbanye/SmilesTokenizer_PubChem_1M"

# Hugging Face Hub repository that hosts the CLaM-NP language models.
# The repository is expected to contain the following subfolders, each a native
# ``transformers`` checkpoint (config.json + model.safetensors + tokenizer):
#   - ``natural``   : model trained on natural products (COCONUT)
#   - ``synthetic`` : model trained on synthetic compounds (ZINC22)
#   - ``general``   : model trained on general/bioactive compounds (ChEMBL)
DEFAULT_HF_REPO = "kohbanye/clamnp"

# Valid scoring modes for :class:`clamnp.scorer.CLaMNPScorer`.
#   - ``stabilized``    : CLaM-NP Score (default; requires the general model)
#   - ``unstabilized``  : CLaM-NP Score (unstabilized); log-likelihood ratio of
#                         the natural vs. synthetic model
#   - ``natural_only``  : log-likelihood under the natural model only
#   - ``synthetic_only``: log-likelihood under the synthetic model only
SCORING_MODES = ("stabilized", "unstabilized", "natural_only", "synthetic_only")
DEFAULT_SCORING_MODE = "stabilized"

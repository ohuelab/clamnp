"""CLaM-NP: Natural-product-likeness scoring with SMILES language models.

Quick start (inference)::

    from clamnp import CLaMNPScorer

    scorer = CLaMNPScorer.from_pretrained()      # downloads models from HF Hub
    print(scorer.score("CC(=O)Oc1ccccc1C(=O)O"))     # CLaM-NP Score

Importing this package is lightweight: the inference path (``CLaMNPScorer``,
``HFLanguageModel``) does not import PyTorch Lightning. Training utilities
(``create_model``, ``BaseLanguageModel``, ``load_checkpoint``,
``SMILESDataModule``) are imported lazily on first access and require the
``[train]`` optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import (
    DEFAULT_HF_REPO,
    DEFAULT_SCORING_MODE,
    DEFAULT_TOKENIZER,
    SCORING_MODES,
)
from .scorer import CLaMNPScorer
from .scoring import HFLanguageModel, LogLikelihoodMixin

__version__ = "0.2.0"

__all__ = [
    "CLaMNPScorer",
    "HFLanguageModel",
    "LogLikelihoodMixin",
    "DEFAULT_HF_REPO",
    "DEFAULT_TOKENIZER",
    "DEFAULT_SCORING_MODE",
    "SCORING_MODES",
    # Lazily imported (require the ``[train]`` extra):
    "create_model",
    "BaseLanguageModel",
    "GPT2Model",
    "load_checkpoint",
    "SMILESDataModule",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .data import SMILESDataModule
    from .loading import load_checkpoint
    from .model import BaseLanguageModel, GPT2Model, create_model


# Lazy imports keep `import clamnp` free of the heavy training dependencies
# (lightning, scikit-learn, molvs, ...). These names resolve on first access.
_LAZY = {
    "create_model": ("clamnp.model", "create_model"),
    "BaseLanguageModel": ("clamnp.model", "BaseLanguageModel"),
    "GPT2Model": ("clamnp.model", "GPT2Model"),
    "load_checkpoint": ("clamnp.loading", "load_checkpoint"),
    "SMILESDataModule": ("clamnp.data", "SMILESDataModule"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

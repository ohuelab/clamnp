"""Load CLaM-NP models from raw Lightning ``.ckpt`` checkpoints.

This is a training/development utility (it depends on PyTorch Lightning via
:mod:`clamnp.model`). For inference from the Hugging Face Hub, prefer
:meth:`clamnp.scorer.CLaMNPScorer.from_pretrained`, which needs no
Lightning.
"""

from __future__ import annotations

import torch

from .constants import DEFAULT_TOKENIZER
from .model import BaseLanguageModel, create_model


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> BaseLanguageModel:
    """Load a :class:`~clamnp.model.BaseLanguageModel` from a ``.ckpt`` file.

    Args:
        checkpoint_path: Path to a Lightning checkpoint.
        device: Device to load the model onto.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint.get("hyper_parameters", {})

    model = create_model(
        learning_rate=hparams.get("learning_rate", 5e-5),
        warmup_steps=hparams.get("warmup_steps", 1000),
        max_length=hparams.get("max_length", 256),
        tokenizer_name=hparams.get("tokenizer_name", DEFAULT_TOKENIZER),
        n_embd=hparams.get("n_embd", 768),
        n_layer=hparams.get("n_layer", 6),
        n_head=hparams.get("n_head", 12),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model

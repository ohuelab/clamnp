"""Training-time Lightning modules for the CLaM-NP language models.

This module depends on PyTorch Lightning and is only needed for *training*
(and for loading raw ``.ckpt`` checkpoints). Inference-only code should import
from :mod:`clamnp.scoring` / :mod:`clamnp.scorer` instead, which are
Lightning-free.
"""

from __future__ import annotations

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .constants import DEFAULT_TOKENIZER
from .scoring import LogLikelihoodMixin


class BaseLanguageModel(LogLikelihoodMixin, L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        max_length: int = 256,
        tokenizer_name: str = DEFAULT_TOKENIZER,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_length = max_length

        # Load SMILES tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model will be initialized in subclasses
        self.model: PreTrainedModel | None = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        max_epochs = self.trainer.max_epochs
        if max_epochs is None:
            raise ValueError("Max epochs is not set.")

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class GPT2Model(BaseLanguageModel):
    def __init__(
        self,
        n_embd: int = 576,
        n_layer: int = 6,
        n_head: int = 12,
        learning_rate: float = 5e-4,
        warmup_steps: int = 1000,
        max_length: int = 512,
        tokenizer_name: str = DEFAULT_TOKENIZER,
    ):
        super().__init__(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_length=max_length,
            tokenizer_name=tokenizer_name,
        )

        # Initialize GPT-2 configuration
        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_length,
            # for kohbanye/SmilesTokenizer_PubChem_1M tokenizer
            bos_token_id=12,
            eos_token_id=13,
            pad_token_id=0,
        )

        # Initialize GPT-2 model
        self.model = GPT2LMHeadModel(config)


def create_model(model_type: str = "gpt2", **kwargs) -> GPT2Model:
    """Factory function to create the language model.

    Only GPT-2 is supported. ``model_type`` is retained for backward
    compatibility with existing callers and checkpoints.
    """
    if model_type.lower() != "gpt2":
        raise ValueError(f"Unsupported model type: {model_type!r} (only 'gpt2')")
    return GPT2Model(**kwargs)

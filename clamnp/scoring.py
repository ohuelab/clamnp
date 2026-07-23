"""Language-model log-likelihood computation (Lightning-free).

This module contains the core inference logic shared by the training-time
Lightning modules (:mod:`clamnp.model`) and the lightweight, dependency-light
:class:`HFLanguageModel` used for inference. Importing this module does **not**
pull in PyTorch Lightning, so it is safe to use in minimal inference installs.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .constants import DEFAULT_TOKENIZER


class LogLikelihoodMixin:
    """Mixin providing per-SMILES log-likelihood / perplexity computation.

    Classes using this mixin must provide the following attributes:
        - ``self.model``: a causal-LM :class:`~transformers.PreTrainedModel`
        - ``self.tokenizer``: a :class:`~transformers.PreTrainedTokenizerBase`
        - ``self.max_length``: int, maximum sequence length
        - ``self.device``: the torch device the model lives on
        - ``self.eval()``: switch the model to evaluation mode
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def tokenize_smiles(self, smiles: list[str]) -> BatchEncoding:
        """Tokenize SMILES strings for model input."""
        return self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def calculate_perplexity(self, smiles: str) -> float:
        """Calculate perplexity for a single SMILES string."""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                [smiles],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Labels are the same as input_ids for language modeling
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            perplexity = torch.exp(outputs.loss).item()

        return perplexity

    def _calculate_log_likelihood(self, smiles: str) -> float:
        """Calculate log likelihood for a single SMILES string."""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                [smiles],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Labels are the same as input_ids for language modeling
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            # Negative log likelihood (cross entropy loss)
            nll = outputs.loss.item()

            # Return log likelihood (negative of NLL)
            return -nll

    def _calculate_log_likelihoods_batch(self, smiles_list: list[str]) -> list[float]:
        """Calculate log likelihoods for a batch of SMILES (internal helper)."""
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Get logits from model
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=None
        )
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate loss per sample
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # Reshape for loss calculation
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)

        # Get per-token losses
        per_token_loss = loss_fct(shift_logits_flat, shift_labels_flat)
        per_token_loss = per_token_loss.view(shift_labels.size())

        # Calculate per-sample loss (average over non-padding tokens)
        # mask out padding tokens (where labels == -100)
        mask = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)

        # Convert to log likelihoods (negative of loss)
        log_likelihoods = (-per_sample_loss).cpu().tolist()

        return log_likelihoods

    def batch_calculate_log_likelihood(
        self, smiles_list: list[str], batch_size: int = 1000
    ) -> list[float]:
        """Calculate log likelihood for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Number of SMILES to process at once (default: 1000)

        Returns:
            List of log likelihoods
        """
        self.eval()
        log_likelihoods = []

        with torch.no_grad():
            # Process in chunks to avoid memory issues
            for i in tqdm(
                range(0, len(smiles_list), batch_size),
                desc="Calculating log likelihoods",
            ):
                batch = smiles_list[i : i + batch_size]
                batch_lls = self._calculate_log_likelihoods_batch(batch)
                log_likelihoods.extend(batch_lls)

        return log_likelihoods


class HFLanguageModel(LogLikelihoodMixin):
    """Lightweight causal-LM wrapper for inference.

    Wraps a native ``transformers`` model + tokenizer and exposes the same
    log-likelihood interface as the training-time Lightning modules, without
    depending on PyTorch Lightning. Load a model hosted on the Hugging Face
    Hub with :meth:`from_pretrained`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        device: str | torch.device = "cpu",
    ):
        self._device = torch.device(device)
        self.model = model.to(self._device).eval()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return self._device

    def eval(self) -> "HFLanguageModel":
        self.model.eval()
        return self

    def to(self, device: str | torch.device) -> "HFLanguageModel":
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        subfolder: str | None = None,
        device: str | torch.device = "cpu",
        max_length: int = 512,
        tokenizer_name: str | None = None,
    ) -> "HFLanguageModel":
        """Load a model (and its tokenizer) from the Hugging Face Hub.

        Args:
            repo_id: Hub repository id, e.g. ``"kohbanye/clamnp"``.
            subfolder: Optional subfolder within the repo (e.g. ``"natural"``).
            device: Device to load the model onto.
            max_length: Maximum sequence length used during scoring.
            tokenizer_name: Override for the tokenizer location. Defaults to the
                model subfolder if it ships a tokenizer, otherwise the shared
                SMILES tokenizer :data:`clamnp.constants.DEFAULT_TOKENIZER`.
        """
        model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder)

        if tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

        return cls(model, tokenizer, max_length=max_length, device=device)

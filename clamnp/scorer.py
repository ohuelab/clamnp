"""Natural-product-likeness scoring (CLaM-NP Score).

Lightning-free: only depends on torch, transformers and RDKit, so it can be
used in a minimal inference install. Load pretrained models straight from the
Hugging Face Hub via :meth:`CLaMNPScorer.from_pretrained`.
"""

from __future__ import annotations

import math
from typing import overload

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .constants import (
    DEFAULT_HF_REPO,
    DEFAULT_SCORING_MODE,
    DEFAULT_SIGMOID_K,
    SCORING_MODES,
)
from .scoring import HFLanguageModel, LogLikelihoodMixin


class CLaMNPScorer:
    """Calculate Natural Product Likeness Score using log-likelihood ratio."""

    def __init__(
        self,
        natural_model: LogLikelihoodMixin,
        synthetic_model: LogLikelihoodMixin,
        general_model: LogLikelihoodMixin | None = None,
        sigmoid_k: float = DEFAULT_SIGMOID_K,
        sigmoid_offset: float = 0.0,
        scaffold_only: bool = False,
        scoring_mode: str = DEFAULT_SCORING_MODE,
    ):
        self.natural_model = natural_model
        self.synthetic_model = synthetic_model
        self.general_model = general_model
        self.sigmoid_k = sigmoid_k
        self.sigmoid_offset = sigmoid_offset
        self.scaffold_only = scaffold_only
        self.scoring_mode = scoring_mode

        # Validate scoring_mode
        if scoring_mode not in SCORING_MODES:
            raise ValueError(
                f"Invalid scoring_mode: {scoring_mode}. Must be one of {SCORING_MODES}"
            )

        # Validate stabilized mode requirements
        if scoring_mode == "stabilized" and general_model is None:
            raise ValueError(
                "scoring_mode='stabilized' requires general_model to be provided"
            )

        # Ensure models are in eval mode
        self.natural_model.eval()
        self.synthetic_model.eval()
        if self.general_model is not None:
            self.general_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_HF_REPO,
        scoring_mode: str = DEFAULT_SCORING_MODE,
        device: str | torch.device | None = None,
        max_length: int = 512,
        tokenizer_name: str | None = None,
        natural_subfolder: str = "natural",
        synthetic_subfolder: str = "synthetic",
        general_subfolder: str = "general",
        **scorer_kwargs,
    ) -> "CLaMNPScorer":
        """Build a scorer from models hosted on the Hugging Face Hub.

        Downloads (and caches) the ``natural`` and ``synthetic`` models, plus the
        ``general`` model when ``scoring_mode='stabilized'``. Each model is
        expected to live in its own subfolder of ``repo_id``.

        Args:
            repo_id: Hub repository id, e.g. ``"kohbanye/clamnp"``.
            scoring_mode: One of :data:`clamnp.constants.SCORING_MODES`.
            device: Torch device. Defaults to CUDA when available, else CPU.
            max_length: Maximum sequence length used during scoring.
            tokenizer_name: Optional tokenizer override for all models.
        """
        if scoring_mode not in SCORING_MODES:
            raise ValueError(
                f"Invalid scoring_mode: {scoring_mode}. Must be one of {SCORING_MODES}"
            )

        resolved_device: str | torch.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        def _load(subfolder: str) -> HFLanguageModel:
            return HFLanguageModel.from_pretrained(
                repo_id,
                subfolder=subfolder,
                device=resolved_device,
                max_length=max_length,
                tokenizer_name=tokenizer_name,
            )

        natural_model = _load(natural_subfolder)
        synthetic_model = _load(synthetic_subfolder)
        general_model = None
        if scoring_mode == "stabilized":
            general_model = _load(general_subfolder)

        return cls(
            natural_model,
            synthetic_model,
            general_model=general_model,
            scoring_mode=scoring_mode,
            **scorer_kwargs,
        )

    def _extract_scaffold(self, smiles: str) -> str:
        """Extract Bemis-Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles  # Return original if parsing fails
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception:
            return smiles  # Return original if scaffold extraction fails

    def _validate_and_canonicalize(self, smiles: str) -> str:
        """Canonical SMILES of the molecule (or its scaffold), or ValueError if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Extract scaffold if scaffold_only mode is enabled
        if self.scaffold_only:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                mol = scaffold
            except Exception:
                pass  # Keep original molecule if scaffold extraction fails

        return Chem.MolToSmiles(mol, doRandom=False, canonical=True)

    def _enumerate_smiles(self, smiles: str, n: int, seed: int) -> list[str]:
        """``n`` randomized SMILES for one molecule, deterministic given ``seed``."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if self.scaffold_only:
            try:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
            except Exception:
                pass
        # Same writings as training's MolToSmiles(doRandom=True), but from a seeded
        # generator; doRandom draws from RDKit's global RNG and cannot be seeded.
        return list(Chem.MolToRandomSmilesVect(mol, n, randomSeed=seed))

    def _calculate_dynamic_alpha(
        self, log_p_synthetic: float, log_p_general: float
    ) -> float:
        """Calculate dynamic alpha (gating weight) based on log likelihoods.

        Args:
            log_p_synthetic: Log probability under synthetic model
            log_p_general: Log probability under general model

        Returns:
            Dynamic alpha value in range [0, 1]
        """
        diff = log_p_synthetic - log_p_general
        return 1.0 / (1.0 + math.exp(-diff))

    def _log_stabilized_mixture(
        self, log_p_syn: float, log_p_gen: float, alpha: float
    ) -> float:
        """Compute log(alpha * P_syn + (1-alpha) * P_gen) via log-sum-exp.

        Args:
            log_p_syn: Log probability under synthetic model
            log_p_gen: Log probability under general model
            alpha: Weighting factor (dynamic or fixed)

        Returns:
            Log of the stabilized mixture probability
        """
        # Handle edge cases
        if alpha == 1.0:
            return log_p_syn
        if alpha == 0.0:
            return log_p_gen

        # Log-sum-exp trick for numerical stability
        max_log_p = max(log_p_syn, log_p_gen)
        log_alpha = math.log(alpha)
        log_one_minus_alpha = math.log(1.0 - alpha)

        result = max_log_p + math.log(
            math.exp(log_alpha + log_p_syn - max_log_p)
            + math.exp(log_one_minus_alpha + log_p_gen - max_log_p)
        )

        return result

    def _score_raw(self, smiles: str) -> float:
        """Raw NP-likeness score (log-likelihood ratio) for a SMILES string.

        Scoring modes:
        - "stabilized": log P(x|natural) - log(a*P(x|synthetic) + (1-a)*P(x|general))
          where a is a dynamic gating weight. This is the CLaM-NP Score (default).
        - "unstabilized": log P(x|natural) - log P(x|synthetic)
          (CLaM-NP Score, unstabilized)
        - "natural_only": log P(x|natural)
        - "synthetic_only": log P(x|synthetic)

        Unbounded; positive = more natural-product-like, negative = more
        synthetic/general-like.
        """
        smiles = self._validate_and_canonicalize(smiles)

        if self.scoring_mode == "unstabilized":
            log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
            log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
            return log_p_natural - log_p_synthetic

        elif self.scoring_mode == "stabilized":
            log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
            log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
            log_p_general = self.general_model._calculate_log_likelihood(smiles)
            # Calculate dynamic alpha
            alpha = self._calculate_dynamic_alpha(log_p_synthetic, log_p_general)
            log_mixture = self._log_stabilized_mixture(
                log_p_synthetic, log_p_general, alpha
            )
            return log_p_natural - log_mixture

        elif self.scoring_mode == "natural_only":
            return self.natural_model._calculate_log_likelihood(smiles)

        elif self.scoring_mode == "synthetic_only":
            return self.synthetic_model._calculate_log_likelihood(smiles)

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")

    @overload
    def score(
        self, smiles: str, raw: bool = ..., n_enum: int = ..., seed: int = ...
    ) -> float: ...

    @overload
    def score(
        self, smiles: list[str], raw: bool = ..., n_enum: int = ..., seed: int = ...
    ) -> list[float]: ...

    def score(
        self,
        smiles: str | list[str],
        raw: bool = False,
        n_enum: int = 0,
        seed: int = 0,
    ) -> float | list[float]:
        """Calculate the CLaM-NP Score for one or more SMILES strings.

        Pass a list of SMILES to score them in a single batched forward pass
        (much faster than scoring one at a time): a list input returns a list of
        scores, a single string returns a single score.

        Returns the sigmoid-normalized score in [0, 1] (0.5 = neutral, higher =
        more natural-product-like). Pass ``raw=True`` for the underlying raw
        log-likelihood ratio instead (unbounded; positive = more natural-like).

        ``n_enum`` > 0 averages the raw score over that many randomized SMILES
        (seeded by ``seed``) instead of scoring the canonical SMILES once.
        """
        if n_enum < 0:
            raise ValueError("n_enum must be >= 0")

        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        raw_scores = self._raw_scores(smiles_list, n_enum, seed)
        scores = raw_scores if raw else [self._sigmoid_normalize(s) for s in raw_scores]
        return scores[0] if single else scores

    def _raw_scores(
        self, smiles_list: list[str], n_enum: int, seed: int
    ) -> list[float]:
        """One raw score per molecule, averaged over ``n_enum`` writings if > 0."""
        if not n_enum:
            return self._batch_score_raw(smiles_list)
        # One flat batch over every writing of every molecule, then regrouped. The
        # mean is over raw scores, before the sigmoid, which would bias it to 0.5.
        writings = [
            w for s in smiles_list for w in self._enumerate_smiles(s, n_enum, seed)
        ]
        values = self._batch_score_processed(writings)
        return [
            sum(values[i : i + n_enum]) / n_enum for i in range(0, len(values), n_enum)
        ]

    def _batch_score_raw(self, smiles_list: list[str]) -> list[float]:
        """Raw NP-likeness scores for a batch of SMILES strings."""
        return self._batch_score_processed(
            [self._validate_and_canonicalize(smiles) for smiles in smiles_list]
        )

    def _batch_score_processed(self, smiles_list: list[str]) -> list[float]:
        """Raw scores for strings that are already in the form to be fed to the models.

        Separate from :meth:`_batch_score_raw` so that enumerated SMILES reach the
        models as written: canonicalising them first would collapse every
        enumeration of a molecule back onto one string.
        """

        if self.scoring_mode == "unstabilized":
            log_p_natural_list = self.natural_model.batch_calculate_log_likelihood(
                smiles_list
            )
            log_p_synthetic_list = self.synthetic_model.batch_calculate_log_likelihood(
                smiles_list
            )
            scores = [
                log_p_nat - log_p_syn
                for log_p_nat, log_p_syn in zip(
                    log_p_natural_list, log_p_synthetic_list
                )
            ]

        elif self.scoring_mode == "stabilized":
            log_p_natural_list = self.natural_model.batch_calculate_log_likelihood(
                smiles_list
            )
            log_p_synthetic_list = self.synthetic_model.batch_calculate_log_likelihood(
                smiles_list
            )
            log_p_general_list = self.general_model.batch_calculate_log_likelihood(
                smiles_list
            )
            scores = []
            for log_p_nat, log_p_syn, log_p_gen in zip(
                log_p_natural_list, log_p_synthetic_list, log_p_general_list
            ):
                # Calculate dynamic alpha for each sample
                alpha = self._calculate_dynamic_alpha(log_p_syn, log_p_gen)
                log_mixture = self._log_stabilized_mixture(log_p_syn, log_p_gen, alpha)
                scores.append(log_p_nat - log_mixture)

        elif self.scoring_mode == "natural_only":
            scores = self.natural_model.batch_calculate_log_likelihood(smiles_list)

        elif self.scoring_mode == "synthetic_only":
            scores = self.synthetic_model.batch_calculate_log_likelihood(smiles_list)

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")

        return scores

    def _sigmoid_normalize(self, llr: float) -> float:
        """Apply sigmoid normalization to a raw log-likelihood-ratio score.

        Args:
            llr: Raw score (log-likelihood ratio)

        Returns:
            Normalized score in range [0, 1]
        """
        adjusted_llr = llr + self.sigmoid_offset
        return 1.0 / (1.0 + math.exp(-self.sigmoid_k * adjusted_llr))

    def score_with_details(self, smiles: str) -> dict[str, float]:
        """Calculate NP-likeness score with a detailed breakdown.

        Returns dictionary with:
        - score: Raw NP-likeness score
        - score_normalized: Normalized score in range [0, 1]
        - log_p_natural: Log likelihood under natural model
        - log_p_synthetic: Log likelihood under synthetic model
        - perplexity_natural: Perplexity under natural model
        - perplexity_synthetic: Perplexity under synthetic model

        If general_model is provided:
        - log_p_general: Log likelihood under general model
        - perplexity_general: Perplexity under general model
        - log_mixture / dynamic_alpha: only when scoring_mode='stabilized'
        """
        smiles = self._validate_and_canonicalize(smiles)

        log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
        log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
        perp_natural = self.natural_model.calculate_perplexity(smiles)
        perp_synthetic = self.synthetic_model.calculate_perplexity(smiles)

        result = {
            "log_p_natural": log_p_natural,
            "log_p_synthetic": log_p_synthetic,
            "perplexity_natural": perp_natural,
            "perplexity_synthetic": perp_synthetic,
        }

        # Add general model details if available
        if self.general_model is not None:
            log_p_general = self.general_model._calculate_log_likelihood(smiles)
            perp_general = self.general_model.calculate_perplexity(smiles)
            result["log_p_general"] = log_p_general
            result["perplexity_general"] = perp_general

            if self.scoring_mode == "stabilized":
                alpha = self._calculate_dynamic_alpha(log_p_synthetic, log_p_general)
                log_mixture = self._log_stabilized_mixture(
                    log_p_synthetic, log_p_general, alpha
                )
                result["log_mixture"] = log_mixture
                result["dynamic_alpha"] = alpha

        # Calculate raw score using current scoring mode
        raw_score = self._score_raw(smiles)
        normalized_score = self._sigmoid_normalize(raw_score)

        result["score"] = raw_score
        result["score_normalized"] = normalized_score

        return result

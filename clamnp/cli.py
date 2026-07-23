"""Command-line interface for CLaM-NP scoring.

Examples::

    # Score a single molecule (models auto-downloaded from the HF Hub)
    clamnp "CC(=O)Oc1ccccc1C(=O)O"

    # Score several molecules at once
    clamnp "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "O=C(O)c1ccccc1O"

    # Score a file (one SMILES per line) and write a CSV
    clamnp --input molecules.smi --output scores.csv

    # Use the unstabilized (2-model) score instead of the default
    clamnp --mode unstabilized "CC(=O)Oc1ccccc1C(=O)O"
"""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from .constants import DEFAULT_HF_REPO, DEFAULT_SCORING_MODE


class Mode(str, Enum):
    stabilized = "stabilized"
    unstabilized = "unstabilized"
    natural_only = "natural_only"
    synthetic_only = "synthetic_only"


def _read_smiles_file(path: Path) -> list[str]:
    smiles: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept "SMILES" or "SMILES<whitespace>id" formats.
        smiles.append(line.split()[0])
    return smiles


def score(
    smiles: Optional[list[str]] = typer.Argument(
        None, help="One or more SMILES strings to score."
    ),
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        exists=True,
        dir_okay=False,
        help="File with one SMILES per line (overrides positional SMILES).",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write results as CSV instead of printing."
    ),
    mode: Mode = typer.Option(
        Mode(DEFAULT_SCORING_MODE), "--mode", help="Scoring mode."
    ),
    repo_id: str = typer.Option(
        DEFAULT_HF_REPO, "--repo-id", help="Hugging Face Hub repo hosting the models."
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device to run on (default: cuda if available, else cpu).",
    ),
    normalized: bool = typer.Option(
        False,
        "--normalized",
        help="Also report the sigmoid-normalized score in [0, 1].",
    ),
    scaffold_only: bool = typer.Option(
        False,
        "--scaffold-only",
        help="Score the Bemis-Murcko scaffold instead of the whole molecule.",
    ),
) -> None:
    """Score molecules for natural-product-likeness (CLaM-NP Score)."""
    smiles_list = _read_smiles_file(input) if input else list(smiles or [])
    if not smiles_list:
        raise typer.BadParameter("Provide at least one SMILES string or --input file.")

    # Imported here so `--help` stays instant and import errors are actionable.
    from .scorer import CLaMNPScorer

    typer.echo(f"Loading models from '{repo_id}' (mode={mode.value})...", err=True)
    scorer = CLaMNPScorer.from_pretrained(
        repo_id=repo_id,
        scoring_mode=mode.value,
        device=device,
        scaffold_only=scaffold_only,
    )

    raw_scores = scorer.batch_score(smiles_list)
    norm_scores = (
        [scorer._sigmoid_normalize(s) for s in raw_scores]
        if normalized
        else [None] * len(raw_scores)
    )

    if output:
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["smiles", "score"] + (["score_normalized"] if normalized else [])
            writer.writerow(header)
            for smi, raw, norm in zip(smiles_list, raw_scores, norm_scores):
                row = [smi, f"{raw:.6f}"] + ([f"{norm:.6f}"] if normalized else [])
                writer.writerow(row)
        typer.echo(f"Wrote {len(smiles_list)} scores to {output}", err=True)
    else:
        for smi, raw, norm in zip(smiles_list, raw_scores, norm_scores):
            if normalized:
                typer.echo(f"{raw:.4f}\t{norm:.4f}\t{smi}")
            else:
                typer.echo(f"{raw:.4f}\t{smi}")


def main() -> None:
    """Console-script entry point (`clamnp`)."""
    typer.run(score)


if __name__ == "__main__":
    main()

"""Analyze recipe-matched scratch against the reference checkpoint trajectory.

The launch matrix and W&B group do not exist while the experiment is Draft.
After launch, this script should be completed by replacing the explicit guard
below with the audited fetch/pair/bootstrap/plot pipeline. Results must come
from ``wandb.Api()`` and the immutable compiled matrices, never hardcoded
values.
"""

from __future__ import annotations

from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[1]
STEM = "20260918-MS-recipe-matched-scratch-control"
PROJECT = "neurosoft_supervised_pretraining"
PARENT_TRANSFER_MATRIX = (
    ROOT
    / "launch"
    / "architecture_transfer"
    / "reference-backbone-minipigs.jsonl"
)
SCRATCH_MATRIX = (
    ROOT
    / "launch"
    / "recipe_matched_scratch"
    / "recipe-matched-scratch-minipigs.jsonl"
)
CSV_DIR = ROOT / "analysis" / "csv"
FIGURE_DIR = ROOT / "analysis" / "figures"


def main() -> None:
    """Refuse analysis until the preregistered scratch inventory is compiled."""
    if not PARENT_TRANSFER_MATRIX.exists():
        raise FileNotFoundError(PARENT_TRANSFER_MATRIX)
    if not SCRATCH_MATRIX.exists():
        raise RuntimeError(
            "Experiment is still Draft: compile and audit the recipe-matched "
            f"scratch matrix at {SCRATCH_MATRIX} before completing analysis."
        )

    # Instantiate the public API here so the completed analysis cannot silently
    # substitute hand-entered endpoint values for W&B-derived results.
    wandb.Api()
    raise NotImplementedError(
        "Complete the audited run fetch, paired subject aggregation, five-way "
        "simultaneous bootstrap intervals, CSV outputs, and recentered main plot."
    )


if __name__ == "__main__":
    main()

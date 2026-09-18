"""Analyze monkey reference checkpoints against recipe-matched scratch.

This species-specific entry point reuses the audited analysis implementation
from the corresponding minipig control while replacing its immutable matrix,
sample-size, label, and artifact-stem constants. It fetches exact W&B run IDs
through ``wandb.Api()`` and does not import the main :mod:`foundry` package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    ROOT / "analysis" / "20260918-MS-recipe-matched-scratch-control_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "recipe_matched_scratch_analysis", IMPLEMENTATION
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        f"Could not load analysis implementation: {IMPLEMENTATION}"
    )
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

analysis.STEM = "20260918-MS-recipe-matched-scratch-monkeys"
analysis.SPECIES_LABEL = "monkey"
analysis.EXPECTED_TRANSFER_CELLS = 195
analysis.EXPECTED_SCRATCH_CELLS = 39
analysis.EXPECTED_SUBJECTS = 5
analysis.PARENT_MATRIX = (
    ROOT
    / "launch"
    / "architecture_transfer"
    / "reference-backbone-monkeys.jsonl"
)
analysis.SCRATCH_MATRIX = (
    ROOT
    / "launch"
    / "recipe_matched_scratch"
    / "recipe-matched-scratch-monkeys.jsonl"
)


if __name__ == "__main__":
    analysis.main()

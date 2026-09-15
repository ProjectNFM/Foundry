"""Unit tests for Phase 4E downstream learning-curve aggregation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE = (
    Path(__file__).parents[1]
    / "analysis"
    / "20260914-MS-validation-loss-checkpoint-transfer-learning-curves_analysis.py"
)
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("phase4e_figures", MODULE)
assert SPEC and SPEC.loader
phase4e_figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase4e_figures)


def test_stable_endpoint_requires_three_consecutive_smoothed_crossings() -> (
    None
):
    metric = phase4e_figures.VALIDATION_F1
    history = pd.DataFrame(
        {
            "trainer/global_step": [10, 20, 30, 40, 50, 60, 70],
            metric: [0.10, 0.20, 0.90, 0.91, 0.20, 1.00, 1.00],
        }
    )
    endpoint = phase4e_figures.stable_endpoint(history)
    assert endpoint == {"stable_step": 40.0, "censored": False}


def test_phase4e_units_average_source_seeds_before_pairing() -> None:
    rows = []
    for source_seed, f1, step in (
        (42, 0.4, 100),
        (43, 0.6, 200),
        (44, 0.5, 300),
    ):
        rows.append(
            {
                "phase": "phase4e",
                "condition": "transfer_validation_loss",
                "species": "minipigs",
                "recording_id": "sub-01_ses-01",
                "subject": "sub-01",
                "fraction": 0.05,
                "target_seed": 42,
                "source_seed": source_seed,
                "test_f1": f1,
                "stable_step": step,
                "censored": False,
            }
        )
    rows.append(
        {
            "phase": "phase4e",
            "condition": "scratch_matched",
            "species": "minipigs",
            "recording_id": "sub-01_ses-01",
            "subject": "sub-01",
            "fraction": 0.05,
            "target_seed": 42,
            "source_seed": None,
            "test_f1": 0.55,
            "stable_step": 180,
            "censored": False,
        }
    )
    recording, subject = phase4e_figures.phase4e_units(pd.DataFrame(rows))
    assert recording.loc[0, "f1_advantage"] == pytest.approx(-0.05)
    assert recording.loc[0, "steps_saved"] == -20
    assert subject.loc[0, "f1_advantage"] == pytest.approx(-0.05)

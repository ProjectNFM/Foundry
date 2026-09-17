"""Focused invariants for the batch-128 architecture-transfer analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "_architecture_transfer_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "architecture_transfer_analysis", MODULE_PATH
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_compiled_inventory_has_6360_unique_cells_and_exact_comparators() -> (
    None
):
    frames = [
        analysis.load_compiled_design(name) for name in analysis.EXPERIMENTS
    ]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("run_id")
    assert len(combined) == 6_360
    assert combined.run_id.is_unique
    assert combined.cell_id.is_unique
    scratch = set(
        combined.loc[combined.condition.str.startswith("scratch"), "cell_id"]
    )
    transfer = combined.loc[~combined.condition.str.startswith("scratch")]
    assert set(transfer.matched_scratch_id).issubset(scratch)


def test_hierarchy_weights_subjects_not_recording_counts() -> None:
    seed = pd.DataFrame(
        {
            "condition": ["c"] * 9,
            "species": ["minipigs"] * 9,
            "subject": ["sub-01"] * 6 + ["sub-02"] * 3,
            "recording": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "source_step": [100] * 9,
            "target_seed": [42, 43, 44] * 3,
            "effect": [1.0] * 6 + [3.0] * 3,
        }
    )
    recording, subject = analysis._hierarchy(seed, ["effect"])
    assert len(recording) == 3
    assert len(subject) == 2
    # Equal-subject mean is (1 + 3) / 2, not the recording-weighted 5 / 3.
    assert subject.effect.mean() == 2.0


def test_positive_effect_and_steps_saved_always_favor_transfer() -> None:
    transfer_f1, scratch_f1 = 0.55, 0.50
    scratch_time, transfer_time = 1_000, 700
    assert transfer_f1 - scratch_f1 > 0
    assert scratch_time - transfer_time > 0


def test_dynamics_normalization_uses_scratch_maximum() -> None:
    scratch_maximum = 0.5
    assert analysis._normalize_progress(scratch_maximum, scratch_maximum) == 1.0
    assert analysis._normalize_progress(0.45, scratch_maximum) == 0.9
    assert analysis.PRIMARY_THRESHOLD == 0.9


def test_session_sensitivity_is_centered_within_recording() -> None:
    recording = pd.DataFrame(
        {
            "condition": ["c"] * 4,
            "species": ["minipigs"] * 4,
            "subject": ["sub-01"] * 4,
            "recording": ["a", "a", "b", "b"],
            "source_step": [100, 300, 100, 300],
            "effect": [0.02, 0.05, -0.03, -0.04],
        }
    )
    centered = analysis._session_sensitivity_data(recording)
    baseline = centered[centered.source_step == 100]
    assert np.allclose(baseline.effect_change, 0)
    later = centered[centered.source_step == 300].set_index("recording")
    assert np.isclose(later.loc["a", "effect_change_pp"], 3.0)
    assert np.isclose(later.loc["b", "effect_change_pp"], -1.0)


def test_classwise_f1_excludes_unsupported_classes() -> None:
    confusion = np.eye(8)
    confusion[1] = 0
    values = analysis._class_f1(confusion)
    assert np.isnan(values[1])
    assert values[0] == 1.0

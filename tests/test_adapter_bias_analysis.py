from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE = (
    Path(__file__).parents[1]
    / "analysis"
    / "20260915-MS-adapter-bias-perturbation_analysis.py"
)
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("adapter_bias_analysis", MODULE)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_subject_slopes_aggregate_source_seeds_first() -> None:
    rows = []
    for species, subject_count in (("minipigs", 3), ("monkeys", 2)):
        for subject_index in range(subject_count):
            for step in analysis.FIXED_STEPS:
                rows.append(
                    {
                        "species": species,
                        "excluded_target_subject": f"sub-{subject_index}",
                        "checkpoint_kind": "milestone",
                        "checkpoint_step": step,
                        "checkpoint_global_step": step,
                        "delta_cross_entropy": 2 * np.log(step) + subject_index,
                    }
                )
    slopes = analysis.subject_slopes(pd.DataFrame(rows), "cross_entropy")
    assert np.allclose(slopes["slope"], 2.0)


def test_subject_slopes_accept_exploratory_three_checkpoint_subset() -> None:
    steps = (100, 1000, 10000)
    frame = pd.DataFrame(
        {
            "species": ["minipigs"] * 3,
            "excluded_target_subject": ["sub-01"] * 3,
            "checkpoint_kind": ["milestone"] * 3,
            "checkpoint_step": steps,
            "delta_cross_entropy": np.log(steps),
        }
    )
    slopes = analysis.subject_slopes(frame, "cross_entropy", steps)
    assert np.isclose(slopes.loc[0, "slope"], 1.0)


def test_bootstrap_resamples_subject_slopes_deterministically() -> None:
    slopes = pd.DataFrame(
        {
            "species": ["minipigs"] * 3,
            "metric": ["cross_entropy"] * 3,
            "slope": [1.0, 2.0, 3.0],
        }
    )
    summary_a, draws_a = analysis.bootstrap_species_slopes(slopes, 100, 7)
    summary_b, draws_b = analysis.bootstrap_species_slopes(slopes, 100, 7)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(draws_a, draws_b)

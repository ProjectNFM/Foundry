"""Analyze Phase 4C 500-step head-reset transfer after runs are complete.

This scaffold deliberately fetches W&B histories rather than embedding result
values.  Populate ``GROUPS`` and the compiled-cell paths when the launch recipe
is committed; the analysis then audits run identities and derives the
preregistered stable time-to-90%-of-peak endpoint from validation histories.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import wandb

from _wandb_utils import default_entity, fetch_metric_history


PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
PREFIX = "20260911-MS-500step-head-reset-transfer"
VALIDATION_F1 = f"val/{TASK}_supported_f1"
GROUPS: dict[str, dict[str, str]] = {
    "head_reset_full": {
        "minipigs": "PHASE4C_STEP500_HEAD_RESET_FULL_FT_MINIPIGS",
        "monkeys": "PHASE4C_STEP500_HEAD_RESET_FULL_FT_MONKEYS",
    },
    "frozen_representation": {
        "minipigs": "PHASE4C_STEP500_FROZEN_REPRESENTATION_MINIPIGS",
        "monkeys": "PHASE4C_STEP500_FROZEN_REPRESENTATION_MONKEYS",
    },
    "frozen_random_control": {
        "minipigs": "PHASE4C_FROZEN_RANDOM_MINIPIGS",
        "monkeys": "PHASE4C_FROZEN_RANDOM_MONKEYS",
    },
    # Reused Phase-4B/Phase-2 control groups are added only after their exact
    # compiled run identities are audited against the reused-cell registry.
}


def stable_time_to_90_percent_peak(
    history: pd.DataFrame,
) -> dict[str, float | bool]:
    """Return the preregistered stable time-to-near-peak endpoint.

    ``history`` must contain ``_step`` and the target validation metric, one
    record per epoch.  A three-validation trailing rolling median is used.  An
    event requires the current and next two smoothed values to be at least 90%
    of the run-specific smoothed peak.  A missing event is right-censored at
    the final observed validation step.
    """
    values = (
        history[["_step", VALIDATION_F1]].dropna().sort_values("_step").copy()
    )
    if len(values) < 3:
        raise ValueError("Need at least three validation evaluations")
    values["smoothed_f1"] = (
        values[VALIDATION_F1].rolling(3, min_periods=3).median()
    )
    values = values.dropna(subset=["smoothed_f1"]).reset_index(drop=True)
    threshold = 0.90 * float(values.smoothed_f1.max())
    stable = values.smoothed_f1.ge(threshold)
    crossings = (
        stable
        & stable.shift(-1, fill_value=False)
        & stable.shift(-2, fill_value=False)
    )
    if crossings.any():
        row = values.loc[crossings.idxmax()]
        return {
            "stable_step": float(row["_step"]),
            "censored": False,
            "peak_smoothed_f1": float(values.smoothed_f1.max()),
        }
    return {
        "stable_step": float(values.iloc[-1]["_step"]),
        "censored": True,
        "peak_smoothed_f1": float(values.smoothed_f1.max()),
    }


def fetch_validation_endpoint(
    run_id: str, *, api: Any | None = None
) -> dict[str, float | bool]:
    """Fetch one run's audited validation history and calculate its endpoint."""
    history = fetch_metric_history(
        run_id,
        VALIDATION_F1,
        PROJECT,
        default_entity(),
        x_axis="_step",
        api=api,
    )
    return stable_time_to_90_percent_peak(history)


def main() -> None:
    if not GROUPS:
        raise RuntimeError(
            "Populate GROUPS and compiled-cell provenance after committing the Phase-4C launch recipe."
        )
    api = wandb.Api()
    rows: list[dict[str, object]] = []
    for condition, species_groups in GROUPS.items():
        for species, group in species_groups.items():
            for run in api.runs(
                f"{default_entity()}/{PROJECT}", filters={"group": group}
            ):
                endpoint = fetch_validation_endpoint(str(run.id), api=api)
                rows.append(
                    {
                        "condition": condition,
                        "species": species,
                        "run_id": str(run.id),
                        **endpoint,
                    }
                )
    frame = pd.DataFrame(rows)
    print(
        frame.groupby(["condition", "species", "censored"], dropna=False).size()
    )


if __name__ == "__main__":
    main()

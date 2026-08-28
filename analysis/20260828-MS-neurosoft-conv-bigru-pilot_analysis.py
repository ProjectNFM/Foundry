"""Summarize the Phase-2 NeuroSoft convolution--BiGRU scratch pilot.

Fetches the declared WandB group, checks the eight production-semantic pilot
cells, and writes a compact run table and validation-learning-curve figure.

Usage:
    uv run python analysis/20260828-MS-neurosoft-conv-bigru-pilot_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import (
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

PREFIX = "20260828-MS-neurosoft-conv-bigru-pilot"
PROJECT = "neurosoft_supervised_pretraining"
GROUP = "PHASE2_CONV_BIGRU_PILOT"
TASK = "neurosoft_acoustic_stim_8band"
MONITOR = f"val/{TASK}_supported_f1"
PILOT_CELLS = {
    ("minipigs", "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw", 0.25, 42),
    ("minipigs", "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw", 1.00, 42),
    ("minipigs", "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw", 1.00, 43),
    ("minipigs", "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw", 1.00, 44),
    ("monkeys", "sub-01_ses-04_task-AcousStim_acq-RH_desc-raw", 0.25, 42),
    ("monkeys", "sub-01_ses-04_task-AcousStim_acq-RH_desc-raw", 1.00, 42),
    ("monkeys", "sub-01_ses-04_task-AcousStim_acq-RH_desc-raw", 1.00, 43),
    ("monkeys", "sub-01_ses-04_task-AcousStim_acq-RH_desc-raw", 1.00, 44),
}
RUN_COLUMNS = [
    "run_id",
    "run_name",
    "state",
    "species",
    "recording_id",
    "fraction",
    "seed",
    "best_val_supported_f1",
    "test_supported_f1",
    "best_step",
    "best_flops",
    "flop_method",
]


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def species_from_config(config: dict[str, Any]) -> str | None:
    root = str(nested(config, "data", "root") or "").lower()
    if "minipig" in root:
        return "minipigs"
    if "monkey" in root:
        return "monkeys"
    return None


def scalar(summary: dict[str, Any], key: str, summary_key: str = "last") -> Any:
    return unwrap_summary_value(summary.get(key), summary_key)


def collect_runs(
    entity: str,
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    histories: list[tuple[str, pd.DataFrame]] = []
    for run in api.runs(f"{entity}/{PROJECT}", filters={"group": GROUP}):
        config = dict(run.config)
        summary = dict(run.summary)
        record_ids = (
            nested(config, "data", "dataset_kwargs", "recording_ids") or []
        )
        recording_id = record_ids[0] if record_ids else None
        fraction = nested(config, "data", "training_fraction")
        seed = nested(config, "run", "seed")
        species = species_from_config(config)
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "species": species,
                "recording_id": recording_id,
                "fraction": fraction,
                "seed": seed,
                "best_val_supported_f1": scalar(summary, MONITOR, "max"),
                "test_supported_f1": scalar(
                    summary, f"test/{TASK}_supported_f1", "max"
                ),
                "best_step": scalar(summary, "compute/best_step"),
                "best_flops": scalar(summary, "compute/best_flops"),
                "flop_method": scalar(summary, "compute/flop_method"),
            }
        )
        history = run.history(keys=["epoch", MONITOR], pandas=True)
        histories.append((run.name, history.dropna(subset=[MONITOR])))
    return pd.DataFrame(rows, columns=RUN_COLUMNS), histories


def validate(rows: pd.DataFrame) -> list[str]:
    observed = set()
    for row in rows.itertuples():
        if row.species is None or row.recording_id is None:
            continue
        try:
            observed.add(
                (
                    row.species,
                    row.recording_id,
                    float(row.fraction),
                    int(row.seed),
                )
            )
        except (TypeError, ValueError):
            issues = [f"unparseable fraction or seed in run: {row.run_name}"]
            break
    else:
        issues = []
    missing = PILOT_CELLS - observed
    issues.extend(f"missing pilot cell: {cell}" for cell in sorted(missing))
    finished = rows[rows["state"] == "finished"]
    if len(finished) != len(PILOT_CELLS):
        issues.append(
            f"expected {len(PILOT_CELLS)} finished pilot runs, found {len(finished)}"
        )
    for field in (
        "best_val_supported_f1",
        "best_step",
        "best_flops",
        "flop_method",
    ):
        if rows[field].isna().any():
            issues.append(
                f"missing required field in at least one run: {field}"
            )
    return issues


def plot_histories(
    histories: list[tuple[str, pd.DataFrame]], output: Path
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, history in histories:
        if not history.empty:
            axis.plot(history["epoch"], history[MONITOR], label=name, alpha=0.8)
    axis.set(
        xlabel="Epoch",
        ylabel="Validation supported macro-F1",
        title="Phase-2 BiGRU pilot",
    )
    if histories:
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=default_entity())
    args = parser.parse_args()
    rows, histories = collect_runs(args.entity)
    output_csv = csv_dir(__file__) / f"{PREFIX}_runs.csv"
    output_figure = figures_dir(__file__) / f"{PREFIX}_validation_curves.png"
    rows.to_csv(output_csv, index=False)
    plot_histories(histories, output_figure)
    issues = validate(rows)
    print(rows.to_string(index=False))
    if issues:
        print("\nGATE ISSUES:")
        print("\n".join(f"- {issue}" for issue in issues))
    else:
        print(
            "\nAll declared pilot cells and required summary fields are present."
        )


if __name__ == "__main__":
    main()

"""Analyze the partial Phase-2 compact Conv--BiGRU capacity experiment.

The experiment was stopped after minipig seed 42 completed and while monkey
seed 42 was training.  This script deliberately reports that incomplete state
rather than treating either result as a three-seed comparison.

Usage:
    uv run python analysis/20260831-MS-neurosoft-conv-bigru-compact-capacity.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir, unwrap_summary_value


PREFIX = "20260831-MS-neurosoft-conv-bigru-compact-capacity"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
MONITOR = f"val/{TASK}_supported_f1"
TRAIN_F1 = f"train/{TASK}_supported_f1"
TEST_F1 = f"test/{TASK}_supported_f1"

# The sole completed minipig seed and the monkey run interrupted on request.
RUNS = (
    {
        "species": "minipig",
        "seed": 42,
        "run_id": "7fb7r2eo",
        "outcome": "completed",
    },
    {
        "species": "monkey",
        "seed": 42,
        "run_id": "tf6hl7wy",
        "outcome": "interrupted",
    },
)


def scalar(summary: dict[str, Any], metric: str, key: str = "max") -> Any:
    """Read WandB's flattened or SummarySubDict metric representation."""
    flat = f"{metric}.{key}"
    if flat in summary:
        return summary[flat]
    return unwrap_summary_value(summary.get(metric), key)


def collect(entity: str | None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Could not resolve WandB entity; set WANDB_ENTITY.")

    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    for declared in RUNS:
        run = api.run(f"{entity}/{PROJECT}/{declared['run_id']}")
        summary = dict(run.summary)
        # W&B can return an empty frame when a step-level training metric and
        # epoch-level validation metric are requested together.  Fetch and
        # merge them independently, matching the shared analysis convention.
        validation = run.history(
            keys=["epoch", MONITOR], samples=10_000, pandas=True
        )
        training = run.history(
            keys=["epoch", TRAIN_F1], samples=10_000, pandas=True
        )
        history = validation.merge(training, on="epoch", how="outer")
        histories[declared["run_id"]] = history
        rows.append(
            {
                **declared,
                "run_name": run.name,
                "wandb_state": run.state,
                "best_val_supported_f1": scalar(summary, MONITOR),
                "best_train_supported_f1": scalar(summary, TRAIN_F1),
                "test_supported_f1": scalar(summary, TEST_F1),
                "last_epoch": summary.get("compute/epoch"),
                "best_step": summary.get("compute/best_step"),
                "parameter_count": summary.get("compute/total_parameters"),
            }
        )
    return pd.DataFrame(rows), histories


def plot_histories(histories: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, declared in zip(axes, RUNS, strict=True):
        run_id = declared["run_id"]
        history = histories[run_id]
        if MONITOR in history:
            curve = history.dropna(subset=[MONITOR])
            axis.plot(curve["epoch"], curve[MONITOR], label="validation F1")
        if TRAIN_F1 in history:
            curve = history.dropna(subset=[TRAIN_F1])
            axis.plot(curve["epoch"], curve[TRAIN_F1], label="training F1")
        axis.set(
            title=f"{declared['species'].title()} seed {declared['seed']}\n"
            f"{declared['outcome']}",
            xlabel="Epoch",
            ylabel="Supported macro-F1",
        )
        if axis.lines:
            axis.legend(loc="best")
        axis.grid(alpha=0.25)
    fig.suptitle("Compact Conv--BiGRU partial capacity experiment")
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    rows, histories = collect(default_entity())
    csv_path = csv_dir(__file__) / f"{PREFIX}_partial_runs.csv"
    figure_path = figures_dir(__file__) / f"{PREFIX}_partial_learning_curves.png"
    rows.to_csv(csv_path, index=False)
    plot_histories(histories, figure_path)

    display = rows[
        [
            "species",
            "seed",
            "outcome",
            "wandb_state",
            "best_val_supported_f1",
            "best_train_supported_f1",
            "test_supported_f1",
            "last_epoch",
            "best_step",
            "run_id",
        ]
    ]
    print("Partial experiment summary (not a three-seed comparison):")
    print(display.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()

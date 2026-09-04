"""Analyze the three-seed NeuroSoft input-normalization replication.

Fetches the completed seed-42 screen and the Phase-3 seed-43/44 replication
from WandB, then writes run-level and three-seed summary CSVs plus an error-bar
comparison figure.

Usage:
    uv run python analysis/20260901-MS-neurosoft-input-normalization-replication_analysis.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir


PREFIX = "20260901-MS-neurosoft-input-normalization-replication"
PROJECT = "neurosoft_supervised_pretraining"
GROUPS = (
    "PHASE2_INPUT_NORMALIZATION_ABLATION",
    "PHASE3_INPUT_NORMALIZATION_REPLICATION",
)
TASK = "neurosoft_acoustic_stim_8band"
MONITOR = f"val/{TASK}_supported_f1"
SEEDS = {42, 43, 44}
NORMALIZATION_MODES = (
    "disabled",
    "recording_train_channel_zscore",
    "recording_train_global_zscore",
)
MODE_LABELS = {
    "disabled": "raw",
    "recording_train_channel_zscore": "train-channel z-score",
    "recording_train_global_zscore": "train-global z-score",
}


def nested_or_flat(config: dict[str, Any], *keys: str) -> Any:
    """Read a config key from nested or dotted W&B config data."""
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def model_label(config: dict[str, Any]) -> str:
    """Return the study's compact model label."""
    target = str(nested_or_flat(config, "model", "_target_") or "")
    if target.endswith("EEGNetEncoder"):
        return "EEGNet"
    if target.endswith((".GRU", ".NeurosoftConvBiGRU")):
        return "GRU"
    return target.rsplit(".", maxsplit=1)[-1] or "unknown"


def best_validation(history: pd.DataFrame) -> tuple[float, int]:
    """Return the best monitor value and its epoch from a run history."""
    curve = history.dropna(subset=[MONITOR]).copy()
    curve[MONITOR] = pd.to_numeric(curve[MONITOR], errors="coerce")
    curve = curve.dropna(subset=[MONITOR])
    if curve.empty:
        raise RuntimeError(f"No values logged for {MONITOR!r}.")
    best_index = curve[MONITOR].idxmax()
    return float(curve.loc[best_index, MONITOR]), int(
        curve.loc[best_index, "epoch"]
    )


def collect(entity: str | None) -> pd.DataFrame:
    """Fetch the 36 planned cells through the WandB API."""
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Could not resolve WandB entity; set WANDB_ENTITY.")

    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for group in GROUPS:
        for run in api.runs(f"{entity}/{PROJECT}", filters={"group": group}):
            if run.id in seen_run_ids:
                continue
            seen_run_ids.add(run.id)
            config = dict(run.config)
            normalization = nested_or_flat(
                config, "data", "input_normalization", "mode"
            )
            seed = nested_or_flat(config, "run", "seed")
            if normalization not in NORMALIZATION_MODES or seed not in SEEDS:
                continue
            history = run.history(keys=["epoch", MONITOR], samples=10_000)
            best_f1, best_epoch = best_validation(history)
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "wandb_group": group,
                    "state": run.state,
                    "species": nested_or_flat(config, "data", "audit_species"),
                    "model": model_label(config),
                    "normalization": normalization,
                    "seed": int(seed),
                    "best_val_supported_f1": best_f1,
                    "best_epoch": best_epoch,
                    "normalization_stats_sha256": dict(run.summary).get(
                        "input_normalization/stats_sha256"
                    ),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No input-normalization replication runs found.")
    duplicates = table.duplicated(
        subset=["species", "model", "normalization", "seed"], keep=False
    )
    if duplicates.any():
        raise RuntimeError(
            "Found duplicate runs for planned cells:\n"
            + table.loc[duplicates].to_string(index=False)
        )
    return table.sort_values(["species", "model", "normalization", "seed"])


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize F1 across the three fixed seeds for every condition."""
    return (
        table.groupby(["species", "model", "normalization"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_val_supported_f1=("best_val_supported_f1", "mean"),
            std_val_supported_f1=("best_val_supported_f1", "std"),
        )
        .sort_values(["species", "model", "normalization"])
    )


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    """Plot mean plus sample standard deviation for each comparison cell."""
    species = ("minipigs", "monkeys")
    models = ("EEGNet", "GRU")
    colors = {
        "disabled": "#777777",
        "recording_train_channel_zscore": "#0072B2",
        "recording_train_global_zscore": "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for row, species_name in enumerate(species):
        for column, model in enumerate(models):
            axis = axes[row, column]
            subset = (
                summary[
                    (summary.species == species_name) & (summary.model == model)
                ]
                .set_index("normalization")
                .reindex(NORMALIZATION_MODES)
            )
            x_values = range(len(NORMALIZATION_MODES))
            axis.bar(
                x_values,
                subset.mean_val_supported_f1,
                yerr=subset.std_val_supported_f1.fillna(0.0),
                capsize=5,
                color=[colors[mode] for mode in NORMALIZATION_MODES],
            )
            axis.set(
                title=f"{species_name.title()} — {model}",
                xticks=list(x_values),
                xticklabels=[MODE_LABELS[mode] for mode in NORMALIZATION_MODES],
                ylabel="Best validation supported macro-F1",
            )
            axis.tick_params(axis="x", labelrotation=18)
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("NeuroSoft input-normalization replication (mean ± sample SD)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    table = collect(default_entity())
    summary = summarize(table)
    csv_root = csv_dir(__file__)
    runs_csv = csv_root / f"{PREFIX}_runs.csv"
    summary_csv = csv_root / f"{PREFIX}_summary.csv"
    figure_path = figures_dir(__file__) / f"{PREFIX}_summary.png"
    table.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, figure_path)

    print("Three-seed validation supported macro-F1 summary:")
    print(summary.to_string(index=False))
    print(f"\nRun CSV: {runs_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()

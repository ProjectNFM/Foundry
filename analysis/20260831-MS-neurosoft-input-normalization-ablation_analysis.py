"""Analyze the NeuroSoft input-normalization validation ablation.

Fetches all supported normalization cells from the experiment group and
compares their validation supported macro-F1 histories.  The known Phase-1
run IDs are retained below; Phase-2 global-z-score runs are discovered by
their shared W&B group after they finish.

Usage:
    uv run python analysis/20260831-MS-neurosoft-input-normalization-ablation_analysis.py
"""

from __future__ import annotations

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
)


PREFIX = "20260831-MS-neurosoft-input-normalization-ablation"
PROJECT = "neurosoft_supervised_pretraining"
GROUP = "PHASE2_INPUT_NORMALIZATION_ABLATION"
TASK = "neurosoft_acoustic_stim_8band"
MONITOR = f"val/{TASK}_supported_f1"
PHASE1_RUN_IDS = (
    "1qwq1x7r",
    "lx7f8e1t",
    "xfo8t5yt",
    "l7gek5ps",
    "gvpixafp",
    "fu7dnhkn",
    "cra6xbbd",
    "677tch9g",
)
NORMALIZATION_MODES = {
    "disabled",
    "recording_train_global_zscore",
    "recording_train_channel_zscore",
}


def nested_or_flat(config: dict[str, Any], *keys: str) -> Any:
    """Read a value from either W&B's nested or dotted config representation."""
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def best_validation(history: pd.DataFrame) -> tuple[float, int]:
    """Return best monitor value and its epoch from the exact run history."""
    curve = history.dropna(subset=[MONITOR]).copy()
    if curve.empty:
        raise RuntimeError(f"No values logged for {MONITOR!r}.")
    curve[MONITOR] = pd.to_numeric(curve[MONITOR], errors="coerce")
    curve = curve.dropna(subset=[MONITOR])
    best_index = curve[MONITOR].idxmax()
    epoch = curve.loc[best_index, "epoch"]
    return float(curve.loc[best_index, MONITOR]), int(epoch)


def model_label(config: dict[str, Any]) -> str:
    target = str(nested_or_flat(config, "model", "_target_") or "")
    if target.endswith("EEGNetEncoder"):
        return "EEGNet"
    if target.endswith((".GRU", ".NeurosoftConvBiGRU")):
        return "GRU"
    return target.rsplit(".", maxsplit=1)[-1] or "unknown"


def collect(entity: str | None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Collect group summaries and validation histories through the W&B API."""
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Could not resolve WandB entity; set WANDB_ENTITY.")

    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    runs = list(api.runs(f"{entity}/{PROJECT}", filters={"group": GROUP}))
    if not runs:
        raise RuntimeError(f"No WandB runs found for group {GROUP!r}.")

    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        normalization = nested_or_flat(
            config, "data", "input_normalization", "mode"
        )
        if normalization not in NORMALIZATION_MODES:
            continue
        history = run.history(
            keys=["epoch", MONITOR], samples=10_000, pandas=True
        )
        best_f1, best_epoch = best_validation(history)
        histories[run.id] = history
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "phase": "phase_1" if run.id in PHASE1_RUN_IDS else "phase_2",
                "model": model_label(config),
                "species": nested_or_flat(config, "data", "audit_species"),
                "normalization": normalization,
                "seed": nested_or_flat(config, "run", "seed"),
                "best_val_supported_f1": best_f1,
                "best_epoch": best_epoch,
                "summary_best_monitor_value": summary.get(
                    "compute/best_monitor_value"
                ),
                "normalization_stats_sha256": summary.get(
                    "input_normalization/stats_sha256"
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(
            "The group contained no normalization-ablation runs."
        )
    table = table.sort_values(["species", "model", "normalization", "run_id"])
    raw = table[table.normalization == "disabled"][
        ["species", "model", "best_val_supported_f1"]
    ].rename(columns={"best_val_supported_f1": "raw_best_val_supported_f1"})
    table = table.merge(raw, on=["species", "model"], how="left")
    table["delta_vs_raw"] = (
        table.best_val_supported_f1 - table.raw_best_val_supported_f1
    )
    return table, histories


def plot_histories(
    table: pd.DataFrame, histories: dict[str, pd.DataFrame], output: Path
) -> None:
    """Plot raw and normalized validation curves for each species/model pair."""
    species = [
        value
        for value in ("minipigs", "monkeys")
        if value in set(table.species)
    ]
    models = [value for value in ("EEGNet", "GRU") if value in set(table.model)]
    fig, axes = plt.subplots(
        len(species),
        len(models),
        figsize=(6 * len(models), 4.5 * len(species)),
        squeeze=False,
    )
    styles = {
        "disabled": {"label": "raw", "color": "#777777"},
        "recording_train_channel_zscore": {
            "label": "train-channel z-score",
            "color": "#0072B2",
        },
        "recording_train_global_zscore": {
            "label": "train-global z-score",
            "color": "#D55E00",
        },
    }
    for species_index, species_name in enumerate(species):
        for model_index, model in enumerate(models):
            axis = axes[species_index, model_index]
            subset = table[
                (table.species == species_name) & (table.model == model)
            ]
            for _, row in subset.iterrows():
                history = histories[row.run_id]
                if MONITOR not in history:
                    continue
                curve = history.dropna(subset=[MONITOR])
                style = styles[row.normalization]
                axis.plot(
                    curve["epoch"],
                    curve[MONITOR],
                    label=style["label"],
                    color=style["color"],
                )
            axis.set(
                title=f"{species_name.title()} — {model}",
                xlabel="Epoch",
                ylabel="Validation supported macro-F1",
            )
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
    fig.suptitle("NeuroSoft input-normalization ablation (single seed)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)


def main() -> None:
    table, histories = collect(default_entity())
    csv_path = csv_dir(__file__) / f"{PREFIX}_runs.csv"
    figure_path = figures_dir(__file__) / f"{PREFIX}_validation_curves.png"
    table.to_csv(csv_path, index=False)
    plot_histories(table, histories, figure_path)

    print("Validation supported macro-F1 (single-seed screen):")
    print(
        table[
            [
                "species",
                "model",
                "normalization",
                "best_val_supported_f1",
                "delta_vs_raw",
                "best_epoch",
                "run_id",
            ]
        ].to_string(index=False)
    )
    print(f"\nCSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()

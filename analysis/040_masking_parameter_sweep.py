"""Analyze the leak-fixed masking-parameter sweep.

Fetches the five pretraining runs in ``MASKING_SEQLEN_LEAK_FIXED`` and their
three-task downstream evaluation from Weights & Biases.  Aggregate F1 values
use only runs whose W&B state is ``finished``; the printed coverage table keeps
incomplete and failed folds visible rather than treating them as zero scores.

Usage:
    uv run python analysis/040_masking_parameter_sweep.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PRETRAIN_PROJECT = "foundry_pretraining"
DOWNSTREAM_PROJECT = "foundry_finetuning"
PRETRAIN_GROUP = "MASKING_SEQLEN_LEAK_FIXED"

# The sweep is identified by its stable W&B run names and group.  Resolved W&B
# run IDs are emitted into the per-fold CSV on every analysis run.
PRETRAIN_RUNS: dict[str, dict[str, Any]] = {
    "M0": {
        "name": "pretrain_M0_baseline_leak_fixed",
        "label": "M0\nratio 0.5, block 10",
        "mask_ratio": 0.5,
        "block_size": 10,
    },
    "M1": {
        "name": "pretrain_M1_ratio70_leak_fixed",
        "label": "M1\nratio 0.7, block 10",
        "mask_ratio": 0.7,
        "block_size": 10,
    },
    "M2": {
        "name": "pretrain_M2_ratio80_leak_fixed",
        "label": "M2\nratio 0.8, block 10",
        "mask_ratio": 0.8,
        "block_size": 10,
    },
    "M3": {
        "name": "pretrain_M3_ratio90_leak_fixed",
        "label": "M3\nratio 0.9, block 10",
        "mask_ratio": 0.9,
        "block_size": 10,
    },
    "M4": {
        "name": "pretrain_M4_block20_leak_fixed",
        "label": "M4\nratio 0.5, block 20",
        "mask_ratio": 0.5,
        "block_size": 20,
    },
}
RUN_ORDER = list(PRETRAIN_RUNS)
COLORS = {
    "M0": "#4C78A8",
    "M1": "#59A14F",
    "M2": "#F28E2B",
    "M3": "#E15759",
    "M4": "#B279A2",
}

DOWNSTREAM_GROUPS = {
    ("Kemp Sleep", "Finetune"): "KEMP_FT_DATA_SCALING",
    ("Kemp Sleep", "Linear probe"): "KEMP_LP_DATA_SCALING",
    ("PhysioNet MI", "Finetune"): "PHYSIONET_FT_DATA_SCALING",
    ("PhysioNet MI", "Linear probe"): "PHYSIONET_LP_DATA_SCALING",
    ("Brain Invaders P300", "Finetune"): "BI_P300_FT_DATA_SCALING",
    ("Brain Invaders P300", "Linear probe"): "BI_P300_LP_DATA_SCALING",
}
METRIC_KEYS = {
    "Kemp Sleep": "val/sleep_stage_5class_f1",
    "PhysioNet MI": "val/motor_imagery_binary_f1",
    "Brain Invaders P300": "val/p300_binary_f1",
}
TASKS = list(METRIC_KEYS)
MODES = ["Finetune", "Linear probe"]


def fold_from_name(name: str) -> int | None:
    match = re.search(r"fold(\d+)", name)
    return int(match.group(1)) if match else None


def sweep_id_from_name(name: str) -> str | None:
    for sweep_id, info in PRETRAIN_RUNS.items():
        if info["name"] in name:
            return sweep_id
    return None


def fetch_pretraining(api: wandb.Api) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long loss curves and one row of metadata for each checkpoint."""
    curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    entity = api.default_entity
    for sweep_id, info in PRETRAIN_RUNS.items():
        matches = list(
            api.runs(
                f"{entity}/{PRETRAIN_PROJECT}",
                filters={"group": PRETRAIN_GROUP, "display_name": info["name"]},
            )
        )
        if not matches:
            print(f"WARNING: pretraining run not found: {info['name']}")
            continue
        run = matches[0]
        # W&B may store these metrics on disjoint logging steps, so fetch them
        # separately instead of relying on an inner join in one history call.
        train_history = run.history(keys=["train/loss"], samples=50_000, pandas=True)
        val_history = run.history(keys=["val/loss"], samples=50_000, pandas=True)
        for _, row in train_history.iterrows():
            curves.append(
                {
                    "sweep_id": sweep_id,
                    "step": row.get("_step"),
                    "train_loss": row.get("train/loss"),
                    "val_loss": np.nan,
                }
            )
        for _, row in val_history.iterrows():
            curves.append(
                {
                    "sweep_id": sweep_id,
                    "step": row.get("_step"),
                    "train_loss": np.nan,
                    "val_loss": row.get("val/loss"),
                }
            )
        values = val_history.get("val/loss", pd.Series(dtype=float)).dropna()
        summaries.append(
            {
                "sweep_id": sweep_id,
                "run_name": info["name"],
                "wandb_run_id": run.id,
                "state": run.state,
                "mask_ratio": info["mask_ratio"],
                "block_size": info["block_size"],
                "steps": run.summary.get("_step"),
                "best_val_loss": values.min() if not values.empty else np.nan,
                "final_val_loss": values.iloc[-1] if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(curves), pd.DataFrame(summaries)


def fetch_downstream(api: wandb.Api) -> pd.DataFrame:
    """Fetch every expected downstream run, including failed runs for coverage."""
    records: list[dict[str, Any]] = []
    entity = api.default_entity
    for (task, mode), group in DOWNSTREAM_GROUPS.items():
        metric_key = METRIC_KEYS[task]
        matches = list(api.runs(f"{entity}/{DOWNSTREAM_PROJECT}", filters={"group": group}))
        selected = [run for run in matches if sweep_id_from_name(run.name)]
        print(f"{task} / {mode}: {len(selected)} sweep runs found")
        for run in selected:
            history = run.history(keys=[metric_key], samples=50_000, pandas=True)
            values = history.get(metric_key, pd.Series(dtype=float)).dropna()
            records.append(
                {
                    "task": task,
                    "mode": mode,
                    "sweep_id": sweep_id_from_name(run.name),
                    "fold": fold_from_name(run.name),
                    "run_name": run.name,
                    "wandb_run_id": run.id,
                    "state": run.state,
                    "last_step": run.summary.get("_step"),
                    "best_f1": values.max() if not values.empty else np.nan,
                    "num_metric_points": len(values),
                }
            )
    return pd.DataFrame(records)


def summarize_finished(downstream: pd.DataFrame) -> pd.DataFrame:
    """Summarize best F1 by condition using completed folds only."""
    completed = downstream[(downstream["state"] == "finished") & downstream["best_f1"].notna()].copy()
    summary = (
        completed.groupby(["task", "mode", "sweep_id"])["best_f1"]
        .agg(mean="mean", std="std", n_finished="count")
        .reset_index()
    )
    coverage = (
        downstream.groupby(["task", "mode", "sweep_id"])
        .agg(n_found=("run_name", "count"), n_finished=("state", lambda x: (x == "finished").sum()))
        .reset_index()
    )
    summary = coverage.merge(
        summary.drop(columns="n_finished"), on=["task", "mode", "sweep_id"], how="left"
    )
    return summary


def plot_pretraining_curves(curves: pd.DataFrame) -> Path | None:
    if curves.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for sweep_id in RUN_ORDER:
        subset = curves[curves["sweep_id"] == sweep_id].sort_values("step")
        if subset.empty:
            continue
        label = PRETRAIN_RUNS[sweep_id]["label"].replace("\n", ": ")
        train_subset = subset.dropna(subset=["step", "train_loss"])
        val_subset = subset.dropna(subset=["step", "val_loss"])
        axes[0].plot(
            train_subset["step"], train_subset["train_loss"], label=label, color=COLORS[sweep_id]
        )
        axes[1].plot(
            val_subset["step"], val_subset["val_loss"], label=label, color=COLORS[sweep_id], marker="o", markersize=3
        )
    for ax, title, ylabel in zip(axes, ["Training loss", "Validation loss"], ["MAE reconstruction loss", "MAE reconstruction loss"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Pretraining step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle("Leak-fixed masking sweep: pretraining reconstruction loss", fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "040_masking_sweep_pretraining_loss.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_downstream_summary(summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    for row, mode in enumerate(MODES):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            subset = summary[(summary["task"] == task) & (summary["mode"] == mode)].set_index("sweep_id")
            means = [subset.loc[s, "mean"] if s in subset.index else np.nan for s in RUN_ORDER]
            stds = [subset.loc[s, "std"] if s in subset.index else np.nan for s in RUN_ORDER]
            coverage = [subset.loc[s, "n_finished"] if s in subset.index else 0 for s in RUN_ORDER]
            bars = ax.bar(np.arange(len(RUN_ORDER)), means, color=[COLORS[s] for s in RUN_ORDER], yerr=np.nan_to_num(stds, nan=0), capsize=3)
            for bar, value, n in zip(bars, means, coverage):
                if pd.notna(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "no finished\nfold", ha="center", va="bottom", fontsize=7)
            ax.set_title(task, fontweight="bold")
            ax.set_xticks(np.arange(len(RUN_ORDER)), RUN_ORDER)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{mode}\nbest validation F1")
    fig.suptitle("Downstream transfer by masking condition (finished folds only)", fontweight="bold")
    fig.text(0.5, 0.01, "Error bars = sample standard deviation; n = completed folds contributing to each mean", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = FIGURES_DIR / "040_masking_sweep_downstream_f1.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_from_m0(summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    for row, mode in enumerate(MODES):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            subset = summary[(summary["task"] == task) & (summary["mode"] == mode)].set_index("sweep_id")
            if "M0" not in subset.index or pd.isna(subset.loc["M0", "mean"]):
                ax.text(0.5, 0.5, "No completed M0 fold\nfor this comparison", transform=ax.transAxes, ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_title(task, fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    ax.set_ylabel(f"{mode}\nΔ best validation F1 vs M0")
                continue
            base = subset.loc["M0", "mean"]
            ids = [s for s in RUN_ORDER if s != "M0" and s in subset.index]
            deltas = [subset.loc[s, "mean"] - base for s in ids]
            bars = ax.bar(ids, deltas, color=[COLORS[s] for s in ids])
            ax.axhline(0, color="black", linewidth=0.9)
            for bar, delta in zip(bars, deltas):
                ax.text(bar.get_x() + bar.get_width() / 2, delta + (0.004 if delta >= 0 else -0.004), f"{delta:+.3f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=8)
            ax.set_title(task, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{mode}\nΔ best validation F1 vs M0")
    fig.suptitle("Transfer change relative to M0 (means from finished folds only)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURES_DIR / "040_masking_sweep_delta_vs_m0.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def print_tables(pretrain: pd.DataFrame, summary: pd.DataFrame, downstream: pd.DataFrame) -> None:
    print("\nPRETRAINING SUMMARY")
    print(pretrain.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDOWNSTREAM F1 SUMMARY — FINISHED FOLDS ONLY")
    display = summary.copy()
    display["f1_mean_std"] = display.apply(lambda r: "—" if pd.isna(r.get("mean")) else f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r['std']) else f"{r['mean']:.4f} (one fold)", axis=1)
    print(display[["task", "mode", "sweep_id", "f1_mean_std", "n_finished", "n_found"]].to_string(index=False))
    print("\nRUN-STATE COVERAGE")
    coverage = pd.crosstab(downstream["state"], downstream["sweep_id"]).reindex(columns=RUN_ORDER, fill_value=0)
    print(coverage.to_string())
    states = downstream[downstream["state"] != "finished"]
    print(f"\nNon-finished runs excluded from F1 aggregates: {len(states)} / {len(downstream)}")
    if not states.empty:
        print(states[["task", "mode", "sweep_id", "fold", "state", "wandb_run_id", "last_step"]].sort_values(["task", "mode", "sweep_id", "fold"]).to_string(index=False))


def main() -> None:
    api = wandb.Api()
    print(f"W&B entity: {api.default_entity}")
    curves, pretrain = fetch_pretraining(api)
    downstream = fetch_downstream(api)
    if downstream.empty:
        raise RuntimeError("No downstream sweep runs were found.")
    summary = summarize_finished(downstream)
    curves.to_csv(RESULTS_DIR / "040_masking_sweep_pretraining_curves.csv", index=False)
    pretrain.to_csv(RESULTS_DIR / "040_masking_sweep_pretraining.csv", index=False)
    downstream.to_csv(RESULTS_DIR / "040_masking_sweep_per_fold.csv", index=False)
    summary.to_csv(RESULTS_DIR / "040_masking_sweep_summary.csv", index=False)
    print_tables(pretrain, summary, downstream)
    figures = [
        plot_pretraining_curves(curves),
        plot_downstream_summary(summary),
        plot_delta_from_m0(summary),
    ]
    print("\nGENERATED FIGURES")
    for figure in figures:
        if figure:
            print(figure)


if __name__ == "__main__":
    main()

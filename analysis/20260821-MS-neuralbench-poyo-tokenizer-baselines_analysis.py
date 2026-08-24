"""Summarize NeuralBench POYO tokenizer baselines against matched EEGNet.

Fetches runs from WandB, reports per-seed and mean±SD metrics, and writes
comparison tables and figures for three tasks:

- **P300 & Motor Imagery:** held-out *test* balanced accuracy (all runs
  finished and ran test evaluation).
- **Sleep Stage:** best *validation* balanced accuracy only — POYO runs
  timed out before test evaluation due to the quadratic latent-sequence
  cost documented in ``docs/neuralbench-poyo-sleep-profiling.md``.

Run after all POYO and parent EEGNet jobs have completed (or timed out):

    uv run python analysis/20260821-MS-neuralbench-poyo-tokenizer-baselines_analysis.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

STEM = Path(__file__).stem
SEEDS = (33, 34, 35)
PROJECT = "foundry-neuralbench"
ENTITY = default_entity()
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)

TASKS = {
    "P300": {
        "task_key": "neuralbench_p300",
        "poyo_group": "NB_P300_POYO_TOKENIZER_BASELINES",
        "eegnet_group": "NB_P300_EEGNET_MATCHED",
        "metric_split": "test",
    },
    "Motor Imagery": {
        "task_key": "neuralbench_motor_imagery",
        "poyo_group": "NB_MI_POYO_TOKENIZER_BASELINES",
        "eegnet_group": "NB_MI_EEGNET_MATCHED",
        "metric_split": "test",
    },
    "Sleep Stage": {
        "task_key": "neuralbench_sleep_stage",
        "poyo_group": "NB_SLEEP_POYO_TOKENIZER_BASELINES",
        "eegnet_group": "NB_SLEEP_EEGNET_MATCHED",
        "metric_split": "val",
    },
}

CORE_METRICS = ("balanced_acc", "f1", "auroc", "acc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(run: wandb.apis.public.Run) -> int:
    config = run.config or {}
    value = config.get("seed") or config.get("run", {}).get("seed")
    if value is None:
        match = re.search(r"seed(\d+)", run.name or "")
        if match is None:
            raise RuntimeError(f"Could not infer seed for run {run.id}")
        value = match.group(1)
    return int(value)


def _tokenizer_label(run: wandb.apis.public.Run) -> str:
    text = " ".join([run.name or "", *map(str, run.tags or [])]).lower()
    if "cwt" in text:
        return "CWT-CNN"
    if "resample" in text or "rcnn" in text:
        return "ResampleCNN"
    raise RuntimeError(
        f"Could not infer tokenizer for run {run.id}: {run.name}"
    )


def _extract_metrics(
    summary: dict,
    task_key: str,
    split: str,
    metrics: tuple[str, ...] = CORE_METRICS,
) -> dict[str, float | None]:
    """Pull metric values from a run summary.

    WandB stores summary metrics with a ``.max`` suffix for accuracy-like
    metrics (e.g. ``test/neuralbench_p300_balanced_acc.max``).
    """
    result: dict[str, float | None] = {}
    for metric in metrics:
        prefix = f"{split}/{task_key}_{metric}"
        val = summary.get(f"{prefix}.max")
        if val is None:
            val = summary.get(prefix)
        if isinstance(val, dict):
            val = val.get("max", val.get("last"))
        try:
            result[metric] = float(val) if val is not None else None
        except (TypeError, ValueError):
            result[metric] = None
    return result


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_poyo_group(
    api: wandb.Api,
    group: str,
    task_key: str,
    split: str,
) -> pd.DataFrame:
    """Fetch POYO tokenizer runs.  Accepts *any* terminal state for val-only tasks."""
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    rows: list[dict] = []
    for run in api.runs(path, filters={"group": group}):
        terminal = run.state in ("finished", "failed", "crashed")
        if split == "test" and run.state != "finished":
            print(f"  [skip] {run.name} ({run.state}) — need finished for test metrics")
            continue
        if not terminal:
            print(f"  [skip] {run.name} ({run.state}) — still running")
            continue
        metrics = _extract_metrics(run.summary, task_key, split)
        if all(v is None for v in metrics.values()):
            print(f"  [skip] {run.name} — no {split} metrics in summary")
            continue
        row: dict = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": _seed(run),
            "condition": _tokenizer_label(run),
            "state": run.state,
            "last_epoch": run.summary.get("epoch", None),
            **metrics,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No usable runs in WandB group {group}")
    return frame.sort_values(["condition", "seed"]).reset_index(drop=True)


def fetch_eegnet_group(
    api: wandb.Api,
    group: str,
    task_key: str,
    split: str,
) -> pd.DataFrame:
    """Fetch matched EEGNet runs (all finished for test; accept any terminal for val)."""
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    rows: list[dict] = []
    for run in api.runs(path, filters={"group": group}):
        terminal = run.state in ("finished", "failed", "crashed")
        if split == "test" and run.state != "finished":
            continue
        if not terminal:
            continue
        metrics = _extract_metrics(run.summary, task_key, split)
        if all(v is None for v in metrics.values()):
            continue
        row: dict = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": _seed(run),
            "condition": "Matched EEGNet",
            "state": run.state,
            "last_epoch": run.summary.get("epoch", None),
            **metrics,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No usable runs in WandB group {group}")
    return frame.sort_values("seed").reset_index(drop=True)


def fetch_val_training_curves(
    api: wandb.Api,
    runs_df: pd.DataFrame,
    task_key: str,
) -> pd.DataFrame:
    """Fetch epoch-level validation balanced accuracy for training curves."""
    path_prefix = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    val_key = f"val/{task_key}_balanced_acc"
    frames: list[pd.DataFrame] = []
    for _, row in runs_df.iterrows():
        try:
            run = api.run(f"{path_prefix}/{row['run_id']}")
        except Exception:
            continue
        history = run.history(keys=["epoch", val_key], samples=10_000, pandas=True)
        if history.empty or val_key not in history.columns:
            continue
        history = history.dropna(subset=[val_key])
        if history.empty:
            continue
        history = history.rename(columns={val_key: "val_balanced_acc"})
        history["seed"] = row["seed"]
        history["condition"] = row["condition"]
        history["run_id"] = row["run_id"]
        frames.append(history[["epoch", "val_balanced_acc", "seed", "condition", "run_id"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["condition", "seed", "epoch"])


# ---------------------------------------------------------------------------
# Summary and printing
# ---------------------------------------------------------------------------

def make_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        task_df.groupby("condition")[list(CORE_METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(filter(None, c)).rstrip("_") for c in agg.columns]
    return agg


def print_task_results(label: str, task_df: pd.DataFrame, split: str) -> str:
    lines: list[str] = []
    split_label = "test" if split == "test" else "best validation"
    lines.append(f"\n{'=' * 72}")
    lines.append(f"{label} — per-seed {split_label} metrics")
    lines.append("=" * 72)

    display_cols = ["condition", "seed", "state", "last_epoch"] + list(CORE_METRICS)
    present = [c for c in display_cols if c in task_df.columns]
    lines.append(
        task_df[present].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    lines.append(f"\n{label} — mean ± SD ({split_label})")
    lines.append("-" * 60)
    summary = make_summary(task_df)
    for _, row in summary.iterrows():
        parts = [f"  {row['condition']:>15s}:"]
        for metric in CORE_METRICS:
            m_key = f"{metric}_mean"
            s_key = f"{metric}_std"
            n_key = f"{metric}_count"
            if m_key in row and pd.notna(row[m_key]):
                parts.append(f"  {metric}={row[m_key]:.4f}±{row[s_key]:.4f} (n={int(row[n_key])})")
        lines.append(" ".join(parts))

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {
    "Matched EEGNet": "#4c78a8",
    "CWT-CNN": "#e45756",
    "ResampleCNN": "#72b7b2",
}
CONDITION_ORDER = ["Matched EEGNet", "CWT-CNN", "ResampleCNN"]


def plot_test_balanced_accuracy(all_results: dict[str, pd.DataFrame]) -> Path:
    """Bar chart of test balanced accuracy for P300 and MI only."""
    test_tasks = {k: v for k, v in all_results.items() if k != "Sleep Stage"}
    fig, axes = plt.subplots(1, len(test_tasks), figsize=(5 * len(test_tasks), 5), sharey=True)
    if len(test_tasks) == 1:
        axes = [axes]

    for ax, (task, df) in zip(axes, test_tasks.items()):
        summary = make_summary(df)
        means, stds, labels = [], [], []
        for cond in CONDITION_ORDER:
            row = summary[summary["condition"] == cond]
            if row.empty:
                continue
            means.append(float(row["balanced_acc_mean"].iloc[0]))
            stds.append(float(row["balanced_acc_std"].iloc[0]))
            labels.append(cond)
        x = np.arange(len(labels))
        bars = ax.bar(
            x, means, yerr=stds, capsize=4,
            color=[COLORS.get(l, "#999") for l in labels],
            edgecolor="white", width=0.6,
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(task, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test balanced accuracy (mean ± SD, 3 seeds)")
    fig.suptitle("POYO Tokenizer Baselines vs Matched EEGNet — Test", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_test_balanced_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_val_balanced_accuracy_sleep(sleep_df: pd.DataFrame) -> Path:
    """Bar chart of best validation balanced accuracy for Sleep Stage."""
    fig, ax = plt.subplots(figsize=(5, 5))
    summary = make_summary(sleep_df)
    means, stds, labels = [], [], []
    for cond in CONDITION_ORDER:
        row = summary[summary["condition"] == cond]
        if row.empty:
            continue
        means.append(float(row["balanced_acc_mean"].iloc[0]))
        stds.append(float(row["balanced_acc_std"].iloc[0]))
        labels.append(cond)
    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=[COLORS.get(l, "#999") for l in labels],
        edgecolor="white", width=0.6,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title("Sleep Stage", fontsize=12)
    ax.set_ylabel("Best validation balanced accuracy\n(mean ± SD, 3 seeds)")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "POYO Tokenizer Baselines vs Matched EEGNet — Val only\n"
        "(POYO runs timed out; no test evaluation)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_sleep_val_balanced_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_tasks_overview(all_results: dict[str, pd.DataFrame]) -> Path:
    """Three-panel bar chart: all tasks side by side.

    P300 and MI show test metrics; Sleep shows val metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, (task, df) in zip(axes, all_results.items()):
        split_label = "val" if task == "Sleep Stage" else "test"
        summary = make_summary(df)
        means, stds, labels = [], [], []
        for cond in CONDITION_ORDER:
            row = summary[summary["condition"] == cond]
            if row.empty:
                continue
            means.append(float(row["balanced_acc_mean"].iloc[0]))
            stds.append(float(row["balanced_acc_std"].iloc[0]))
            labels.append(cond)
        x = np.arange(len(labels))
        bars = ax.bar(
            x, means, yerr=stds, capsize=4,
            color=[COLORS.get(l, "#999") for l in labels],
            edgecolor="white", width=0.6,
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        suffix = " (val*)" if split_label == "val" else ""
        ax.set_title(f"{task}{suffix}", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Balanced accuracy (mean ± SD, 3 seeds)")
    fig.suptitle(
        "POYO Tokenizer Baselines vs Matched EEGNet\n"
        "P300 & MI = test; *Sleep = best validation (POYO timed out)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_all_tasks_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(
    curves: dict[str, pd.DataFrame],
) -> Path:
    """Epoch-level val balanced accuracy curves for all tasks."""
    n_tasks = len(curves)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5.5 * n_tasks, 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, (task, df) in zip(axes, curves.items()):
        if df.empty:
            ax.set_title(f"{task} — no curve data")
            continue
        for cond in CONDITION_ORDER:
            for seed in SEEDS:
                sub = df[(df["condition"] == cond) & (df["seed"] == seed)].sort_values("epoch")
                if sub.empty:
                    continue
                ax.plot(
                    sub["epoch"], sub["val_balanced_acc"],
                    color=COLORS.get(cond, "#999"), alpha=0.5, linewidth=1,
                )
            cond_all = df[df["condition"] == cond].groupby("epoch")["val_balanced_acc"].mean()
            if not cond_all.empty:
                ax.plot(
                    cond_all.index, cond_all.values,
                    color=COLORS.get(cond, "#999"), linewidth=2.5, label=cond,
                )
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Val balanced accuracy")
    fig.suptitle("Training curves — val balanced accuracy", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_seed_scatter(all_results: dict[str, pd.DataFrame]) -> Path:
    """Per-seed scatter: POYO (CWT-CNN and ResampleCNN) vs EEGNet baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (task, df) in zip(axes, all_results.items()):
        eegnet = df[df["condition"] == "Matched EEGNet"].set_index("seed")["balanced_acc"]
        for cond in ["CWT-CNN", "ResampleCNN"]:
            poyo = df[df["condition"] == cond].set_index("seed")["balanced_acc"]
            common_seeds = sorted(set(eegnet.index) & set(poyo.index))
            if not common_seeds:
                continue
            ax.scatter(
                [eegnet[s] for s in common_seeds],
                [poyo[s] for s in common_seeds],
                label=cond, color=COLORS.get(cond, "#999"),
                s=60, zorder=5,
            )
        lims = ax.get_xlim()
        ax.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.6, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        split_label = "val" if task == "Sleep Stage" else "test"
        ax.set_xlabel(f"EEGNet {split_label} bal. acc.")
        ax.set_title(task, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("POYO balanced accuracy")
    fig.suptitle("Per-seed POYO vs Matched EEGNet", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_per_seed_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api = wandb.Api()
    all_results: dict[str, pd.DataFrame] = {}
    all_curves: dict[str, pd.DataFrame] = {}

    for label, spec in TASKS.items():
        task_key = spec["task_key"]
        split = spec["metric_split"]
        print(f"\n{'=' * 72}")
        print(f"Fetching {label} (split={split})...")
        print("=" * 72)

        poyo = fetch_poyo_group(api, spec["poyo_group"], task_key, split)
        eegnet = fetch_eegnet_group(api, spec["eegnet_group"], task_key, split)
        task_df = pd.concat([poyo, eegnet], ignore_index=True)
        task_df.insert(0, "task", label)
        task_df.to_csv(CSV_DIR / f"{STEM}_{task_key}.csv", index=False)
        all_results[label] = task_df

        print_task_results(label, task_df, split)

        print(f"\n  Fetching training curves for {label}...")
        curves = fetch_val_training_curves(api, task_df, task_key)
        if not curves.empty:
            curves.to_csv(CSV_DIR / f"{STEM}_{task_key}_curves.csv", index=False)
        all_curves[label] = curves

    # --- Combined summary table ---
    print("\n\n" + "=" * 72)
    print("COMBINED SUMMARY")
    print("=" * 72)
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined_summary = (
        combined.groupby(["task", "condition"])[list(CORE_METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    combined_summary.columns = [
        "_".join(filter(None, c)).rstrip("_") for c in combined_summary.columns
    ]
    combined_summary.to_csv(CSV_DIR / f"{STEM}_combined_summary.csv", index=False)
    print(
        combined_summary.to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    # --- Figures ---
    print("\n\nGenerating figures...")
    figs: list[Path] = []
    figs.append(plot_test_balanced_accuracy(all_results))
    print(f"  Saved: {figs[-1]}")

    if "Sleep Stage" in all_results:
        figs.append(plot_val_balanced_accuracy_sleep(all_results["Sleep Stage"]))
        print(f"  Saved: {figs[-1]}")

    figs.append(plot_all_tasks_overview(all_results))
    print(f"  Saved: {figs[-1]}")

    non_empty_curves = {k: v for k, v in all_curves.items() if not v.empty}
    if non_empty_curves:
        figs.append(plot_training_curves(non_empty_curves))
        print(f"  Saved: {figs[-1]}")

    figs.append(plot_per_seed_scatter(all_results))
    print(f"  Saved: {figs[-1]}")

    print("\nDone. Generated figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

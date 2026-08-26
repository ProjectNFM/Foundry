"""Analyze the multi-length pretraining experiment (S1 vs M0 baseline).

S1 was pretrained on mixed window lengths [1, 2, 5, 10]s while M0 used
fixed 2s windows.  Downstream evaluation had partial coverage: PhysioNet MI
completed fully (6/6), Kemp Sleep LP had 2/3 folds, and all other tasks
(Sleep FT, P300 FT/LP) failed.  This script reports whatever data exists
and documents the failure inventory.

Usage:
    uv run python analysis/20260811-MS-multi-length-pretraining_analysis.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, figures_dir

warnings.filterwarnings("ignore", category=FutureWarning)

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)

PRETRAIN_PROJECT = "foundry_pretraining"
DOWNSTREAM_PROJECT = "foundry_finetuning"
PRETRAIN_GROUP = "MASKING_SEQLEN_LEAK_FIXED"

PRETRAIN_RUNS: dict[str, dict[str, Any]] = {
    "M0": {
        "name": "pretrain_M0_baseline_leak_fixed",
        "label": "M0 (fixed 2s)",
        "color": "#4C78A8",
    },
    "S1": {
        "name": "pretrain_S1_multilength_leak_fixed",
        "label": "S1 (multi-length)",
        "color": "#E45756",
    },
}
RUN_ORDER = ["M0", "S1"]

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


def _run_id_tag(name: str) -> str | None:
    for tag, info in PRETRAIN_RUNS.items():
        if info["name"] in name:
            return tag
    return None


def _fold(name: str) -> int | None:
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_pretraining(api: wandb.Api) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return loss curves and summary for M0 and S1 pretraining runs."""
    entity = api.default_entity
    curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for tag, info in PRETRAIN_RUNS.items():
        matches = list(
            api.runs(
                f"{entity}/{PRETRAIN_PROJECT}",
                filters={"group": PRETRAIN_GROUP, "display_name": info["name"]},
            )
        )
        if not matches:
            print(f"  WARNING: pretraining run not found: {info['name']}")
            continue
        run = matches[0]
        for kind, key in [("train", "train/loss"), ("val", "val/loss")]:
            history = run.history(keys=[key], samples=50_000, pandas=True)
            for _, row in history.iterrows():
                curves.append(
                    {
                        "run_tag": tag,
                        "step": row.get("_step"),
                        "kind": kind,
                        "loss": row.get(key),
                    }
                )

        val_h = run.history(keys=["val/loss"], samples=50_000, pandas=True)
        vals = val_h.get("val/loss", pd.Series(dtype=float)).dropna()
        summaries.append(
            {
                "run_tag": tag,
                "run_name": info["name"],
                "wandb_id": run.id,
                "state": run.state,
                "steps": run.summary.get("_step"),
                "best_val_loss": vals.min() if not vals.empty else np.nan,
                "final_val_loss": vals.iloc[-1] if not vals.empty else np.nan,
            }
        )

    return pd.DataFrame(curves), pd.DataFrame(summaries)


def fetch_downstream(api: wandb.Api) -> pd.DataFrame:
    """Fetch all downstream runs for M0 and S1, including failed ones."""
    entity = api.default_entity
    records: list[dict[str, Any]] = []

    for (task, mode), group in DOWNSTREAM_GROUPS.items():
        metric_key = METRIC_KEYS[task]
        all_runs = list(
            api.runs(f"{entity}/{DOWNSTREAM_PROJECT}", filters={"group": group})
        )
        relevant = [r for r in all_runs if _run_id_tag(r.name) in RUN_ORDER]
        print(f"  {task} / {mode}: {len(relevant)} runs found")

        for run in relevant:
            history = run.history(
                keys=[metric_key], samples=50_000, pandas=True
            )
            values = history.get(metric_key, pd.Series(dtype=float)).dropna()
            records.append(
                {
                    "task": task,
                    "mode": mode,
                    "run_tag": _run_id_tag(run.name),
                    "fold": _fold(run.name),
                    "run_name": run.name,
                    "wandb_id": run.id,
                    "state": run.state,
                    "last_step": run.summary.get("_step"),
                    "best_f1": values.max() if not values.empty else np.nan,
                    "n_metric_pts": len(values),
                }
            )

    return pd.DataFrame(records)


def summarize(downstream: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of best F1 from finished folds, plus coverage counts."""
    finished = downstream[
        (downstream["state"] == "finished") & downstream["best_f1"].notna()
    ].copy()
    agg = (
        finished.groupby(["task", "mode", "run_tag"])["best_f1"]
        .agg(mean="mean", std="std", n_finished="count")
        .reset_index()
    )
    coverage = (
        downstream.groupby(["task", "mode", "run_tag"])
        .agg(
            n_found=("run_name", "count"),
            n_finished=("state", lambda x: (x == "finished").sum()),
        )
        .reset_index()
    )
    return coverage.merge(
        agg.drop(columns="n_finished"),
        on=["task", "mode", "run_tag"],
        how="left",
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_pretraining_curves(curves: pd.DataFrame) -> Path | None:
    if curves.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag in RUN_ORDER:
        info = PRETRAIN_RUNS[tag]
        sub = curves[curves["run_tag"] == tag].sort_values("step")
        for ax, kind in zip(axes, ["train", "val"]):
            part = sub[sub["kind"] == kind].dropna(subset=["loss"])
            if part.empty:
                continue
            ax.plot(
                part["step"],
                part["loss"],
                label=info["label"],
                color=info["color"],
                marker="o" if kind == "val" else None,
                markersize=2,
            )
    for ax, title in zip(axes, ["Training loss", "Validation loss"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Pretraining step")
        ax.set_ylabel("MAE reconstruction loss")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=9)
    fig.suptitle("M0 vs S1: pretraining reconstruction loss", fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / f"{STEM}_pretraining_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_downstream_comparison(summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None

    combos = [(t, m) for t in TASKS for m in MODES]
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5), sharey=True)

    for ax, (task, mode) in zip(axes, combos):
        sub = summary[(summary["task"] == task) & (summary["mode"] == mode)]
        sub = sub.set_index("run_tag")
        means, stds, colors, labels = [], [], [], []
        for tag in RUN_ORDER:
            info = PRETRAIN_RUNS[tag]
            if tag in sub.index and pd.notna(sub.loc[tag, "mean"]):
                means.append(sub.loc[tag, "mean"])
                stds.append(
                    sub.loc[tag, "std"] if pd.notna(sub.loc[tag, "std"]) else 0
                )
                n_fin = int(sub.loc[tag, "n_finished"])
            else:
                means.append(0)
                stds.append(0)
                n_fin = 0
            colors.append(info["color"])
            labels.append(f"{info['label']}\n(n={n_fin})")

        x = np.arange(len(RUN_ORDER))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.55)
        for bar, m, n_lbl in zip(bars, means, labels):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    m + 0.015,
                    f"{m:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "no finished\nfolds",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{task}\n{mode}", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Best validation F1 (mean ± SD)")
    fig.suptitle(
        "S1 (multi-length) vs M0 (fixed 2s): downstream transfer",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGURES_DIR / f"{STEM}_downstream_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_chart(summary: pd.DataFrame) -> Path | None:
    """Bar chart of S1 delta vs M0 for task/mode combos with data."""
    pairs: list[tuple[str, str, float]] = []
    for task in TASKS:
        for mode in MODES:
            sub = summary[(summary["task"] == task) & (summary["mode"] == mode)]
            sub = sub.set_index("run_tag")
            if "M0" in sub.index and "S1" in sub.index:
                m0_val = sub.loc["M0", "mean"]
                s1_val = sub.loc["S1", "mean"]
                if pd.notna(m0_val) and pd.notna(s1_val):
                    pairs.append((task, mode, s1_val - m0_val))

    if not pairs:
        return None

    labels = [f"{t}\n{m}" for t, m, _ in pairs]
    deltas = [d for _, _, d in pairs]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(pairs)), 4.5))
    bar_colors = ["#E45756" if d >= 0 else "#4C78A8" for d in deltas]
    bars = ax.bar(range(len(deltas)), deltas, color=bar_colors, width=0.55)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, d in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + (0.003 if d >= 0 else -0.003),
            f"{d:+.4f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=9,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ F1 (S1 − M0)")
    ax.set_title(
        "S1 vs M0: downstream F1 delta (finished folds only)", fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / f"{STEM}_delta_vs_m0.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_results(
    pretrain: pd.DataFrame, summary: pd.DataFrame, downstream: pd.DataFrame
) -> str:
    lines: list[str] = []

    lines.append("\nPRETRAINING SUMMARY")
    lines.append("=" * 60)
    lines.append(
        pretrain.to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    lines.append("\n\nDOWNSTREAM F1 SUMMARY — FINISHED FOLDS ONLY")
    lines.append("=" * 60)
    display = summary.copy()
    display["f1_mean_std"] = display.apply(
        lambda r: (
            "—"
            if pd.isna(r.get("mean"))
            else f"{r['mean']:.4f} ± {r['std']:.4f}"
            if pd.notna(r.get("std"))
            else f"{r['mean']:.4f} (one fold)"
        ),
        axis=1,
    )
    table_str = display[
        ["task", "mode", "run_tag", "f1_mean_std", "n_finished", "n_found"]
    ].to_string(index=False)
    lines.append(table_str)

    lines.append("\n\nRUN-STATE INVENTORY")
    lines.append("-" * 60)
    for tag in RUN_ORDER:
        sub = downstream[downstream["run_tag"] == tag]
        states = sub["state"].value_counts().to_dict()
        lines.append(f"  {tag}: {dict(states)}")

    failed = downstream[downstream["state"] != "finished"]
    lines.append(f"\nNon-finished runs: {len(failed)} / {len(downstream)}")
    if not failed.empty:
        lines.append(
            failed[
                [
                    "task",
                    "mode",
                    "run_tag",
                    "fold",
                    "state",
                    "wandb_id",
                    "last_step",
                ]
            ]
            .sort_values(["task", "mode", "run_tag", "fold"])
            .to_string(index=False)
        )

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    api = wandb.Api()
    print(f"W&B entity: {api.default_entity}")

    print("\nFetching pretraining runs...")
    curves, pretrain = fetch_pretraining(api)
    curves.to_csv(CSV_DIR / f"{STEM}_pretraining_curves.csv", index=False)
    pretrain.to_csv(CSV_DIR / f"{STEM}_pretraining_summary.csv", index=False)

    print("\nFetching downstream runs...")
    downstream = fetch_downstream(api)
    if downstream.empty:
        print("ERROR: No downstream runs found.")
        return
    downstream.to_csv(CSV_DIR / f"{STEM}_downstream_per_fold.csv", index=False)

    summary = summarize(downstream)
    summary.to_csv(CSV_DIR / f"{STEM}_downstream_summary.csv", index=False)

    print_results(pretrain, summary, downstream)

    print("\n\nGenerating figures...")
    figs: list[Path] = []
    fig = plot_pretraining_curves(curves)
    if fig:
        figs.append(fig)
    fig = plot_downstream_comparison(summary)
    if fig:
        figs.append(fig)
    fig = plot_delta_chart(summary)
    if fig:
        figs.append(fig)

    print("\nGenerated figures:")
    for f in figs:
        print(f"  {f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

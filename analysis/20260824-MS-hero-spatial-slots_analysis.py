"""Analyze the HERO spatial-slot ablation experiment.

Queries W&B for 1-slot vs 8-slot HERO runs (flat temporal mode) across
P300, Motor Imagery, and Sleep Stage.  Compares both conditions against
each other and against the matched EEGNet baselines.

Run after all 18 spatial-slot jobs have finished:

    uv run python analysis/20260824-MS-hero-spatial-slots_analysis.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

STEM = Path(__file__).stem
PROJECT = "foundry-neuralbench"
ENTITY = default_entity()
SEEDS = (33, 34, 35)
MARGIN = 0.02
CSV_DIR = csv_dir(__file__)
FIGURES_DIR = figures_dir(__file__)

TASKS = {
    "P300": {
        "hero_group": "NB_P300_HERO_SPATIAL_SLOTS",
        "eegnet_group": "NB_P300_EEGNET_MATCHED",
        "task_key": "neuralbench_p300",
        "channels": 16,
        "eegnet_reference": 0.625,
    },
    "Motor Imagery": {
        "hero_group": "NB_MI_HERO_SPATIAL_SLOTS",
        "eegnet_group": "NB_MI_EEGNET_MATCHED",
        "task_key": "neuralbench_motor_imagery",
        "channels": 64,
        "eegnet_reference": 0.571,
    },
    "Sleep Stage": {
        "hero_group": "NB_SLEEP_HERO_SPATIAL_SLOTS",
        "eegnet_group": "NB_SLEEP_EEGNET_MATCHED",
        "task_key": "neuralbench_sleep_stage",
        "channels": 2,
        "eegnet_reference": 0.680,
    },
}

SLOT_CONDITIONS = (1, 8)

COLORS = {
    1: "#e45756",
    8: "#4c78a8",
    "eegnet": "#72b7b2",
}


def _project_path() -> str:
    return f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT


def _seed(run: Any) -> int:
    value = run.config.get("seed")
    if value is None:
        nested = run.config.get("run", {})
        value = nested.get("seed") if isinstance(nested, dict) else None
    if value is None:
        match = re.search(r"seed[_-]?(\d+)", run.name or "", re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"Cannot infer seed for {run.name} ({run.id})")
        value = match.group(1)
    return int(value)


def _num_spatial_slots(run: Any) -> int:
    value = run.config.get("num_spatial_slots")
    if value is None:
        nested = run.config.get("model", {})
        value = nested.get("num_spatial_slots") if isinstance(nested, dict) else None
    if value is None:
        match = re.search(r"slots?[_-]?(\d+)", run.name or "", re.IGNORECASE)
        if match is None:
            raise RuntimeError(
                f"Cannot infer num_spatial_slots for {run.name} ({run.id})"
            )
        value = match.group(1)
    return int(value)


def _summary_float(summary: Any, key: str, aggregate: str = "max") -> float | None:
    value = summary.get(f"{key}.{aggregate}")
    if value is None:
        value = summary.get(key)
    if isinstance(value, dict):
        value = value.get(aggregate, value.get("last"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_hero_group(
    api: wandb.Api, group: str, task_key: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in api.runs(_project_path(), filters={"group": group}):
        if run.state != "finished":
            print(f"[skip] {run.name} ({run.id}): state={run.state}")
            continue
        test_bal = _summary_float(run.summary, f"test/{task_key}_balanced_acc")
        if test_bal is None:
            print(f"[skip] {run.name} ({run.id}): no test balanced accuracy")
            continue
        rows.append(
            {
                "num_spatial_slots": _num_spatial_slots(run),
                "seed": _seed(run),
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "test_balanced_acc": test_bal,
                "test_f1": _summary_float(run.summary, f"test/{task_key}_f1"),
                "test_auroc": _summary_float(run.summary, f"test/{task_key}_auroc"),
                "test_acc": _summary_float(run.summary, f"test/{task_key}_acc"),
                "val_balanced_acc": _summary_float(
                    run.summary, f"val/{task_key}_balanced_acc"
                ),
                "selected_epoch": _summary_float(run.summary, "epoch", "max"),
                "parameter_count": _summary_float(
                    run.summary, "diagnostics/parameter_count", "last"
                ),
                "peak_memory_mb": _summary_float(
                    run.summary, "diagnostics/peak_memory_mb", "max"
                ),
            }
        )
    return pd.DataFrame(rows)


def fetch_eegnet_group(
    api: wandb.Api, group: str, task_key: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in api.runs(_project_path(), filters={"group": group}):
        if run.state != "finished":
            continue
        test_bal = _summary_float(run.summary, f"test/{task_key}_balanced_acc")
        if test_bal is None:
            continue
        rows.append(
            {
                "seed": _seed(run),
                "run_id": run.id,
                "test_balanced_acc": test_bal,
            }
        )
    return pd.DataFrame(rows)


def validate_hero_runs(frame: pd.DataFrame, task: str) -> None:
    if frame.empty:
        raise RuntimeError(f"No finished usable HERO runs for {task}")
    dupes = frame.duplicated(["num_spatial_slots", "seed"], keep=False)
    if dupes.any():
        detail = frame.loc[dupes, ["num_spatial_slots", "seed", "run_id"]]
        raise RuntimeError(f"Duplicate slot/seed runs for {task}:\n{detail}")
    expected = {(s, seed) for s in SLOT_CONDITIONS for seed in SEEDS}
    observed = set(zip(frame["num_spatial_slots"], frame["seed"], strict=False))
    missing = sorted(expected - observed)
    if missing:
        print(f"[warn] Missing HERO slot/seed cells for {task}: {missing}")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("num_spatial_slots", as_index=False)
        .agg(
            mean_test_balanced_acc=("test_balanced_acc", "mean"),
            std_test_balanced_acc=("test_balanced_acc", "std"),
            n=("test_balanced_acc", "count"),
        )
    )


def print_task_results(
    task: str,
    hero: pd.DataFrame,
    eegnet: pd.DataFrame,
    eegnet_ref: float,
) -> str:
    lines: list[str] = []
    lines.append(f"\n{'=' * 72}")
    lines.append(f"  {task}")
    lines.append("=" * 72)

    per_seed = hero.sort_values(["num_spatial_slots", "seed"])
    display_cols = [
        "num_spatial_slots",
        "seed",
        "test_balanced_acc",
        "val_balanced_acc",
        "selected_epoch",
    ]
    present = [c for c in display_cols if c in per_seed.columns]
    lines.append("\n  Per-seed HERO results:")
    lines.append(per_seed[present].to_string(index=False, float_format="{:.4f}".format))

    summary = aggregate(hero)
    lines.append("\n  Aggregate (mean +/- SD):")
    lines.append(summary.to_string(index=False, float_format="{:.4f}".format))

    slot1 = hero[hero["num_spatial_slots"] == 1]["test_balanced_acc"]
    slot8 = hero[hero["num_spatial_slots"] == 8]["test_balanced_acc"]
    delta = slot8.mean() - slot1.mean()
    lines.append(f"\n  8-slot - 1-slot delta: {delta:+.4f} ({delta:+.1%})")
    lines.append(f"  Meets 2-pp threshold: {'YES' if delta >= MARGIN else 'NO'}")

    if not eegnet.empty:
        eeg_mean = eegnet["test_balanced_acc"].mean()
        eeg_std = eegnet["test_balanced_acc"].std()
        lines.append(f"\n  Matched EEGNet reference: {eeg_mean:.4f} +/- {eeg_std:.4f}")
        lines.append(f"  Manuscript EEGNet reference: {eegnet_ref:.3f}")
        d1 = slot1.mean() - eeg_mean
        d8 = slot8.mean() - eeg_mean
        lines.append(f"  HERO 1-slot vs EEGNet: {d1:+.4f}")
        lines.append(f"  HERO 8-slot vs EEGNet: {d8:+.4f}")

    output = "\n".join(lines)
    print(output)
    return output


def evaluate_hypothesis(all_hero: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Evaluate the pre-registered hypothesis."""
    results: dict[str, Any] = {}
    for task, hero in all_hero.items():
        slot1 = hero[hero["num_spatial_slots"] == 1]["test_balanced_acc"]
        slot8 = hero[hero["num_spatial_slots"] == 8]["test_balanced_acc"]
        delta = slot8.mean() - slot1.mean()
        results[task] = {
            "delta": delta,
            "meets_2pp": delta >= MARGIN,
            "slot1_mean": slot1.mean(),
            "slot8_mean": slot8.mean(),
        }

    mi = results.get("Motor Imagery", {})
    results["hypothesis_supported"] = mi.get("meets_2pp", False)
    return results


def plot_slot_comparison(
    all_hero: dict[str, pd.DataFrame],
    all_eegnet: dict[str, pd.DataFrame],
) -> Path:
    """Grouped bar chart: 1-slot vs 8-slot vs EEGNet per task."""
    n = len(all_hero)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (task, hero) in zip(axes, all_hero.items()):
        slot1 = hero[hero["num_spatial_slots"] == 1]["test_balanced_acc"]
        slot8 = hero[hero["num_spatial_slots"] == 8]["test_balanced_acc"]
        eegnet = all_eegnet.get(task, pd.DataFrame())

        groups: list[tuple[str, float, float, str]] = [
            ("1-slot", slot1.mean(), slot1.std(), COLORS[1]),
            ("8-slot", slot8.mean(), slot8.std(), COLORS[8]),
        ]
        if not eegnet.empty:
            eeg_vals = eegnet["test_balanced_acc"]
            groups.append(("EEGNet", eeg_vals.mean(), eeg_vals.std(), COLORS["eegnet"]))

        x = np.arange(len(groups))
        labels = [g[0] for g in groups]
        means = [g[1] for g in groups]
        stds = [g[2] for g in groups]
        colors = [g[3] for g in groups]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.6)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        delta = slot8.mean() - slot1.mean()
        channels = TASKS[task]["channels"]
        ax.set_title(f"{task} ({channels}ch)\ndelta = {delta:+.4f}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test balanced accuracy (mean +/- SD, 3 seeds)")
    fig.suptitle(
        "HERO spatial-slot ablation: 1-slot vs 8-slot (flat temporal)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_slot_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_seed(all_hero: dict[str, pd.DataFrame]) -> Path:
    """Per-seed paired comparison: 1-slot vs 8-slot."""
    n = len(all_hero)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (task, hero) in zip(axes, all_hero.items()):
        pivot = hero.pivot(
            index="seed", columns="num_spatial_slots", values="test_balanced_acc"
        )
        if 1 not in pivot.columns or 8 not in pivot.columns:
            ax.set_title(f"{task} — incomplete data")
            continue

        for seed in SEEDS:
            if seed not in pivot.index:
                continue
            v1, v8 = pivot.loc[seed, 1], pivot.loc[seed, 8]
            ax.plot([0, 1], [v1, v8], "o-", color="#555", markersize=7, linewidth=1.2)
            ax.annotate(f"s{seed}", (1, v8), textcoords="offset points", xytext=(8, 0), fontsize=8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["1-slot", "8-slot"])
        ax.set_title(f"{task} ({TASKS[task]['channels']}ch)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test balanced accuracy")
    fig.suptitle(
        "Per-seed paired comparison: 1-slot vs 8-slot",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_per_seed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    api = wandb.Api()
    all_hero: dict[str, pd.DataFrame] = {}
    all_eegnet: dict[str, pd.DataFrame] = {}
    all_text: list[str] = []

    for task, spec in TASKS.items():
        print(f"\nFetching {task}...")
        hero = fetch_hero_group(api, spec["hero_group"], spec["task_key"])
        validate_hero_runs(hero, task)
        all_hero[task] = hero

        eegnet = fetch_eegnet_group(api, spec["eegnet_group"], spec["task_key"])
        all_eegnet[task] = eegnet

        txt = print_task_results(task, hero, eegnet, spec["eegnet_reference"])
        all_text.append(txt)

    runs = pd.concat(
        [df.assign(task=task) for task, df in all_hero.items()],
        ignore_index=True,
    )
    runs_path = CSV_DIR / f"{STEM}_runs.csv"
    runs.to_csv(runs_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for task, hero in all_hero.items():
        agg = aggregate(hero)
        for _, row in agg.iterrows():
            summary_rows.append({"task": task, **row.to_dict()})
    summary = pd.DataFrame(summary_rows)
    summary_path = CSV_DIR / f"{STEM}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n\n" + "=" * 72)
    print("SPATIAL-SLOT ABLATION SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    hyp = evaluate_hypothesis(all_hero)
    print("\n\nHypothesis evaluation:")
    for task in TASKS:
        r = hyp[task]
        print(
            f"  {task}: 8-slot mean={r['slot8_mean']:.4f}, "
            f"1-slot mean={r['slot1_mean']:.4f}, "
            f"delta={r['delta']:+.4f}, meets 2pp: {'YES' if r['meets_2pp'] else 'NO'}"
        )
    verdict = "SUPPORTED" if hyp["hypothesis_supported"] else "NOT SUPPORTED"
    print(f"\n  MI hypothesis (8-slot >= 1-slot + 2pp): {verdict}")

    print("\n\nGenerating figures...")
    figs: list[Path] = []
    figs.append(plot_slot_comparison(all_hero, all_eegnet))
    print(f"  Saved: {figs[-1]}")
    figs.append(plot_per_seed(all_hero))
    print(f"  Saved: {figs[-1]}")

    print(f"\nSaved run table: {runs_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

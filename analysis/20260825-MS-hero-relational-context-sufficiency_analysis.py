"""Analyze the HERO Motor Imagery relational-context sufficiency experiment.

The script queries W&B by group, prints per-seed and aggregate held-out test
balanced accuracy, evaluates the pre-registered sufficiency criteria, saves a
CSV cache, and writes a comparison figure.

Run after all Phase 4 jobs finish:

    uv run python analysis/20260825-MS-hero-relational-context-sufficiency_analysis.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

STEM = Path(__file__).stem
PROJECT = "foundry-neuralbench"
ENTITY = default_entity()
HERO_GROUP = "NB_MI_HERO_RELATIONAL_CONTEXT"
EEGNET_GROUP = "NB_MI_EEGNET_MATCHED"
TASK_KEY = "neuralbench_motor_imagery"
SEEDS = (33, 34, 35)
MARGIN = 0.02
CSV_DIR = csv_dir(__file__)
FIGURES_DIR = figures_dir(__file__)

CONDITION_ORDER = (
    "signal-only",
    "type-only",
    "local-context",
    "relational-only",
    "position-only",
    "relational-position",
    "shuffled-relational",
)

CONDITION_LABELS = {
    "signal-only": "Signal-only",
    "type-only": "Type-only",
    "local-context": "Local context",
    "relational-only": "Relational-only",
    "position-only": "Position-only",
    "relational-position": "Relational + position",
    "shuffled-relational": "Shuffled relational",
    "eegnet": "Matched EEGNet",
}


def _project_path() -> str:
    return f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _seed(run: Any) -> int:
    value = run.config.get("seed") or _nested(run.config, "run", "seed")
    if value is None:
        match = re.search(r"seed[_-]?(\d+)", run.name or "", re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"Cannot infer seed for {run.name} ({run.id})")
        value = match.group(1)
    return int(value)


def _condition(run: Any) -> str:
    """Read the pre-launch condition label, with a run-name fallback."""
    candidates = (
        run.config.get("experiment_condition"),
        _nested(run.config, "run", "condition"),
        _nested(run.config, "experiment", "condition"),
    )
    for candidate in candidates:
        if candidate in CONDITION_ORDER:
            return str(candidate)

    normalized = re.sub(r"[_\s]+", "-", (run.name or "").lower())
    for condition in sorted(CONDITION_ORDER, key=len, reverse=True):
        if condition in normalized:
            return condition
    raise RuntimeError(
        f"Run {run.name} ({run.id}) has no recognized condition label. "
        "Set experiment_condition in the launch config."
    )


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


def fetch_group(api: wandb.Api, group: str, *, eegnet: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in api.runs(_project_path(), filters={"group": group}):
        if run.state != "finished":
            print(f"[skip] {run.name} ({run.id}): state={run.state}")
            continue
        condition = "eegnet" if eegnet else _condition(run)
        test_balanced_acc = _summary_float(
            run.summary, f"test/{TASK_KEY}_balanced_acc"
        )
        if test_balanced_acc is None:
            print(f"[skip] {run.name} ({run.id}): no test balanced accuracy")
            continue
        rows.append(
            {
                "condition": condition,
                "seed": _seed(run),
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "test_balanced_acc": test_balanced_acc,
                "val_balanced_acc": _summary_float(
                    run.summary, f"val/{TASK_KEY}_balanced_acc"
                ),
                "selected_epoch": _summary_float(run.summary, "epoch", "max"),
                "parameter_count": _summary_float(
                    run.summary, "diagnostics/parameter_count", "last"
                ),
                "peak_memory_mb": _summary_float(
                    run.summary, "diagnostics/peak_memory_mb", "max"
                ),
                "wall_clock_seconds": _summary_float(
                    run.summary, "diagnostics/wall_clock_seconds", "last"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_hero_runs(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError(f"No finished usable runs found in {HERO_GROUP}")
    duplicates = frame.duplicated(["condition", "seed"], keep=False)
    if duplicates.any():
        detail = frame.loc[duplicates, ["condition", "seed", "run_id"]]
        raise RuntimeError(f"Duplicate condition/seed runs:\n{detail}")
    expected = {(condition, seed) for condition in CONDITION_ORDER for seed in SEEDS}
    observed = set(zip(frame["condition"], frame["seed"], strict=False))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing HERO condition/seed cells: {missing}")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("condition", as_index=False)
        .agg(
            mean_test_balanced_acc=("test_balanced_acc", "mean"),
            std_test_balanced_acc=("test_balanced_acc", "std"),
            n=("test_balanced_acc", "count"),
        )
    )


def evaluate_sufficiency(hero: pd.DataFrame) -> dict[str, bool]:
    means = hero.groupby("condition")["test_balanced_acc"].mean()
    paired = hero.pivot(index="seed", columns="condition", values="test_balanced_acc")

    relational = means["relational-only"]
    improves_signal = relational - means["signal-only"] >= MARGIN
    improves_local = relational - means["local-context"] >= MARGIN
    near_position = means["relational-position"] - relational <= MARGIN
    binding_specific = relational - means["shuffled-relational"] >= MARGIN

    wins = (
        (paired["relational-only"] > paired["signal-only"])
        & (paired["relational-only"] > paired["local-context"])
    ).sum()
    no_large_seed_regression = (
        paired["relational-only"] - paired["signal-only"] >= -MARGIN
    ).all()
    seed_consistent = bool(wins >= 2 and no_large_seed_regression)

    return {
        "relational >= signal-only + 0.02": bool(improves_signal),
        "relational >= local-context + 0.02": bool(improves_local),
        "relational within 0.02 of relational + position": bool(near_position),
        "relational >= shuffled relational + 0.02": bool(binding_specific),
        "matched-seed consistency": seed_consistent,
    }


def plot_balanced_accuracy(summary: pd.DataFrame) -> Path:
    order = list(CONDITION_ORDER) + ["eegnet"]
    data = summary.set_index("condition").reindex(order).dropna(how="all")
    labels = [CONDITION_LABELS[index] for index in data.index]
    colors = ["#4c78a8"] * len(data)
    if "eegnet" in data.index:
        colors[data.index.get_loc("eegnet")] = "#e45756"

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(
        labels,
        data["mean_test_balanced_acc"],
        yerr=data["std_test_balanced_acc"].fillna(0),
        capsize=4,
        color=colors,
    )
    ax.set_ylabel("Held-out test balanced accuracy")
    ax.set_title("HERO relational-context sufficiency on NeuralBench MI")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_test_balanced_accuracy.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    api = wandb.Api()
    hero = fetch_group(api, HERO_GROUP)
    validate_hero_runs(hero)
    eegnet = fetch_group(api, EEGNET_GROUP, eegnet=True)

    runs = pd.concat([hero, eegnet], ignore_index=True)
    runs_path = CSV_DIR / f"{STEM}_runs.csv"
    runs.to_csv(runs_path, index=False)

    summary = aggregate(runs)
    summary_path = CSV_DIR / f"{STEM}_summary.csv"
    summary.to_csv(summary_path, index=False)

    display = summary.copy()
    display["condition"] = display["condition"].map(CONDITION_LABELS)
    print("\nThree-seed held-out test balanced accuracy")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    checks = evaluate_sufficiency(hero)
    print("\nPre-registered relational-sufficiency criteria")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print(f"\nOverall verdict: {'SUPPORTED' if all(checks.values()) else 'NOT SUPPORTED'}")

    figure_path = plot_balanced_accuracy(summary)
    print(f"\nSaved run table: {runs_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()

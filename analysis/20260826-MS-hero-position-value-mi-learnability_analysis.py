"""Analyze validation-only HERO position-value Motor Imagery pilot runs.

Fetches the two-condition pilot W&B group, saves run/summary CSVs, prints
the pre-registered learnability checks, and plots validation balanced accuracy.
No test metric is requested or consumed.

Run with:

    uv run python analysis/20260826-MS-hero-position-value-mi-learnability_analysis.py
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
TASK_KEY = "neuralbench_motor_imagery"
GROUP = "NB_MI_HERO_POSITION_VALUE_PILOT"
CONDITIONS = ("anonymous", "position_values")
RUN_IDS = {
    "anonymous": "h051ctmv",
    "position_values": "5o9glkt1",
}
LABELS = {
    "anonymous": "Anonymous control",
    "position_values": "Position values",
}
CSV_DIR = csv_dir(__file__)
FIGURES_DIR = figures_dir(__file__)


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
    candidate = run.config.get("condition_name")
    if candidate in CONDITIONS:
        return str(candidate)
    normalized = re.sub(r"[-\s]+", "_", (run.name or "").lower())
    for condition in sorted(CONDITIONS, key=len, reverse=True):
        if condition in normalized:
            return condition
    raise RuntimeError(
        f"Run {run.name} ({run.id}) has no recognized condition_name."
    )


def _summary_float(summary: Any, key: str, aggregate: str) -> float | None:
    value = summary.get(f"{key}.{aggregate}")
    if value is None:
        value = summary.get(key)
    if isinstance(value, dict):
        value = value.get(aggregate, value.get("last"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_group(api: wandb.Api) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Pin the completed runs rather than selecting the first run of each
    # condition from the group.  This keeps the completed-pilot report stable
    # if retries or future runs are later added to the same W&B group.
    for expected_condition, run_id in RUN_IDS.items():
        run = api.run(f"{_project_path()}/{run_id}")
        if run.group != GROUP:
            raise RuntimeError(
                f"Run {run.name} ({run.id}) is in group {run.group!r}, "
                f"expected {GROUP!r}."
            )
        if run.state != "finished":
            print(f"[skip] {run.name} ({run.id}): state={run.state}")
            continue
        condition = _condition(run)
        if condition != expected_condition:
            raise RuntimeError(
                f"Run {run.name} ({run.id}) resolved to {condition!r}, "
                f"expected {expected_condition!r}."
            )
        val_balanced_acc = _summary_float(
            run.summary, f"val/{TASK_KEY}_balanced_acc", "max"
        )
        if val_balanced_acc is None:
            print(
                f"[skip] {run.name} ({run.id}): no validation balanced accuracy"
            )
            continue
        rows.append(
            {
                "condition": condition,
                "seed": _seed(run),
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "val_balanced_acc": val_balanced_acc,
                "train_balanced_acc": _summary_float(
                    run.summary, f"train/{TASK_KEY}_balanced_acc", "max"
                ),
                "min_train_loss": _summary_float(
                    run.summary, f"train/{TASK_KEY}_loss", "min"
                ),
                # Lightning's epoch summary records the final logged epoch;
                # it is not necessarily the checkpoint-selected best epoch.
                "last_logged_epoch": _summary_float(run.summary, "epoch", "max"),
                "routing_entropy": _summary_float(
                    run.summary, "val/hero/routing/attention_entropy", "last"
                ),
            }
        )
    return pd.DataFrame(rows)


def pilot_checks(runs: pd.DataFrame) -> dict[str, bool]:
    if runs.empty:
        return {}
    anon = runs.loc[runs["condition"] == "anonymous"]
    posval = runs.loc[runs["condition"] == "position_values"]
    if anon.empty or posval.empty:
        print("Missing one or both conditions; cannot evaluate pilot gates.")
        return {}

    anon_val = anon.iloc[0]["val_balanced_acc"]
    posval_val = posval.iloc[0]["val_balanced_acc"]
    anon_train = anon.iloc[0]["train_balanced_acc"]
    posval_train = posval.iloc[0]["train_balanced_acc"]
    posval_train_loss = posval.iloc[0]["min_train_loss"]

    val_delta = posval_val - anon_val
    train_delta = (posval_train or 0) - (anon_train or 0)

    return {
        "Gate 1: position-values best val balanced acc >= 0.40": bool(
            posval_val >= 0.40
        ),
        "Gate 2: position-values val balanced acc improvement >= 0.05 over anonymous": bool(
            val_delta >= 0.05
        ),
        f"Gate 3a: position-values peak train balanced acc >= 0.10 above anonymous (delta={train_delta:.4f})": bool(
            train_delta >= 0.10
        ),
        f"Gate 3b: position-values min train CE < 1.386 (value={posval_train_loss})": bool(
            posval_train_loss is not None and posval_train_loss < 1.386
        ),
    }


def plot_validation(runs: pd.DataFrame) -> Path | None:
    if runs.empty:
        return None
    data = runs.set_index("condition").reindex(CONDITIONS).dropna(how="all")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        [LABELS[idx] for idx in data.index],
        data["val_balanced_acc"],
        color=["#9ecae1", "#3182bd"],
        width=0.5,
    )
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.axhline(0.40, color="#e6550d", linestyle=":", linewidth=1, label="Gate")
    ax.set_title("Position-value pilot: validation balanced accuracy")
    ax.set_ylabel("Best validation balanced accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_validation_balanced_accuracy.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_training_curves(api: wandb.Api, runs: pd.DataFrame) -> Path | None:
    if runs.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"anonymous": "#9ecae1", "position_values": "#3182bd"}
    for _, row in runs.iterrows():
        run = api.run(f"{_project_path()}/{row['run_id']}")
        for metric_key, ax, ylabel in [
            (f"val/{TASK_KEY}_balanced_acc", axes[0], "Validation balanced accuracy"),
            (f"train/{TASK_KEY}_loss", axes[1], "Training loss"),
        ]:
            history = run.history(keys=["epoch", metric_key], samples=10_000, pandas=True)
            history = history.dropna(subset=[metric_key])
            if not history.empty:
                ax.plot(
                    history["epoch"],
                    history[metric_key],
                    label=LABELS[row["condition"]],
                    color=colors[row["condition"]],
                    linewidth=1.5,
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                ax.legend()
    axes[0].axhline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].axhline(0.40, color="#e6550d", linestyle=":", linewidth=1, alpha=0.5)
    axes[1].axhline(1.386, color="#e6550d", linestyle=":", linewidth=1, alpha=0.5, label="4-class uniform")
    axes[1].legend()
    fig.suptitle("Position-value pilot: training curves", fontsize=13)
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_training_curves.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _print_checks(title: str, checks: dict[str, bool]) -> None:
    if not checks:
        return
    print(f"\n{title}")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")


def main() -> None:
    api = wandb.Api()
    runs = fetch_group(api)
    if runs.empty:
        raise RuntimeError("No finished position-value pilot runs found.")

    runs_path = CSV_DIR / f"{STEM}_runs.csv"
    runs.to_csv(runs_path, index=False)

    print("\nPer-run results")
    print(
        runs[
            [
                "condition",
                "seed",
                "run_name",
                "run_id",
                "val_balanced_acc",
                "train_balanced_acc",
                "min_train_loss",
                "last_logged_epoch",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    _print_checks("Pre-registered pilot gates", pilot_checks(runs))

    val_fig = plot_validation(runs)
    curve_fig = plot_training_curves(api, runs)

    print(f"\nSaved run table: {runs_path}")
    if val_fig is not None:
        print(f"Saved figure: {val_fig}")
    if curve_fig is not None:
        print(f"Saved figure: {curve_fig}")


if __name__ == "__main__":
    main()

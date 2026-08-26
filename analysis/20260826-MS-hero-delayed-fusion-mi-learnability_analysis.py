"""Analyze validation-only HERO delayed-fusion Motor Imagery runs.

Fetches the pilot and full W&B groups, saves run/summary CSVs, prints the
pre-registered learnability checks, and plots validation balanced accuracy.
No test metric is requested or consumed.

Run with:

    uv run python analysis/20260826-MS-hero-delayed-fusion-mi-learnability_analysis.py
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
PILOT_GROUP = "NB_MI_HERO_DELAYED_FUSION_PILOT"
FULL_GROUP = "NB_MI_HERO_DELAYED_FUSION_FULL"
SEEDS = (33, 34, 35)
CONDITIONS = ("early_fusion", "delayed_fusion", "delayed_fusion_position")
LABELS = {
    "early_fusion": "Early fusion",
    "delayed_fusion": "Delayed fusion",
    "delayed_fusion_position": "Delayed fusion + position",
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


def fetch_group(api: wandb.Api, group: str, phase: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in api.runs(_project_path(), filters={"group": group}):
        if run.state != "finished":
            print(f"[skip] {run.name} ({run.id}): state={run.state}")
            continue
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
                "phase": phase,
                "condition": _condition(run),
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
                "selected_epoch": _summary_float(run.summary, "epoch", "max"),
                "position_gate_mean": _position_gate_mean(run.summary),
                "routing_entropy": _summary_float(
                    run.summary, "val/hero/routing/attention_entropy", "last"
                ),
            }
        )
    return pd.DataFrame(rows)


def _position_gate_mean(summary: Any) -> float | None:
    values = [
        _summary_float(
            summary,
            f"val/hero/routing/position_gate_head{head}",
            "last",
        )
        for head in range(8)
    ]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["phase", "condition"], as_index=False)
        .agg(
            mean_val_balanced_acc=("val_balanced_acc", "mean"),
            std_val_balanced_acc=("val_balanced_acc", "std"),
            mean_train_balanced_acc=("train_balanced_acc", "mean"),
            min_train_loss=("min_train_loss", "min"),
            n=("val_balanced_acc", "count"),
        )
        .sort_values(["phase", "condition"])
    )


def pilot_checks(pilot: pd.DataFrame) -> dict[str, bool]:
    delayed = pilot.loc[pilot["condition"] == "delayed_fusion"]
    if delayed.empty:
        return {}
    train_balanced_acc = delayed.iloc[0]["train_balanced_acc"]
    return {
        "delayed-fusion training balanced accuracy >= 0.95": bool(
            pd.notna(train_balanced_acc) and train_balanced_acc >= 0.95
        )
    }


def full_checks(full: pd.DataFrame) -> dict[str, bool]:
    if full.empty:
        print(
            "\nFull experiment not launched; pre-registered full-run criteria not evaluated."
        )
        return {}
    required = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    observed = set(zip(full["condition"], full["seed"], strict=False))
    if missing := sorted(required - observed):
        print(f"\nFull experiment incomplete; missing cells: {missing}")
        return {}

    paired = full.pivot(
        index="seed", columns="condition", values="val_balanced_acc"
    )
    means = full.groupby("condition")["val_balanced_acc"].mean()
    delayed_delta = means["delayed_fusion"] - means["early_fusion"]
    position_delta = means["delayed_fusion_position"] - means["delayed_fusion"]
    delayed_rows = full.loc[full["condition"] == "delayed_fusion"]
    return {
        "delayed-fusion mean validation balanced accuracy >= 0.40": bool(
            means["delayed_fusion"] >= 0.40
        ),
        "delayed-fusion mean improvement >= 0.05": bool(delayed_delta >= 0.05),
        "delayed fusion wins all matched seeds": bool(
            (paired["delayed_fusion"] > paired["early_fusion"]).all()
        ),
        "delayed-fusion minimum train loss < 1.386": bool(
            delayed_rows["min_train_loss"].lt(1.386).all()
        ),
        "position mean improvement >= 0.02": bool(position_delta >= 0.02),
        "position wins at least two matched seeds": bool(
            (paired["delayed_fusion_position"] > paired["delayed_fusion"]).sum()
            >= 2
        ),
        "position has no matched-seed regression > 0.02": bool(
            (paired["delayed_fusion_position"] - paired["delayed_fusion"])
            .ge(-0.02)
            .all()
        ),
    }


def plot_validation(summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None
    phases = [
        phase for phase in ("pilot", "full") if phase in set(summary["phase"])
    ]
    fig, axes = plt.subplots(
        1, len(phases), figsize=(6 * len(phases), 5), squeeze=False
    )
    for axis, phase in zip(axes[0], phases, strict=True):
        data = summary.loc[summary["phase"] == phase].set_index("condition")
        data = data.reindex(CONDITIONS).dropna(how="all")
        axis.bar(
            [LABELS[index] for index in data.index],
            data["mean_val_balanced_acc"],
            yerr=data["std_val_balanced_acc"].fillna(0),
            capsize=4,
            color=["#9ecae1", "#3182bd", "#31a354"][: len(data)],
        )
        axis.axhline(
            0.25, color="black", linestyle="--", linewidth=1, label="Chance"
        )
        axis.axhline(
            0.40, color="#e6550d", linestyle=":", linewidth=1, label="Gate"
        )
        axis.set_title(f"{phase.title()} validation")
        axis.set_ylabel("Best validation balanced accuracy")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_validation_balanced_accuracy.png"
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
    pilot = fetch_group(api, PILOT_GROUP, "pilot")
    full = fetch_group(api, FULL_GROUP, "full")
    frames = [frame for frame in (pilot, full) if not frame.empty]
    if not frames:
        raise RuntimeError(
            "No finished delayed-fusion pilot or full runs found."
        )

    runs = pd.concat(frames, ignore_index=True)
    runs_path = CSV_DIR / f"{STEM}_runs.csv"
    runs.to_csv(runs_path, index=False)
    summary = aggregate(runs)
    summary_path = CSV_DIR / f"{STEM}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nValidation-only summary")
    print(
        summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    _print_checks("Pilot learnability gate", pilot_checks(pilot))
    _print_checks("Full pre-registered criteria", full_checks(full))

    figure_path = plot_validation(summary)
    print(f"\nSaved run table: {runs_path}")
    print(f"Saved summary: {summary_path}")
    if figure_path is not None:
        print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()

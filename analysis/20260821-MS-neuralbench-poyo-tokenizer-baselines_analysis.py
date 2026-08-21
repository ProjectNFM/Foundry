"""Summarize NeuralBench POYO tokenizer baselines against matched EEGNet.

Fetches finished runs from WandB, reports per-seed and mean-plus-standard-
deviation held-out test metrics, and writes a balanced-accuracy comparison.
Run after the POYO and parent EEGNet jobs have finished:

    uv run python analysis/20260821-MS-neuralbench-poyo-tokenizer-baselines_analysis.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
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
    "P300": (
        "neuralbench_p300",
        "NB_P300_POYO_TOKENIZER_BASELINES",
        "NB_P300_EEGNET_MATCHED",
    ),
    "Motor Imagery": (
        "neuralbench_motor_imagery",
        "NB_MI_POYO_TOKENIZER_BASELINES",
        "NB_MI_EEGNET_MATCHED",
    ),
    "Sleep Stage": (
        "neuralbench_sleep_stage",
        "NB_SLEEP_POYO_TOKENIZER_BASELINES",
        "NB_SLEEP_EEGNET_MATCHED",
    ),
}
METRICS = ("balanced_acc", "f1", "auroc", "acc")


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
        f"Could not infer tokenizer condition for run {run.id}: {run.name}"
    )


def fetch_group(
    api: wandb.Api, group: str, task: str, *, tokenizer: bool
) -> pd.DataFrame:
    """Fetch finished held-out-test summaries for one WandB group."""
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    rows = []
    for run in api.runs(path, filters={"group": group}):
        if run.state != "finished":
            continue
        row: dict[str, object] = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": _seed(run),
            "condition": _tokenizer_label(run)
            if tokenizer
            else "Matched EEGNet",
        }
        for metric in METRICS:
            key = f"test/{task}_{metric}"
            value = run.summary.get(key)
            if value is None:
                raise RuntimeError(f"Missing {key} in finished run {run.id}")
            row[metric] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No finished runs in WandB group {group}")
    return frame.sort_values(["condition", "seed"]).reset_index(drop=True)


def validate_coverage(frame: pd.DataFrame, task_label: str) -> None:
    expected = {"CWT-CNN", "ResampleCNN", "Matched EEGNet"}
    actual = set(frame["condition"])
    if actual != expected:
        raise RuntimeError(f"{task_label}: expected {expected}, found {actual}")
    for condition in expected:
        observed = set(frame.loc[frame.condition == condition, "seed"])
        if observed != set(SEEDS):
            raise RuntimeError(
                f"{task_label} / {condition}: expected seeds {SEEDS}, found {sorted(observed)}"
            )


def plot_balanced_accuracy(summary: pd.DataFrame) -> Path:
    task_order = list(TASKS)
    condition_order = ["Matched EEGNet", "CWT-CNN", "ResampleCNN"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, task in zip(axes, task_order, strict=True):
        subset = (
            summary[summary.task == task]
            .set_index("condition")
            .loc[condition_order]
        )
        ax.bar(
            condition_order,
            subset["balanced_acc_mean"],
            yerr=subset["balanced_acc_std"],
            capsize=3,
        )
        ax.set_title(task)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test balanced accuracy")
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_test_balanced_accuracy.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    api = wandb.Api()
    frames = []
    for label, (task, poyo_group, eegnet_group) in TASKS.items():
        poyo = fetch_group(api, poyo_group, task, tokenizer=True)
        eegnet = fetch_group(api, eegnet_group, task, tokenizer=False)
        task_frame = pd.concat([poyo, eegnet], ignore_index=True)
        validate_coverage(task_frame, label)
        task_frame.insert(0, "task", label)
        task_frame.to_csv(CSV_DIR / f"{STEM}_{task}.csv", index=False)
        frames.append(task_frame)

    results = pd.concat(frames, ignore_index=True)
    summary = (
        results.groupby(["task", "condition"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(filter(None, column)).rstrip("_") for column in summary.columns
    ]
    summary.to_csv(CSV_DIR / f"{STEM}_summary.csv", index=False)
    print(
        summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"Saved figure: {plot_balanced_accuracy(summary)}")


if __name__ == "__main__":
    main()

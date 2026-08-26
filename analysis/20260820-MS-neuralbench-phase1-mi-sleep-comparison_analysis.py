"""Compare Foundry and NeuralBench EEGNet on MI and sleep staging.

Foundry metrics are best validation values fetched from WandB. NeuralBench
metrics are test values from the local completed ``job.pkl`` artifacts.
Run with:
    uv run python analysis/20260820-MS-neuralbench-phase1-mi-sleep-comparison_analysis.py
"""

from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)
RESULTS_ROOT = Path("/network/scratch/s/sobralm/neuralbench-results")
EXPERIMENT_PREFIX = "neuralbench.main.Experiment.run,1"
SEEDS = (33, 34, 35)
PROJECT = "foundry-neuralbench"
ENTITY = default_entity()

TASKS = {
    "Motor Imagery": {
        "nb_task": "motor_imagery",
        "foundry_task": "neuralbench_motor_imagery",
        "group": "NB_MI_EEGNET_COMPARISON",
    },
    "Sleep Stage": {
        "nb_task": "sleep_stage",
        "foundry_task": "neuralbench_sleep_stage",
        "group": "NB_SLEEP_EEGNET_COMPARISON",
    },
}
NB_KEYS = {
    "balanced_acc": "test/bal_acc",
    "auroc": "test/auroc",
    "f1_macro": "test/f1_score_macro",
    "accuracy": "test/acc",
    "loss": "test/loss",
    "training_time_s": "training_time_s",
}


def _seed_from_run(run: wandb.apis.public.Run) -> int:
    seed = (run.config or {}).get("seed") or (run.config or {}).get(
        "run", {}
    ).get("seed")
    if seed is None:
        match = re.search(r"seed(\d+)", run.name or "")
        if match is None:
            raise RuntimeError(
                f"Could not determine seed for WandB run {run.id}"
            )
        seed = match.group(1)
    return int(seed)


def load_neuralbench(task_name: str) -> pd.DataFrame:
    """Load successful NeuralBench LocalJob artifacts for all experiment seeds."""
    rows: list[dict[str, float | int]] = []
    pattern = str(
        RESULTS_ROOT
        / EXPERIMENT_PREFIX
        / f"seed=*,task_name={task_name},*"
        / "job.pkl"
    )
    for path_text in glob.glob(pattern):
        path = Path(path_text)
        match = re.search(r"seed=(\d+),", str(path))
        if match is None or int(match.group(1)) not in SEEDS:
            continue
        with path.open("rb") as handle:
            job = pickle.load(handle)  # noqa: S301 -- approved local artifacts
        status, result = job.__dict__.get("_result", (None, None))
        if status != "success" or not isinstance(result, dict):
            raise RuntimeError(f"NeuralBench job did not succeed: {path}")
        rows.append(
            {
                "seed": int(match.group(1)),
                **{short: float(result[key]) for short, key in NB_KEYS.items()},
            }
        )
    frame = pd.DataFrame(rows).sort_values("seed").drop_duplicates("seed")
    missing = (
        sorted(set(SEEDS) - set(frame["seed"]))
        if not frame.empty
        else list(SEEDS)
    )
    if missing:
        raise RuntimeError(f"Missing NeuralBench {task_name} seeds: {missing}")
    return frame.reset_index(drop=True)


def fetch_foundry(api: wandb.Api, group: str, task_name: str) -> pd.DataFrame:
    """Fetch the finished Foundry runs and their best validation summaries."""
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    rows: list[dict[str, float | int | str]] = []
    for run in api.runs(path, filters={"group": group}):
        if run.state != "finished":
            continue
        summary = run.summary
        rows.append(
            {
                "seed": _seed_from_run(run),
                "run_name": run.name,
                "run_id": run.id,
                "balanced_acc": float(
                    summary[f"val/{task_name}_balanced_acc.max"]
                ),
                "auroc": float(summary[f"val/{task_name}_auroc.max"]),
                "f1_macro": float(summary[f"val/{task_name}_f1.max"]),
                "accuracy": float(summary[f"val/{task_name}_acc.max"]),
                "wandb_runtime_s": float(summary.get("_runtime", float("nan"))),
                "last_epoch": int(summary.get("epoch", -1)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("seed").drop_duplicates("seed")
    missing = (
        sorted(set(SEEDS) - set(frame["seed"]))
        if not frame.empty
        else list(SEEDS)
    )
    if missing:
        raise RuntimeError(f"Missing finished Foundry {group} seeds: {missing}")
    return frame.reset_index(drop=True)


def summarize(
    task: str, nb: pd.DataFrame, foundry: pd.DataFrame
) -> pd.DataFrame:
    metrics = ("balanced_acc", "auroc", "f1_macro", "accuracy")
    rows = []
    for metric in metrics:
        nb_mean, nb_std = nb[metric].mean(), nb[metric].std()
        f_mean, f_std = foundry[metric].mean(), foundry[metric].std()
        rows.append(
            {
                "task": task,
                "metric": metric,
                "foundry_val_mean": f_mean,
                "foundry_val_sd": f_std,
                "neuralbench_test_mean": nb_mean,
                "neuralbench_test_sd": nb_std,
                "delta_foundry_minus_neuralbench": f_mean - nb_mean,
            }
        )
    return pd.DataFrame(rows)


def plot_balanced_accuracy(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (task, (nb, foundry)) in zip(axes, results.items(), strict=True):
        x = range(len(SEEDS))
        ax.bar(
            [v - 0.2 for v in x],
            nb["balanced_acc"],
            0.4,
            label="NeuralBench test",
        )
        ax.bar(
            [v + 0.2 for v in x],
            foundry["balanced_acc"],
            0.4,
            label="Foundry val",
        )
        ax.set_title(task)
        ax.set_xticks(list(x), [str(seed) for seed in SEEDS])
        ax.set_xlabel("Seed")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False)
    fig.suptitle("EEGNet comparison: Foundry validation vs NeuralBench test")
    fig.tight_layout()
    output = FIGURES_DIR / f"{STEM}_balanced_accuracy.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    api = wandb.Api()
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    summaries = []
    for label, spec in TASKS.items():
        nb = load_neuralbench(spec["nb_task"])
        foundry = fetch_foundry(api, spec["group"], spec["foundry_task"])
        nb.to_csv(
            CSV_DIR / f"{STEM}_{spec['nb_task']}_neuralbench.csv", index=False
        )
        foundry.to_csv(
            CSV_DIR / f"{STEM}_{spec['nb_task']}_foundry.csv", index=False
        )
        results[label] = (nb, foundry)
        summaries.append(summarize(label, nb, foundry))

        print(f"\\n{label} — per-seed results")
        print("NeuralBench (test):")
        print(
            nb.to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print("Foundry (best validation):")
        print(
            foundry.to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(CSV_DIR / f"{STEM}_summary.csv", index=False)
    print("\\nMean ± SD comparison")
    print(
        summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"\\nSaved figure: {plot_balanced_accuracy(results)}")


if __name__ == "__main__":
    main()

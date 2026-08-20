"""Summarize local NeuralBench P3 EEGNet reference-grid results.

The NeuralBench CLI stores completed local runs as ``exca.task.LocalJob``
pickles rather than creating WandB runs.  This script is deliberately based on
those primary result artifacts; Foundry/WandB runs will be added in the later
comparison phase.

Run with:
    uv run python analysis/20260820-MS-neuralbench-p300-eegnet-comparison_analysis.py
"""

from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _wandb_utils import csv_dir, figures_dir


RESULTS_ROOT = Path("/network/scratch/s/sobralm/neuralbench-results")
EXPERIMENT_PREFIX = "neuralbench.main.Experiment.run,1"
SEEDS = (33, 34, 35)
METRICS = (
    "test/bal_acc",
    "test/auroc",
    "test/auprc",
    "test/f1_score_macro",
    "test/acc",
    "test/loss",
    "training_time_s",
)


def load_results() -> pd.DataFrame:
    """Load successful grid results from NeuralBench's LocalJob artifacts."""
    rows: list[dict[str, float | int]] = []
    pattern = str(RESULTS_ROOT / EXPERIMENT_PREFIX / "seed=*/job.pkl")
    for job_path_text in glob.glob(pattern):
        job_path = Path(job_path_text)
        match = re.search(r"seed=(\d+),", str(job_path))
        if match is None:
            continue
        seed = int(match.group(1))
        if seed not in SEEDS:
            continue
        with job_path.open("rb") as handle:
            job = pickle.load(handle)  # noqa: S301 -- local experiment artifact
        status, result = job.__dict__.get("_result", (None, None))
        if status != "success" or not isinstance(result, dict):
            raise RuntimeError(f"NeuralBench job did not succeed: {job_path}")
        rows.append(
            {"seed": seed, **{metric: result[metric] for metric in METRICS}}
        )

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    missing = (
        sorted(set(SEEDS) - set(frame["seed"]))
        if not frame.empty
        else list(SEEDS)
    )
    if missing:
        raise RuntimeError(
            f"Missing completed NeuralBench result(s) for seed(s): {missing}"
        )
    return frame


def main() -> None:
    results = load_results()
    script_path = Path(__file__)
    results.to_csv(
        csv_dir(script_path)
        / "20260820-MS-neuralbench-p300-eegnet-comparison_results.csv",
        index=False,
    )

    summary = results.loc[:, METRICS].agg(["mean", "std"]).T
    summary.index.name = "metric"
    print("NeuralBench P3 / Korczowski2014A EEGNet test metrics")
    print(
        results.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nMean ± sample standard deviation")
    for metric, values in summary.iterrows():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        results["seed"].astype(str), results["test/bal_acc"], color="#4c78a8"
    )
    ax.set(xlabel="Seed", ylabel="Test balanced accuracy", ylim=(0, 1))
    ax.set_title("NeuralBench P3 EEGNet reference")
    for seed, value in zip(results["seed"], results["test/bal_acc"]):
        ax.text(str(seed), value + 0.012, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(
        figures_dir(script_path)
        / "20260820-MS-neuralbench-p300-eegnet-comparison_balanced_accuracy.png",
        dpi=150,
    )


if __name__ == "__main__":
    main()

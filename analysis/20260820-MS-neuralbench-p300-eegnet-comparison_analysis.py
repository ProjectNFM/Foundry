"""Compare Foundry EEGNet vs NeuralBench EEGNet on P300 / Korczowski2014A.

Phase 1 (existing): loads NeuralBench local grid results (test metrics).
Phase 2 (this update): fetches Foundry WandB runs (validation metrics) and
produces side-by-side comparison tables and figures.

Note: Foundry reports *best validation* metrics (no test-set evaluation yet).
NeuralBench reports *test* metrics evaluated at the best validation checkpoint.
The comparison is explicitly val-vs-test; a follow-up with test-set evaluation
on the Foundry side is needed for a full apples-to-apples comparison.

Run with:
    uv run python analysis/20260820-MS-neuralbench-p300-eegnet-comparison_analysis.py
"""

from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)

RESULTS_ROOT = Path("/network/scratch/s/sobralm/neuralbench-results")
EXPERIMENT_PREFIX = "neuralbench.main.Experiment.run,1"
SEEDS = (33, 34, 35)

NB_TEST_METRICS = (
    "test/bal_acc",
    "test/auroc",
    "test/auprc",
    "test/f1_score_macro",
    "test/acc",
    "test/loss",
    "training_time_s",
)

WANDB_PROJECT = "foundry-neuralbench"
WANDB_GROUP = "NB_P300_EEGNET_COMPARISON"
ENTITY = default_entity()

TASK = "neuralbench_p300"
FOUNDRY_VAL_METRICS = {
    "val_balanced_acc": f"val/{TASK}_balanced_acc.max",
    "val_auroc": f"val/{TASK}_auroc.max",
    "val_f1": f"val/{TASK}_f1.max",
    "val_acc": f"val/{TASK}_acc.max",
    "val_loss": f"val/{TASK}_loss.min",
}
FOUNDRY_HISTORY_METRICS = {
    "val_balanced_acc": f"val/{TASK}_balanced_acc",
    "val_auroc": f"val/{TASK}_auroc",
    "val_f1": f"val/{TASK}_f1",
    "val_loss": f"val/{TASK}_loss",
    "train_loss": f"train/{TASK}_loss",
}

COLORS = {
    "foundry": "#4c78a8",
    "neuralbench": "#e45756",
}


def load_neuralbench_results() -> pd.DataFrame:
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
            {"seed": seed, **{m: result[m] for m in NB_TEST_METRICS}}
        )
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    missing = (
        sorted(set(SEEDS) - set(frame["seed"]))
        if not frame.empty
        else list(SEEDS)
    )
    if missing:
        raise RuntimeError(
            f"Missing NeuralBench result(s) for seed(s): {missing}"
        )
    return frame


def _run_path(run_id: str) -> str:
    if ENTITY:
        return f"{ENTITY}/{WANDB_PROJECT}/{run_id}"
    return f"{WANDB_PROJECT}/{run_id}"


def _seed_from_run(run: wandb.apis.public.Run) -> int | None:
    config = run.config or {}
    seed = config.get("seed") or config.get("run", {}).get("seed")
    if seed is not None:
        return int(seed)
    name = run.name or ""
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def fetch_foundry_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    """Fetch finished Foundry runs from the comparison WandB group."""
    if api is None:
        api = wandb.Api()
    path = f"{ENTITY}/{WANDB_PROJECT}" if ENTITY else WANDB_PROJECT
    print(f"Fetching Foundry runs: {path} group={WANDB_GROUP}")
    runs = api.runs(path, filters={"group": WANDB_GROUP})
    rows: list[dict] = []
    for run in runs:
        if run.state != "finished":
            print(f"  skipping {run.name} (state={run.state})")
            continue
        seed = _seed_from_run(run)
        row: dict = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": seed,
        }
        for short, wandb_key in FOUNDRY_VAL_METRICS.items():
            val = run.summary.get(wandb_key)
            try:
                row[short] = float(val)
            except (TypeError, ValueError):
                row[short] = None
        rows.append(row)
    print(f"  found {len(rows)} finished run(s)")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("seed").reset_index(drop=True)
    return df


def fetch_foundry_training_curves(
    foundry_runs: pd.DataFrame, api: wandb.Api | None = None
) -> pd.DataFrame:
    """Fetch epoch-level val balanced acc and val loss for training curves."""
    if api is None:
        api = wandb.Api()
    keep_cols = ["epoch"] + list(FOUNDRY_HISTORY_METRICS.values())
    val_bacc_key = FOUNDRY_HISTORY_METRICS["val_balanced_acc"]
    frames: list[pd.DataFrame] = []
    for _, row in foundry_runs.iterrows():
        run = api.run(_run_path(row["run_id"]))
        history = run.history(samples=10_000, pandas=True)
        if history.empty:
            continue
        present = [c for c in keep_cols if c in history.columns]
        history = history[present]
        if val_bacc_key in history.columns:
            history = history.dropna(subset=[val_bacc_key])
        history["seed"] = row["seed"]
        history["run_id"] = row["run_id"]
        frames.append(history)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def print_comparison(nb: pd.DataFrame, foundry: pd.DataFrame) -> str:
    """Print side-by-side metrics and return the summary string."""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("NeuralBench P300 EEGNet — reference test metrics (3 seeds)")
    lines.append("=" * 72)
    lines.append(
        nb.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    nb_mean = nb[list(NB_TEST_METRICS)].mean()
    nb_std = nb[list(NB_TEST_METRICS)].std()
    lines.append("\nMean ± SD:")
    for m in NB_TEST_METRICS:
        lines.append(f"  {m}: {nb_mean[m]:.4f} ± {nb_std[m]:.4f}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("Foundry EEGNet — best validation metrics (3 seeds)")
    lines.append("=" * 72)
    val_cols = list(FOUNDRY_VAL_METRICS.keys())
    lines.append(
        foundry[["seed"] + val_cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    f_mean = foundry[val_cols].mean()
    f_std = foundry[val_cols].std()
    lines.append("\nMean ± SD:")
    for m in val_cols:
        lines.append(f"  {m}: {f_mean[m]:.4f} ± {f_std[m]:.4f}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(
        "Head-to-head comparison (Foundry best-val vs NeuralBench test)"
    )
    lines.append("=" * 72)
    metric_pairs = [
        ("balanced_acc", "val_balanced_acc", "test/bal_acc"),
        ("auroc", "val_auroc", "test/auroc"),
        ("f1", "val_f1", "test/f1_score_macro"),
        ("acc", "val_acc", "test/acc"),
        ("loss", "val_loss", "test/loss"),
    ]
    header = f"{'Metric':<16s} {'Foundry (val)':>16s} {'NeuralBench (test)':>20s} {'Delta':>10s}"
    lines.append(header)
    lines.append("-" * len(header))
    for label, f_key, nb_key in metric_pairs:
        f_val = f_mean[f_key]
        n_val = nb_mean[nb_key]
        delta = f_val - n_val
        lines.append(
            f"{label:<16s} {f_val:>16.4f} {n_val:>20.4f} {delta:>+10.4f}"
        )

    output = "\n".join(lines)
    print(output)
    return output


def plot_comparison_bars(nb: pd.DataFrame, foundry: pd.DataFrame) -> Path:
    """Grouped bar chart: Foundry val balanced acc vs NeuralBench test balanced acc per seed."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SEEDS))
    width = 0.35

    nb_vals = [
        float(nb.loc[nb["seed"] == s, "test/bal_acc"].iloc[0]) for s in SEEDS
    ]
    f_vals = [
        float(foundry.loc[foundry["seed"] == s, "val_balanced_acc"].iloc[0])
        for s in SEEDS
    ]

    bars_nb = ax.bar(
        x - width / 2, nb_vals, width,
        label="NeuralBench (test)", color=COLORS["neuralbench"], edgecolor="white",
    )
    bars_f = ax.bar(
        x + width / 2, f_vals, width,
        label="Foundry (val)", color=COLORS["foundry"], edgecolor="white",
    )

    for bars in (bars_nb, bars_f):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    nb_mean = np.mean(nb_vals)
    f_mean = np.mean(f_vals)
    ax.axhline(nb_mean, color=COLORS["neuralbench"], ls="--", lw=1, alpha=0.6)
    ax.axhline(f_mean, color=COLORS["foundry"], ls="--", lw=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Foundry EEGNet (val) vs NeuralBench EEGNet (test)\nP300 balanced accuracy")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_balanced_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_all_metrics_comparison(nb: pd.DataFrame, foundry: pd.DataFrame) -> Path:
    """Bar chart comparing mean±std across all comparable metrics."""
    metric_pairs = [
        ("Balanced\nacc.", "val_balanced_acc", "test/bal_acc"),
        ("AUROC", "val_auroc", "test/auroc"),
        ("F1 (macro)", "val_f1", "test/f1_score_macro"),
        ("Accuracy", "val_acc", "test/acc"),
    ]
    labels = [p[0] for p in metric_pairs]
    f_means = [float(foundry[p[1]].mean()) for p in metric_pairs]
    f_stds = [float(foundry[p[1]].std()) for p in metric_pairs]
    nb_means = [float(nb[p[2]].mean()) for p in metric_pairs]
    nb_stds = [float(nb[p[2]].std()) for p in metric_pairs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2, nb_means, width, yerr=nb_stds, capsize=4,
        label="NeuralBench (test)", color=COLORS["neuralbench"], edgecolor="white",
    )
    ax.bar(
        x + width / 2, f_means, width, yerr=f_stds, capsize=4,
        label="Foundry (val)", color=COLORS["foundry"], edgecolor="white",
    )

    for i in range(len(labels)):
        ax.text(
            x[i] - width / 2, nb_means[i] + nb_stds[i] + 0.012,
            f"{nb_means[i]:.3f}", ha="center", va="bottom", fontsize=7.5,
        )
        ax.text(
            x[i] + width / 2, f_means[i] + f_stds[i] + 0.012,
            f"{f_means[i]:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value (mean ± SD over 3 seeds)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Foundry EEGNet (val) vs NeuralBench EEGNet (test) — all metrics")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_all_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_training_curves(curves: pd.DataFrame) -> Path:
    """Epoch-level val balanced acc and val loss curves per seed."""
    val_bacc_key = FOUNDRY_HISTORY_METRICS["val_balanced_acc"]
    val_loss_key = FOUNDRY_HISTORY_METRICS["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for seed in SEEDS:
        sub = curves[curves["seed"] == seed].sort_values("epoch")
        if sub.empty:
            continue
        label = f"seed {seed}"
        if val_bacc_key in sub.columns:
            vals = sub.dropna(subset=[val_bacc_key])
            if not vals.empty:
                axes[0].plot(vals["epoch"], vals[val_bacc_key], marker="o", markersize=3, label=label)
        if val_loss_key in sub.columns:
            vals = sub.dropna(subset=[val_loss_key])
            if not vals.empty:
                axes[1].plot(vals["epoch"], vals[val_loss_key], marker="o", markersize=3, label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val balanced accuracy")
    axes[0].set_title("Foundry EEGNet — validation balanced accuracy")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val loss")
    axes[1].set_title("Foundry EEGNet — validation loss")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    api = wandb.Api()

    print("Phase 1: Loading NeuralBench reference results...")
    nb = load_neuralbench_results()

    print("\nPhase 2: Fetching Foundry WandB runs...")
    foundry = fetch_foundry_runs(api)
    if foundry.empty:
        raise SystemExit("No finished Foundry runs found in WandB group.")

    foundry_seeds = set(foundry["seed"].dropna().astype(int))
    missing = sorted(set(SEEDS) - foundry_seeds)
    if missing:
        print(f"  WARNING: missing Foundry seeds: {missing}")

    nb.to_csv(CSV_DIR / f"{STEM}_neuralbench_results.csv", index=False)
    foundry.to_csv(CSV_DIR / f"{STEM}_foundry_results.csv", index=False)

    print("\nPhase 3: Fetching training curves...")
    curves = fetch_foundry_training_curves(foundry, api)
    if not curves.empty:
        curves.to_csv(CSV_DIR / f"{STEM}_foundry_curves.csv", index=False)

    print("\n")
    print_comparison(nb, foundry)

    print("\nPhase 4: Generating figures...")
    plot_comparison_bars(nb, foundry)
    plot_all_metrics_comparison(nb, foundry)
    if not curves.empty:
        plot_training_curves(curves)

    print("\nDone.")


if __name__ == "__main__":
    main()

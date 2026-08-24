"""Compare Foundry Matched EEGNet (test) vs NeuralBench EEGNet (test).

Fetches matched-config Foundry EEGNet runs from WandB and compares their
test-set balanced accuracy against NeuralBench EEGNet test metrics (loaded
from local job artifacts) on P300, Motor Imagery, and Sleep Stage.

Unlike Phase 1, both sides now evaluate their best-validation checkpoint
on the held-out test split, making this an apples-to-apples comparison.

Run after all nine matched EEGNet jobs have completed:

    uv run python analysis/20260821-MS-neuralbench-matched-test-parity_analysis.py
"""

from __future__ import annotations

import glob
import pickle
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

RESULTS_ROOT = Path("/network/scratch/s/sobralm/neuralbench-results")
NB_EXPERIMENT_PREFIX = "neuralbench.main.Experiment.run,1"

TASKS = {
    "P300": {
        "foundry_task_key": "neuralbench_p300",
        "group": "NB_P300_EEGNET_MATCHED",
        "nb_task_name": None,
    },
    "Motor Imagery": {
        "foundry_task_key": "neuralbench_motor_imagery",
        "group": "NB_MI_EEGNET_MATCHED",
        "nb_task_name": "motor_imagery",
    },
    "Sleep Stage": {
        "foundry_task_key": "neuralbench_sleep_stage",
        "group": "NB_SLEEP_EEGNET_MATCHED",
        "nb_task_name": "sleep_stage",
    },
}

CORE_METRICS = ("balanced_acc", "f1", "auroc", "acc")

NB_METRIC_MAP = {
    "balanced_acc": "test/bal_acc",
    "auroc": "test/auroc",
    "f1": "test/f1_score_macro",
    "acc": "test/acc",
    "loss": "test/loss",
    "training_time_s": "training_time_s",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_from_run(run: wandb.apis.public.Run) -> int:
    config = run.config or {}
    value = config.get("seed") or config.get("run", {}).get("seed")
    if value is None:
        match = re.search(r"seed(\d+)", run.name or "")
        if match is None:
            raise RuntimeError(f"Could not infer seed for run {run.id}")
        value = match.group(1)
    return int(value)


def _extract_foundry_metrics(
    summary: dict,
    task_key: str,
    split: str = "test",
    metrics: tuple[str, ...] = CORE_METRICS,
) -> dict[str, float | None]:
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
# NeuralBench reference loading
# ---------------------------------------------------------------------------

def load_neuralbench_reference(nb_task_name: str | None) -> pd.DataFrame:
    """Load NeuralBench test metrics from local job.pkl artifacts."""
    if nb_task_name is None:
        pattern = str(RESULTS_ROOT / NB_EXPERIMENT_PREFIX / "seed=*/job.pkl")
    else:
        pattern = str(
            RESULTS_ROOT / NB_EXPERIMENT_PREFIX
            / f"seed=*,task_name={nb_task_name},*" / "job.pkl"
        )

    rows: list[dict] = []
    for path_text in glob.glob(pattern):
        path = Path(path_text)
        match = re.search(r"seed=(\d+)", str(path))
        if match is None or int(match.group(1)) not in SEEDS:
            continue
        with path.open("rb") as handle:
            job = pickle.load(handle)  # noqa: S301 -- local experiment artifact
        status, result = job.__dict__.get("_result", (None, None))
        if status != "success" or not isinstance(result, dict):
            print(f"  [warn] NeuralBench job not success: {path}")
            continue
        rows.append({
            "seed": int(match.group(1)),
            **{short: float(result[key]) for short, key in NB_METRIC_MAP.items()
               if key in result},
        })

    frame = pd.DataFrame(rows).sort_values("seed").drop_duplicates("seed")
    missing = sorted(set(SEEDS) - set(frame["seed"])) if not frame.empty else list(SEEDS)
    if missing:
        raise RuntimeError(
            f"Missing NeuralBench seeds {missing} for task={nb_task_name}. "
            f"Pattern: {pattern}"
        )
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Foundry run fetching
# ---------------------------------------------------------------------------

def fetch_foundry_matched(
    api: wandb.Api,
    group: str,
    task_key: str,
) -> pd.DataFrame:
    """Fetch finished matched EEGNet runs with test metrics."""
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    rows: list[dict] = []
    for run in api.runs(path, filters={"group": group}):
        if run.state != "finished":
            print(f"  [skip] {run.name} ({run.state})")
            continue
        test_metrics = _extract_foundry_metrics(run.summary, task_key, "test")
        val_metrics = _extract_foundry_metrics(run.summary, task_key, "val")
        if all(v is None for v in test_metrics.values()):
            print(f"  [skip] {run.name} — no test metrics")
            continue
        row: dict = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": _seed_from_run(run),
            "state": run.state,
            "last_epoch": run.summary.get("epoch", None),
        }
        for m, v in test_metrics.items():
            row[f"test_{m}"] = v
        for m, v in val_metrics.items():
            row[f"val_{m}"] = v
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("seed").drop_duplicates("seed")
    if frame.empty:
        raise RuntimeError(f"No usable Foundry runs in group {group}")
    return frame.reset_index(drop=True)


def fetch_val_training_curves(
    api: wandb.Api,
    runs_df: pd.DataFrame,
    task_key: str,
) -> pd.DataFrame:
    """Fetch epoch-level validation balanced accuracy for training curve plots."""
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
        history["run_id"] = row["run_id"]
        frames.append(history[["epoch", "val_balanced_acc", "seed", "run_id"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["seed", "epoch"])


# ---------------------------------------------------------------------------
# Summary and printing
# ---------------------------------------------------------------------------

def print_task_comparison(
    label: str,
    foundry: pd.DataFrame,
    nb: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"\n{'=' * 72}")
    lines.append(f"{label} — Test-vs-Test Comparison")
    lines.append("=" * 72)

    lines.append(f"\n  Foundry Matched EEGNet — per-seed test metrics:")
    test_cols = ["seed", "last_epoch"] + [f"test_{m}" for m in CORE_METRICS]
    present = [c for c in test_cols if c in foundry.columns]
    lines.append(
        foundry[present].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    lines.append(f"\n  NeuralBench EEGNet — per-seed test metrics:")
    nb_cols = ["seed"] + [m for m in CORE_METRICS if m in nb.columns]
    lines.append(
        nb[nb_cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    lines.append(f"\n  Head-to-head (mean ± SD):")
    lines.append(f"  {'Metric':<20s} {'Foundry (test)':>18s} {'NeuralBench (test)':>20s} {'Delta':>10s} {'|Delta|':>10s}")
    lines.append("  " + "-" * 82)
    for metric in CORE_METRICS:
        f_col = f"test_{metric}"
        if f_col not in foundry.columns or metric not in nb.columns:
            continue
        f_mean = foundry[f_col].mean()
        f_std = foundry[f_col].std()
        n_mean = nb[metric].mean()
        n_std = nb[metric].std()
        delta = f_mean - n_mean
        lines.append(
            f"  {metric:<20s} {f_mean:>7.4f}±{f_std:.4f}   {n_mean:>7.4f}±{n_std:.4f}   {delta:>+8.4f}   {abs(delta):>8.4f}"
        )

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {
    "Foundry Matched EEGNet": "#4c78a8",
    "NeuralBench EEGNet": "#e45756",
}


def plot_test_comparison_bars(
    all_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Path:
    """Three-panel grouped bar chart: Foundry vs NeuralBench test balanced accuracy."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (task, (foundry, nb)) in zip(axes, all_results.items()):
        f_mean = foundry["test_balanced_acc"].mean()
        f_std = foundry["test_balanced_acc"].std()
        n_mean = nb["balanced_acc"].mean()
        n_std = nb["balanced_acc"].std()

        x = np.arange(2)
        means = [n_mean, f_mean]
        stds = [n_std, f_std]
        labels = ["NeuralBench\nEEGNet", "Foundry\nMatched EEGNet"]
        colors = [COLORS["NeuralBench EEGNet"], COLORS["Foundry Matched EEGNet"]]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                      edgecolor="white", width=0.6)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=10)

        delta = f_mean - n_mean
        ax.set_title(f"{task}\n(delta = {delta:+.1%} pp)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test balanced accuracy (mean ± SD, 3 seeds)")
    fig.suptitle(
        "Foundry Matched EEGNet vs NeuralBench EEGNet — Test Set",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_test_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_seed_scatter(
    all_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Path:
    """Per-seed scatter: Foundry vs NeuralBench test balanced accuracy."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (task, (foundry, nb)) in zip(axes, all_results.items()):
        foundry_by_seed = foundry.set_index("seed")["test_balanced_acc"]
        nb_by_seed = nb.set_index("seed")["balanced_acc"]
        common = sorted(set(foundry_by_seed.index) & set(nb_by_seed.index))
        if not common:
            ax.set_title(f"{task} — no common seeds")
            continue

        nb_vals = [nb_by_seed[s] for s in common]
        f_vals = [foundry_by_seed[s] for s in common]

        ax.scatter(nb_vals, f_vals, color=COLORS["Foundry Matched EEGNet"],
                   s=80, zorder=5, edgecolors="white")
        for s, nv, fv in zip(common, nb_vals, f_vals):
            ax.annotate(f"s{s}", (nv, fv), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)

        all_vals = nb_vals + f_vals
        lo = min(all_vals) - 0.05
        hi = max(all_vals) + 0.05
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8, alpha=0.6, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("NeuralBench test bal. acc.")
        ax.set_title(task, fontsize=11)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Foundry test bal. acc.")
    fig.suptitle(
        "Per-seed parity: Foundry Matched EEGNet vs NeuralBench",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_per_seed_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(
    all_curves: dict[str, pd.DataFrame],
) -> Path:
    """Epoch-level val balanced accuracy training curves for each task."""
    n = len(all_curves)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    seed_colors = {33: "#4c78a8", 34: "#e45756", 35: "#72b7b2"}

    for ax, (task, df) in zip(axes, all_curves.items()):
        if df.empty:
            ax.set_title(f"{task} — no curve data")
            continue
        for seed in SEEDS:
            sub = df[df["seed"] == seed].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["val_balanced_acc"],
                    color=seed_colors.get(seed, "#999"), linewidth=1.8,
                    label=f"seed {seed}", marker="o", markersize=3)
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Val balanced accuracy")
    fig.suptitle(
        "Foundry Matched EEGNet — Validation Training Curves",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta_summary(
    all_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> Path:
    """Horizontal bar chart showing delta (Foundry - NeuralBench) per task and metric."""
    tasks = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    y_positions = []
    y_labels = []
    colors = []
    vals = []

    pos = 0
    for task in tasks:
        foundry, nb = all_results[task]
        for metric in CORE_METRICS:
            f_col = f"test_{metric}"
            if f_col not in foundry.columns or metric not in nb.columns:
                continue
            delta = foundry[f_col].mean() - nb[metric].mean()
            y_positions.append(pos)
            y_labels.append(f"{task}\n{metric}")
            vals.append(delta)
            colors.append("#4c78a8" if delta >= 0 else "#e45756")
            pos += 1
        pos += 0.5

    ax.barh(y_positions, vals, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.axvline(-0.02, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label="±2 pp threshold")
    ax.axvline(0.02, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

    for yp, v in zip(y_positions, vals):
        ax.text(v + 0.002 * np.sign(v), yp, f"{v:+.4f}", va="center",
                fontsize=8, ha="left" if v >= 0 else "right")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Delta (Foundry − NeuralBench)")
    ax.set_title("Test metric deltas: Foundry Matched EEGNet vs NeuralBench", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_delta_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api = wandb.Api()
    all_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_curves: dict[str, pd.DataFrame] = {}
    all_text: list[str] = []

    for label, spec in TASKS.items():
        print(f"\n{'=' * 72}")
        print(f"Processing {label}...")
        print("=" * 72)

        print(f"  Loading NeuralBench reference ({spec['nb_task_name'] or 'p3'})...")
        nb = load_neuralbench_reference(spec["nb_task_name"])
        nb.to_csv(CSV_DIR / f"{STEM}_{label.lower().replace(' ', '_')}_neuralbench.csv",
                  index=False)

        print(f"  Fetching Foundry matched runs (group={spec['group']})...")
        foundry = fetch_foundry_matched(api, spec["group"], spec["foundry_task_key"])
        foundry.to_csv(CSV_DIR / f"{STEM}_{label.lower().replace(' ', '_')}_foundry.csv",
                       index=False)

        all_results[label] = (foundry, nb)
        txt = print_task_comparison(label, foundry, nb)
        all_text.append(txt)

        print(f"\n  Fetching training curves for {label}...")
        curves = fetch_val_training_curves(api, foundry, spec["foundry_task_key"])
        if not curves.empty:
            curves.to_csv(
                CSV_DIR / f"{STEM}_{label.lower().replace(' ', '_')}_curves.csv",
                index=False,
            )
        all_curves[label] = curves

    # --- Combined summary table ---
    print("\n\n" + "=" * 72)
    print("PARITY SUMMARY — All Tasks (test vs test)")
    print("=" * 72)
    summary_rows = []
    for task, (foundry, nb) in all_results.items():
        for metric in CORE_METRICS:
            f_col = f"test_{metric}"
            if f_col not in foundry.columns or metric not in nb.columns:
                continue
            f_mean = foundry[f_col].mean()
            f_std = foundry[f_col].std()
            n_mean = nb[metric].mean()
            n_std = nb[metric].std()
            delta = f_mean - n_mean
            summary_rows.append({
                "task": task,
                "metric": metric,
                "foundry_mean": f_mean,
                "foundry_std": f_std,
                "neuralbench_mean": n_mean,
                "neuralbench_std": n_std,
                "delta": delta,
                "abs_delta": abs(delta),
                "within_2pp": abs(delta) <= 0.02,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CSV_DIR / f"{STEM}_parity_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    parity_pass = summary["within_2pp"].all()
    bal_acc_summary = summary[summary["metric"] == "balanced_acc"]
    bal_acc_pass = bal_acc_summary["within_2pp"].all()
    print(f"\nBalanced accuracy within ±2 pp for ALL tasks: {'YES' if bal_acc_pass else 'NO'}")
    print(f"All metrics within ±2 pp for ALL tasks: {'YES' if parity_pass else 'NO'}")

    for _, row in bal_acc_summary.iterrows():
        print(f"  {row['task']}: delta = {row['delta']:+.4f} "
              f"({'PASS' if row['within_2pp'] else 'FAIL'})")

    # --- Figures ---
    print("\n\nGenerating figures...")
    figs: list[Path] = []

    figs.append(plot_test_comparison_bars(all_results))
    print(f"  Saved: {figs[-1]}")

    figs.append(plot_per_seed_scatter(all_results))
    print(f"  Saved: {figs[-1]}")

    figs.append(plot_delta_summary(all_results))
    print(f"  Saved: {figs[-1]}")

    non_empty_curves = {k: v for k, v in all_curves.items() if not v.empty}
    if non_empty_curves:
        figs.append(plot_training_curves(non_empty_curves))
        print(f"  Saved: {figs[-1]}")

    print("\nDone. Generated figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

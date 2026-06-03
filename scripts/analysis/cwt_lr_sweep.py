"""Generate figures for Experiment 3b: CWT Learning Rate Multiplier Sweep.

Compares CWT tokenizer with different learning rate multipliers for the
CWT parameters (freqs, n_cycles) at 200 Hz target token rate:
    CWT 1×   (baseline from TOKEN_RATE_SWEEP)
    CWT 10×  (from CWT_LR_AND_PARAM_MATCH local runs)
    CWT 50×  (from CWT_LR_AND_PARAM_MATCH local runs)
    CWT 100× (from CWT_LR_AND_PARAM_MATCH local runs)

Outputs (in docs/figures/):
    cwt_lr_sweep_bar.pdf        — bar chart of AUROC by multiplier
    cwt_lr_sweep_freqs.pdf      — learned frequencies vs initialization

Usage:
    uv run scripts/analysis/cwt_lr_sweep.py
    uv run scripts/analysis/cwt_lr_sweep.py --plot-only
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PALETTE_TEMPORAL  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (7, 4),
    }
)

OUTPUT_DIR = Path("docs/figures")
CACHE_DIR = Path("docs/figures/data")
RUNS_DIR = Path("outputs/runs/CWT_LR_AND_PARAM_MATCH")

COL_CWT = PALETTE_TEMPORAL["cwt"]
AUROC_KEY = "val/ajile_active_behavior_auroc"

MULTIPLIERS = [1, 10, 50, 100]
MULTIPLIER_COLORS = {
    1: "#56B4E9",  # sky blue (baseline)
    10: "#0072B2",  # blue
    50: "#009E73",  # bluish green
    100: "#D55E00",  # vermillion
}

INIT_FREQS = np.logspace(np.log10(0.5), np.log10(30), 9)
INIT_NCYCLES = 2.5


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def collect_lr_sweep_data(plot_only: bool = False) -> pd.DataFrame:
    """Build a dataframe with AUROC and learned params for all multipliers."""
    cache_csv = CACHE_DIR / "CWT_LR_SWEEP.csv"

    if plot_only and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        print(f"  Loaded {len(df)} runs from cache")
        return df

    rows: list[dict] = []

    # Baseline CWT (1×) at 200 Hz from TOKEN_RATE_SWEEP cache
    sweep_csv = CACHE_DIR / "TOKEN_RATE_SWEEP.csv"
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        for _, r in sweep_df[
            (sweep_df["rate"] == 200) & (sweep_df["tokenizer"] == "CWT")
        ].iterrows():
            rows.append(
                {
                    "run_name": r["run_name"],
                    "multiplier": 1,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "TOKEN_RATE_SWEEP",
                }
            )
    else:
        print(f"  Warning: {sweep_csv} not found, baseline will be missing")

    # LR sweep runs (10×, 50×, 100×)
    for d in sorted(os.listdir(RUNS_DIR)):
        run_path = RUNS_DIR / d
        if not run_path.is_dir() or "cwtlr" not in d:
            continue

        # Extract multiplier from name like per_channel_cwt_cwtlr50x_rate200_fold0
        mult_str = d.split("cwtlr")[1].split("x")[0]
        try:
            multiplier = int(mult_str)
        except ValueError:
            continue

        summary_files = sorted(run_path.rglob("wandb-summary.json"))
        if not summary_files:
            continue
        with open(summary_files[-1]) as f:
            summary = json.load(f)

        auroc = summary.get(AUROC_KEY, {})
        if isinstance(auroc, dict):
            auroc = auroc.get("max")
        if auroc is None:
            continue

        parts = d.split("_")
        fold = int(
            [p for p in parts if p.startswith("fold")][0].replace("fold", "")
        )

        # Extract learned frequencies
        freq_keys = sorted(
            [
                k
                for k in summary.keys()
                if "freqs_hz/" in k
                and not any(
                    x in k for x in ["mean", "std", "min", "max", "norm"]
                )
            ]
        )
        learned_freqs = [summary[k] for k in freq_keys] if freq_keys else []

        # Extract learned n_cycles
        ncycles_keys = sorted(
            [
                k
                for k in summary.keys()
                if "n_cycles/" in k
                and "unconstrained" not in k
                and not any(
                    x in k for x in ["mean", "std", "min", "max", "norm"]
                )
            ]
        )
        learned_ncycles = (
            [summary[k] for k in ncycles_keys] if ncycles_keys else []
        )

        # Extract update-to-param ratios
        upr_freqs = summary.get(
            "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
            "/optimizer/update_to_param_ratio"
        )
        upr_ncycles = summary.get(
            "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
            "/optimizer/update_to_param_ratio"
        )

        rows.append(
            {
                "run_name": d,
                "multiplier": multiplier,
                "fold": fold,
                "auroc": auroc,
                "source": "CWT_LR_AND_PARAM_MATCH",
                "learned_freqs": json.dumps(learned_freqs),
                "learned_ncycles": json.dumps(learned_ncycles),
                "upr_freqs": upr_freqs,
                "upr_ncycles": upr_ncycles,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"  Cached {len(df)} runs to {cache_csv}")
    return df


def plot_lr_sweep_bar(df: pd.DataFrame) -> None:
    """Bar chart: AUROC by LR multiplier."""
    agg = (
        df.groupby("multiplier")["auroc"]
        .agg(["mean", "std"])
        .reindex(MULTIPLIERS)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(agg))

    bar_colors = [MULTIPLIER_COLORS[m] for m in agg["multiplier"]]
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=5,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.55,
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.002,
            f"{row['mean']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    xlabels = [f"{int(m)}×" for m in agg["multiplier"]]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_xlabel("CWT Learning Rate Multiplier")
    ax.set_ylabel("Behavior AUROC")
    ax.set_title("CWT LR Multiplier Sweep at 200 Hz\n(mean ± std over 2 folds)")

    ymin = agg["mean"].min() - 0.015
    ymax = agg["mean"].max() + agg["std"].max() + 0.012
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "cwt_lr_sweep_bar.pdf")


def plot_lr_sweep_freqs(df: pd.DataFrame) -> None:
    """Learned frequencies vs initialization for each multiplier."""
    sweep_runs = df[df["multiplier"] > 1].copy()
    if sweep_runs.empty or "learned_freqs" not in sweep_runs.columns:
        print("  Skipping frequency plot (no learned_freqs data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: learned frequencies
    ax = axes[0]
    freq_indices = np.arange(len(INIT_FREQS))

    ax.plot(
        freq_indices,
        INIT_FREQS,
        "k--",
        linewidth=1.5,
        label="Initialization",
        zorder=10,
    )

    for mult in [10, 50, 100]:
        mult_runs = sweep_runs[sweep_runs["multiplier"] == mult]
        all_freqs = []
        for _, row in mult_runs.iterrows():
            freqs = json.loads(row["learned_freqs"])
            if freqs:
                all_freqs.append(freqs)
        if not all_freqs:
            continue

        mean_freqs = np.mean(all_freqs, axis=0)
        std_freqs = np.std(all_freqs, axis=0)
        color = MULTIPLIER_COLORS[mult]
        ax.errorbar(
            freq_indices,
            mean_freqs,
            yerr=std_freqs,
            fmt="o-",
            color=color,
            capsize=3,
            markersize=5,
            linewidth=1.2,
            label=f"{mult}× LR",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Frequency Bin Index")
    ax.set_ylabel("Center Frequency (Hz)")
    ax.set_title("Learned Frequencies")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(freq_indices)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right panel: learned n_cycles
    ax = axes[1]
    ax.axhline(
        INIT_NCYCLES,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Initialization (2.5)",
    )

    for mult in [10, 50, 100]:
        mult_runs = sweep_runs[sweep_runs["multiplier"] == mult]
        all_ncycles = []
        for _, row in mult_runs.iterrows():
            if pd.isna(row.get("learned_ncycles")):
                continue
            ncycles = json.loads(row["learned_ncycles"])
            if ncycles:
                all_ncycles.append(ncycles)
        if not all_ncycles:
            continue

        mean_nc = np.mean(all_ncycles, axis=0)
        std_nc = np.std(all_ncycles, axis=0)
        color = MULTIPLIER_COLORS[mult]
        ax.errorbar(
            freq_indices,
            mean_nc,
            yerr=std_nc,
            fmt="o-",
            color=color,
            capsize=3,
            markersize=5,
            linewidth=1.2,
            label=f"{mult}× LR",
        )

    ax.set_xlabel("Frequency Bin Index")
    ax.set_ylabel("Cycle Count")
    ax.set_title("Learned Cycle Counts")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(freq_indices)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, "cwt_lr_sweep_freqs.pdf")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Experiment 3b (CWT LR sweep) figures...")
    df = collect_lr_sweep_data(plot_only=args.plot_only)
    print("\nAUROC by multiplier:")
    print(df.groupby("multiplier")["auroc"].agg(["mean", "std", "count"]))
    plot_lr_sweep_bar(df)
    plot_lr_sweep_freqs(df)
    print("\nDone!")


if __name__ == "__main__":
    main()

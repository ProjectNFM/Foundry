"""Generate figures for the TOKEN_RATE_SWEEP experiments.

Experiments:
  1. Token rate scaling (CWT vs CNN at 100/200/400 Hz)
  2. CWT+CNN hybrid comparison
  3a. CWT gradient diagnostics

Outputs (in docs/figures/):
    token_rate_sweep_bar.pdf         — grouped bar: all 3 tokenizers × 3 rates
    token_rate_scaling_line.pdf      — AUROC vs token rate line plot
    cwt_gradient_diagnostics.pdf     — gradient/update diagnostics for CWT params

Usage:
    uv run scripts/analysis/token_rate_sweep.py
    uv run scripts/analysis/token_rate_sweep.py --plot-only
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

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
RUNS_DIR = Path("outputs/runs/TOKEN_RATE_SWEEP")

COL_CWT = PALETTE_TEMPORAL["cwt"]
COL_CNN = PALETTE_TEMPORAL["cnn"]
COL_CWT_CNN = "#E69F00"  # Okabe-Ito yellow/amber for hybrid

TOKENIZER_COLORS = {
    "CWT": COL_CWT,
    "CNN": COL_CNN,
    "CWT+CNN": COL_CWT_CNN,
}

TOKENIZER_PARAMS = {
    "CWT": 3666,
    "CNN": 3924,
    "CWT+CNN": 59858,
}

RATES = [100, 200, 400]
AUROC_KEY = "val/ajile_active_behavior_auroc"


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def _fetch_from_wandb() -> tuple[list[dict], dict]:
    """Fetch TOKEN_RATE_SWEEP data from W&B API when local runs aren't available."""
    import wandb
    from common import WANDB_ENTITY, WANDB_PROJECT

    api = wandb.Api(timeout=60)
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": "TOKEN_RATE_SWEEP"},
    )

    rows = []
    grad_data = {}

    for run in runs:
        s = run.summary._json_dict
        d = run.display_name

        auroc = s.get(AUROC_KEY, {})
        if isinstance(auroc, dict):
            auroc = auroc.get("max")
        if auroc is None:
            continue

        if "cwt_cnn" in d:
            tokenizer = "CWT+CNN"
        elif "cwt" in d:
            tokenizer = "CWT"
        else:
            tokenizer = "CNN"

        parts = d.split("_")
        try:
            rate = int(
                [p for p in parts if p.startswith("rate")][0].replace(
                    "rate", ""
                )
            )
            fold = int(
                [p for p in parts if p.startswith("fold")][0].replace(
                    "fold", ""
                )
            )
        except (IndexError, ValueError):
            continue

        rows.append(
            {
                "run_name": d,
                "tokenizer": tokenizer,
                "rate": rate,
                "fold": fold,
                "auroc": auroc,
            }
        )

        if tokenizer in ("CWT", "CWT+CNN"):
            freqs_upr = s.get(
                "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                "/optimizer/update_to_param_ratio"
            )
            ncycles_upr = s.get(
                "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                "/optimizer/update_to_param_ratio"
            )
            grad_norm_freqs = s.get(
                "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                "/grad/norm"
            )
            grad_norm_ncycles = s.get(
                "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                "/grad/norm"
            )
            eff_step_freqs = s.get(
                "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                "/optimizer/effective_step_norm"
            )
            eff_step_ncycles = s.get(
                "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                "/optimizer/effective_step_norm"
            )

            grad_data[d] = {
                "tokenizer": tokenizer,
                "rate": rate,
                "fold": fold,
                "freqs_update_to_param_ratio": freqs_upr,
                "ncycles_update_to_param_ratio": ncycles_upr,
                "freqs_grad_norm": grad_norm_freqs,
                "ncycles_grad_norm": grad_norm_ncycles,
                "freqs_effective_step_norm": eff_step_freqs,
                "ncycles_effective_step_norm": eff_step_ncycles,
            }

    return rows, grad_data


def collect_data(plot_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Extract metrics and gradient diagnostics from local wandb summaries."""
    cache_csv = CACHE_DIR / "TOKEN_RATE_SWEEP.csv"
    cache_grad = CACHE_DIR / "TOKEN_RATE_SWEEP_gradients.json"

    if plot_only and cache_csv.exists() and cache_grad.exists():
        df = pd.read_csv(cache_csv)
        with open(cache_grad) as f:
            grad_data = json.load(f)
        print(f"  Loaded {len(df)} runs from cache")
        return df, grad_data

    rows = []
    grad_data = {}

    if RUNS_DIR.exists():
        import os

        for d in sorted(os.listdir(RUNS_DIR)):
            run_path = RUNS_DIR / d
            if not run_path.is_dir() or d.startswith("."):
                continue

            summary_files = list(run_path.rglob("wandb-summary.json"))
            if not summary_files:
                continue
            with open(sorted(summary_files)[-1]) as f:
                summary = json.load(f)

            auroc = summary.get(AUROC_KEY, {})
            if isinstance(auroc, dict):
                auroc = auroc.get("max")
            if auroc is None:
                continue

            if "cwt_cnn" in d:
                tokenizer = "CWT+CNN"
            elif "cwt" in d:
                tokenizer = "CWT"
            else:
                tokenizer = "CNN"

            parts = d.split("_")
            rate = int(
                [p for p in parts if p.startswith("rate")][0].replace(
                    "rate", ""
                )
            )
            fold = int(
                [p for p in parts if p.startswith("fold")][0].replace(
                    "fold", ""
                )
            )

            rows.append(
                {
                    "run_name": d,
                    "tokenizer": tokenizer,
                    "rate": rate,
                    "fold": fold,
                    "auroc": auroc,
                }
            )

            if tokenizer in ("CWT", "CWT+CNN"):
                freqs_upr = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                    "/optimizer/update_to_param_ratio"
                )
                ncycles_upr = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                    "/optimizer/update_to_param_ratio"
                )
                grad_norm_freqs = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                    "/grad/norm"
                )
                grad_norm_ncycles = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                    "/grad/norm"
                )
                eff_step_freqs = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.freqs_unconstrained"
                    "/optimizer/effective_step_norm"
                )
                eff_step_ncycles = summary.get(
                    "params/tokenizer.temporal_embedding.cwt.n_cycles_unconstrained"
                    "/optimizer/effective_step_norm"
                )

                grad_data[d] = {
                    "tokenizer": tokenizer,
                    "rate": rate,
                    "fold": fold,
                    "freqs_update_to_param_ratio": freqs_upr,
                    "ncycles_update_to_param_ratio": ncycles_upr,
                    "freqs_grad_norm": grad_norm_freqs,
                    "ncycles_grad_norm": grad_norm_ncycles,
                    "freqs_effective_step_norm": eff_step_freqs,
                    "ncycles_effective_step_norm": eff_step_ncycles,
                }
    else:
        print(f"  {RUNS_DIR} not found, fetching from W&B API...")
        rows, grad_data = _fetch_from_wandb()
        print(f"  Fetched {len(rows)} runs from W&B")

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)

    with open(cache_grad, "w") as f:
        json.dump(grad_data, f, indent=2)
    print(f"  Cached {len(df)} runs and {len(grad_data)} gradient records")

    return df, grad_data


def plot_grouped_bar(df: pd.DataFrame) -> None:
    """Grouped bar chart: tokenizer × rate, colored by tokenizer family."""
    agg = (
        df.groupby(["tokenizer", "rate"])["auroc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    tokenizers = ["CWT", "CNN", "CWT+CNN"]
    n_tok = len(tokenizers)
    n_rates = len(RATES)
    width = 0.22
    x = np.arange(n_rates)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, tok in enumerate(tokenizers):
        sub = agg[agg["tokenizer"] == tok].set_index("rate")
        vals = [sub.loc[r, "mean"] if r in sub.index else 0 for r in RATES]
        stds = [sub.loc[r, "std"] if r in sub.index else 0 for r in RATES]

        offset = (i - (n_tok - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=stds,
            capsize=3,
            color=TOKENIZER_COLORS[tok],
            edgecolor="white",
            linewidth=0.5,
            label=f"{tok} ({TOKENIZER_PARAMS[tok]:,} params)",
        )
        for bar, val, std in zip(bars, vals, stds):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.002,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} Hz" for r in RATES], fontsize=9)
    ax.set_xlabel("Target Token Rate")
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "Token Rate Sweep — CWT vs CNN vs CWT+CNN\n(mean ± std over 2 folds)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = agg["mean"].min() - 0.02
    ymax = agg["mean"].max() + 0.02
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    _save(fig, "token_rate_sweep_bar.pdf")


def plot_scaling_line(df: pd.DataFrame) -> None:
    """Line plot: AUROC vs token rate for each tokenizer."""
    agg = (
        df.groupby(["tokenizer", "rate"])["auroc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    tokenizers = ["CWT", "CNN", "CWT+CNN"]
    markers = {"CWT": "o", "CNN": "s", "CWT+CNN": "D"}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for tok in tokenizers:
        sub = agg[agg["tokenizer"] == tok].sort_values("rate")
        ax.errorbar(
            sub["rate"],
            sub["mean"],
            yerr=sub["std"],
            marker=markers[tok],
            color=TOKENIZER_COLORS[tok],
            linewidth=1.5,
            markersize=7,
            capsize=4,
            label=f"{tok} ({TOKENIZER_PARAMS[tok]:,} params)",
        )

    ax.set_xlabel("Target Token Rate (Hz)")
    ax.set_ylabel("Behavior AUROC")
    ax.set_title("AUROC vs Token Rate")
    ax.set_xticks(RATES)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = agg["mean"].min() - 0.015
    ymax = agg["mean"].max() + 0.015
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    _save(fig, "token_rate_scaling_line.pdf")


def plot_gradient_diagnostics(grad_data: dict) -> None:
    """Visualize CWT gradient diagnostics across runs."""
    if not grad_data:
        print("  No gradient data, skipping diagnostics plot")
        return

    cwt_runs = {k: v for k, v in grad_data.items() if v["tokenizer"] == "CWT"}
    cwt_cnn_runs = {
        k: v for k, v in grad_data.items() if v["tokenizer"] == "CWT+CNN"
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: update_to_param_ratio for freqs and n_cycles
    ax = axes[0]
    categories = []
    vals_freqs = []
    vals_ncycles = []

    for runs, label_prefix, color in [
        (cwt_runs, "CWT", COL_CWT),
        (cwt_cnn_runs, "CWT+CNN", COL_CWT_CNN),
    ]:
        for name, data in sorted(runs.items()):
            rate = data["rate"]
            fold = data["fold"]
            categories.append(f"{label_prefix}\n{rate}Hz f{fold}")
            vals_freqs.append(data.get("freqs_update_to_param_ratio", 0))
            vals_ncycles.append(data.get("ncycles_update_to_param_ratio", 0))

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(
        x - width / 2,
        vals_freqs,
        width,
        color=COL_CWT,
        alpha=0.8,
        label="freqs",
    )
    ax.bar(
        x + width / 2,
        vals_ncycles,
        width,
        color="#CC79A7",
        alpha=0.8,
        label="n_cycles",
    )
    ax.set_yscale("log")
    ax.set_ylabel("update-to-param ratio")
    ax.set_title("Adam Effective Update / Parameter Magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1e-4, color="#999", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(
        len(x) - 0.5,
        1.2e-4,
        "1e-4 reference",
        fontsize=6,
        color="#999",
        ha="right",
    )

    # Panel 2: gradient norm vs effective step
    ax = axes[1]
    for runs, label, color, marker in [
        (cwt_runs, "CWT", COL_CWT, "o"),
        (cwt_cnn_runs, "CWT+CNN", COL_CWT_CNN, "D"),
    ]:
        gn = [d.get("freqs_grad_norm", 0) for d in runs.values()]
        es = [d.get("freqs_effective_step_norm", 0) for d in runs.values()]
        ax.scatter(
            gn,
            es,
            color=color,
            marker=marker,
            s=50,
            label=f"{label} (freqs)",
            zorder=3,
        )

        gn_c = [d.get("ncycles_grad_norm", 0) for d in runs.values()]
        es_c = [d.get("ncycles_effective_step_norm", 0) for d in runs.values()]
        ax.scatter(
            gn_c,
            es_c,
            color=color,
            marker=marker,
            s=50,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            label=f"{label} (n_cycles)",
            zorder=3,
        )

    ax.set_xlabel("Gradient Norm")
    ax.set_ylabel("Effective Step Norm")
    ax.set_title("Gradient Norm vs Effective Update")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "CWT Gradient Diagnostics — TOKEN_RATE_SWEEP", fontsize=12, y=1.02
    )
    fig.tight_layout()
    _save(fig, "cwt_gradient_diagnostics.pdf")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating TOKEN_RATE_SWEEP figures...")
    df, grad_data = collect_data(plot_only=args.plot_only)

    plot_grouped_bar(df)
    plot_scaling_line(df)
    plot_gradient_diagnostics(grad_data)
    print("Done!")


if __name__ == "__main__":
    main()

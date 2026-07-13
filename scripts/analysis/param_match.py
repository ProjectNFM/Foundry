"""Generate figures for Experiments 4 & 5: CWT+CNN capacity comparison.

Compares five tokenizers at 200 Hz target token rate:
    CWT          (3,666 params)  — from TOKEN_RATE_SWEEP cache
    CNN 12f      (3,924 params)  — from TOKEN_RATE_SWEEP cache
    CWT+CNN 12f  (5,778 params)  — from CWT_CNN_12F local runs
    CNN 64f     (50,048 params)  — from CWT_LR_AND_PARAM_MATCH local runs
    CWT+CNN 64f (59,858 params)  — from CWT_LR_AND_PARAM_MATCH local runs

Outputs (in docs/figures/):
    param_match_bar.pdf  — grouped bar chart with parameter counts

Usage:
    uv run scripts/analysis/param_match.py
    uv run scripts/analysis/param_match.py --plot-only
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
RUNS_DIR_PARAM_MATCH = Path("outputs/runs/CWT_LR_AND_PARAM_MATCH")
RUNS_DIR_12F = Path("outputs/runs/CWT_CNN_12F")

COL_CWT = PALETTE_TEMPORAL["cwt"]
COL_CNN = PALETTE_TEMPORAL["cnn"]
COL_CWT_CNN = "#E69F00"  # Okabe-Ito amber for hybrid
COL_CWT_CNN_12F = "#F0E442"  # Okabe-Ito yellow for low-capacity hybrid
COL_CNN_64F = "#CC79A7"  # Okabe-Ito reddish purple for param-matched CNN

AUROC_KEY = "val/ajile_active_behavior_auroc"

TOKENIZER_SPECS = [
    ("CWT", 3_666, COL_CWT),
    ("CNN (12f)", 3_924, COL_CNN),
    ("CWT+CNN (12f)", 5_778, COL_CWT_CNN_12F),
    ("CNN (64f)", 50_048, COL_CNN_64F),
    ("CWT+CNN (64f)", 59_858, COL_CWT_CNN),
]


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def _fetch_param_match_from_wandb(
    group: str, prefixes: dict[str, str]
) -> list[dict]:
    """Fetch param match runs from W&B API."""
    import wandb
    from common import WANDB_ENTITY, WANDB_PROJECT

    api = wandb.Api(timeout=60)
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": group},
    )

    rows = []
    for run in runs:
        d = run.display_name
        s = run.summary._json_dict

        matched_label = None
        for prefix, label in prefixes.items():
            if d.startswith(prefix):
                matched_label = label
                break
        if matched_label is None:
            continue

        auroc = s.get(AUROC_KEY, {})
        if isinstance(auroc, dict):
            auroc = auroc.get("max")
        if auroc is None:
            continue

        parts = d.split("_")
        try:
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
                "tokenizer": matched_label,
                "fold": fold,
                "auroc": auroc,
                "source": group,
            }
        )

    return rows


def collect_exp4_data(plot_only: bool = False) -> pd.DataFrame:
    """Build a dataframe with AUROC for all four tokenizers at 200 Hz."""
    cache_csv = CACHE_DIR / "PARAM_MATCH.csv"

    if plot_only and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        print(f"  Loaded {len(df)} runs from cache")
        return df

    rows: list[dict] = []

    # --- Pull CWT and CNN (12f) at 200 Hz from TOKEN_RATE_SWEEP cache ---
    sweep_csv = CACHE_DIR / "TOKEN_RATE_SWEEP.csv"
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        for _, r in sweep_df[
            (sweep_df["rate"] == 200)
            & (sweep_df["tokenizer"].isin(["CWT", "CNN"]))
        ].iterrows():
            label = "CWT" if r["tokenizer"] == "CWT" else "CNN (12f)"
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": label,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "TOKEN_RATE_SWEEP",
                }
            )
    else:
        print(
            f"  Warning: {sweep_csv} not found, CWT/CNN 12f bars will be missing"
        )

    # --- Pull from local run directories or W&B ---
    run_sources = [
        (
            RUNS_DIR_PARAM_MATCH,
            {
                "per_channel_cwt_cnn_rate200": "CWT+CNN (64f)",
                "per_channel_resample_cnn_64f_rate200": "CNN (64f)",
            },
            "CWT_LR_AND_PARAM_MATCH",
        ),
        (
            RUNS_DIR_12F,
            {"per_channel_cwt_cnn_12f_rate200": "CWT+CNN (12f)"},
            "CWT_CNN_12F",
        ),
    ]

    for runs_dir, prefixes, source_name in run_sources:
        if runs_dir.exists():
            for d in sorted(os.listdir(runs_dir)):
                run_path = runs_dir / d
                if not run_path.is_dir() or d.startswith("."):
                    continue

                matched_label = None
                for prefix, label in prefixes.items():
                    if d.startswith(prefix):
                        matched_label = label
                        break
                if matched_label is None:
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

                parts = d.split("_")
                fold = int(
                    [p for p in parts if p.startswith("fold")][0].replace(
                        "fold", ""
                    )
                )

                rows.append(
                    {
                        "run_name": d,
                        "tokenizer": matched_label,
                        "fold": fold,
                        "auroc": auroc,
                        "source": source_name,
                    }
                )
        else:
            print(
                f"  {runs_dir} not found, fetching {source_name} from W&B API..."
            )
            wandb_rows = _fetch_param_match_from_wandb(source_name, prefixes)
            rows.extend(wandb_rows)
            print(f"  Fetched {len(wandb_rows)} runs from W&B")

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"  Cached {len(df)} runs to {cache_csv}")
    return df


def plot_param_match_bar(df: pd.DataFrame) -> None:
    """Bar chart comparing the four tokenizers, annotated with param counts."""
    tok_order = [name for name, _, _ in TOKENIZER_SPECS]
    colors = {name: color for name, _, color in TOKENIZER_SPECS}
    params = {name: p for name, p, _ in TOKENIZER_SPECS}

    agg = (
        df.groupby("tokenizer")["auroc"]
        .agg(["mean", "std"])
        .reindex(tok_order)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(agg))

    bar_colors = [colors[t] for t in agg["tokenizer"]]
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
            bar.get_height() + row["std"] + 0.003,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    xlabels = [f"{t}\n({params[t]:,} params)" for t in agg["tokenizer"]]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "CWT+CNN Capacity Comparison at 200 Hz\n(mean ± std over 2 folds)"
    )

    ymin = agg["mean"].min() - 0.025
    ymax = agg["mean"].max() + agg["std"].max() + 0.015
    ax.set_ylim(ymin, ymax)

    ax.axvline(2.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(
        1.0,
        ymin + 0.002,
        "low capacity (12 filters)",
        ha="center",
        fontsize=7,
        color="#999999",
    )
    ax.text(
        3.5,
        ymin + 0.002,
        "high capacity (64 filters)",
        ha="center",
        fontsize=7,
        color="#999999",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "param_match_bar.pdf")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Experiments 4 & 5 (param-match + 12f) figures...")
    df = collect_exp4_data(plot_only=args.plot_only)
    if df.empty:
        print("  No data available, skipping plots")
        return
    print(df.groupby("tokenizer")["auroc"].agg(["mean", "std", "count"]))
    plot_param_match_bar(df)
    print("Done!")


if __name__ == "__main__":
    main()

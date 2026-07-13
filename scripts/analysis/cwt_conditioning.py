"""Generate figures for Experiment 7: CWT+CNN Scalogram Conditioning.

Tests whether highpass filtering, log-magnitude compression, and
per-frequency instance normalization improve downstream behavior
classification when applied to the CWT+CNN scalogram.

Outputs (in docs/figures/):
    cwt_conditioning_bar.pdf — bar chart of all conditioning variants + baselines

Usage:
    uv run scripts/analysis/cwt_conditioning.py
    uv run scripts/analysis/cwt_conditioning.py --plot-only
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
from common import PALETTE_TEMPORAL, make_base_parser  # noqa: E402

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

AUROC_KEY = "val/ajile_active_behavior_auroc"

COL_CWT_CNN = "#E69F00"  # Okabe-Ito amber — hybrid baseline
COL_CNN = PALETTE_TEMPORAL["cnn"]
COL_HP = "#009E73"  # Okabe-Ito bluish green — best single
COL_HP_LOG = "#56B4E9"  # sky blue
COL_LOG = "#0072B2"  # blue
COL_FNORM = "#CC79A7"  # reddish purple
COL_HP_FNORM = "#882255"  # dark purple
COL_HP_LOG_FNORM = "#332288"  # indigo

RUNS_DIR = Path("outputs/runs/CWT_CONDITIONING")

CONFIG_SPECS = [
    ("CWT+CNN", 59_858, COL_CWT_CNN),
    ("CWT+CNN+hp", 59_858, COL_HP),
    ("CWT+CNN+log", 59_858, COL_LOG),
    ("CWT+CNN+fnorm", 59_858, COL_FNORM),
    ("CWT+CNN+hp+log", 59_858, COL_HP_LOG),
    ("CWT+CNN+hp+fnorm", 59_858, COL_HP_FNORM),
    ("CWT+CNN+hp+log+fnorm", 59_858, COL_HP_LOG_FNORM),
    ("CNN 64f", 50_048, COL_CNN),
]

RUN_PREFIX_MAP = {
    "per_channel_cwt_cnn_hp_log_fnorm_rate200": "CWT+CNN+hp+log+fnorm",
    "per_channel_cwt_cnn_hp_fnorm_rate200": "CWT+CNN+hp+fnorm",
    "per_channel_cwt_cnn_hp_log_rate200": "CWT+CNN+hp+log",
    "per_channel_cwt_cnn_fnorm_rate200": "CWT+CNN+fnorm",
    "per_channel_cwt_cnn_hp_rate200": "CWT+CNN+hp",
    "per_channel_cwt_cnn_log_rate200": "CWT+CNN+log",
}


def _save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def _fetch_conditioning_from_wandb() -> list[dict]:
    """Fetch CWT_CONDITIONING runs from W&B API."""
    import wandb
    from common import WANDB_ENTITY, WANDB_PROJECT

    api = wandb.Api(timeout=60)
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": "CWT_CONDITIONING"},
    )

    rows = []
    for run in runs:
        d = run.display_name
        s = run.summary._json_dict

        matched_label = None
        for prefix, label in RUN_PREFIX_MAP.items():
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
                "source": "CWT_CONDITIONING",
            }
        )

    return rows


def collect_data(cache_dir: Path, *, plot_only: bool = False) -> pd.DataFrame:
    cache_csv = cache_dir / "CWT_CONDITIONING.csv"

    if plot_only and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        print(f"  Loaded {len(df)} rows from cache")
        return df

    rows: list[dict] = []

    # New conditioning runs from local outputs or W&B
    if RUNS_DIR.exists():
        for d in sorted(os.listdir(RUNS_DIR)):
            run_path = RUNS_DIR / d
            if not run_path.is_dir() or d.startswith("."):
                continue

            matched_label = None
            for prefix, label in RUN_PREFIX_MAP.items():
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
                    "source": "CWT_CONDITIONING",
                }
            )
    else:
        print(f"  {RUNS_DIR} not found, fetching from W&B API...")
        wandb_rows = _fetch_conditioning_from_wandb()
        rows.extend(wandb_rows)
        print(f"  Fetched {len(wandb_rows)} conditioning runs from W&B")

    # Baselines from PARAM_MATCH cache
    param_csv = cache_dir / "PARAM_MATCH.csv"
    if param_csv.exists():
        pm = pd.read_csv(param_csv)
        if not pm.empty:
            for _, r in pm[pm["tokenizer"] == "CWT+CNN (64f)"].iterrows():
                rows.append(
                    {
                        "run_name": r["run_name"],
                        "tokenizer": "CWT+CNN",
                        "fold": r["fold"],
                        "auroc": r["auroc"],
                        "source": "CWT_LR_AND_PARAM_MATCH",
                    }
                )
            for _, r in pm[pm["tokenizer"] == "CNN (64f)"].iterrows():
                rows.append(
                    {
                        "run_name": r["run_name"],
                        "tokenizer": "CNN 64f",
                        "fold": r["fold"],
                        "auroc": r["auroc"],
                        "source": "CWT_LR_AND_PARAM_MATCH",
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"  Cached {len(df)} rows to {cache_csv}")
    return df


def plot_conditioning_bar(df: pd.DataFrame, output_dir: Path) -> None:
    tok_order = [name for name, _, _ in CONFIG_SPECS]
    colors = {name: color for name, _, color in CONFIG_SPECS}
    params = {name: p for name, p, _ in CONFIG_SPECS}

    agg = (
        df.groupby("tokenizer")["auroc"]
        .agg(["mean", "std"])
        .reindex(tok_order)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    agg = agg.dropna(subset=["mean"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(agg))

    bar_colors = [colors.get(t, "#888888") for t in agg["tokenizer"]]
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=4,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.55,
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.002,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    xlabels = []
    for t in agg["tokenizer"]:
        p = params.get(t)
        if p:
            xlabels.append(f"{t}\n({p:,}p)")
        else:
            xlabels.append(t)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, ha="center")
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "CWT+CNN Scalogram Conditioning: Highpass × Log-Mag × Freq-Norm\n"
        "(mean ± std over 2 folds)"
    )

    # Divider between CWT+CNN variants and CNN baseline
    n_cwt = sum(1 for t in agg["tokenizer"] if t != "CNN 64f")
    if n_cwt < len(agg):
        ax.axvline(
            n_cwt - 0.5,
            color="#cccccc",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )

    ymin = agg["mean"].min() - 0.025
    ymax = agg["mean"].max() + agg["std"].max() + 0.015
    ax.set_ylim(ymin, ymax)

    if n_cwt > 0:
        ax.text(
            (n_cwt - 1) / 2,
            ymin + 0.002,
            "CWT+CNN variants",
            ha="center",
            fontsize=8,
            color="#666666",
        )
    if n_cwt < len(agg):
        ax.text(
            n_cwt,
            ymin + 0.002,
            "CNN baseline",
            ha="center",
            fontsize=8,
            color="#666666",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "cwt_conditioning_bar.pdf", output_dir)


def main() -> None:
    parser = make_base_parser(
        "Experiment 7: CWT+CNN Scalogram Conditioning figures"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Experiment 7 (CWT+CNN Scalogram Conditioning) figures...")
    df = collect_data(args.cache_dir, plot_only=args.plot_only)

    print("\nAUROC summary:")
    summary = (
        df.groupby("tokenizer")["auroc"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())

    plot_conditioning_bar(df, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()

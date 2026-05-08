"""Generate additional experiment-specific figures for the tokenizer journal.

Uses cached CSV data from W&B (run the per-group scripts first).

Outputs (in docs/figures/):
    concat_vs_add_comparison.pdf    — Concat vs add fusion comparison
    token_count_scaling.pdf         — Token count vs sampling rate / approach

Usage:
    uv run scripts/analysis/experiment_figures.py
    uv run scripts/analysis/experiment_figures.py --cache-dir docs/figures/data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
BLUE = "#2196F3"
ORANGE = "#FF9800"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
RED = "#E53935"


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def plot_concat_vs_add() -> None:
    """Concat vs add channel fusion from AJILE_NEW_ALLTOKS data."""
    csv_path = CACHE_DIR / "AJILE_NEW_ALLTOKS.csv"
    if not csv_path.exists():
        print(f"  Skipping concat vs add: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    pairs = {
        "CWT": ("per_channel_cwt", "per_channel_cwt_add"),
        "ResampleCNN": (
            "per_channel_resample_cnn",
            "per_channel_resample_cnn_add",
        ),
    }

    results = []
    for family, (concat_key, add_key) in pairs.items():
        for fusion, key in [("Concat", concat_key), ("Add", add_key)]:
            sub = df[df["tokenizer"] == key]
            if sub.empty:
                continue
            results.append(
                {
                    "family": family,
                    "fusion": fusion,
                    "mean": sub["best_metric"].mean(),
                    "std": sub["best_metric"].std(),
                }
            )

    rdf = pd.DataFrame(results)
    families = rdf["family"].unique()
    x = np.arange(len(families))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (fusion, color) in enumerate([("Concat", BLUE), ("Add", ORANGE)]):
        sub = rdf[rdf["fusion"] == fusion]
        vals = [sub[sub["family"] == f]["mean"].values[0] for f in families]
        stds = [sub[sub["family"] == f]["std"].values[0] for f in families]
        bars = ax.bar(
            x + i * width - width / 2,
            vals,
            width,
            yerr=stds,
            capsize=4,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=fusion,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ymin = rdf["mean"].min() - rdf["std"].max() - 0.02
    ax.set_ylim(bottom=max(0.78, ymin))
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "Channel Embedding Fusion: Concat vs Add\n(AJILE\\_NEW\\_ALLTOKS, mean ± std over folds)"
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "concat_vs_add_comparison.pdf")


def plot_token_count_scaling() -> None:
    """Illustrate how token count scales with sampling rate and approach."""
    sampling_rates = np.array([128, 256, 500, 1000, 2000])
    duration = 1.0
    n_channels = 64

    per_timepoint_tokens = sampling_rates * duration * n_channels
    cwt_100hz_tokens = np.full_like(
        sampling_rates, 100 * duration * n_channels, dtype=float
    )
    spatial_cwt_tokens = np.full_like(
        sampling_rates, 100 * duration, dtype=float
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        sampling_rates,
        per_timepoint_tokens,
        "o-",
        color=RED,
        linewidth=2,
        markersize=6,
        label="Per-Ch Per-Timepoint Linear",
        zorder=3,
    )
    ax.plot(
        sampling_rates,
        cwt_100hz_tokens,
        "s-",
        color=BLUE,
        linewidth=2,
        markersize=6,
        label="Per-Ch CWT/CNN (100 Hz tokens)",
        zorder=3,
    )
    ax.plot(
        sampling_rates,
        spatial_cwt_tokens,
        "^-",
        color=GREEN,
        linewidth=2,
        markersize=6,
        label="Spatial Session CWT/CNN (100 Hz tokens)",
        zorder=3,
    )

    ax.set_xlabel("Input sampling rate (Hz)")
    ax.set_ylabel("Tokens per 1 s window")
    ax.set_title(f"Token Count Scaling ({n_channels} channels, 1 s window)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, fs in enumerate(sampling_rates):
        ax.annotate(
            f"{int(per_timepoint_tokens[i]):,}",
            (fs, per_timepoint_tokens[i]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
            color=RED,
        )

    fig.tight_layout()
    _save(fig, "token_count_scaling.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating experiment-specific figures...")
    plot_concat_vs_add()
    plot_token_count_scaling()
    print("Done!")


if __name__ == "__main__":
    main()

"""Generate additional experiment-specific figures for the tokenizer journal.

Uses cached CSV data from W&B (run the per-group scripts first).

Outputs (in docs/figures/):
    concat_vs_add_comparison.pdf        — Concat vs add fusion comparison
    token_count_scaling.pdf             — Token count vs sampling rate / approach
    ablation_behavior_filtered_bar.pdf  — Ablation bar chart (crashed run removed)
    spatial_comparison.pdf              — Common layer + MLP vs Linear comparison

Usage:
    uv run scripts/analysis/experiment_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PALETTE_TEMPORAL, get_color  # noqa: E402

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

COL_CWT = PALETTE_TEMPORAL["cwt"]
COL_CNN = PALETTE_TEMPORAL["cnn"]
COL_LIN = PALETTE_TEMPORAL["linear"]
COL_IDT = PALETTE_TEMPORAL["identity"]

# Darker variant for "common layer" CWT
COL_CWT_DARK = "#004C7F"

# Semantic colors for non-tokenizer-family distinctions
COL_ACCENT_A = "#56B4E9"  # light blue — concat / linear projector
COL_ACCENT_B = "#E69F00"  # amber      — add / MLP projector


def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


def _temporal_legend(
    labels: list[str], loc: str = "lower right"
) -> list[mpatches.Patch]:
    """Build legend handles for the temporal-embedding families present in *labels*."""
    seen: set[str] = set()
    handles: list[mpatches.Patch] = []
    for name, key in [
        ("CWT", "cwt"),
        ("CNN", "cnn"),
        ("Linear", "linear"),
        ("Identity", "identity"),
    ]:
        c = PALETTE_TEMPORAL[key]
        if c not in seen and any(get_color(label) == c for label in labels):
            handles.append(mpatches.Patch(color=c, label=name))
            seen.add(c)
    return handles


def plot_ablation_behavior_filtered() -> None:
    """Ablation bar chart with crashed Per-Ch Linear fold 0 removed."""
    csv_path = CACHE_DIR / "AJILE12_TOKENIZER_ABLATION.csv"
    if not csv_path.exists():
        print(f"  Skipping ablation filtered: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    mask = ~(
        (df["tokenizer"] == "per_channel_per_timepoint_linear")
        & (df["fold"] == 0)
    )
    df = df[mask]

    agg = (
        df.groupby("tokenizer_label")["best_metric"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    colors = [get_color(label) for label in agg["tokenizer_label"]]

    fig, ax = plt.subplots(figsize=(max(7, len(agg) * 0.9), 4.5))
    x = np.arange(len(agg))
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.005,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel("AUROC")
    ax.set_title(
        "Full Tokenizer Ablation — Behavior AUROC\n(mean ± std over folds)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        agg["tokenizer_label"], rotation=35, ha="right", fontsize=8
    )
    ax.set_ylim(bottom=0.78)

    handles = _temporal_legend(list(agg["tokenizer_label"]))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "ablation_behavior_filtered_bar.pdf")


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

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, (fusion, color) in enumerate(
        [("Concat", COL_ACCENT_A), ("Add", COL_ACCENT_B)]
    ):
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
        for bar, val, std in zip(bars, vals, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ymin = rdf["mean"].min() - rdf["std"].max() - 0.02
    ax.set_ylim(bottom=max(0.78, ymin))
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "Channel Embedding Fusion: Concat vs Add\n(mean ± std over folds)"
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "concat_vs_add_comparison.pdf")


def plot_spatial_comparison() -> None:
    """Compare common layer effect and MLP vs Linear for spatial session."""
    ablation_path = CACHE_DIR / "AJILE12_TOKENIZER_ABLATION.csv"
    alltoks_path = CACHE_DIR / "AJILE_NEW_ALLTOKS.csv"

    if not ablation_path.exists() or not alltoks_path.exists():
        print("  Skipping spatial comparison: cached CSVs not found")
        return

    df_abl = pd.read_csv(ablation_path)
    df_all = pd.read_csv(alltoks_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Panel 1: Common layer effect (from ablation data) ---
    ax = axes[0]

    results: list[dict] = []
    for variant, key in [
        ("No common layer", "spatial_session_cwt"),
        ("With common layer", "spatial_session_cwt_common"),
    ]:
        sub = df_abl[df_abl["tokenizer"] == key]
        if sub.empty:
            continue
        results.append(
            {
                "family": "CWT",
                "variant": variant,
                "mean": sub["best_metric"].mean(),
                "std": sub["best_metric"].std(),
            }
        )

    for name, key in [
        ("Linear", "spatial_session_per_timepoint_linear"),
        ("Identity", "spatial_session_per_timepoint_identity"),
    ]:
        sub = df_abl[df_abl["tokenizer"] == key]
        if sub.empty:
            continue
        results.append(
            {
                "family": name,
                "variant": "Baseline",
                "mean": sub["best_metric"].mean(),
                "std": sub["best_metric"].std(),
            }
        )

    rdf = pd.DataFrame(results)

    bar_specs = [
        ("CWT\n(no common)", "No common layer", "CWT", COL_CWT),
        ("CWT\n(common)", "With common layer", "CWT", COL_CWT_DARK),
        ("Linear", "Baseline", "Linear", COL_LIN),
        ("Identity", "Baseline", "Identity", COL_IDT),
    ]
    labels, vals, stds, colors = [], [], [], []
    for lab, variant, family, color in bar_specs:
        row = rdf[(rdf["variant"] == variant) & (rdf["family"] == family)]
        labels.append(lab)
        colors.append(color)
        if not row.empty:
            vals.append(row["mean"].values[0])
            s = row["std"].values[0]
            stds.append(s if not np.isnan(s) else 0)
        else:
            vals.append(0)
            stds.append(0)

    xp = np.arange(len(labels))
    bars = ax.bar(
        xp,
        vals,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val, std in zip(bars, vals, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(xp)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Behavior AUROC")
    ax.set_title("Common Layer Effect\n(Ablation group)")
    ax.set_ylim(bottom=0.82)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel 2: MLP vs Linear projector (from AJILE_NEW_ALLTOKS) ---
    ax = axes[1]
    projector_pairs = {
        "CWT": ("spatial_session_cwt", "spatial_session_mlp_cwt"),
        "CNN": (
            "spatial_session_resample_cnn",
            "spatial_session_mlp_resample_cnn",
        ),
    }

    x_pos = np.arange(len(projector_pairs))
    width = 0.3
    for i, (variant, color, lab) in enumerate(
        [
            ("linear", COL_ACCENT_A, "Linear projector"),
            ("mlp", COL_ACCENT_B, "MLP projector"),
        ]
    ):
        v, s = [], []
        for family, (lin_key, mlp_key) in projector_pairs.items():
            key = lin_key if variant == "linear" else mlp_key
            sub = df_all[df_all["tokenizer"] == key]
            if sub.empty:
                v.append(0)
                s.append(0)
            else:
                v.append(sub["best_metric"].mean())
                s.append(sub["best_metric"].std() if len(sub) > 1 else 0)

        bars = ax.bar(
            x_pos + i * width - width / 2,
            v,
            width,
            yerr=s,
            capsize=4,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=lab,
        )
        for bar, val, std in zip(bars, v, s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(projector_pairs.keys()), fontsize=9)
    ax.set_ylabel("Behavior AUROC")
    ax.set_title("Spatial Projector: Linear vs MLP\n(Extended sweep group)")
    ax.set_ylim(bottom=0.82)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, "spatial_comparison.pdf")


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
        color=COL_LIN,
        linewidth=2,
        markersize=6,
        label="Per-Ch Per-Timepoint Linear",
        zorder=3,
    )
    ax.plot(
        sampling_rates,
        cwt_100hz_tokens,
        "s-",
        color=COL_CWT,
        linewidth=2,
        markersize=6,
        label="Per-Ch CWT/CNN (100 Hz tokens)",
        zorder=3,
    )
    ax.plot(
        sampling_rates,
        spatial_cwt_tokens,
        "^-",
        color=COL_IDT,
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
            color=COL_LIN,
        )

    fig.tight_layout()
    _save(fig, "token_count_scaling.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating experiment-specific figures...")
    plot_ablation_behavior_filtered()
    plot_concat_vs_add()
    plot_spatial_comparison()
    plot_token_count_scaling()
    print("Done!")


if __name__ == "__main__":
    main()

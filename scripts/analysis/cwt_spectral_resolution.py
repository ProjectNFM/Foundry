"""Generate figures for Experiment 6: CWT Spectral Resolution.

Tests whether increasing CWT frequency bins (9 → 24) and raising n_cycles
(2.5 → 7.0) improves downstream behavior classification.  2×2 factorial
for both CWT-only and CWT+CNN, with CNN baselines for context.

Outputs (in docs/figures/):
    cwt_spectral_resolution_bar.pdf    — factorial bar chart + CNN baselines
    cwt_spectral_resolution_freqs.pdf  — learned frequencies vs initialization

Usage:
    uv run scripts/analysis/cwt_spectral_resolution.py
    uv run scripts/analysis/cwt_spectral_resolution.py --plot-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PALETTE_TEMPORAL,
    WANDB_ENTITY,
    WANDB_PROJECT,
    make_base_parser,
)

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

COL_CWT = PALETTE_TEMPORAL["cwt"]
COL_CNN = PALETTE_TEMPORAL["cnn"]
COL_CWT_CNN = "#E69F00"  # Okabe-Ito amber for hybrid

# Lighter / darker shades for the factorial grid
COL_CWT_LIGHT = "#56B4E9"  # sky blue  — low n_cycles (2.5)
COL_CWT_DARK = "#0072B2"  # blue      — high n_cycles (7)
COL_HYBRID_LIGHT = "#F0E442"  # yellow — low n_cycles (2.5)
COL_HYBRID_DARK = "#E69F00"  # amber  — high n_cycles (7)
COL_CNN_12F = PALETTE_TEMPORAL["cnn"]
COL_CNN_64F = "#CC79A7"  # Okabe-Ito reddish purple

INIT_FREQS_9 = np.logspace(np.log10(0.5), np.log10(30), 9)
INIT_FREQS_24 = np.logspace(np.log10(0.5), np.log10(30), 24)


def _save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")


# ── Data collection ──────────────────────────────────────────────────────


def _fetch_spectral_runs(api: wandb.Api) -> list[dict]:
    """Fetch CWT_SPECTRAL_RESOLUTION runs from W&B."""
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": "CWT_SPECTRAL_RESOLUTION"},
    )
    results = []
    for run in runs:
        s = run.summary._json_dict
        auroc = s.get(AUROC_KEY, {})
        if isinstance(auroc, dict):
            auroc = auroc.get("max")
        if auroc is None:
            continue

        freq_keys = sorted(
            k
            for k in s
            if "freqs_hz/" in k
            and not any(x in k for x in ["mean", "std", "min", "max", "norm"])
        )
        ncycles_keys = sorted(
            k
            for k in s
            if "n_cycles/" in k
            and "unconstrained" not in k
            and not any(x in k for x in ["mean", "std", "min", "max", "norm"])
        )

        results.append(
            {
                "run_name": run.display_name,
                "auroc": auroc,
                "learned_freqs": json.dumps(
                    [s[k] for k in freq_keys] if freq_keys else []
                ),
                "learned_ncycles": json.dumps(
                    [s[k] for k in ncycles_keys] if ncycles_keys else []
                ),
            }
        )
    return results


def _parse_run_name(name: str) -> dict:
    """Extract tokenizer family, num_freqs, n_cycles, num_filters, fold."""
    is_hybrid = "cwt_cnn" in name
    fold = int(name.rsplit("fold", 1)[1])

    if "24f" in name:
        num_freqs = 24
    elif "9f" in name:
        num_freqs = 9
    else:
        num_freqs = 9

    if "nc7" in name:
        nc = 7.0
    elif "nc2p5" in name or "nc2.5" in name:
        nc = 2.5
    else:
        nc = 2.5

    if is_hybrid:
        if "48F" in name:
            num_filters = 48
        else:
            num_filters = 64
    else:
        num_filters = 0

    return {
        "family": "CWT+CNN" if is_hybrid else "CWT",
        "num_freqs": num_freqs,
        "n_cycles": nc,
        "num_filters": num_filters,
        "fold": fold,
    }


def collect_data(
    api: wandb.Api | None,
    cache_dir: Path,
    *,
    plot_only: bool = False,
) -> pd.DataFrame:
    cache_csv = cache_dir / "CWT_SPECTRAL_RESOLUTION.csv"

    if plot_only and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        print(f"  Loaded {len(df)} rows from cache")
        return df

    rows: list[dict] = []

    # ── New Experiment 6 runs ──
    if api is not None:
        print("  Fetching CWT_SPECTRAL_RESOLUTION from W&B...")
        raw = _fetch_spectral_runs(api)
        print(f"  Fetched {len(raw)} runs")
        for r in raw:
            parsed = _parse_run_name(r["run_name"])
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": _make_label(parsed),
                    "family": parsed["family"],
                    "num_freqs": parsed["num_freqs"],
                    "n_cycles": parsed["n_cycles"],
                    "num_filters": parsed["num_filters"],
                    "fold": parsed["fold"],
                    "auroc": r["auroc"],
                    "source": "CWT_SPECTRAL_RESOLUTION",
                    "learned_freqs": r["learned_freqs"],
                    "learned_ncycles": r["learned_ncycles"],
                }
            )

    # ── Baselines from CWT_LR_AND_PARAM_MATCH (CWT 9f 100×LR, CWT+CNN 64f, CNN 64f) ──
    param_csv = cache_dir / "PARAM_MATCH.csv"
    if param_csv.exists():
        pm = pd.read_csv(param_csv)
        for _, r in pm[pm["tokenizer"] == "CWT+CNN (64f)"].iterrows():
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": "CWT+CNN 9f-64F nc2.5",
                    "family": "CWT+CNN",
                    "num_freqs": 9,
                    "n_cycles": 2.5,
                    "num_filters": 64,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "CWT_LR_AND_PARAM_MATCH",
                    "learned_freqs": "[]",
                    "learned_ncycles": "[]",
                }
            )
        for _, r in pm[pm["tokenizer"] == "CNN (64f)"].iterrows():
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": "CNN 64f",
                    "family": "CNN",
                    "num_freqs": 0,
                    "n_cycles": 0,
                    "num_filters": 64,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "CWT_LR_AND_PARAM_MATCH",
                    "learned_freqs": "[]",
                    "learned_ncycles": "[]",
                }
            )

    # ── CWT 9f baseline from CWT_LR_SWEEP (100× multiplier) ──
    lr_csv = cache_dir / "CWT_LR_SWEEP.csv"
    if lr_csv.exists():
        lr = pd.read_csv(lr_csv)
        for _, r in lr[lr["multiplier"] == 100].iterrows():
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": "CWT 9f nc2.5",
                    "family": "CWT",
                    "num_freqs": 9,
                    "n_cycles": 2.5,
                    "num_filters": 0,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "CWT_LR_AND_PARAM_MATCH",
                    "learned_freqs": r.get("learned_freqs", "[]"),
                    "learned_ncycles": r.get("learned_ncycles", "[]"),
                }
            )

    # ── CNN 12f from TOKEN_RATE_SWEEP at 200 Hz ──
    sweep_csv = cache_dir / "TOKEN_RATE_SWEEP.csv"
    if sweep_csv.exists():
        sweep = pd.read_csv(sweep_csv)
        for _, r in sweep[
            (sweep["rate"] == 200) & (sweep["tokenizer"] == "CNN")
        ].iterrows():
            rows.append(
                {
                    "run_name": r["run_name"],
                    "tokenizer": "CNN 12f",
                    "family": "CNN",
                    "num_freqs": 0,
                    "n_cycles": 0,
                    "num_filters": 12,
                    "fold": r["fold"],
                    "auroc": r["auroc"],
                    "source": "TOKEN_RATE_SWEEP",
                    "learned_freqs": "[]",
                    "learned_ncycles": "[]",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"  Cached {len(df)} rows to {cache_csv}")
    return df


def _make_label(parsed: dict) -> str:
    nf = parsed["num_freqs"]
    nc = parsed["n_cycles"]
    nc_str = "nc2.5" if nc == 2.5 else f"nc{int(nc)}"
    if parsed["family"] == "CWT+CNN":
        filt = parsed["num_filters"]
        return f"CWT+CNN {nf}f-{filt}F {nc_str}"
    return f"CWT {nf}f {nc_str}"


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_spectral_resolution_bar(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: CWT-only factorial, CWT+CNN factorial, CNN baselines."""

    cwt_order = [
        "CWT 9f nc2.5",
        "CWT 9f nc7",
        "CWT 24f nc2.5",
        "CWT 24f nc7",
    ]
    hybrid_order = [
        "CWT+CNN 9f-64F nc2.5",
        "CWT+CNN 9f-64F nc7",
        "CWT+CNN 24f-48F nc2.5",
        "CWT+CNN 24f-48F nc7",
    ]
    cnn_order = ["CNN 12f", "CNN 64f"]
    full_order = cwt_order + hybrid_order + cnn_order

    color_map = {
        "CWT 9f nc2.5": COL_CWT_LIGHT,
        "CWT 9f nc7": COL_CWT_DARK,
        "CWT 24f nc2.5": COL_CWT_LIGHT,
        "CWT 24f nc7": COL_CWT_DARK,
        "CWT+CNN 9f-64F nc2.5": COL_HYBRID_LIGHT,
        "CWT+CNN 9f-64F nc7": COL_HYBRID_DARK,
        "CWT+CNN 24f-48F nc2.5": COL_HYBRID_LIGHT,
        "CWT+CNN 24f-48F nc7": COL_HYBRID_DARK,
        "CNN 12f": COL_CNN_12F,
        "CNN 64f": COL_CNN_64F,
    }

    param_counts = {
        "CWT 9f nc2.5": 3_666,
        "CWT 9f nc7": 3_666,
        "CWT 24f nc2.5": 9_456,
        "CWT 24f nc7": 9_456,
        "CWT+CNN 9f-64F nc2.5": 59_858,
        "CWT+CNN 9f-64F nc7": 59_858,
        "CWT+CNN 24f-48F nc2.5": 51_024,
        "CWT+CNN 24f-48F nc7": 51_024,
        "CNN 12f": 3_924,
        "CNN 64f": 50_048,
    }

    agg = (
        df.groupby("tokenizer")["auroc"]
        .agg(["mean", "std"])
        .reindex(full_order)
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    agg = agg.dropna(subset=["mean"])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(agg))

    bar_colors = [color_map.get(t, "#888888") for t in agg["tokenizer"]]
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=4,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.002,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    xlabels = []
    for t in agg["tokenizer"]:
        p = param_counts.get(t)
        short = t.replace("CWT+CNN ", "CWT+CNN\n").replace("CWT ", "CWT\n")
        if p:
            xlabels.append(f"{short}\n({p:,}p)")
        else:
            xlabels.append(short)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7, ha="center")
    ax.set_ylabel("Behavior AUROC")
    ax.set_title(
        "CWT Spectral Resolution: num_freqs × n_cycles Factorial\n"
        "(mean ± std over 2 folds, all CWT configs at 100× LR multiplier)"
    )

    n_cwt = sum(1 for t in agg["tokenizer"] if t in cwt_order)
    n_hybrid = sum(1 for t in agg["tokenizer"] if t in hybrid_order)
    if n_cwt > 0 and n_hybrid > 0:
        ax.axvline(
            n_cwt - 0.5,
            color="#cccccc",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
    if n_hybrid > 0:
        ax.axvline(
            n_cwt + n_hybrid - 0.5,
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
            "CWT-only",
            ha="center",
            fontsize=8,
            color="#666666",
        )
    if n_hybrid > 0:
        ax.text(
            n_cwt + (n_hybrid - 1) / 2,
            ymin + 0.002,
            "CWT+CNN",
            ha="center",
            fontsize=8,
            color="#666666",
        )
    n_cnn = sum(1 for t in agg["tokenizer"] if t in cnn_order)
    if n_cnn > 0:
        ax.text(
            n_cwt + n_hybrid + (n_cnn - 1) / 2,
            ymin + 0.002,
            "CNN baselines",
            ha="center",
            fontsize=8,
            color="#666666",
        )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=COL_CWT_LIGHT, label="nc=2.5 (CWT)"),
        Patch(facecolor=COL_CWT_DARK, label="nc=7 (CWT)"),
        Patch(facecolor=COL_HYBRID_LIGHT, label="nc=2.5 (CWT+CNN)"),
        Patch(facecolor=COL_HYBRID_DARK, label="nc=7 (CWT+CNN)"),
        Patch(facecolor=COL_CNN_12F, label="CNN 12f"),
        Patch(facecolor=COL_CNN_64F, label="CNN 64f"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right", ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "cwt_spectral_resolution_bar.pdf", output_dir)


def plot_learned_freqs(df: pd.DataFrame, output_dir: Path) -> None:
    """Learned frequencies vs initialization for 9f and 24f configs."""
    cwt_df = df[
        (df["family"].isin(["CWT", "CWT+CNN"]))
        & (df["source"] == "CWT_SPECTRAL_RESOLUTION")
    ].copy()

    if cwt_df.empty or "learned_freqs" not in cwt_df.columns:
        print("  Skipping frequency plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    configs_9f = [
        ("CWT 9f nc7", COL_CWT_DARK, "o", "CWT 9f nc7"),
        ("CWT+CNN 9f-64F nc7", COL_HYBRID_DARK, "D", "CWT+CNN 9f nc7"),
    ]
    configs_24f = [
        ("CWT 24f nc2.5", COL_CWT_LIGHT, "s", "CWT 24f nc2.5"),
        ("CWT 24f nc7", COL_CWT_DARK, "o", "CWT 24f nc7"),
        ("CWT+CNN 24f-48F nc2.5", COL_HYBRID_LIGHT, "s", "CWT+CNN 24f nc2.5"),
        ("CWT+CNN 24f-48F nc7", COL_HYBRID_DARK, "D", "CWT+CNN 24f nc7"),
    ]

    # Left panel: 9-frequency configs
    ax = axes[0]
    idx_9 = np.arange(len(INIT_FREQS_9))
    ax.plot(
        idx_9,
        INIT_FREQS_9,
        "k--",
        linewidth=1.5,
        label="Init (log-spaced)",
        zorder=10,
    )

    for tok_label, color, marker, legend_label in configs_9f:
        runs = cwt_df[cwt_df["tokenizer"] == tok_label]
        all_freqs = []
        for _, row in runs.iterrows():
            freqs = json.loads(row["learned_freqs"])
            if freqs and len(freqs) == 9:
                all_freqs.append(sorted(freqs))
        if not all_freqs:
            continue
        mean_f = np.mean(all_freqs, axis=0)
        std_f = np.std(all_freqs, axis=0)
        ax.errorbar(
            idx_9,
            mean_f,
            yerr=std_f,
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            markersize=5,
            linewidth=1.2,
            label=legend_label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Frequency Bin Index")
    ax.set_ylabel("Center Frequency (Hz)")
    ax.set_title("9-Frequency Configs")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xticks(idx_9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right panel: 24-frequency configs
    ax = axes[1]
    idx_24 = np.arange(len(INIT_FREQS_24))
    ax.plot(
        idx_24,
        INIT_FREQS_24,
        "k--",
        linewidth=1.5,
        label="Init (log-spaced)",
        zorder=10,
    )

    for tok_label, color, marker, legend_label in configs_24f:
        runs = cwt_df[cwt_df["tokenizer"] == tok_label]
        all_freqs = []
        for _, row in runs.iterrows():
            freqs = json.loads(row["learned_freqs"])
            if freqs and len(freqs) == 24:
                all_freqs.append(sorted(freqs))
        if not all_freqs:
            continue
        mean_f = np.mean(all_freqs, axis=0)
        std_f = np.std(all_freqs, axis=0)
        ax.errorbar(
            idx_24,
            mean_f,
            yerr=std_f,
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            markersize=4,
            linewidth=1.0,
            label=legend_label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Frequency Bin Index (sorted)")
    ax.set_ylabel("Center Frequency (Hz)")
    ax.set_title("24-Frequency Configs")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xticks(idx_24[::2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Learned CWT Frequencies vs Initialization (100× LR multiplier)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "cwt_spectral_resolution_freqs.pdf", output_dir)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = make_base_parser("Experiment 6: CWT Spectral Resolution figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    api = None if args.plot_only else wandb.Api(timeout=60)

    print("Generating Experiment 6 (CWT Spectral Resolution) figures...")
    df = collect_data(api, args.cache_dir, plot_only=args.plot_only)

    print("\nAUROC summary:")
    summary = (
        df.groupby("tokenizer")["auroc"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())

    plot_spectral_resolution_bar(df, args.output_dir)
    plot_learned_freqs(df, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()

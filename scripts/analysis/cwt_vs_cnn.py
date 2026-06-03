"""Generate figures for the CWT vs CNN comparison experiments.

Groups: CWT_VS_CNN_BEHAVIOR, CWT_VS_CNN_POSE

Outputs:
    cwt_vs_cnn_behavior_bar.pdf   — bar chart of behavior AUROC
    cwt_vs_cnn_pose_bar.pdf       — bar chart of pose R²
    cwt_vs_cnn_scaling.pdf        — dim256 vs dim512 scaling comparison
    cwt_vs_cnn_*_curves.pdf       — training curves (optional)

Usage:
    uv run scripts/analysis/cwt_vs_cnn.py
    uv run scripts/analysis/cwt_vs_cnn.py --plot-only
    uv run scripts/analysis/cwt_vs_cnn.py --no-curves
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_GROUPS,
    get_color,
    load_or_fetch_group,
    make_base_parser,
    plot_group_bar_chart,
    plot_training_curves,
    setup_dirs_and_api,
)

GROUPS = ["CWT_VS_CNN_BEHAVIOR", "CWT_VS_CNN_POSE"]


def plot_scaling(df_behavior, output_dir: Path) -> Path | None:
    """Behavior-only comparison for CWT vs CNN at dim256 and dim512.

    Bars are colored by tokenizer family, hatched by dimension.
    The dimension is inferred from tokenizer_label since the CSV's
    embed_dim column always reads the backbone dim (256).
    """
    if df_behavior is None or df_behavior.empty:
        return None

    families = ["CWT", "CNN"]
    dim_labels = {
        "CWT": {"256": "Per-Ch CWT", "512": "Per-Ch CWT (512)"},
        "CNN": {"256": "Per-Ch CNN", "512": "Per-Ch CNN (512)"},
    }

    agg = (
        df_behavior.groupby("tokenizer_label")["best_metric"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    width = 0.35
    x = np.arange(len(families))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, (dim_key, hatch) in enumerate([("256", ""), ("512", "///")]):
        vals, stds = [], []
        for fam in families:
            target = dim_labels[fam][dim_key]
            row = agg[agg["tokenizer_label"] == target]
            vals.append(row["mean"].values[0] if not row.empty else 0)
            stds.append(row["std"].values[0] if not row.empty else 0)

        bar_colors = [get_color(fam) for fam in families]
        bars = ax.bar(
            x + i * width - width / 2,
            vals,
            width,
            yerr=stds,
            capsize=3,
            color=bar_colors,
            edgecolor="white" if not hatch else "grey",
            linewidth=0.5,
            hatch=hatch,
        )
        for bar, val, std in zip(bars, vals, stds):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.003,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_title("CWT vs CNN — Embedding Dimension Scaling\n(Behavior AUROC)")
    ax.set_ylabel("Behavior AUROC")

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=get_color("CWT"), label="CWT"),
        mpatches.Patch(facecolor=get_color("CNN"), label="CNN"),
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="grey", label="dim 256"),
        mpatches.Patch(
            facecolor="#CCCCCC", edgecolor="grey", hatch="///", label="dim 512"
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    path = output_dir / "cwt_vs_cnn_scaling.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _filter_dim256(df):
    """Keep only dim-256 runs (drop dim512 tokenizer configs)."""
    if df is None:
        return None
    return df[~df["tokenizer"].str.contains("dim512")]


def main():
    parser = make_base_parser("CWT vs CNN comparison figures")
    args = parser.parse_args()
    api = setup_dirs_and_api(args)

    dfs = {}
    for name in GROUPS:
        group = EXPERIMENT_GROUPS[name]
        df = load_or_fetch_group(
            group, api, args.cache_dir, plot_only=args.plot_only
        )
        if df is None:
            continue
        dfs[name] = df

        df_256 = _filter_dim256(df)
        plot_group_bar_chart(df_256, group, args.output_dir)
        if not args.no_curves and api is not None:
            plot_training_curves(api, df_256, group, args.output_dir)

    plot_scaling(dfs.get("CWT_VS_CNN_BEHAVIOR"), args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()

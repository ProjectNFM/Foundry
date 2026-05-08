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
    BEHAVIOR_METRIC,
    EXPERIMENT_GROUPS,
    POSE_METRIC,
    load_or_fetch_group,
    make_base_parser,
    plot_group_bar_chart,
    plot_training_curves,
    setup_dirs_and_api,
)

GROUPS = ["CWT_VS_CNN_BEHAVIOR", "CWT_VS_CNN_POSE"]


def plot_scaling(df_behavior, df_pose, output_dir: Path) -> Path | None:
    """Side-by-side comparison for CWT vs CNN at dim256 and dim512."""
    import pandas as pd

    panels: list[tuple[str, pd.DataFrame, str]] = []
    if df_behavior is not None and not df_behavior.empty:
        panels.append(("Behavior AUROC", df_behavior, BEHAVIOR_METRIC))
    if df_pose is not None and not df_pose.empty:
        panels.append(("Pose R²", df_pose, POSE_METRIC))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, df, metric) in zip(axes, panels):
        agg = (
            df.groupby(["tokenizer_label", "embed_dim"])["best_metric"]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg["std"] = agg["std"].fillna(0)

        dims = sorted(agg["embed_dim"].unique())
        tokenizers = agg["tokenizer_label"].unique()
        width = 0.35
        x = np.arange(len(tokenizers))

        for i, dim in enumerate(dims):
            sub = agg[agg["embed_dim"] == dim]
            vals = [
                sub[sub["tokenizer_label"] == t]["mean"].values[0]
                if t in sub["tokenizer_label"].values
                else 0
                for t in tokenizers
            ]
            stds = [
                sub[sub["tokenizer_label"] == t]["std"].values[0]
                if t in sub["tokenizer_label"].values
                else 0
                for t in tokenizers
            ]
            ax.bar(
                x + i * width - width / 2,
                vals,
                width,
                yerr=stds,
                capsize=3,
                label=f"dim={dim}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(tokenizers, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        metric_label = metric.split("/")[-1].replace("_", " ").title()
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("CWT vs CNN — Embedding Dimension Scaling", fontsize=13)
    fig.tight_layout()

    path = output_dir / "cwt_vs_cnn_scaling.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")
    return path


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
        plot_group_bar_chart(df, group, args.output_dir)
        if not args.no_curves and api is not None:
            plot_training_curves(api, df, group, args.output_dir)

    plot_scaling(
        dfs.get("CWT_VS_CNN_BEHAVIOR"),
        dfs.get("CWT_VS_CNN_POSE"),
        args.output_dir,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()

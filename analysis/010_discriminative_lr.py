"""Discriminative LR finetuning: pretrained backbone vs task head learning rates.

Fetches runs from KEMP_DISCRIMINATIVE_LR_SEARCH (Phase 1a grid) and compares
against the from-scratch baselines in KEMP_SCRATCH_HP_SEARCH (exp 009).

WandB project: foundry_finetuning

Usage:
    uv run python analysis/010_discriminative_lr.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()

SEARCH_GROUP = "KEMP_DISCRIMINATIVE_LR_SEARCH"
BASELINE_GROUP = "KEMP_SCRATCH_HP_SEARCH"
VALIDATION_GROUP = "KEMP_DISCRIMINATIVE_LR_VALIDATION"

EXP_009_PRETRAINED_BEST = 0.5425  # best uniform-lr pretrained from exp 009

VAL_F1 = "val/sleep_stage_5class_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)


def _fetch_group_runs(group: str, api: wandb.Api) -> list:
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    return list(api.runs(path, filters={"group": group}))


def _extract_hp_from_run(run) -> dict:
    config = run.config
    hp = config.get("hyperparameters", {})
    return {
        "backbone_lr": hp.get("backbone_learning_rate"),
        "head_lr": hp.get("learning_rate"),
        "lr": hp.get("learning_rate"),
        "warmup": config.get("module", {}).get("warmup_epochs", 0),
        "fold": hp.get("fold_number", 0),
        "frozen": config.get("run", {}).get("freeze_pretrained", False),
        "scratch": config.get("run", {}).get("pretrained_checkpoint") is None,
    }


def _extract_metrics(run) -> dict:
    return {
        "best_val_f1": unwrap_summary_value(run.summary.get(VAL_F1), "max"),
        "best_val_loss": unwrap_summary_value(run.summary.get(VAL_LOSS), "min"),
        "final_train_loss": unwrap_summary_value(
            run.summary.get(TRAIN_LOSS), "min"
        ),
        "best_epoch": unwrap_summary_value(run.summary.get("epoch"), "max"),
    }


def fetch_search_runs(api: wandb.Api) -> pd.DataFrame:
    rows = []
    runs = _fetch_group_runs(SEARCH_GROUP, api)
    if not runs:
        print(f"  No runs found for group '{SEARCH_GROUP}'")
        return pd.DataFrame(rows)
    print(
        f"  Found {len(runs)} runs for discriminative LR search ({SEARCH_GROUP})"
    )

    for run in runs:
        if run.state != "finished":
            print(f"    Skipping {run.id} (state={run.state})")
            continue
        hp = _extract_hp_from_run(run)
        metrics = _extract_metrics(run)
        rows.append(
            {
                "Backbone LR": hp["backbone_lr"],
                "Head LR": hp["head_lr"],
                "Fold": hp["fold"],
                "Run ID": run.id,
                "Run Name": run.name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def fetch_baseline_runs(api: wandb.Api) -> pd.DataFrame:
    rows = []
    runs = _fetch_group_runs(BASELINE_GROUP, api)
    if not runs:
        print(f"  No runs found for group '{BASELINE_GROUP}'")
        return pd.DataFrame(rows)
    print(f"  Found {len(runs)} runs for scratch baselines ({BASELINE_GROUP})")

    for run in runs:
        if run.state != "finished":
            print(f"    Skipping {run.id} (state={run.state})")
            continue
        hp = _extract_hp_from_run(run)
        metrics = _extract_metrics(run)
        rows.append(
            {
                "LR": hp["lr"],
                "Warmup": hp["warmup"],
                "Fold": hp["fold"],
                "Run ID": run.id,
                "Run Name": run.name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def print_grid(search_df: pd.DataFrame) -> None:
    if search_df.empty:
        print("\nNo search runs to grid.")
        return

    pivot = search_df.pivot_table(
        values="best_val_f1",
        index="Backbone LR",
        columns="Head LR",
        aggfunc="mean",
    ).sort_index()

    print(f"\n{'=' * 60}")
    print("  Pretrained Discriminative LR — Val F1 by (Backbone LR, Head LR)")
    print(f"{'=' * 60}")
    print(pivot.to_string(float_format="%.4f"))

    best_idx = search_df["best_val_f1"].idxmax()
    best = search_df.loc[best_idx]
    print(
        f"\n  Best: backbone_lr={best['Backbone LR']}, head_lr={best['Head LR']}"
        f" → Val F1 = {best['best_val_f1']:.4f}"
        f" (run {best['Run ID']})"
    )


def plot_heatmap(search_df: pd.DataFrame) -> None:
    if search_df.empty:
        return

    pivot = search_df.pivot_table(
        values="best_val_f1",
        index="Backbone LR",
        columns="Head LR",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = pivot.values.min() - 0.01
    vmax = pivot.values.max() + 0.01
    im = ax.imshow(
        pivot.values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in pivot.index])
    ax.set_xlabel("Head LR")
    ax.set_ylabel("Backbone LR")
    ax.set_title("Discriminative LR — Val F1 (fold 0)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=ax, label="Val F1", shrink=0.8)
    plt.tight_layout()
    out = FIGURES_DIR / "010_discriminative_lr_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_comparison(search_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    labels = []
    vals = []
    colors = []

    # Exp 009 pretrained uniform best (hardcoded reference)
    labels.append("Pretrained uniform\n(exp 009)")
    vals.append(EXP_009_PRETRAINED_BEST)
    colors.append("#E8A0A0")

    # From-scratch baselines from KEMP_SCRATCH_HP_SEARCH (fold 0 only)
    if not baseline_df.empty:
        fold0 = baseline_df[baseline_df["Fold"] == 0]
        if not fold0.empty:
            best_scratch = fold0.loc[fold0["best_val_f1"].idxmax()]
            labels.append(
                f"Scratch best\n(lr={best_scratch['LR']:.0e}, "
                f"wu={int(best_scratch['Warmup'])})"
            )
            vals.append(best_scratch["best_val_f1"])
            colors.append("#A0C0E8")

    # Best discriminative LR run
    if not search_df.empty:
        best = search_df.loc[search_df["best_val_f1"].idxmax()]
        labels.append(
            f"Discriminative LR\n(bb={best['Backbone LR']:.0e}, "
            f"hd={best['Head LR']:.0e})"
        )
        vals.append(best["best_val_f1"])
        colors.append("#DD8452")

    # All discriminative LR configs as smaller bars
    if not search_df.empty:
        for _, row in search_df.iterrows():
            if row.name == search_df["best_val_f1"].idxmax():
                continue
            labels.append(
                f"Disc. LR\n(bb={row['Backbone LR']:.0e},\nhd={row['Head LR']:.0e})"
            )
            vals.append(row["best_val_f1"])
            colors.append("#FFBB78")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("Discriminative LR vs Scratch Baselines (fold 0)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(vals) - 0.05))
    plt.tight_layout()

    out = FIGURES_DIR / "010_discriminative_lr_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_learning_curves(
    search_df: pd.DataFrame, baseline_df: pd.DataFrame, api: wandb.Api
) -> None:
    """Learning curves comparing best discriminative LR vs best scratch."""
    if search_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best discriminative LR runs (top 3)
    ax = axes[0]
    top3 = search_df.nlargest(3, "best_val_f1")
    for _, row in top3.iterrows():
        try:
            path = (
                f"{WANDB_ENTITY}/{WANDB_PROJECT}"
                if WANDB_ENTITY
                else WANDB_PROJECT
            )
            run = api.run(f"{path}/{row['Run ID']}")
            history = run.history(
                keys=["epoch", VAL_F1], samples=10_000, pandas=True
            )
            history = history.dropna(subset=[VAL_F1])
            if not history.empty:
                ax.plot(
                    history["epoch"],
                    history[VAL_F1],
                    label=f"bb={row['Backbone LR']:.0e}, hd={row['Head LR']:.0e} ({row['best_val_f1']:.3f})",
                    linewidth=1.5,
                )
        except Exception as e:
            print(f"    Could not fetch history for {row['Run ID']}: {e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Discriminative LR — Top 3 Configs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Best scratch baselines (top 3, fold 0)
    ax = axes[1]
    if not baseline_df.empty:
        fold0 = baseline_df[baseline_df["Fold"] == 0]
        top3_baseline = fold0.nlargest(3, "best_val_f1")
        for _, row in top3_baseline.iterrows():
            try:
                path = (
                    f"{WANDB_ENTITY}/{WANDB_PROJECT}"
                    if WANDB_ENTITY
                    else WANDB_PROJECT
                )
                run = api.run(f"{path}/{row['Run ID']}")
                history = run.history(
                    keys=["epoch", VAL_F1], samples=10_000, pandas=True
                )
                history = history.dropna(subset=[VAL_F1])
                if not history.empty:
                    ax.plot(
                        history["epoch"],
                        history[VAL_F1],
                        label=f"lr={row['LR']:.0e}, wu={int(row['Warmup'])} ({row['best_val_f1']:.3f})",
                        linewidth=1.5,
                    )
            except Exception as e:
                print(f"    Could not fetch history for {row['Run ID']}: {e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Scratch Baselines — Top 3 Configs (fold 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "010_discriminative_lr_curves.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()

    print("Fetching discriminative LR search runs...")
    search_df = fetch_search_runs(api)

    print("\nFetching scratch baseline runs...")
    baseline_df = fetch_baseline_runs(api)

    if search_df.empty:
        print("\nNo completed discriminative LR runs found.")
        return

    print(f"\nDiscriminative LR runs: {len(search_df)}")
    print_grid(search_df)

    print("\n" + "=" * 60)
    print("  Reference baselines")
    print("=" * 60)
    print(f"  Exp 009 pretrained uniform best: {EXP_009_PRETRAINED_BEST:.4f}")
    if not baseline_df.empty:
        fold0 = baseline_df[baseline_df["Fold"] == 0]
        if not fold0.empty:
            best_scratch = fold0.loc[fold0["best_val_f1"].idxmax()]
            print(
                f"  Exp 009 scratch best (lr={best_scratch['LR']:.0e}, "
                f"wu={int(best_scratch['Warmup'])}): {best_scratch['best_val_f1']:.4f}"
            )
            print("\n  All scratch configs (fold 0):")
            for _, row in fold0.sort_values(
                "best_val_f1", ascending=False
            ).iterrows():
                print(
                    f"    lr={row['LR']:.0e}, wu={int(row['Warmup'])}: "
                    f"F1={row['best_val_f1']:.4f}"
                )

    plot_heatmap(search_df)
    plot_comparison(search_df, baseline_df)
    plot_learning_curves(search_df, baseline_df, api)


if __name__ == "__main__":
    main()

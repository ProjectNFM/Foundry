"""Discriminative LR finetuning: pretrained backbone vs task head learning rates.

Fetches runs from KEMP_DISCRIMINATIVE_LR_SEARCH (Phase 1 grid),
KEMP_DISCRIMINATIVE_LR_CONTROLS (frozen backbone + scratch baselines), and
KEMP_DISCRIMINATIVE_LR_VALIDATION (3-fold validation, if launched).

Compares against exp 009 best configs on fold 0.

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
CONTROLS_GROUP = "KEMP_DISCRIMINATIVE_LR_CONTROLS"
VALIDATION_GROUP = "KEMP_DISCRIMINATIVE_LR_VALIDATION"

EXP_009_BEST = {
    "Pretrained uniform (exp 009)": 0.5425,
    "Scratch uniform (exp 009)": 0.5629,
}
EXP_008_LINEAR_PROBE = 0.418

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


def fetch_runs(groups: dict[str, str], api: wandb.Api) -> pd.DataFrame:
    rows = []
    for label, group in groups.items():
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs found for group '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for {label} ({group})")

        for run in runs:
            if run.state != "finished":
                print(f"    Skipping {run.id} (state={run.state})")
                continue

            hp = _extract_hp_from_run(run)
            metrics = _extract_metrics(run)
            condition = label
            if hp["frozen"]:
                condition = "Frozen backbone"
            elif hp["scratch"]:
                condition = "Scratch uniform"

            rows.append(
                {
                    "Group": label,
                    "Condition": condition,
                    "Backbone LR": hp["backbone_lr"],
                    "Head LR": hp["head_lr"],
                    "Fold": hp["fold"],
                    "Run ID": run.id,
                    "Run Name": run.name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def print_grid(df: pd.DataFrame) -> None:
    search = df[df["Group"] == "Search"].copy()
    if search.empty:
        print("\nNo search runs to grid.")
        return

    pivot = search.pivot_table(
        values="best_val_f1",
        index="Backbone LR",
        columns="Head LR",
        aggfunc="mean",
    ).sort_index()

    print(f"\n{'=' * 60}")
    print("  Pretrained — Val F1 by (Backbone LR, Head LR)")
    print(f"{'=' * 60}")
    print(pivot.to_string(float_format="%.4f"))

    best_idx = search["best_val_f1"].idxmax()
    best = search.loc[best_idx]
    print(
        f"\n  Best: backbone_lr={best['Backbone LR']}, head_lr={best['Head LR']}"
        f" → Val F1 = {best['best_val_f1']:.4f}"
        f" (run {best['Run ID']})"
    )


def plot_heatmap(df: pd.DataFrame) -> None:
    search = df[df["Group"] == "Search"].copy()
    if search.empty:
        return

    pivot = search.pivot_table(
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


def plot_comparison(df: pd.DataFrame) -> None:
    labels = []
    vals = []
    colors = []

    labels.append("Exp 008\nLinear probe")
    vals.append(EXP_008_LINEAR_PROBE)
    colors.append("#C0C0C0")

    for name, val in EXP_009_BEST.items():
        labels.append(name.replace(" (exp 009)", "\n(exp 009)"))
        vals.append(val)
        colors.append("#E8A0A0" if "Pretrained" in name else "#A0C0E8")

    search = df[df["Group"] == "Search"]
    if not search.empty:
        best = search.loc[search["best_val_f1"].idxmax()]
        labels.append(
            f"Exp 010\nBest discriminative\n(bb={best['Backbone LR']:.0e}, "
            f"hd={best['Head LR']:.0e})"
        )
        vals.append(best["best_val_f1"])
        colors.append("#DD8452")

    controls = df[df["Group"] == "Controls"]
    for _, row in controls.iterrows():
        labels.append(row["Condition"].replace(" ", "\n"))
        vals.append(row["best_val_f1"])
        colors.append("#55A868")

    fig, ax = plt.subplots(figsize=(11, 6))
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
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("Discriminative LR vs Baselines (fold 0)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(vals) - 0.05))
    plt.tight_layout()

    out = FIGURES_DIR / "010_discriminative_lr_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()

    print("Fetching discriminative LR runs...")
    df = fetch_runs(
        {"Search": SEARCH_GROUP, "Controls": CONTROLS_GROUP},
        api,
    )

    if df.empty:
        print("\nNo completed runs found. Launch Phase 1 first.")
        return

    print(f"\nTotal completed runs: {len(df)}")
    print_grid(df)

    print("\n" + "=" * 60)
    print("  Reference baselines (fold 0)")
    print("=" * 60)
    print(
        f"  Exp 008 linear probe (frozen backbone): {EXP_008_LINEAR_PROBE:.4f}"
    )
    for name, val in EXP_009_BEST.items():
        print(f"  {name}: {val:.4f}")

    plot_heatmap(df)
    plot_comparison(df)

    print("\nChecking for validation runs...")
    val_df = fetch_runs({"Validation": VALIDATION_GROUP}, api)
    if not val_df.empty:
        agg = (
            val_df.groupby("Condition")["best_val_f1"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print("\n" + "=" * 60)
        print("  Phase 2 — 3-fold validation (mean ± std)")
        print("=" * 60)
        print(agg.to_string())
    else:
        print("  No validation runs yet (Phase 2 not launched).")


if __name__ == "__main__":
    main()

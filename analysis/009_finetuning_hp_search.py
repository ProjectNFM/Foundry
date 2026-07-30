"""Finetuning hyperparameter search: pretrained vs from-scratch CWT-CNN.

Fetches all runs from the Phase 1 (KEMP_FINETUNE_HP_SEARCH) and Phase 2
(KEMP_SCRATCH_HP_SEARCH) groups, builds a comparison table across the
(learning_rate, warmup_epochs) grid, and identifies the best configuration
for each condition.

If Phase 3 validation runs exist (KEMP_FINETUNE_HP_VALIDATION,
KEMP_SCRATCH_HP_VALIDATION), includes 3-fold aggregated results.

WandB project: foundry_finetuning

Usage:
    uv run python analysis/009_finetuning_hp_search.py
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

HP_SEARCH_GROUPS = {
    "Pretrained": "KEMP_FINETUNE_HP_SEARCH",
    "Scratch": "KEMP_SCRATCH_HP_SEARCH",
}

VALIDATION_GROUPS = {
    "Pretrained": "KEMP_FINETUNE_HP_VALIDATION",
    "Scratch": "KEMP_SCRATCH_HP_VALIDATION",
}

EXP_006_SCRATCH_CWT_CNN_F1 = {
    "fold0": 0.5612,
    "fold1": 0.5632,
    "fold2": 0.5363,
}
EXP_006_SCRATCH_CWT_CNN_MEAN = np.mean(
    list(EXP_006_SCRATCH_CWT_CNN_F1.values())
)

EXP_007_FINETUNE_CWT_CNN_F1 = {
    "fold0": 0.5419,
    "fold1": 0.5610,
    "fold2": 0.4989,
}
EXP_007_FINETUNE_CWT_CNN_MEAN = np.mean(
    list(EXP_007_FINETUNE_CWT_CNN_F1.values())
)

VAL_F1 = "val/sleep_stage_5class_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)


def _fetch_group_runs(group: str, api: wandb.Api) -> list:
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    return list(api.runs(path, filters={"group": group}))


def _extract_hp_from_run(run) -> dict:
    """Extract learning rate and warmup epochs from a run's config."""
    config = run.config
    lr = config.get("hyperparameters", {}).get("learning_rate")
    warmup = config.get("module", {}).get("warmup_epochs", 0)
    fold = config.get("hyperparameters", {}).get("fold_number", 0)
    return {"lr": lr, "warmup": warmup, "fold": fold}


def _extract_metrics(run) -> dict:
    """Extract key metrics from a run's summary."""
    return {
        "best_val_f1": unwrap_summary_value(run.summary.get(VAL_F1), "max"),
        "best_val_loss": unwrap_summary_value(run.summary.get(VAL_LOSS), "min"),
        "final_train_loss": unwrap_summary_value(
            run.summary.get(TRAIN_LOSS), "min"
        ),
        "best_epoch": unwrap_summary_value(run.summary.get("epoch"), "max"),
    }


def fetch_sweep_results(groups: dict[str, str], api: wandb.Api) -> pd.DataFrame:
    rows = []
    for condition, group in groups.items():
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs found for group '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for {condition} ({group})")

        for run in runs:
            if run.state != "finished":
                print(f"    Skipping {run.id} (state={run.state})")
                continue

            hp = _extract_hp_from_run(run)
            metrics = _extract_metrics(run)
            rows.append(
                {
                    "Condition": condition,
                    "LR": hp["lr"],
                    "Warmup": hp["warmup"],
                    "Fold": hp["fold"],
                    "Run ID": run.id,
                    "Run Name": run.name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def print_hp_grid(df: pd.DataFrame) -> None:
    """Print a pivoted HP grid showing val F1 for each (LR, Warmup) combo."""
    for condition in df["Condition"].unique():
        sub = df[df["Condition"] == condition]
        pivot = sub.pivot_table(
            values="best_val_f1",
            index="LR",
            columns="Warmup",
            aggfunc="mean",
        )
        pivot = pivot.sort_index()
        print(f"\n{'=' * 60}")
        print(f"  {condition} — Val F1 by (LR, Warmup Epochs)")
        print(f"{'=' * 60}")
        print(pivot.to_string(float_format="%.4f"))

        best_idx = sub["best_val_f1"].idxmax()
        best = sub.loc[best_idx]
        print(
            f"\n  Best: LR={best['LR']}, Warmup={best['Warmup']}"
            f" → Val F1 = {best['best_val_f1']:.4f}"
            f" (epoch {best['best_epoch']}, run {best['Run ID']})"
        )


def plot_hp_heatmaps(df: pd.DataFrame) -> None:
    """Plot side-by-side heatmaps of val F1 across the HP grid."""
    conditions = sorted(df["Condition"].unique())
    if len(conditions) < 2:
        conditions = list(df["Condition"].unique())

    fig, axes = plt.subplots(
        1, len(conditions), figsize=(6 * len(conditions), 5)
    )
    if len(conditions) == 1:
        axes = [axes]

    vmin = df["best_val_f1"].min() - 0.01
    vmax = df["best_val_f1"].max() + 0.01

    for ax, condition in zip(axes, conditions):
        sub = df[df["Condition"] == condition]
        pivot = sub.pivot_table(
            values="best_val_f1",
            index="LR",
            columns="Warmup",
            aggfunc="mean",
        ).sort_index()

        im = ax.imshow(
            pivot.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in pivot.index])
        ax.set_xlabel("Warmup Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{condition}")

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

    fig.suptitle("Val F1 — HP Grid Search (fold 0)", fontsize=14)
    fig.colorbar(im, ax=axes, label="Val F1", shrink=0.8)
    plt.tight_layout()

    out = FIGURES_DIR / "009_hp_search_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_best_comparison(df: pd.DataFrame) -> None:
    """Bar chart: best pretrained vs best scratch vs exp 006/007 baselines."""
    best_per_condition = (
        df.groupby("Condition", as_index=False)
        .apply(lambda g: g.loc[g["best_val_f1"].idxmax()], include_groups=False)
        .reset_index(drop=True)
    )

    labels = []
    vals = []
    colors = []

    labels.append("Exp 007\nFinetune\n(lr=1e-4)")
    vals.append(EXP_007_FINETUNE_CWT_CNN_MEAN)
    colors.append("#E8A0A0")

    labels.append("Exp 006\nScratch\n(lr=1e-4)")
    vals.append(EXP_006_SCRATCH_CWT_CNN_MEAN)
    colors.append("#A0C0E8")

    for _, row in best_per_condition.iterrows():
        cond = row["Condition"]
        lr = row["LR"]
        wu = int(row["Warmup"])
        label = f"Exp 009\n{cond}\n(lr={lr:.0e}, wu={wu})"
        labels.append(label)
        color = "#DD8452" if cond == "Pretrained" else "#4C72B0"
        vals.append(row["best_val_f1"])
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 6))
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
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("CWT-CNN — Best HP Config vs Baselines (fold 0)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(vals) - 0.05))
    plt.tight_layout()

    out = FIGURES_DIR / "009_best_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_lr_curves_by_condition(df: pd.DataFrame, api: wandb.Api) -> None:
    """Learning curves for the best run per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conditions = sorted(df["Condition"].unique())

    for ax, condition in zip(axes, conditions):
        sub = df[df["Condition"] == condition].copy()
        sub = sub.sort_values("best_val_f1", ascending=False)

        for _, row in sub.head(3).iterrows():
            run_id = row["Run ID"]
            lr = row["LR"]
            wu = int(row["Warmup"])

            try:
                path = (
                    f"{WANDB_ENTITY}/{WANDB_PROJECT}"
                    if WANDB_ENTITY
                    else WANDB_PROJECT
                )
                run = api.run(f"{path}/{run_id}")
                history = run.history(
                    keys=["epoch", VAL_F1], samples=10_000, pandas=True
                )
                history = history.dropna(subset=[VAL_F1])

                if not history.empty:
                    ax.plot(
                        history["epoch"],
                        history[VAL_F1],
                        label=f"lr={lr:.0e}, wu={wu} (F1={row['best_val_f1']:.3f})",
                        linewidth=1.5,
                    )
            except Exception as e:
                print(f"    Could not fetch history for {run_id}: {e}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val F1")
        ax.set_title(f"{condition} — Top 3 Configs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "009_learning_curves_top3.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()

    print("Fetching HP search runs...")
    df = fetch_sweep_results(HP_SEARCH_GROUPS, api)

    if df.empty:
        print("\nNo completed runs found. Check WandB groups and run state.")
        return

    print(f"\nTotal completed runs: {len(df)}")
    print_hp_grid(df)

    print("\n" + "=" * 60)
    print("  Reference baselines (CWT-CNN, mean across 3 folds)")
    print("=" * 60)
    print(f"  Exp 006 from scratch:  {EXP_006_SCRATCH_CWT_CNN_MEAN:.4f}")
    print(f"  Exp 007 finetuned:     {EXP_007_FINETUNE_CWT_CNN_MEAN:.4f}")

    plot_hp_heatmaps(df)
    plot_best_comparison(df)
    plot_lr_curves_by_condition(df, api)

    print("\nChecking for Phase 3 validation runs...")
    val_df = fetch_sweep_results(VALIDATION_GROUPS, api)

    if not val_df.empty:
        print(f"\nValidation runs found: {len(val_df)}")
        agg = (
            val_df.groupby("Condition")["best_val_f1"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print("\n" + "=" * 60)
        print("  Phase 3 — 3-fold validation (mean ± std)")
        print("=" * 60)
        print(agg.to_string())
    else:
        print("  No validation runs yet (Phase 3 not launched).")


if __name__ == "__main__":
    main()

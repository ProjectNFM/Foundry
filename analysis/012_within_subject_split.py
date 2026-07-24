"""Within-subject (intrasession) split control — train-val gap diagnostic.

Compares train-val loss gap between intrasession and intersubject splits
to determine whether the large gap in inter-subject evaluation reflects
model overfitting or inherent subject-level distribution shift.

Fetches runs from KEMP_INTRASUBJECT_SPLIT (intrasession) and compares
against baselines from KEMP_SCRATCH_HP_SEARCH (intersubject, exp 009)
on the same fold.

WandB project: foundry_finetuning

Usage:
    uv run python analysis/012_within_subject_split.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
    unwrap_summary_value,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()

INTRASESSION_GROUP = "KEMP_INTRASUBJECT_SPLIT"
INTRASESSION_CONTROLS_GROUP = "KEMP_INTRASUBJECT_SPLIT_CONTROLS"
INTERSUBJECT_BASELINE_GROUP = "KEMP_SCRATCH_HP_SEARCH"

VAL_F1 = "val/sleep_stage_5class_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)


def _fetch_group_runs(group: str, api: wandb.Api) -> list:
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    return list(api.runs(path, filters={"group": group}))


def _extract_from_run(run) -> dict:
    config = run.config
    hp = config.get("hyperparameters", {})
    data_cfg = config.get("data", {})

    s = run.summary
    train_loss = unwrap_summary_value(s.get(TRAIN_LOSS), "min")
    val_loss = unwrap_summary_value(s.get(VAL_LOSS), "min")
    val_f1 = unwrap_summary_value(s.get(VAL_F1), "max")

    gap = (
        val_loss - train_loss
        if isinstance(train_loss, float) and isinstance(val_loss, float)
        else None
    )

    split_type = data_cfg.get("split_type", "unknown")

    return {
        "LR": hp.get("learning_rate"),
        "Fold": hp.get("fold_number", 0),
        "Split": split_type,
        "Run ID": run.id,
        "Run Name": run.name,
        "Group": run.group,
        "best_val_f1": val_f1,
        "best_val_loss": val_loss,
        "final_train_loss": train_loss,
        "train_val_gap": gap,
        "best_epoch": unwrap_summary_value(s.get("epoch"), "max"),
    }


def fetch_all_runs(api: wandb.Api, fold: int = 0) -> pd.DataFrame:
    """Fetch intrasession runs and intersubject baselines (same fold)."""
    rows = []

    for label, group in [
        ("Intrasession", INTRASESSION_GROUP),
        ("Intrasession Controls", INTRASESSION_CONTROLS_GROUP),
    ]:
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs for '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for {label} ({group})")
        for run in runs:
            if run.state != "finished":
                print(f"    Skipping {run.id} (state={run.state})")
                continue
            rows.append(_extract_from_run(run))

    # Intersubject baseline (best scratch from exp 009, same fold)
    runs = _fetch_group_runs(INTERSUBJECT_BASELINE_GROUP, api)
    if runs:
        print(
            f"  Found {len(runs)} runs for intersubject baseline "
            f"({INTERSUBJECT_BASELINE_GROUP})"
        )
        for run in runs:
            if run.state != "finished":
                continue
            row = _extract_from_run(run)
            if row["Fold"] != fold:
                continue
            rows.append(row)
    else:
        print(f"  No runs for baseline group '{INTERSUBJECT_BASELINE_GROUP}'")

    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print("  Within-Subject Split Control — Gap Comparison")
    print(f"{'=' * 70}")

    for split in df["Split"].unique():
        sub = df[df["Split"] == split].sort_values(
            "best_val_f1", ascending=False
        )
        if sub.empty:
            continue
        print(f"\n--- Split: {split} ---")
        for _, row in sub.iterrows():
            gap_str = (
                f"{row['train_val_gap']:.4f}"
                if row["train_val_gap"] is not None
                else "?"
            )
            print(
                f"  lr={row['LR']:.0e}  "
                f"train={row['final_train_loss']:.4f}  "
                f"val={row['best_val_loss']:.4f}  "
                f"gap={gap_str}  "
                f"f1={row['best_val_f1']:.4f}  "
                f"({row['Run ID']}, {row['Group']})"
            )

    # Summary comparison
    intra = df[df["Split"] == "intrasession"]
    inter = df[df["Split"] == "intersubject"]

    if not intra.empty and not inter.empty:
        intra_best = intra.loc[intra["best_val_f1"].idxmax()]
        inter_best = inter.loc[inter["best_val_f1"].idxmax()]

        print(f"\n{'=' * 70}")
        print("  Key comparison (best run per split)")
        print(f"{'=' * 70}")
        print(
            f"  Intrasession: gap={intra_best['train_val_gap']:.4f}  "
            f"f1={intra_best['best_val_f1']:.4f}"
        )
        print(
            f"  Intersubject: gap={inter_best['train_val_gap']:.4f}  "
            f"f1={inter_best['best_val_f1']:.4f}"
        )
        gap_reduction = (
            inter_best["train_val_gap"] - intra_best["train_val_gap"]
        )
        print(
            f"\n  Gap reduction (inter→intra): {gap_reduction:.4f} "
            f"({gap_reduction / inter_best['train_val_gap'] * 100:.0f}%)"
        )
        f1_diff = intra_best["best_val_f1"] - inter_best["best_val_f1"]
        print(
            f"  F1 improvement (inter→intra): {f1_diff:+.4f} ({f1_diff * 100:+.1f} pp)"
        )


def plot_gap_comparison(df: pd.DataFrame) -> None:
    """Bar chart comparing train-val gap between split types."""
    intra = df[df["Split"] == "intrasession"]
    inter = df[df["Split"] == "intersubject"]

    if intra.empty or inter.empty:
        print("  Cannot plot — need both split types.")
        return

    intra_best = intra.loc[intra["best_val_f1"].idxmax()]
    inter_best = inter.loc[inter["best_val_f1"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Train-val gap
    ax = axes[0]
    labels = [
        "Intrasession\n(within-subject)",
        "Intersubject\n(between-subject)",
    ]
    gaps = [intra_best["train_val_gap"], inter_best["train_val_gap"]]
    colors = ["#4C72B0", "#DD8452"]

    bars = ax.bar(
        labels, gaps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Train-Val Loss Gap by Split Type")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Val F1
    ax = axes[1]
    f1s = [intra_best["best_val_f1"], inter_best["best_val_f1"]]
    bars = ax.bar(
        labels, f1s, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("Val F1 by Split Type")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(f1s) - 0.1))

    plt.tight_layout()
    out = FIGURES_DIR / "012_split_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_loss_curves(df: pd.DataFrame, api: wandb.Api) -> None:
    """Learning curves showing train/val loss over epochs for both splits."""
    intra = df[df["Split"] == "intrasession"]
    inter = df[df["Split"] == "intersubject"]

    if intra.empty or inter.empty:
        print("  Cannot plot loss curves — need both split types.")
        return

    intra_id = intra.loc[intra["best_val_f1"].idxmax(), "Run ID"]
    inter_id = inter.loc[inter["best_val_f1"].idxmax(), "Run ID"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (run_id, label, color) in zip(
        axes,
        [
            (intra_id, "Intrasession (within-subject)", "#4C72B0"),
            (inter_id, "Intersubject (between-subject)", "#DD8452"),
        ],
    ):
        try:
            history = fetch_metric_history(
                run_id,
                [TRAIN_LOSS, VAL_LOSS],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                api=api,
            )
            if not history.empty:
                ax.plot(
                    history["epoch"],
                    history[TRAIN_LOSS],
                    label="Train loss",
                    color=color,
                    linewidth=1.5,
                )
                ax.plot(
                    history["epoch"],
                    history[VAL_LOSS],
                    label="Val loss",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.fill_between(
                    history["epoch"],
                    history[TRAIN_LOSS],
                    history[VAL_LOSS],
                    alpha=0.15,
                    color=color,
                )
        except Exception as e:
            print(f"  Could not fetch history for {run_id}: {e}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{label}\n(run: {run_id})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "012_loss_curves.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()

    print("Fetching runs for within-subject split control...")
    df = fetch_all_runs(api, fold=0)

    if df.empty:
        print("\nNo completed runs found. Launch the experiment first.")
        return

    print(f"\nTotal runs fetched: {len(df)}")
    print_results(df)
    plot_gap_comparison(df)
    plot_loss_curves(df, api)


if __name__ == "__main__":
    main()

"""Within-subject (intrasession) split control — train-val gap diagnostic.

Compares train-val loss gap and val F1 across three conditions:
  1. Intrasession scratch — within-subject split, trained from scratch
  2. Intrasession pretrained — within-subject split, finetuned from SSL checkpoint
  3. Intersubject scratch — between-subject split, trained from scratch (control)

Fetches runs from KEMP_INTRASUBJECT_SPLIT (intrasession conditions) and
KEMP_INTRASUBJECT_SPLIT_CONTROLS (intersubject control). Also pulls the
best intersubject baseline from KEMP_SCRATCH_HP_SEARCH (exp 009) for
additional context.

WandB project: foundry_finetuning

Usage:
    uv run python analysis/012_within_subject_split.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
    unwrap_summary_value,
)

USABLE_RUN_STATES = frozenset({"finished", "failed", "crashed", "killed"})
MIN_EPOCHS = 5

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()

INTRASESSION_GROUP = "KEMP_INTRASUBJECT_SPLIT"
INTRASESSION_CONTROLS_GROUP = "KEMP_INTRASUBJECT_SPLIT_CONTROLS"
INTERSUBJECT_BASELINE_GROUP = "KEMP_SCRATCH_HP_SEARCH"

VAL_F1 = "val/sleep_stage_5class_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)

CONDITION_COLORS = {
    "Intrasession\n(scratch)": "#4C72B0",
    "Intrasession\n(pretrained)": "#55A868",
    "Intersubject\n(scratch)": "#DD8452",
}


def _fetch_group_runs(group: str, api: wandb.Api) -> list:
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    return list(api.runs(path, filters={"group": group}))


def _extract_from_run(run) -> dict:
    config = run.config
    hp = config.get("hyperparameters", {})
    data_cfg = config.get("data", {})
    run_cfg = config.get("run", {})

    is_pretrained = run_cfg.get("pretrained_checkpoint") is not None
    init_mode = run_cfg.get("init_mode", "scratch")
    if is_pretrained or init_mode == "pretrained":
        init_label = "pretrained"
    else:
        init_label = "scratch"

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
        "Init": init_label,
        "Run ID": run.id,
        "Run Name": run.name,
        "Group": run.group,
        "State": run.state,
        "best_val_f1": val_f1,
        "best_val_loss": val_loss,
        "final_train_loss": train_loss,
        "train_val_gap": gap,
        "best_epoch": unwrap_summary_value(s.get("epoch"), "max"),
    }


def _condition_label(row: pd.Series) -> str:
    split = row["Split"]
    init = row["Init"]
    if split == "intrasession":
        return f"Intrasession\n({init})"
    else:
        return f"Intersubject\n({init})"


def _epochs_completed(run) -> float | None:
    epoch = unwrap_summary_value(run.summary.get("epoch"), "max")
    if isinstance(epoch, (int, float)) and not np.isnan(epoch):
        return float(epoch)
    return None


def _is_usable_run(run) -> tuple[bool, str]:
    if run.state not in USABLE_RUN_STATES:
        return False, f"state={run.state}"
    epochs = _epochs_completed(run)
    if epochs is None:
        return False, "no epoch logged"
    if epochs < MIN_EPOCHS:
        return False, f"epoch={epochs:.0f} < min_epochs={MIN_EPOCHS}"
    return True, ""


def fetch_all_runs(api: wandb.Api, fold: int = 0) -> pd.DataFrame:
    """Fetch intrasession runs, controls, and intersubject baselines."""
    rows = []

    for label, group in [
        ("Intrasession", INTRASESSION_GROUP),
        ("Intersubject Controls", INTRASESSION_CONTROLS_GROUP),
    ]:
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs for '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for {label} ({group})")
        for run in runs:
            usable, reason = _is_usable_run(run)
            if not usable:
                print(f"    Skipping {run.id} ({reason})")
                continue
            row = _extract_from_run(run)
            if run.state != "finished":
                epochs = _epochs_completed(run)
                print(
                    f"    Including partial run {run.id} "
                    f"(state={run.state}, epochs={epochs:.0f})"
                )
            rows.append(row)

    # Intersubject baseline from exp 009 (same fold) for extra context
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

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Condition"] = df.apply(_condition_label, axis=1)
    return df


def print_results(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}")
    print("  Within-Subject Split Control — Three-Way Comparison")
    print(f"{'=' * 78}")

    conditions = df["Condition"].unique() if "Condition" in df.columns else []
    for cond in sorted(conditions):
        sub = df[df["Condition"] == cond].sort_values(
            "best_val_f1", ascending=False
        )
        if sub.empty:
            continue
        print(f"\n--- {cond.replace(chr(10), ' ')} ---")
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
                f"({row['Run Name']}, {row['Run ID']}, {row['Group']})"
            )

    # Summary comparison: best per condition
    print(f"\n{'=' * 78}")
    print("  Best run per condition")
    print(f"{'=' * 78}")

    best_rows = {}
    for cond in sorted(conditions):
        sub = df[df["Condition"] == cond]
        if sub.empty:
            continue
        best = sub.loc[sub["best_val_f1"].idxmax()]
        best_rows[cond] = best
        label = cond.replace("\n", " ")
        print(
            f"  {label:<30s}  "
            f"gap={best['train_val_gap']:.4f}  "
            f"train={best['final_train_loss']:.4f}  "
            f"val={best['best_val_loss']:.4f}  "
            f"f1={best['best_val_f1']:.4f}  "
            f"(run={best['Run ID']})"
        )

    # Pairwise comparisons
    intra_scratch = best_rows.get("Intrasession\n(scratch)")
    intra_pt = best_rows.get("Intrasession\n(pretrained)")
    inter_scratch = best_rows.get("Intersubject\n(scratch)")

    if intra_scratch is not None and inter_scratch is not None:
        gap_reduction = (
            inter_scratch["train_val_gap"] - intra_scratch["train_val_gap"]
        )
        f1_diff = intra_scratch["best_val_f1"] - inter_scratch["best_val_f1"]
        print("\n  --- Intrasession scratch vs Intersubject scratch ---")
        print(
            f"  Gap reduction (inter→intra): "
            f"{gap_reduction:.4f} "
            f"({gap_reduction / inter_scratch['train_val_gap'] * 100:.0f}%)"
        )
        print(
            f"  F1 improvement (inter→intra): "
            f"{f1_diff:+.4f} ({f1_diff * 100:+.1f} pp)"
        )

    if intra_pt is not None and intra_scratch is not None:
        f1_diff = intra_pt["best_val_f1"] - intra_scratch["best_val_f1"]
        gap_diff = intra_pt["train_val_gap"] - intra_scratch["train_val_gap"]
        print("\n  --- Intrasession pretrained vs Intrasession scratch ---")
        print(f"  F1 difference: {f1_diff:+.4f} ({f1_diff * 100:+.1f} pp)")
        print(f"  Gap difference: {gap_diff:+.4f}")

    if intra_pt is not None and inter_scratch is not None:
        f1_diff = intra_pt["best_val_f1"] - inter_scratch["best_val_f1"]
        gap_diff = intra_pt["train_val_gap"] - inter_scratch["train_val_gap"]
        print("\n  --- Intrasession pretrained vs Intersubject scratch ---")
        print(f"  F1 difference: {f1_diff:+.4f} ({f1_diff * 100:+.1f} pp)")
        print(f"  Gap difference: {gap_diff:+.4f}")


def plot_three_way_comparison(df: pd.DataFrame) -> None:
    """Bar chart comparing train-val gap and val F1 across all three conditions."""
    conditions_ordered = [
        "Intrasession\n(scratch)",
        "Intrasession\n(pretrained)",
        "Intersubject\n(scratch)",
    ]

    best_per_cond = {}
    for cond in conditions_ordered:
        sub = df[df["Condition"] == cond]
        if sub.empty:
            continue
        best_per_cond[cond] = sub.loc[sub["best_val_f1"].idxmax()]

    present = [c for c in conditions_ordered if c in best_per_cond]
    if len(present) < 2:
        print("  Cannot plot — need at least 2 conditions.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x = np.arange(len(present))
    colors = [CONDITION_COLORS[c] for c in present]

    # Panel 1: Train-Val Gap
    ax = axes[0]
    gaps = [best_per_cond[c]["train_val_gap"] for c in present]
    bars = ax.bar(
        x, gaps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
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
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=9)
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Train-Val Loss Gap")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Val F1
    ax = axes[1]
    f1s = [best_per_cond[c]["best_val_f1"] for c in present]
    bars = ax.bar(
        x, f1s, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
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
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=9)
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("Validation F1")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(f1s) - 0.1))

    # Panel 3: Train Loss vs Val Loss (stacked comparison)
    ax = axes[2]
    train_losses = [best_per_cond[c]["final_train_loss"] for c in present]
    val_losses = [best_per_cond[c]["best_val_loss"] for c in present]
    width = 0.35
    bars_t = ax.bar(
        x - width / 2,
        train_losses,
        width,
        label="Train loss",
        color=colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_v = ax.bar(
        x + width / 2,
        val_losses,
        width,
        label="Val loss",
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars_t, train_losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars_v, val_losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=9)
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Within-Subject Split Control — Three-Way Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "012_three_way_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_loss_curves(df: pd.DataFrame, api: wandb.Api) -> None:
    """Learning curves for all conditions side by side."""
    conditions_ordered = [
        "Intrasession\n(scratch)",
        "Intrasession\n(pretrained)",
        "Intersubject\n(scratch)",
    ]

    present = []
    for cond in conditions_ordered:
        sub = df[df["Condition"] == cond]
        if not sub.empty:
            best = sub.loc[sub["best_val_f1"].idxmax()]
            present.append((cond, best["Run ID"], best["Run Name"]))

    if len(present) < 2:
        print("  Cannot plot loss curves — need at least 2 conditions.")
        return

    fig, axes = plt.subplots(
        1, len(present), figsize=(6 * len(present), 5), sharey=True
    )
    if len(present) == 1:
        axes = [axes]

    plotted_any = False
    ymax = 0.0

    for ax, (cond, run_id, run_name) in zip(axes, present):
        color = CONDITION_COLORS[cond]
        try:
            history = fetch_metric_history(
                run_id,
                [TRAIN_LOSS, VAL_LOSS],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                aggregate_epoch=True,
                api=api,
            )
            if not history.empty:
                plotted_any = True
                ymax = max(
                    ymax,
                    history[TRAIN_LOSS].max(),
                    history[VAL_LOSS].max(),
                )
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
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No history available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                )
        except Exception as e:
            print(f"  Could not fetch history for {run_id}: {e}")
            ax.text(
                0.5,
                0.5,
                "History fetch failed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        label = cond.replace("\n", " ")
        ax.set_title(f"{label}\n{run_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if not plotted_any:
        print("  Skipping loss-curve figure — no history data for any run.")
        plt.close()
        return

    for ax in axes:
        ax.set_ylim(bottom=0, top=ymax * 1.1)

    plt.suptitle(
        "Train / Val Loss Curves by Condition",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "012_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_gap_bar(df: pd.DataFrame) -> None:
    """Simple two-panel bar chart: gap + F1, kept for backward compat."""
    intra_scratch = df[
        (df["Split"] == "intrasession") & (df["Init"] == "scratch")
    ]
    inter = df[df["Split"] == "intersubject"]

    if intra_scratch.empty or inter.empty:
        print("  Cannot plot gap bar — need both split types.")
        return

    intra_best = intra_scratch.loc[intra_scratch["best_val_f1"].idxmax()]
    inter_best = inter.loc[inter["best_val_f1"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [
        "Intrasession\n(within-subject)",
        "Intersubject\n(between-subject)",
    ]
    colors = ["#4C72B0", "#DD8452"]

    ax = axes[0]
    gaps = [intra_best["train_val_gap"], inter_best["train_val_gap"]]
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


def main():
    api = wandb.Api()

    print("Fetching runs for within-subject split control...")
    df = fetch_all_runs(api, fold=0)

    if df.empty:
        print("\nNo completed runs found. Launch the experiment first.")
        return

    print(f"\nTotal runs fetched: {len(df)}")
    print_results(df)
    plot_three_way_comparison(df)
    plot_loss_curves(df, api)
    plot_gap_bar(df)


if __name__ == "__main__":
    main()

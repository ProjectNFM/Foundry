"""Full dataset pretraining — embedding mode scaling (exp 017).

Compares embedding configurations from exp 014/016 at full Klinzing
dataset scale.  Conditions and run IDs to be filled in after exp 014/016
results are available.

WandB project: foundry_pretraining
Group: PRETRAIN_FULL_DATASET_SCALING

Runs (fill in after launch):
  TBD

Usage:
    uv run python analysis/017_full_dataset_scaling.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
    fetch_run_summary,
)

WANDB_PROJECT = "foundry_pretraining"
WANDB_ENTITY = default_entity()

# TODO: fill in run IDs after launching
# Conditions TBD based on exp 014/016 results; starting with likely grid
RUNS = {
    "sess-D ch-S": "<TBD>",
    "sess-S ch-S": "<TBD>",
    "sess-D ch-D": "<TBD>",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FIGURES_DIR = figures_dir(__file__)

CONDITION_COLORS = {
    "sess-D ch-S": "#DD8452",
    "sess-S ch-S": "#4C72B0",
    "sess-D ch-D": "#8172B2",
    "sess-S ch-D": "#C44E52",
}


def fetch_all_data() -> dict[str, dict]:
    """Fetch per-epoch metrics and summary for every run."""
    results = {}
    for cond, run_id in RUNS.items():
        print(f"Fetching {cond} ({run_id})...")

        epoch_df = fetch_metric_history(
            run_id,
            [TRAIN_LOSS, VAL_LOSS],
            WANDB_PROJECT,
            WANDB_ENTITY,
            x_axis="epoch",
            aggregate_epoch=True,
        )

        summary = fetch_run_summary(
            run_id,
            WANDB_PROJECT,
            {
                "best_val_loss": (VAL_LOSS, "min"),
                "max_epoch": ("epoch", "max"),
            },
            WANDB_ENTITY,
        )

        best_val_epoch = None
        best_val_loss = None
        train_at_best_val = None
        gap_at_best_val = None
        if not epoch_df.empty and VAL_LOSS in epoch_df.columns:
            valid = epoch_df.dropna(subset=[VAL_LOSS])
            if not valid.empty:
                best_idx = valid[VAL_LOSS].idxmin()
                best_row = valid.loc[best_idx]
                best_val_epoch = int(best_row["epoch"])
                best_val_loss = float(best_row[VAL_LOSS])
                if TRAIN_LOSS in best_row and pd.notna(best_row[TRAIN_LOSS]):
                    train_at_best_val = float(best_row[TRAIN_LOSS])
                    gap_at_best_val = best_val_loss - train_at_best_val

        overfit_epoch = None
        if not epoch_df.empty and VAL_LOSS in epoch_df.columns:
            val_series = epoch_df.dropna(subset=[VAL_LOSS]).sort_values("epoch")
            if len(val_series) >= 3:
                val_vals = val_series[VAL_LOSS].values
                for i in range(1, len(val_vals)):
                    if val_vals[i] > val_vals[i - 1]:
                        overfit_epoch = int(val_series["epoch"].iloc[i])
                        break

        results[cond] = {
            "run_id": run_id,
            "state": summary["state"],
            "best_val_loss": best_val_loss or summary["best_val_loss"],
            "train_at_best_val": train_at_best_val,
            "gap_at_best_val": gap_at_best_val,
            "best_val_epoch": best_val_epoch,
            "overfit_epoch": overfit_epoch,
            "max_epoch": summary["max_epoch"],
            "epoch_df": epoch_df,
        }
    return results


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 100}")
    print("  Full Dataset Pretraining — Embedding Mode Scaling (exp 017)")
    print(f"{'=' * 100}")

    header = (
        f"{'Condition':<15s}  {'Best Val':>10s}  {'Train@BV':>10s}  "
        f"{'Gap':>8s}  {'BV Epoch':>8s}  {'OF Epoch':>8s}  "
        f"{'Max Ep':>6s}  {'State':<10s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    for cond in RUNS:
        d = data[cond]
        val_s = f"{d['best_val_loss']:.4f}" if d["best_val_loss"] else "?"
        train_s = (
            f"{d['train_at_best_val']:.4f}"
            if d["train_at_best_val"] is not None
            else "?"
        )
        gap_s = (
            f"{d['gap_at_best_val']:.4f}"
            if d["gap_at_best_val"] is not None
            else "?"
        )
        of_s = (
            str(d["overfit_epoch"]) if d["overfit_epoch"] is not None else "—"
        )
        print(
            f"{cond:<15s}  {val_s:>10s}  {train_s:>10s}  "
            f"{gap_s:>8s}  {d['best_val_epoch']:>8}  {of_s:>8s}  "
            f"{d['max_epoch']:>6}  {d['state']:<10s}  {d['run_id']}"
        )

    print()
    best_cond = min(
        data, key=lambda c: data[c]["best_val_loss"] or float("inf")
    )
    print(
        f"  Best validation loss: {best_cond} ({data[best_cond]['best_val_loss']:.4f})"
    )


def plot_val_comparison(data: dict) -> None:
    """Bar chart comparing best validation loss and train-val gap."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    vals = [data[c]["best_val_loss"] for c in conds]
    colors = [CONDITION_COLORS.get(c, "#999999") for c in conds]
    bars = ax.bar(
        conds,
        vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Validation Loss")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    gaps = [data[c]["gap_at_best_val"] or 0 for c in conds]
    bars = ax.bar(
        conds,
        gaps,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Train-Val Gap at Best Val Epoch")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Full Dataset Pretraining — Scaling (exp 017)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "017_val_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_val_overlay(data: dict) -> None:
    """Overlay val loss curves for all conditions."""
    conds = list(RUNS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conds:
        d = data[cond]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        label = f"{cond} (best={d['best_val_loss']:.4f})"
        color = CONDITION_COLORS.get(cond, "#999999")
        ax.plot(
            valid_val["epoch"],
            valid_val[VAL_LOSS],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=label,
        )

        if d["best_val_epoch"] is not None:
            best_row = valid_val[valid_val["epoch"] == d["best_val_epoch"]]
            if not best_row.empty:
                ax.plot(
                    d["best_val_epoch"],
                    best_row[VAL_LOSS].values[0],
                    marker="*",
                    markersize=14,
                    color=color,
                    zorder=5,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "Validation Loss — Full Dataset Scaling (exp 017)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "017_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Train/val loss curves, one subplot per condition."""
    conds = list(RUNS.keys())
    n = len(conds)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for i, cond in enumerate(conds):
        ax = axes[i]
        d = data[cond]
        edf = d["epoch_df"]
        color = CONDITION_COLORS.get(cond, "#999999")

        if TRAIN_LOSS in edf.columns:
            valid_train = edf.dropna(subset=[TRAIN_LOSS]).sort_values("epoch")
            ax.plot(
                valid_train["epoch"],
                valid_train[TRAIN_LOSS],
                color=color,
                linewidth=2,
                label="Train",
            )

        if VAL_LOSS in edf.columns:
            valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
            ax.plot(
                valid_val["epoch"],
                valid_val[VAL_LOSS],
                color=color,
                linewidth=2,
                linestyle="--",
                label="Val",
            )

        if TRAIN_LOSS in edf.columns and VAL_LOSS in edf.columns:
            both = edf.dropna(subset=[TRAIN_LOSS, VAL_LOSS]).sort_values(
                "epoch"
            )
            if not both.empty:
                ax.fill_between(
                    both["epoch"],
                    both[TRAIN_LOSS],
                    both[VAL_LOSS],
                    alpha=0.15,
                    color=color,
                )

        gap_s = (
            f"gap={d['gap_at_best_val']:.3f}"
            if d["gap_at_best_val"] is not None
            else ""
        )
        ax.set_title(
            f"{cond}\n(val={d['best_val_loss']:.4f}, {gap_s})", fontsize=10
        )
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Learning Curves — Full Dataset Scaling (exp 017)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "017_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for full dataset scaling (exp 017)...")
    data = fetch_all_data()
    print_summary(data)
    plot_val_comparison(data)
    plot_val_overlay(data)
    plot_learning_curves(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

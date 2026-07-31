"""Session embedding mode comparison for intersubject pretraining (exp 014).

Compares train-val loss curves and best validation performance across three
session embedding modes: static, disabled, and dynamic.

WandB project: foundry_pretraining
Group: PRETRAIN_SESSION_EMB_COMPARISON

Runs:
  pretrain_sessemb_static:   zjkkc5j6
  pretrain_sessemb_disabled: 0bsi4w78
  pretrain_sessemb_dynamic:  owetriji

Usage:
    uv run python analysis/014_session_emb_mode_comparison.py
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

RUNS = {
    "Static": "zjkkc5j6",
    "Disabled": "0bsi4w78",
    "Dynamic": "owetriji",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FIGURES_DIR = figures_dir(__file__)

MODE_COLORS = {
    "Static": "#4C72B0",
    "Disabled": "#DD8452",
    "Dynamic": "#55A868",
}


def fetch_all_data() -> dict[str, dict]:
    """Fetch per-epoch metrics and summary for every run."""
    results = {}
    for mode, run_id in RUNS.items():
        print(f"Fetching {mode} ({run_id})...")

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

        # Find overfitting onset: first epoch where val loss increases
        overfit_epoch = None
        if not epoch_df.empty and VAL_LOSS in epoch_df.columns:
            val_series = epoch_df.dropna(subset=[VAL_LOSS]).sort_values("epoch")
            if len(val_series) >= 3:
                val_vals = val_series[VAL_LOSS].values
                for i in range(1, len(val_vals)):
                    if val_vals[i] > val_vals[i - 1]:
                        overfit_epoch = int(val_series["epoch"].iloc[i])
                        break

        results[mode] = {
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
    print(f"\n{'=' * 90}")
    print("  Session Embedding Mode Comparison — Summary (exp 014)")
    print(f"{'=' * 90}")

    header = (
        f"{'Mode':<10s}  {'Best Val':>10s}  {'Train@BV':>10s}  "
        f"{'Gap':>8s}  {'BV Epoch':>8s}  {'OF Epoch':>8s}  "
        f"{'Max Ep':>6s}  {'State':<10s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    for mode in ["Static", "Disabled", "Dynamic"]:
        d = data[mode]
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
            f"{mode:<10s}  {val_s:>10s}  {train_s:>10s}  "
            f"{gap_s:>8s}  {d['best_val_epoch']:>8}  {of_s:>8s}  "
            f"{d['max_epoch']:>6}  {d['state']:<10s}  {d['run_id']}"
        )

    print()
    best_mode = min(
        data, key=lambda m: data[m]["best_val_loss"] or float("inf")
    )
    print(
        f"  Best validation loss: {best_mode} ({data[best_mode]['best_val_loss']:.4f})"
    )


def plot_val_comparison(data: dict) -> None:
    """Bar chart comparing best validation loss and train-val gap."""
    modes = ["Static", "Disabled", "Dynamic"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Best val loss
    ax = axes[0]
    vals = [data[m]["best_val_loss"] for m in modes]
    colors = [MODE_COLORS[m] for m in modes]
    bars = ax.bar(
        modes,
        vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Validation Loss by Mode")
    ax.grid(axis="y", alpha=0.3)
    y_min = min(vals) * 0.95
    y_max = max(vals) * 1.05
    ax.set_ylim(y_min, y_max)

    # Panel 2: Train-val gap at best val epoch
    ax = axes[1]
    gaps = [data[m]["gap_at_best_val"] or 0 for m in modes]
    bars = ax.bar(
        modes,
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
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Train-Val Gap at Best Val Epoch")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Session Embedding Mode Comparison (exp 014)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "014_val_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Train/val loss curves overlaid, with overfitting onset marked."""
    modes = ["Static", "Disabled", "Dynamic"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, mode in enumerate(modes):
        ax = axes[i]
        d = data[mode]
        edf = d["epoch_df"]
        color = MODE_COLORS[mode]

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

        if d["best_val_epoch"] is not None and d["best_val_loss"] is not None:
            ax.axvline(
                d["best_val_epoch"],
                color="gray",
                linewidth=1,
                linestyle=":",
                alpha=0.7,
            )
            ax.annotate(
                f"best val\nepoch {d['best_val_epoch']}",
                xy=(d["best_val_epoch"], d["best_val_loss"]),
                xytext=(d["best_val_epoch"] + 1, d["best_val_loss"] + 0.02),
                fontsize=8,
                color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        gap_s = (
            f"gap={d['gap_at_best_val']:.3f}"
            if d["gap_at_best_val"] is not None
            else ""
        )
        of_s = (
            f"overfit@ep{d['overfit_epoch']}"
            if d["overfit_epoch"] is not None
            else "no overfit detected"
        )
        ax.set_title(
            f"{mode}\n(val={d['best_val_loss']:.4f}, {gap_s}, {of_s})",
            fontsize=10,
        )
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Learning Curves — Session Embedding Modes",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "014_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_val_overlay(data: dict) -> None:
    """Overlay val loss curves for all three modes on a single axis."""
    modes = ["Static", "Disabled", "Dynamic"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for mode in modes:
        d = data[mode]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        ax.plot(
            valid_val["epoch"],
            valid_val[VAL_LOSS],
            color=MODE_COLORS[mode],
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=f"{mode} (best={d['best_val_loss']:.4f})",
        )

        if d["best_val_epoch"] is not None:
            best_row = valid_val[valid_val["epoch"] == d["best_val_epoch"]]
            if not best_row.empty:
                ax.plot(
                    d["best_val_epoch"],
                    best_row[VAL_LOSS].values[0],
                    marker="*",
                    markersize=14,
                    color=MODE_COLORS[mode],
                    zorder=5,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "Validation Loss by Session Embedding Mode",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "014_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for session embedding mode comparison (exp 014)...")
    data = fetch_all_data()
    print_summary(data)
    plot_val_comparison(data)
    plot_learning_curves(data)
    plot_val_overlay(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

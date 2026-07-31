"""Full dataset pretraining scaling — exp 017.

Compares the single surviving run (sess-static, ch-disabled) on the FULL
Klinzing dataset against the equivalent condition from exp 016 on the SMALL
subset.  Three of the four exp 017 runs crashed due to a race condition in
the data staging archive step (see experiment markdown for details).

WandB project: foundry_pretraining
Group: PRETRAIN_FULL_DATASET_SCALING

Exp 017 runs:
  sess-static, ch-disabled (FULL):  qw6q86bw  (only survivor)

Comparable exp 016 runs (SMALL subset):
  sess-S ch-S:    zftehsnf
  sess-S ch-D:    gp79rubc   (direct comparator)
  sess-D ch-S:    574sq9ay
  sess-D ch-D:    6htgoclv

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

EXP017_RUN = {
    "Full sess-S ch-D": "qw6q86bw",
}

EXP016_RUNS = {
    "Small sess-S ch-S": "zftehsnf",
    "Small sess-S ch-D": "gp79rubc",
    "Small sess-D ch-S": "574sq9ay",
    "Small sess-D ch-D": "6htgoclv",
}

ALL_RUNS = {**EXP017_RUN, **EXP016_RUNS}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FIGURES_DIR = figures_dir(__file__)

COLORS = {
    "Full sess-S ch-D": "#E63946",
    "Small sess-S ch-S": "#4C72B0",
    "Small sess-S ch-D": "#C44E52",
    "Small sess-D ch-S": "#DD8452",
    "Small sess-D ch-D": "#8172B2",
}

SUMMARY_KEYS = {
    "best_val_loss": (VAL_LOSS, "min"),
    "best_train_loss": (TRAIN_LOSS, "min"),
    "max_epoch": ("epoch", "max"),
}


def fetch_all_data() -> dict[str, dict]:
    """Fetch per-epoch metrics and summary for every run."""
    results = {}
    for label, run_id in ALL_RUNS.items():
        print(f"Fetching {label} ({run_id})...")

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
            SUMMARY_KEYS,
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

        results[label] = {
            "run_id": run_id,
            "state": summary["state"],
            "best_val_loss": best_val_loss or summary["best_val_loss"],
            "best_train_loss": summary["best_train_loss"],
            "train_at_best_val": train_at_best_val,
            "gap_at_best_val": gap_at_best_val,
            "best_val_epoch": best_val_epoch,
            "max_epoch": summary["max_epoch"],
            "epoch_df": epoch_df,
        }
    return results


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 110}")
    print("  Full Dataset Scaling — exp 017 vs exp 016 (small subset)")
    print(f"{'=' * 110}")

    header = (
        f"{'Condition':<20s}  {'Dataset':<8s}  {'Best Val':>10s}  "
        f"{'Train@BV':>10s}  {'Gap':>8s}  {'BV Epoch':>8s}  "
        f"{'Max Ep':>6s}  {'State':<10s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    for label in ALL_RUNS:
        d = data[label]
        dataset = "Full" if label.startswith("Full") else "Small"
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
        print(
            f"{label:<20s}  {dataset:<8s}  {val_s:>10s}  {train_s:>10s}  "
            f"{gap_s:>8s}  {d['best_val_epoch']:>8}  "
            f"{d['max_epoch']:>6}  {d['state']:<10s}  {d['run_id']}"
        )

    print()
    full_d = data["Full sess-S ch-D"]
    small_d = data["Small sess-S ch-D"]
    print(
        f"  Full dataset  (sess-S ch-D): val={full_d['best_val_loss']:.4f} @ epoch {full_d['best_val_epoch']}"
    )
    print(
        f"  Small dataset (sess-S ch-D): val={small_d['best_val_loss']:.4f} @ epoch {small_d['best_val_epoch']}"
    )
    if full_d["best_val_loss"] and small_d["best_val_loss"]:
        delta = full_d["best_val_loss"] - small_d["best_val_loss"]
        print(f"  Delta (full - small): {delta:+.4f}")


def plot_val_overlay(data: dict) -> None:
    """Overlay val loss curves: full-dataset run vs all small-dataset runs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for label in EXP016_RUNS:
        d = data[label]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        curve_label = f"{label} (best={d['best_val_loss']:.4f})"
        ax.plot(
            valid_val["epoch"],
            valid_val[VAL_LOSS],
            color=COLORS[label],
            linewidth=1.5,
            alpha=0.6,
            linestyle="--",
            marker=".",
            markersize=3,
            label=curve_label,
        )

    for label in EXP017_RUN:
        d = data[label]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        curve_label = f"{label} (best={d['best_val_loss']:.4f})"
        ax.plot(
            valid_val["epoch"],
            valid_val[VAL_LOSS],
            color=COLORS[label],
            linewidth=3,
            marker="o",
            markersize=6,
            label=curve_label,
            zorder=10,
        )
        if d["best_val_epoch"] is not None:
            best_row = valid_val[valid_val["epoch"] == d["best_val_epoch"]]
            if not best_row.empty:
                ax.plot(
                    d["best_val_epoch"],
                    best_row[VAL_LOSS].values[0],
                    marker="*",
                    markersize=16,
                    color=COLORS[label],
                    zorder=15,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "Validation Loss — Full Dataset (exp 017) vs Small Subset (exp 016)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "017_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Train/val loss for the full-dataset run vs its direct small-dataset comparator."""
    comparisons = {
        "Full Dataset (sess-S ch-D)": "Full sess-S ch-D",
        "Small Subset (sess-S ch-D)": "Small sess-S ch-D",
    }
    comp_colors = {
        "Full Dataset (sess-S ch-D)": "#E63946",
        "Small Subset (sess-S ch-D)": "#C44E52",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i, (title, key) in enumerate(comparisons.items()):
        ax = axes[i]
        d = data[key]
        edf = d["epoch_df"]
        color = comp_colors[title]

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
            f"gap={d['gap_at_best_val']:.4f}"
            if d["gap_at_best_val"] is not None
            else ""
        )
        ax.set_title(
            f"{title}\n(val={d['best_val_loss']:.4f}, {gap_s})",
            fontsize=11,
        )
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Learning Curves — Full Dataset vs Small Subset (sess-S ch-D)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "017_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_bar_comparison(data: dict) -> None:
    """Bar chart comparing best val loss across all runs."""
    labels_order = list(ALL_RUNS.keys())
    vals = [data[label]["best_val_loss"] for label in labels_order]
    colors = [COLORS[label] for label in labels_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(labels_order)),
        vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(range(len(labels_order)))
    x_labels = []
    for label in labels_order:
        dataset = "FULL" if label.startswith("Full") else "small"
        cond = label.replace("Full ", "").replace("Small ", "")
        epochs = data[label]["max_epoch"]
        x_labels.append(f"{cond}\n({dataset}, {epochs}ep)")
    ax.set_xticklabels(x_labels, fontsize=9)

    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel("Best Validation Loss")
    ax.set_title(
        "Best Validation Loss — Full Dataset (exp 017) vs Small Subset (exp 016)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    valid_vals = [v for v in vals if v is not None]
    if valid_vals:
        y_min = min(valid_vals) * 0.97
        y_max = max(valid_vals) * 1.03
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    out = FIGURES_DIR / "017_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for full dataset scaling (exp 017)...")
    data = fetch_all_data()
    print_summary(data)
    plot_val_overlay(data)
    plot_learning_curves(data)
    plot_bar_comparison(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Dynamic channel embedding comparison (exp 018).

Compares ch-disabled (exp 016 winner) vs ch-dynamic (new relative
inter-channel attention) for intersubject pretraining on Klinzing subset.

WandB project: foundry_pretraining
Group: PRETRAIN_DYNAMIC_CHANNEL_EMB

Runs:
  ch-disabled:  zmxyua36
  ch-dynamic:   hggeonah

Usage:
    uv run python analysis/018_dynamic_channel_emb.py
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
    "ch-disabled": "zmxyua36",
    "ch-dynamic": "hggeonah",
}

CONDITION_LABELS = {
    "ch-disabled": "Channel=Disabled\n(exp 016 baseline)",
    "ch-dynamic": "Channel=Dynamic\n(relative attention)",
}

CONDITION_COLORS = {
    "ch-disabled": "#8172B2",
    "ch-dynamic": "#55A868",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FIGURES_DIR = figures_dir(__file__)


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

        final_val_loss = None
        final_train_loss = None
        if not epoch_df.empty:
            last_row = epoch_df.sort_values("epoch").iloc[-1]
            if VAL_LOSS in last_row and pd.notna(last_row[VAL_LOSS]):
                final_val_loss = float(last_row[VAL_LOSS])
            if TRAIN_LOSS in last_row and pd.notna(last_row[TRAIN_LOSS]):
                final_train_loss = float(last_row[TRAIN_LOSS])

        results[cond] = {
            "run_id": run_id,
            "state": summary["state"],
            "best_val_loss": best_val_loss or summary["best_val_loss"],
            "train_at_best_val": train_at_best_val,
            "gap_at_best_val": gap_at_best_val,
            "best_val_epoch": best_val_epoch,
            "final_val_loss": final_val_loss,
            "final_train_loss": final_train_loss,
            "max_epoch": summary["max_epoch"],
            "epoch_df": epoch_df,
        }
    return results


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 105}")
    print("  Dynamic Channel Embeddings — disabled vs dynamic (exp 018)")
    print(f"{'=' * 105}")

    header = (
        f"{'Condition':<15s}  {'Best Val':>10s}  {'Train@BV':>10s}  "
        f"{'Gap':>8s}  {'BV Epoch':>8s}  {'Final Val':>10s}  "
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
        final_s = (
            f"{d['final_val_loss']:.4f}"
            if d["final_val_loss"] is not None
            else "?"
        )
        print(
            f"{cond:<15s}  {val_s:>10s}  {train_s:>10s}  "
            f"{gap_s:>8s}  {d['best_val_epoch']:>8}  {final_s:>10s}  "
            f"{d['max_epoch']:>6}  {d['state']:<10s}  {d['run_id']}"
        )

    print()
    best_cond = min(
        data, key=lambda c: data[c]["best_val_loss"] or float("inf")
    )
    d_best = data[best_cond]
    d_other = data[[c for c in data if c != best_cond][0]]
    if d_best["best_val_loss"] and d_other["best_val_loss"]:
        rel_change = (
            (d_best["best_val_loss"] - d_other["best_val_loss"])
            / d_other["best_val_loss"]
            * 100
        )
        print(
            f"  Best: {best_cond} ({d_best['best_val_loss']:.4f}) — "
            f"{rel_change:+.1f}% relative to other condition"
        )


def plot_bar_comparison(data: dict) -> None:
    """Bar chart comparing best val loss and train-val gap."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    vals = [data[c]["best_val_loss"] for c in conds]
    colors = [CONDITION_COLORS[c] for c in conds]
    x_labels = [CONDITION_LABELS[c] for c in conds]
    bars = ax.bar(
        range(len(conds)),
        vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=10)
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
    ax.set_title("Best Validation Loss")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    gaps = [data[c]["gap_at_best_val"] or 0 for c in conds]
    bars = ax.bar(
        range(len(conds)),
        gaps,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=10)
    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Train-Val Gap at Best Val Epoch")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

    plt.suptitle(
        "Dynamic Channel Embeddings — Disabled vs Dynamic (exp 018)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "018_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Train/val loss curves, one subplot per condition."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, cond in enumerate(conds):
        ax = axes[i]
        d = data[cond]
        edf = d["epoch_df"]
        color = CONDITION_COLORS[cond]

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
            f"{CONDITION_LABELS[cond]}\n(best val={d['best_val_loss']:.4f}, {gap_s})",
            fontsize=10,
        )
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Learning Curves — Dynamic Channel Embeddings (exp 018)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "018_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_val_overlay(data: dict) -> None:
    """Overlay val loss curves for both conditions."""
    conds = list(RUNS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conds:
        d = data[cond]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        label = f"{cond} (best={d['best_val_loss']:.4f})"
        ax.plot(
            valid_val["epoch"],
            valid_val[VAL_LOSS],
            color=CONDITION_COLORS[cond],
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
                    color=CONDITION_COLORS[cond],
                    zorder=5,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "Validation Loss — Disabled vs Dynamic Channel Embeddings (exp 018)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "018_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for dynamic channel embedding comparison (exp 018)...")
    data = fetch_all_data()
    print_summary(data)
    plot_bar_comparison(data)
    plot_learning_curves(data)
    plot_val_overlay(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

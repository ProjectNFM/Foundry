"""Channel embedding ablation for intersubject pretraining (exp 016).

Compares a 2×2 grid of session_emb (static, disabled) × channel_emb
(static, disabled) on the Klinzing subset.

WandB project: foundry_pretraining
Group: PRETRAIN_CHANNEL_EMB_ABLATION

Runs (fill in after launch):
  sess-static,  ch-static:   <TBD>
  sess-static,  ch-disabled: <TBD>
  sess-disabled, ch-static:  <TBD>
  sess-disabled, ch-disabled:<TBD>

Usage:
    uv run python analysis/016_channel_emb_ablation.py
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
RUNS = {
    "sess-S ch-S": "<TBD>",
    "sess-S ch-D": "<TBD>",
    "sess-D ch-S": "<TBD>",
    "sess-D ch-D": "<TBD>",
}

CONDITION_LABELS = {
    "sess-S ch-S": "Session=Static\nChannel=Static",
    "sess-S ch-D": "Session=Static\nChannel=Disabled",
    "sess-D ch-S": "Session=Disabled\nChannel=Static",
    "sess-D ch-D": "Session=Disabled\nChannel=Disabled",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
FIGURES_DIR = figures_dir(__file__)

CONDITION_COLORS = {
    "sess-S ch-S": "#4C72B0",
    "sess-S ch-D": "#C44E52",
    "sess-D ch-S": "#DD8452",
    "sess-D ch-D": "#8172B2",
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
    print("  Channel Embedding Ablation — 2×2 Grid (exp 016)")
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


def plot_grid_comparison(data: dict) -> None:
    """2×2 heatmap-style bar chart comparing best val loss."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
        width=0.6,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=9)
    for bar, val in zip(bars, vals):
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
    if all(v is not None for v in vals):
        y_min = min(v for v in vals if v is not None) * 0.95
        y_max = max(v for v in vals if v is not None) * 1.05
        ax.set_ylim(y_min, y_max)

    ax = axes[1]
    gaps = [data[c]["gap_at_best_val"] or 0 for c in conds]
    bars = ax.bar(
        range(len(conds)),
        gaps,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=9)
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
        "Channel Embedding Ablation — 2×2 Grid (exp 016)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "016_grid_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_val_overlay(data: dict) -> None:
    """Overlay val loss curves for all four conditions."""
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
        "Validation Loss — Channel Embedding Ablation (exp 016)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "016_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Train/val loss curves, one subplot per condition."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

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
            f"gap={d['gap_at_best_val']:.3f}"
            if d["gap_at_best_val"] is not None
            else ""
        )
        ax.set_title(
            f"{CONDITION_LABELS[cond]}\n(val={d['best_val_loss']:.4f}, {gap_s})",
            fontsize=9,
        )
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Learning Curves — Channel Embedding Ablation (exp 016)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "016_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for channel embedding ablation (exp 016)...")
    data = fetch_all_data()
    print_summary(data)
    plot_grid_comparison(data)
    plot_val_overlay(data)
    plot_learning_curves(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

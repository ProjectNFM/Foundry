"""Brain Invaders Reprocessed Long Training — HP Search Analysis.

Fetches runs from WandB groups:
  - BI_P300_EEGNET_REPROC_LONG (EEGNet, 5 LR values, ~94 epochs before cancel)
  - BI_P300_POYO_RCNN_REPROC_LONG (POYO ResampleCNN, 5 LR values, ~44 epochs before cancel)
  - BI_P300_HP_EEGNET_REPROCESSED_V2 (EEGNet patience=50, for comparison)

Runs were cancelled because of clear overfitting. Uses WandB summary
(max/min aggregates) since epoch-level history is unavailable for cancelled runs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY = "poyo-eeg"
PROJECT = "foundry_finetuning"

GROUPS = {
    "eegnet_long": "BI_P300_EEGNET_REPROC_LONG",
    "poyo_rcnn_long": "BI_P300_POYO_RCNN_REPROC_LONG",
    "eegnet_v2": "BI_P300_HP_EEGNET_REPROCESSED_V2",
}

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = "031_bi_reproc_long"


def unwrap(val, key="max"):
    if hasattr(val, "get"):
        return float(val.get(key, 0.0))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def fetch_runs_from_summary(group: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})

    records = []
    for run in runs:
        try:
            lr = float(
                run.config.get("hyperparameters", {}).get("learning_rate", 0)
            )
        except (TypeError, ValueError):
            lr_match = re.search(r"lr([\deE.+-]+)", run.name)
            lr = float(lr_match.group(1)) if lr_match else None

        s = run.summary
        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "lr": lr,
                "epoch": s.get("epoch", 0),
                "best_val_f1": unwrap(s.get("val/p300_binary_f1", 0)),
                "best_val_auroc": unwrap(s.get("val/p300_binary_auroc", 0)),
                "best_val_acc": unwrap(s.get("val/p300_binary_acc", 0)),
                "best_val_recall": unwrap(s.get("val/p300_binary_recall", 0)),
                "best_val_precision": unwrap(
                    s.get("val/p300_binary_precision", 0)
                ),
                "best_val_recall_min": unwrap(
                    s.get("val/p300_binary_recall", 0), "min"
                ),
                "best_val_precision_min": unwrap(
                    s.get("val/p300_binary_precision", 0), "min"
                ),
                "best_train_f1": unwrap(s.get("train/p300_binary_f1", 0)),
                "train_loss_min": unwrap(s.get("train/loss", 0), "min"),
                "val_loss_min": unwrap(s.get("val/loss", 0), "min"),
            }
        )

    return pd.DataFrame(records)


def print_summary(
    model_name: str, df: pd.DataFrame, df_comparison: pd.DataFrame = None
):
    print(f"\n{'=' * 100}")
    print(f"  {model_name.upper()}")
    print(f"{'=' * 100}")

    if df.empty:
        print("  No runs found.")
        return

    df_sorted = df.sort_values("best_val_f1", ascending=False)

    print(
        f"\n  {'LR':>8s}  {'Val F1':>8s}  {'AUROC':>7s}  {'Recall†':>9s}  "
        f"{'Prec†':>7s}  {'Train F1':>9s}  {'Overfit':>8s}  "
        f"{'Train Loss':>10s}  {'Val Loss':>8s}  {'Epochs':>6s}  {'State':>8s}"
    )
    print(f"  {'─' * 107}")
    print(
        "  † Recall/Precision are min-aggregated by WandB (not at best-F1 epoch)"
    )

    for _, row in df_sorted.iterrows():
        gap = row["best_train_f1"] - row["best_val_f1"]
        recall_val = row.get(
            "best_val_recall_min", row.get("best_val_recall", 0)
        )
        prec_val = row.get(
            "best_val_precision_min", row.get("best_val_precision", 0)
        )
        print(
            f"  {row['lr']:>8.0e}  {row['best_val_f1']:>8.4f}  "
            f"{row['best_val_auroc']:>7.4f}  {recall_val:>9.4f}  "
            f"{prec_val:>7.4f}  {row['best_train_f1']:>9.4f}  "
            f"{gap:>+8.3f}  {row['train_loss_min']:>10.6f}  "
            f"{row['val_loss_min']:>8.4f}  {int(row['epoch']):>6d}  "
            f"{row['state']:>8s}"
        )

    best = df_sorted.iloc[0]
    print(
        f"\n  ★ Best config: lr={best['lr']:.0e}, "
        f"val F1={best['best_val_f1']:.4f}"
    )

    if df_comparison is not None and not df_comparison.empty:
        best_comp = df_comparison.sort_values(
            "best_val_f1", ascending=False
        ).iloc[0]
        print("\n  Comparison with patience=50 (parent experiment):")
        print(
            f"    Parent best:  val F1={best_comp['best_val_f1']:.4f} (lr={best_comp['lr']:.0e})"
        )
        print(
            f"    Long best:    val F1={best['best_val_f1']:.4f} (lr={best['lr']:.0e})"
        )
        delta = best["best_val_f1"] - best_comp["best_val_f1"]
        print(f"    Delta:        {delta:+.4f}")


def plot_lr_comparison(model_name: str, df: pd.DataFrame, prefix_tag: str):
    if df.empty:
        return

    df_sorted = df.sort_values("lr")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(df_sorted))
    lr_labels = [f"{lr:.0e}" for lr in df_sorted["lr"]]
    width = 0.35

    # F1
    ax = axes[0]
    train_f1 = df_sorted["best_train_f1"].fillna(0).values
    val_f1 = df_sorted["best_val_f1"].fillna(0).values
    ax.bar(
        x - width / 2,
        train_f1,
        width,
        label="Train F1 (max)",
        color="#2196F3",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        val_f1,
        width,
        label="Val F1 (max)",
        color="#FF9800",
        alpha=0.8,
    )
    for i, (t, v) in enumerate(zip(train_f1, val_f1)):
        ax.text(i - width / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("F1 Score")
    ax.set_title("Train vs Val F1")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Recall & Precision
    ax = axes[1]
    val_recall = df_sorted["best_val_recall"].fillna(0).values
    val_prec = df_sorted["best_val_precision"].fillna(0).values
    ax.bar(
        x - width / 2,
        val_recall,
        width,
        label="Val Recall (max)",
        color="#4CAF50",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        val_prec,
        width,
        label="Val Precision (max)",
        color="#9C27B0",
        alpha=0.8,
    )
    for i, (r, p) in enumerate(zip(val_recall, val_prec)):
        ax.text(i - width / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Score")
    ax.set_title("Val Recall & Precision")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Loss
    ax = axes[2]
    train_loss = df_sorted["train_loss_min"].fillna(0).values
    val_loss = df_sorted["val_loss_min"].fillna(0).values
    ax.bar(
        x - width / 2,
        train_loss,
        width,
        label="Train Loss (min)",
        color="#2196F3",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        val_loss,
        width,
        label="Val Loss (min)",
        color="#FF9800",
        alpha=0.8,
    )
    for i, (t, v) in enumerate(zip(train_loss, val_loss)):
        ax.text(i - width / 2, t + 0.01, f"{t:.4f}", ha="center", fontsize=7)
        ax.text(i + width / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"{model_name}: LR Comparison (Best Values)", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_{prefix_tag}_lr_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_combined_comparison(results: dict):
    """Compare best configs across all groups."""
    bests = {}
    for key, df in results.items():
        if df.empty:
            continue
        best = df.sort_values("best_val_f1", ascending=False).iloc[0]
        bests[key] = best

    if len(bests) < 2:
        return

    model_labels = {
        "eegnet_long": f"EEGNet Long\n(lr={bests.get('eegnet_long', {}).get('lr', 0):.0e})",
        "poyo_rcnn_long": f"POYO RCNN Long\n(lr={bests.get('poyo_rcnn_long', {}).get('lr', 0):.0e})",
        "eegnet_v2": f"EEGNet V2\n(lr={bests.get('eegnet_v2', {}).get('lr', 0):.0e})",
    }

    metrics = [
        "best_val_f1",
        "best_val_auroc",
        "best_val_recall",
        "best_val_precision",
    ]
    metric_labels = ["F1", "AUROC", "Recall", "Precision"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    n_models = len(bests)
    width = 0.8 / n_models
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    for i, (key, best_row) in enumerate(bests.items()):
        vals = [best_row.get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model_labels.get(key, key),
            color=colors[i % len(colors)],
            alpha=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title(
        "Best Config Comparison: All Models\n(Reprocessed Brain Invaders P300, Fold 0)"
    )
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    results = {}

    for model_key, group in GROUPS.items():
        labels = {
            "eegnet_long": "EEGNet Long Training (no early stopping)",
            "poyo_rcnn_long": "POYO ResampleCNN Long Training (no early stopping)",
            "eegnet_v2": "EEGNet V2 (patience=50, parent reference)",
        }
        model_name = labels.get(model_key, model_key)

        print(f"\n{'#' * 100}")
        print(f"# Fetching: {model_name}")
        print(f"# Group: {group}")
        print(f"{'#' * 100}")

        df = fetch_runs_from_summary(group)
        results[model_key] = df
        print(f"  Found {len(df)} runs")

    # Print summaries
    print_summary(
        "EEGNet Long Training (~94 epochs, cancelled)",
        results["eegnet_long"],
        df_comparison=results.get("eegnet_v2"),
    )
    print_summary(
        "POYO ResampleCNN Long Training (~44 epochs, cancelled)",
        results["poyo_rcnn_long"],
    )
    print_summary(
        "EEGNet V2 (patience=50, finished)",
        results["eegnet_v2"],
    )

    # Plots
    for key in ["eegnet_long", "poyo_rcnn_long"]:
        df = results[key]
        name = "EEGNet" if "eegnet" in key else "POYO ResampleCNN"
        plot_lr_comparison(f"{name} (Long Training)", df, key)

    plot_combined_comparison(results)

    # Final summary
    print(f"\n{'#' * 100}")
    print("# FINAL SUMMARY — BEST HP PER MODEL")
    print(f"{'#' * 100}")
    for key in ["eegnet_long", "poyo_rcnn_long"]:
        df = results[key]
        if df.empty:
            continue
        best = df.sort_values("best_val_f1", ascending=False).iloc[0]
        name = "EEGNet" if "eegnet" in key else "POYO ResampleCNN"
        print(f"\n  {name}:")
        print(f"    Best LR: {best['lr']:.0e}")
        print(f"    Val F1:  {best['best_val_f1']:.4f}")
        print(f"    AUROC:   {best['best_val_auroc']:.4f}")
        print(f"    Recall:  {best['best_val_recall']:.4f}")
        print(f"    Prec:    {best['best_val_precision']:.4f}")
        print(
            f"    Overfitting gap (train-val F1): {best['best_train_f1'] - best['best_val_f1']:+.3f}"
        )

    print(f"\n{'=' * 100}")
    print("DONE. All figures saved to:", FIGURES_DIR)
    print(f"{'=' * 100}")

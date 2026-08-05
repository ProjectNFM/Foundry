"""POYO Overfitting Diagnosis — Regularization & Frozen Tokenizer Ablation.

Fetches all 7 runs from WandB group BI_P300_OVERFIT_REGULARIZATION:
  1. baseline     — WD=0.01, dropout=default, unfrozen
  2. wd005        — WD=0.05
  3. wd01         — WD=0.1
  4. drop03       — all dropouts=0.3
  5. drop05       — all dropouts=0.5
  6. frozen_tok   — CWT-CNN frozen (lr_mult=0)
  7. combined     — WD=0.1 + dropout=0.5 + frozen tokenizer

Usage:
    uv run python analysis/033_poyo_overfit_regularization.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from analysis._wandb_utils import default_entity, figures_dir

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity() or "poyo-eeg"
WANDB_GROUP = "BI_P300_OVERFIT_REGULARIZATION"
FIGURES_DIR = figures_dir(__file__)
PREFIX = "033_overfit_reg"

VAL_F1 = "val/p300_binary_f1"
VAL_AUROC = "val/p300_binary_auroc"
VAL_ACC = "val/p300_binary_acc"
VAL_RECALL = "val/p300_binary_recall"
VAL_PRECISION = "val/p300_binary_precision"
TRAIN_F1 = "train/p300_binary_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

CONDITIONS = [
    "baseline",
    "wd005",
    "wd01",
    "drop03",
    "drop05",
    "frozen_tok",
    "combined",
]
CONDITION_LABELS = {
    "baseline": "Baseline\n(WD=0.01)",
    "wd005": "WD=0.05",
    "wd01": "WD=0.1",
    "drop03": "Dropout\n0.3",
    "drop05": "Dropout\n0.5",
    "frozen_tok": "Frozen\nTokenizer",
    "combined": "Combined\n(All)",
}
CONDITION_SHORT = {
    "baseline": "Baseline",
    "wd005": "WD 0.05",
    "wd01": "WD 0.1",
    "drop03": "Dropout 0.3",
    "drop05": "Dropout 0.5",
    "frozen_tok": "Frozen Tok",
    "combined": "Combined",
}
CONDITION_COLORS = {
    "baseline": "#666666",
    "wd005": "#4C72B0",
    "wd01": "#2A4A80",
    "drop03": "#55A868",
    "drop05": "#2D7A3E",
    "frozen_tok": "#E8963E",
    "combined": "#C44E52",
}


def unwrap(val, key="max"):
    if hasattr(val, "get"):
        try:
            return float(val.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def parse_condition(name: str) -> str | None:
    """Extract condition tag from run name like 'bi_p300_overfit_reg_<tag>'."""
    prefix = "bi_p300_overfit_reg_"
    if prefix in name:
        tag = name.split(prefix)[-1]
        if tag in CONDITIONS:
            return tag
    return None


def fetch_all_runs() -> pd.DataFrame:
    """Fetch all runs from the WandB group and return a DataFrame."""
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": WANDB_GROUP},
    )

    records = []
    for run in runs:
        cond = parse_condition(run.name)
        if cond is None:
            print(f"  WARNING: Could not parse run name: {run.name} ({run.id})")
            continue

        s = run.summary
        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "condition": cond,
                "best_val_f1": unwrap(s.get(VAL_F1, 0)),
                "best_val_auroc": unwrap(s.get(VAL_AUROC, 0)),
                "best_val_acc": unwrap(s.get(VAL_ACC, 0)),
                "best_val_recall": unwrap(s.get(VAL_RECALL, 0)),
                "best_val_precision": unwrap(s.get(VAL_PRECISION, 0)),
                "best_train_f1": unwrap(s.get(TRAIN_F1, 0)),
                "train_loss_min": unwrap(s.get(TRAIN_LOSS, 0), "min"),
                "val_loss_min": unwrap(s.get(VAL_LOSS, 0), "min"),
                "epoch": s.get("epoch", 0),
            }
        )
        print(f"  Mapped: {run.name} -> {cond} ({run.id}, state={run.state})")

    df = pd.DataFrame(records)
    print(f"\n  Total runs fetched: {len(df)}")

    missing = [c for c in CONDITIONS if c not in df["condition"].values]
    if missing:
        print(f"  WARNING: Missing conditions: {missing}")

    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary DataFrame with overfit gap."""
    df = df.copy()
    df["overfit_gap"] = df["best_train_f1"] - df["best_val_f1"]
    return (
        df.set_index("condition")
        .loc[[c for c in CONDITIONS if c in df["condition"].values]]
        .reset_index()
    )


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print formatted summary table."""
    print(f"\n{'=' * 110}")
    print(
        "  REGULARIZATION ABLATION — Overfitting Diagnosis (Intersubject, Fold 0)"
    )
    print(f"{'=' * 110}")
    print(
        f"\n  {'Condition':<14s}  {'Val F1':>8s}  {'AUROC':>8s}  "
        f"{'Acc':>7s}  {'Recall':>7s}  {'Prec':>7s}  "
        f"{'Train F1':>9s}  {'Overfit Gap':>11s}  {'Epochs':>7s}  {'State':>10s}"
    )
    print(f"  {'─' * 104}")

    for _, row in summary.iterrows():
        print(
            f"  {CONDITION_SHORT[row['condition']]:<14s}  "
            f"{row['best_val_f1']:.4f}  "
            f"{row['best_val_auroc']:.4f}  "
            f"{row['best_val_acc']:.4f}  "
            f"{row['best_val_recall']:.4f}  "
            f"{row['best_val_precision']:.4f}  "
            f"{row['best_train_f1']:.4f}   "
            f"{row['overfit_gap']:+.4f}   "
            f"{int(row['epoch']):>5d}  "
            f"{row['state']:>10s}"
        )

    # Relative change vs baseline
    baseline = summary[summary["condition"] == "baseline"]
    if not baseline.empty:
        bl_gap = baseline.iloc[0]["overfit_gap"]
        bl_f1 = baseline.iloc[0]["best_val_f1"]
        print(f"\n  {'─' * 104}")
        print(f"  Baseline overfit gap: {bl_gap:+.4f}")
        print(f"  Baseline val F1:      {bl_f1:.4f}")
        print(
            f"\n  {'Condition':<14s}  {'Δ Val F1':>10s}  {'Δ Overfit Gap':>14s}  {'Gap Reduction %':>16s}"
        )
        print(f"  {'─' * 60}")
        for _, row in summary.iterrows():
            if row["condition"] == "baseline":
                continue
            delta_f1 = row["best_val_f1"] - bl_f1
            delta_gap = row["overfit_gap"] - bl_gap
            gap_reduction = (
                (bl_gap - row["overfit_gap"]) / bl_gap * 100
                if bl_gap != 0
                else 0
            )
            print(
                f"  {CONDITION_SHORT[row['condition']]:<14s}  "
                f"{delta_f1:+.4f}     "
                f"{delta_gap:+.4f}         "
                f"{gap_reduction:+.1f}%"
            )


def plot_overfit_gap_bar(summary: pd.DataFrame) -> str:
    """Bar chart comparing overfit gap across conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary))
    colors = [CONDITION_COLORS[c] for c in summary["condition"]]
    bars = ax.bar(
        x,
        summary["overfit_gap"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in summary["condition"]], fontsize=9
    )
    ax.set_ylabel("Overfit Gap (Train F1 − Val F1)", fontsize=10)
    ax.set_title(
        "POYO Overfitting: Regularization Ablation\n(Brain Invaders P300, Intersubject, Fold 0)",
        fontsize=11,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(0, max(summary["overfit_gap"].max() * 1.15, 0.1))
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, summary["overfit_gap"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_overfit_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


def plot_val_f1_bar(summary: pd.DataFrame) -> str:
    """Bar chart of val F1 across conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary))
    colors = [CONDITION_COLORS[c] for c in summary["condition"]]
    bars = ax.bar(
        x,
        summary["best_val_f1"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in summary["condition"]], fontsize=9
    )
    ax.set_ylabel("Best Validation F1", fontsize=10)
    ax.set_title(
        "Validation F1: Regularization Ablation\n(Brain Invaders P300, Intersubject, Fold 0)",
        fontsize=11,
    )
    ax.set_ylim(0, min(1.0, summary["best_val_f1"].max() * 1.3))
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, summary["best_val_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_val_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


def plot_train_vs_val_f1(summary: pd.DataFrame) -> str:
    """Grouped bar chart showing train F1 and val F1 side by side."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary))
    width = 0.35

    bars_train = ax.bar(
        x - width / 2,
        summary["best_train_f1"],
        width,
        label="Train F1",
        color="#C44E52",
        alpha=0.8,
        edgecolor="white",
    )
    bars_val = ax.bar(
        x + width / 2,
        summary["best_val_f1"],
        width,
        label="Val F1",
        color="#4C72B0",
        alpha=0.8,
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in summary["condition"]], fontsize=9
    )
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title(
        "Train vs Val F1: Regularization Ablation\n(Brain Invaders P300, Intersubject, Fold 0)",
        fontsize=11,
    )
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars_train, summary["best_train_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val in zip(bars_val, summary["best_val_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_train_vs_val_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


def plot_training_curves(df: pd.DataFrame) -> str:
    """Plot train/val loss and F1 curves for all conditions."""
    api = wandb.Api()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_train_loss, ax_val_loss = axes[0]
    ax_train_f1, ax_val_f1 = axes[1]

    for _, row in df.iterrows():
        cond = row["condition"]
        color = CONDITION_COLORS[cond]
        label = CONDITION_SHORT[cond]
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{row['run_id']}")

        # Fetch train loss
        hist_train = run.history(
            keys=["epoch", TRAIN_LOSS], samples=10000, pandas=True
        )
        if TRAIN_LOSS in hist_train.columns:
            hist_train = hist_train.dropna(subset=[TRAIN_LOSS])
            if "epoch" in hist_train.columns:
                epoch_train = (
                    hist_train.groupby("epoch")[TRAIN_LOSS].mean().reset_index()
                )
                ax_train_loss.plot(
                    epoch_train["epoch"],
                    epoch_train[TRAIN_LOSS],
                    color=color,
                    label=label,
                    linewidth=1.5,
                )

        # Fetch val loss
        hist_val_loss = run.history(
            keys=["epoch", VAL_LOSS], samples=10000, pandas=True
        )
        if VAL_LOSS in hist_val_loss.columns:
            hist_val_loss = hist_val_loss.dropna(subset=[VAL_LOSS])
            if "epoch" in hist_val_loss.columns:
                ax_val_loss.plot(
                    hist_val_loss["epoch"],
                    hist_val_loss[VAL_LOSS],
                    color=color,
                    label=label,
                    linewidth=1.5,
                )

        # Fetch train F1
        hist_train_f1 = run.history(
            keys=["epoch", TRAIN_F1], samples=10000, pandas=True
        )
        if TRAIN_F1 in hist_train_f1.columns:
            hist_train_f1 = hist_train_f1.dropna(subset=[TRAIN_F1])
            if "epoch" in hist_train_f1.columns:
                ax_train_f1.plot(
                    hist_train_f1["epoch"],
                    hist_train_f1[TRAIN_F1],
                    color=color,
                    label=label,
                    linewidth=1.5,
                )

        # Fetch val F1
        hist_val_f1 = run.history(
            keys=["epoch", VAL_F1], samples=10000, pandas=True
        )
        if VAL_F1 in hist_val_f1.columns:
            hist_val_f1 = hist_val_f1.dropna(subset=[VAL_F1])
            if "epoch" in hist_val_f1.columns:
                ax_val_f1.plot(
                    hist_val_f1["epoch"],
                    hist_val_f1[VAL_F1],
                    color=color,
                    label=label,
                    linewidth=1.5,
                )

    ax_train_loss.set_title("Train Loss")
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.legend(fontsize=8)
    ax_train_loss.grid(alpha=0.3)

    ax_val_loss.set_title("Val Loss")
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Loss")
    ax_val_loss.legend(fontsize=8)
    ax_val_loss.grid(alpha=0.3)

    ax_train_f1.set_title("Train F1")
    ax_train_f1.set_xlabel("Epoch")
    ax_train_f1.set_ylabel("F1")
    ax_train_f1.legend(fontsize=8)
    ax_train_f1.grid(alpha=0.3)
    ax_train_f1.set_ylim(0, 1.05)

    ax_val_f1.set_title("Val F1")
    ax_val_f1.set_xlabel("Epoch")
    ax_val_f1.set_ylabel("F1")
    ax_val_f1.legend(fontsize=8)
    ax_val_f1.grid(alpha=0.3)
    ax_val_f1.set_ylim(0, 1.05)

    fig.suptitle(
        "Training Curves: Regularization Ablation\n(Brain Invaders P300, Intersubject, Fold 0)",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


if __name__ == "__main__":
    print("Fetching runs from WandB...")
    df = fetch_all_runs()

    if df.empty:
        print("ERROR: No runs found. Check group name and entity.")
        exit(1)

    summary = compute_summary(df)
    print_summary_table(summary)

    print("\n\nGenerating figures...")
    plot_overfit_gap_bar(summary)
    plot_val_f1_bar(summary)
    plot_train_vs_val_f1(summary)
    plot_training_curves(summary)

    print("\nDone!")

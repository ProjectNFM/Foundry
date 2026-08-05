"""Brain Invaders P300 Reprocessed — 3-Fold Baselines Analysis.

Fetches all 30 runs from WandB group BI_P300_REPROC_3FOLD:
  5 conditions × 3 folds × 2 split_types (intersubject, intrasession)

Conditions:
  - EEGNet (lr=1e-3)
  - POYO CWT-CNN, channel_emb disabled (lr=1e-4)
  - POYO CWT-CNN, channel_emb dynamic (lr=1e-4)
  - POYO ResampleCNN, channel_emb disabled (lr=1e-4)
  - POYO ResampleCNN, channel_emb dynamic (lr=1e-4)

Usage:
    uv run python analysis/032_brain_invaders_reproc_3fold.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from analysis._wandb_utils import default_entity, figures_dir

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity() or "poyo-eeg"
WANDB_GROUP = "BI_P300_REPROC_3FOLD"
FIGURES_DIR = figures_dir(__file__)
PREFIX = "032_bi_reproc_3fold"

VAL_F1 = "val/p300_binary_f1"
VAL_AUROC = "val/p300_binary_auroc"
VAL_ACC = "val/p300_binary_acc"
VAL_RECALL = "val/p300_binary_recall"
VAL_PRECISION = "val/p300_binary_precision"
TRAIN_F1 = "train/p300_binary_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

CONDITIONS = ["eegnet", "cwt_dis", "cwt_dyn", "rcnn_dis", "rcnn_dyn"]
CONDITION_LABELS = {
    "eegnet": "EEGNet",
    "cwt_dis": "CWT-CNN\nDisabled",
    "cwt_dyn": "CWT-CNN\nDynamic",
    "rcnn_dis": "RCNN\nDisabled",
    "rcnn_dyn": "RCNN\nDynamic",
}
CONDITION_SHORT = {
    "eegnet": "EEGNet",
    "cwt_dis": "CWT Disabled",
    "cwt_dyn": "CWT Dynamic",
    "rcnn_dis": "RCNN Disabled",
    "rcnn_dyn": "RCNN Dynamic",
}
CONDITION_COLORS = {
    "eegnet": "#E8963E",
    "cwt_dis": "#4C72B0",
    "cwt_dyn": "#55A868",
    "rcnn_dis": "#C44E52",
    "rcnn_dyn": "#8172B2",
}
SPLIT_TYPES = ["intersubject", "intrasession"]
FOLDS = [0, 1, 2]


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


def parse_run_name(name: str) -> tuple[str, str, int] | None:
    """Parse run name -> (condition_key, split_type, fold).

    Patterns:
      bi_p300_reproc_intersubject_eegnet_fold0
      bi_p300_reproc_intrasession_per_channel_cwt_cnn_ch-disabled_fold1
      bi_p300_reproc_intersubject_per_channel_resample_cnn_ch-dynamic_fold2
    """
    split_type = None
    for st in SPLIT_TYPES:
        if st in name:
            split_type = st
            break
    if split_type is None:
        return None

    fold = None
    for f in FOLDS:
        if f"fold{f}" in name:
            fold = f
            break
    if fold is None:
        return None

    if "eegnet" in name:
        return "eegnet", split_type, fold

    if "cwt_cnn" in name:
        tok = "cwt"
    elif "resample_cnn" in name:
        tok = "rcnn"
    else:
        return None

    if "ch-disabled" in name:
        ch = "dis"
    elif "ch-dynamic" in name:
        ch = "dyn"
    else:
        return None

    return f"{tok}_{ch}", split_type, fold


def fetch_all_runs() -> pd.DataFrame:
    """Fetch all runs from the WandB group and return a DataFrame."""
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": WANDB_GROUP},
    )

    records = []
    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"  WARNING: Could not parse run name: {run.name} ({run.id})")
            continue

        cond, split_type, fold = parsed
        s = run.summary

        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "condition": cond,
                "split_type": split_type,
                "fold": fold,
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
        print(
            f"  Mapped: {run.name} -> {cond}/{split_type}/fold{fold} ({run.id})"
        )

    df = pd.DataFrame(records)
    print(f"\n  Total runs fetched: {len(df)}")

    expected = {
        (c, st, f) for c in CONDITIONS for st in SPLIT_TYPES for f in FOLDS
    }
    found = set(zip(df["condition"], df["split_type"], df["fold"]))
    missing = expected - found
    if missing:
        print(f"  WARNING: Missing {len(missing)} runs:")
        for c, st, f in sorted(missing):
            print(f"    {c} / {st} / fold{f}")

    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std across folds for each condition × split_type."""
    grouped = (
        df.groupby(["condition", "split_type"])
        .agg(
            mean_f1=("best_val_f1", "mean"),
            std_f1=("best_val_f1", "std"),
            mean_auroc=("best_val_auroc", "mean"),
            std_auroc=("best_val_auroc", "std"),
            mean_acc=("best_val_acc", "mean"),
            mean_recall=("best_val_recall", "mean"),
            mean_precision=("best_val_precision", "mean"),
            mean_train_f1=("best_train_f1", "mean"),
            n_runs=("run_id", "count"),
        )
        .reset_index()
    )
    grouped["overfit_gap"] = grouped["mean_train_f1"] - grouped["mean_f1"]
    return grouped


def print_summary_table(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print formatted summary tables."""
    for split in SPLIT_TYPES:
        sub = summary[summary["split_type"] == split].copy()
        sub["cond_order"] = sub["condition"].map(
            {c: i for i, c in enumerate(CONDITIONS)}
        )
        sub = sub.sort_values("cond_order")

        print(f"\n{'=' * 100}")
        print(f"  {split.upper()} — 3-Fold Results")
        print(f"{'=' * 100}")
        print(
            f"\n  {'Condition':<16s}  {'Val F1':>10s}  {'AUROC':>10s}  "
            f"{'Acc':>8s}  {'Recall':>8s}  {'Prec':>8s}  "
            f"{'Train F1':>9s}  {'Overfit':>8s}  {'N':>3s}"
        )
        print(f"  {'─' * 95}")

        for _, row in sub.iterrows():
            print(
                f"  {CONDITION_SHORT[row['condition']]:<16s}  "
                f"{row['mean_f1']:.4f}±{row['std_f1']:.3f}  "
                f"{row['mean_auroc']:.4f}±{row['std_auroc']:.3f}  "
                f"{row['mean_acc']:.4f}  {row['mean_recall']:.4f}  "
                f"{row['mean_precision']:.4f}  "
                f"{row['mean_train_f1']:.4f}   {row['overfit_gap']:+.3f}  "
                f"{int(row['n_runs']):>3d}"
            )

    # Per-fold detail table
    print(f"\n{'=' * 100}")
    print("  PER-FOLD DETAIL (Val F1)")
    print(f"{'=' * 100}")
    for split in SPLIT_TYPES:
        sub = df[df["split_type"] == split].copy()
        print(f"\n  ── {split} ──")
        print(
            f"  {'Condition':<16s}  {'Fold 0':>8s}  {'Fold 1':>8s}  "
            f"{'Fold 2':>8s}  {'Mean':>8s}  {'Std':>6s}"
        )
        print(f"  {'─' * 60}")
        for cond in CONDITIONS:
            csub = sub[sub["condition"] == cond].sort_values("fold")
            vals = csub["best_val_f1"].tolist()
            fold_strs = [f"{v:.4f}" for v in vals]
            while len(fold_strs) < 3:
                fold_strs.append("  N/A ")
            mean = np.mean(vals) if vals else 0
            std = np.std(vals) if vals else 0
            print(
                f"  {CONDITION_SHORT[cond]:<16s}  "
                + "  ".join(f"{s:>8s}" for s in fold_strs)
                + f"  {mean:>8.4f}  {std:>6.4f}"
            )

    # Inter vs Intra comparison
    print(f"\n{'=' * 100}")
    print("  INTER vs INTRA COMPARISON")
    print(f"{'=' * 100}")
    print(
        f"\n  {'Condition':<16s}  {'Inter F1':>10s}  {'Intra F1':>10s}  "
        f"{'Δ (pp)':>8s}  {'Ratio':>6s}"
    )
    print(f"  {'─' * 60}")
    for cond in CONDITIONS:
        inter = summary[
            (summary["condition"] == cond)
            & (summary["split_type"] == "intersubject")
        ]
        intra = summary[
            (summary["condition"] == cond)
            & (summary["split_type"] == "intrasession")
        ]
        if inter.empty or intra.empty:
            continue
        inter_f1 = inter.iloc[0]["mean_f1"]
        intra_f1 = intra.iloc[0]["mean_f1"]
        delta_pp = (intra_f1 - inter_f1) * 100
        ratio = intra_f1 / inter_f1 if inter_f1 > 0 else float("inf")
        print(
            f"  {CONDITION_SHORT[cond]:<16s}  "
            f"{inter_f1:.4f}      {intra_f1:.4f}      "
            f"{delta_pp:+.1f}    {ratio:.2f}x"
        )

    # Tokenizer and channel emb comparisons
    print(f"\n{'=' * 100}")
    print("  TOKENIZER COMPARISON (CWT-CNN vs ResampleCNN)")
    print(f"{'=' * 100}")
    for split in SPLIT_TYPES:
        sub = summary[summary["split_type"] == split]
        print(f"\n  ── {split} ──")
        print(
            f"  {'Channel Emb':<15s}  {'CWT F1':>10s}  {'RCNN F1':>10s}  {'Δ (pp)':>8s}"
        )
        print(f"  {'─' * 48}")
        for ch, ch_label in [("dis", "Disabled"), ("dyn", "Dynamic")]:
            cwt = sub[sub["condition"] == f"cwt_{ch}"]
            rcnn = sub[sub["condition"] == f"rcnn_{ch}"]
            if cwt.empty or rcnn.empty:
                continue
            cwt_f1 = cwt.iloc[0]["mean_f1"]
            rcnn_f1 = rcnn.iloc[0]["mean_f1"]
            delta = (cwt_f1 - rcnn_f1) * 100
            print(
                f"  {ch_label:<15s}  {cwt_f1:.4f}      {rcnn_f1:.4f}      {delta:+.1f}"
            )

    print(f"\n{'=' * 100}")
    print("  CHANNEL EMBEDDING EFFECT (Disabled vs Dynamic)")
    print(f"{'=' * 100}")
    for split in SPLIT_TYPES:
        sub = summary[summary["split_type"] == split]
        print(f"\n  ── {split} ──")
        print(
            f"  {'Tokenizer':<15s}  {'Disabled F1':>11s}  "
            f"{'Dynamic F1':>10s}  {'Δ (pp)':>8s}"
        )
        print(f"  {'─' * 50}")
        for tok, tok_label in [("cwt", "CWT-CNN"), ("rcnn", "ResampleCNN")]:
            dis = sub[sub["condition"] == f"{tok}_dis"]
            dyn = sub[sub["condition"] == f"{tok}_dyn"]
            if dis.empty or dyn.empty:
                continue
            dis_f1 = dis.iloc[0]["mean_f1"]
            dyn_f1 = dyn.iloc[0]["mean_f1"]
            delta = (dyn_f1 - dis_f1) * 100
            print(
                f"  {tok_label:<15s}  {dis_f1:.4f}       "
                f"{dyn_f1:.4f}      {delta:+.1f}"
            )


def plot_main_results(summary: pd.DataFrame) -> None:
    """Grouped bar chart: all conditions, inter vs intra side by side."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(CONDITIONS))
    width = 0.35

    inter = summary[summary["split_type"] == "intersubject"].set_index(
        "condition"
    )
    intra = summary[summary["split_type"] == "intrasession"].set_index(
        "condition"
    )

    inter_vals = [
        inter.loc[c, "mean_f1"] if c in inter.index else 0 for c in CONDITIONS
    ]
    inter_errs = [
        inter.loc[c, "std_f1"] if c in inter.index else 0 for c in CONDITIONS
    ]
    intra_vals = [
        intra.loc[c, "mean_f1"] if c in intra.index else 0 for c in CONDITIONS
    ]
    intra_errs = [
        intra.loc[c, "std_f1"] if c in intra.index else 0 for c in CONDITIONS
    ]

    bars1 = ax.bar(
        x - width / 2,
        inter_vals,
        width,
        yerr=inter_errs,
        label="Intersubject",
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )
    bars2 = ax.bar(
        x + width / 2,
        intra_vals,
        width,
        yerr=intra_errs,
        label="Intrasession",
        color="#55A868",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=12)
    ax.set_title(
        "Brain Invaders P300 (Reprocessed): Intersubject vs Intrasession",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    all_vals = inter_vals + intra_vals
    nonzero = [v for v in all_vals if v > 0]
    if nonzero:
        ax.set_ylim(0, max(nonzero) + 0.08)

    plt.tight_layout()
    out = FIGURES_DIR / f"{PREFIX}_main_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_inter_vs_intra_delta(summary: pd.DataFrame) -> None:
    """Bar chart showing the intrasession advantage (delta pp) per condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    deltas = []
    labels = []
    colors = []
    for cond in CONDITIONS:
        inter = summary[
            (summary["condition"] == cond)
            & (summary["split_type"] == "intersubject")
        ]
        intra = summary[
            (summary["condition"] == cond)
            & (summary["split_type"] == "intrasession")
        ]
        if inter.empty or intra.empty:
            continue
        delta = (intra.iloc[0]["mean_f1"] - inter.iloc[0]["mean_f1"]) * 100
        deltas.append(delta)
        labels.append(CONDITION_SHORT[cond])
        colors.append(CONDITION_COLORS[cond])

    x = np.arange(len(labels))
    bars = ax.bar(
        x, deltas, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )

    for bar, val in zip(bars, deltas):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.3 if y >= 0 else -1.2),
            f"{val:+.1f}pp",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Intrasession − Intersubject (pp F1)", fontsize=12)
    ax.set_title(
        "Intrasession Advantage per Condition",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / f"{PREFIX}_intra_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_fold_variance(df: pd.DataFrame) -> None:
    """Strip plot: individual fold results per condition, paneled by split type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, split in zip(axes, SPLIT_TYPES):
        sub = df[df["split_type"] == split]
        for i, cond in enumerate(CONDITIONS):
            vals = sub[sub["condition"] == cond]["best_val_f1"].values
            if len(vals) == 0:
                continue
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                [i + j for j in jitter],
                vals,
                color=CONDITION_COLORS[cond],
                s=100,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            mean = np.mean(vals)
            ax.hlines(
                mean,
                i - 0.2,
                i + 0.2,
                color=CONDITION_COLORS[cond],
                linewidth=3,
                zorder=4,
            )

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(
            [CONDITION_SHORT[c] for c in CONDITIONS],
            fontsize=9,
            rotation=15,
        )
        ax.set_title(split.capitalize(), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Macro F1", fontsize=12)
    fig.suptitle(
        "Brain Invaders P300 (Reprocessed): Cross-Fold Variance",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"{PREFIX}_fold_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_overfitting(summary: pd.DataFrame) -> None:
    """Train F1 vs Val F1 for each condition and split type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, split in zip(axes, SPLIT_TYPES):
        sub = summary[summary["split_type"] == split]
        x = np.arange(len(CONDITIONS))
        width = 0.35

        train_vals = [
            sub[sub["condition"] == c]["mean_train_f1"].values[0]
            if c in sub["condition"].values
            else 0
            for c in CONDITIONS
        ]
        val_vals = [
            sub[sub["condition"] == c]["mean_f1"].values[0]
            if c in sub["condition"].values
            else 0
            for c in CONDITIONS
        ]

        ax.bar(
            x - width / 2,
            train_vals,
            width,
            label="Train F1",
            color="#2196F3",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            val_vals,
            width,
            label="Val F1",
            color="#FF9800",
            alpha=0.8,
        )

        for i, (t, v) in enumerate(zip(train_vals, val_vals)):
            if t > 0:
                ax.text(
                    i - width / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8
                )
            if v > 0:
                ax.text(
                    i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_SHORT[c] for c in CONDITIONS],
            fontsize=9,
            rotation=15,
        )
        ax.set_title(split.capitalize(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.1)

    axes[0].set_ylabel("F1 Score", fontsize=12)
    fig.suptitle(
        "Brain Invaders P300 (Reprocessed): Train vs Val F1 (Overfitting)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"{PREFIX}_overfitting.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 70)
    print("  Brain Invaders P300 Reprocessed — 3-Fold Baselines Analysis")
    print("=" * 70)

    print("\n── Fetching runs from WandB group ──")
    df = fetch_all_runs()

    if df.empty:
        print("  ERROR: No runs found. Check group name and entity.")
        return

    summary = compute_summary(df)

    print_summary_table(df, summary)

    print(f"\n{'=' * 70}")
    print("  Generating plots...")
    print(f"{'=' * 70}")
    plot_main_results(summary)
    plot_inter_vs_intra_delta(summary)
    plot_fold_variance(df)
    plot_overfitting(summary)

    print(f"\n{'=' * 70}")
    print(f"  Done. All figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

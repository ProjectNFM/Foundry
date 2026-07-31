"""KempSleep finetuning: 2×2×2 comparison (exp 022).

Compares 8 conditions: {CWT-CNN, ResampleCNN} × {disabled, dynamic} × {scratch, finetuned}
on 5-class KempSleep sleep staging with full parameter tuning.

WandB project: foundry_finetuning
Group: KEMP_FINETUNE_CWT_DYNCH

Runs:
  scratch-cwt-ch-disabled:          g3mfdwj6
  scratch-cwt-ch-dynamic:           pew03xnz
  scratch-rcnn-ch-disabled:         x130d6jj
  scratch-rcnn-ch-dynamic:          lhutmecj
  finetuned-cwt-ch-disabled:        g52jwdde
  finetuned-cwt-ch-dynamic:         n755mbdx
  finetuned-rcnn-ch-disabled:       lwqqqnup
  finetuned-rcnn-ch-dynamic:        m7n84fve

Usage:
    uv run python analysis/022_kemp_baselines_finetune_cwt_dynch.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
    fetch_run_summary,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()
FIGURES_DIR = figures_dir(__file__)

RUNS = {
    "scratch-cwt-disabled": "g3mfdwj6",
    "scratch-cwt-dynamic": "pew03xnz",
    "scratch-rcnn-disabled": "x130d6jj",
    "scratch-rcnn-dynamic": "lhutmecj",
    "finetuned-cwt-disabled": "g52jwdde",
    "finetuned-cwt-dynamic": "n755mbdx",
    "finetuned-rcnn-disabled": "lwqqqnup",
    "finetuned-rcnn-dynamic": "m7n84fve",
}

CONDITION_LABELS = {
    "scratch-cwt-disabled": "Scratch\nCWT\nDisabled",
    "scratch-cwt-dynamic": "Scratch\nCWT\nDynamic",
    "scratch-rcnn-disabled": "Scratch\nRCNN\nDisabled",
    "scratch-rcnn-dynamic": "Scratch\nRCNN\nDynamic",
    "finetuned-cwt-disabled": "Finetuned\nCWT\nDisabled",
    "finetuned-cwt-dynamic": "Finetuned\nCWT\nDynamic",
    "finetuned-rcnn-disabled": "Finetuned\nRCNN\nDisabled",
    "finetuned-rcnn-dynamic": "Finetuned\nRCNN\nDynamic",
}

CONDITION_COLORS = {
    "scratch-cwt-disabled": "#B8B8D0",
    "scratch-cwt-dynamic": "#95C8A0",
    "scratch-rcnn-disabled": "#D8A0A0",
    "scratch-rcnn-dynamic": "#A0B8D0",
    "finetuned-cwt-disabled": "#8172B2",
    "finetuned-cwt-dynamic": "#55A868",
    "finetuned-rcnn-disabled": "#C44E52",
    "finetuned-rcnn-dynamic": "#4C72B0",
}

VAL_F1 = "val/sleep_stage_5class_f1"
VAL_ACC = "val/sleep_stage_5class_acc"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"


def fetch_all_data() -> dict[str, dict]:
    results = {}
    for cond, run_id in RUNS.items():
        print(f"  Fetching {cond} ({run_id})...")
        epoch_df = fetch_metric_history(
            run_id,
            [TRAIN_LOSS, VAL_LOSS, VAL_F1, VAL_ACC],
            WANDB_PROJECT,
            WANDB_ENTITY,
            x_axis="epoch",
            aggregate_epoch=True,
        )
        summary = fetch_run_summary(
            run_id,
            WANDB_PROJECT,
            {
                "best_val_f1": (VAL_F1, "max"),
                "best_val_acc": (VAL_ACC, "max"),
                "best_val_loss": (VAL_LOSS, "min"),
                "max_epoch": ("epoch", "max"),
            },
            WANDB_ENTITY,
        )

        best_val_f1 = None
        best_f1_epoch = None
        if not epoch_df.empty and VAL_F1 in epoch_df.columns:
            valid = epoch_df.dropna(subset=[VAL_F1])
            if not valid.empty:
                best_idx = valid[VAL_F1].idxmax()
                best_row = valid.loc[best_idx]
                best_val_f1 = float(best_row[VAL_F1])
                best_f1_epoch = int(best_row["epoch"])

        results[cond] = {
            "run_id": run_id,
            "state": summary["state"],
            "best_val_f1": best_val_f1 or summary["best_val_f1"],
            "best_val_acc": summary["best_val_acc"],
            "best_val_loss": summary["best_val_loss"],
            "best_f1_epoch": best_f1_epoch,
            "max_epoch": summary["max_epoch"],
            "epoch_df": epoch_df,
        }
    return results


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 115}")
    print("  KempSleep Finetuning: Tokenizer × Channel Emb × Init (exp 022)")
    print(f"{'=' * 115}")

    header = (
        f"{'Condition':<30s}  {'Val F1':>8s}  {'Val Acc':>8s}  "
        f"{'Val Loss':>9s}  {'BF1 Ep':>6s}  {'Max Ep':>6s}  "
        f"{'State':<10s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    sorted_conds = sorted(
        RUNS.keys(), key=lambda c: data[c]["best_val_f1"] or 0, reverse=True
    )
    for cond in sorted_conds:
        d = data[cond]
        f1_s = f"{d['best_val_f1']:.4f}" if d["best_val_f1"] else "?"
        acc_s = f"{d['best_val_acc']:.4f}" if d["best_val_acc"] else "?"
        loss_s = f"{d['best_val_loss']:.4f}" if d["best_val_loss"] else "?"
        ep_s = (
            str(d["best_f1_epoch"]) if d["best_f1_epoch"] is not None else "?"
        )
        print(
            f"{cond:<30s}  {f1_s:>8s}  {acc_s:>8s}  "
            f"{loss_s:>9s}  {ep_s:>6s}  {d['max_epoch']:>6}  "
            f"{d['state']:<10s}  {d['run_id']}"
        )

    print("\n  Pairwise comparisons (F1):")
    comparisons = [
        (
            "Dynamic vs Disabled (finetuned CWT)",
            "finetuned-cwt-dynamic",
            "finetuned-cwt-disabled",
        ),
        (
            "Dynamic vs Disabled (finetuned RCNN)",
            "finetuned-rcnn-dynamic",
            "finetuned-rcnn-disabled",
        ),
        (
            "Dynamic vs Disabled (scratch CWT)",
            "scratch-cwt-dynamic",
            "scratch-cwt-disabled",
        ),
        (
            "Dynamic vs Disabled (scratch RCNN)",
            "scratch-rcnn-dynamic",
            "scratch-rcnn-disabled",
        ),
        (
            "Finetuned vs Scratch (CWT disabled)",
            "finetuned-cwt-disabled",
            "scratch-cwt-disabled",
        ),
        (
            "Finetuned vs Scratch (CWT dynamic)",
            "finetuned-cwt-dynamic",
            "scratch-cwt-dynamic",
        ),
        (
            "Finetuned vs Scratch (RCNN disabled)",
            "finetuned-rcnn-disabled",
            "scratch-rcnn-disabled",
        ),
        (
            "Finetuned vs Scratch (RCNN dynamic)",
            "finetuned-rcnn-dynamic",
            "scratch-rcnn-dynamic",
        ),
        (
            "CWT vs RCNN (finetuned disabled)",
            "finetuned-cwt-disabled",
            "finetuned-rcnn-disabled",
        ),
        (
            "CWT vs RCNN (finetuned dynamic)",
            "finetuned-cwt-dynamic",
            "finetuned-rcnn-dynamic",
        ),
        (
            "CWT vs RCNN (scratch disabled)",
            "scratch-cwt-disabled",
            "scratch-rcnn-disabled",
        ),
        (
            "CWT vs RCNN (scratch dynamic)",
            "scratch-cwt-dynamic",
            "scratch-rcnn-dynamic",
        ),
    ]
    for desc, a, b in comparisons:
        f1_a = data[a]["best_val_f1"]
        f1_b = data[b]["best_val_f1"]
        if f1_a and f1_b:
            diff = (f1_a - f1_b) * 100
            print(f"    {desc}: {diff:+.1f} pp F1")


def plot_bar_comparison(data: dict) -> None:
    """Bar chart of best F1 for all 8 conditions, sorted by F1."""
    sorted_conds = sorted(
        RUNS.keys(), key=lambda c: data[c]["best_val_f1"] or 0, reverse=True
    )
    f1_vals = [data[c]["best_val_f1"] for c in sorted_conds]
    acc_vals = [data[c]["best_val_acc"] for c in sorted_conds]
    colors = [CONDITION_COLORS[c] for c in sorted_conds]
    x_labels = [CONDITION_LABELS[c] for c in sorted_conds]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    ax = axes[0]
    bars = ax.bar(
        range(len(sorted_conds)),
        f1_vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.65,
    )
    ax.set_xticks(range(len(sorted_conds)))
    ax.set_xticklabels(x_labels, fontsize=8)
    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Best Validation F1 (5-class)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(f1_vals) * 1.12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(
        range(len(sorted_conds)),
        acc_vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.65,
    )
    ax.set_xticks(range(len(sorted_conds)))
    ax.set_xticklabels(x_labels, fontsize=8)
    for bar, val in zip(bars, acc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Best Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(acc_vals) * 1.12)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "KempSleep Finetuning: All 8 Conditions (exp 022)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "022_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_grouped_bar(data: dict) -> None:
    """Grouped bar: init × channel_emb, grouped by tokenizer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (tok, tok_label) in enumerate(
        [("cwt", "CWT-CNN"), ("rcnn", "ResampleCNN")]
    ):
        ax = axes[ax_i]
        groups = [
            ("Scratch\nDisabled", f"scratch-{tok}-disabled"),
            ("Scratch\nDynamic", f"scratch-{tok}-dynamic"),
            ("Finetuned\nDisabled", f"finetuned-{tok}-disabled"),
            ("Finetuned\nDynamic", f"finetuned-{tok}-dynamic"),
        ]

        x = np.arange(len(groups))
        f1_vals = [data[g[1]]["best_val_f1"] for g in groups]
        colors = [CONDITION_COLORS[g[1]] for g in groups]

        bars = ax.bar(
            x,
            f1_vals,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            width=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        for bar, val in zip(bars, f1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylabel("Macro F1", fontsize=11)
        ax.set_title(tok_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0.5, max(f1_vals) * 1.08)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "KempSleep F1 by Tokenizer: Init × Channel Emb (exp 022)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "022_grouped_by_tokenizer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_interaction(data: dict) -> None:
    """Interaction plots: tokenizer × channel_emb for each init mode."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Plot 1: Init effect (scratch vs finetuned) for each tokenizer × channel_emb
    ax = axes[0]
    conditions_pairs = [
        (
            "CWT-Disabled",
            "scratch-cwt-disabled",
            "finetuned-cwt-disabled",
            "#8172B2",
        ),
        (
            "CWT-Dynamic",
            "scratch-cwt-dynamic",
            "finetuned-cwt-dynamic",
            "#55A868",
        ),
        (
            "RCNN-Disabled",
            "scratch-rcnn-disabled",
            "finetuned-rcnn-disabled",
            "#C44E52",
        ),
        (
            "RCNN-Dynamic",
            "scratch-rcnn-dynamic",
            "finetuned-rcnn-dynamic",
            "#4C72B0",
        ),
    ]
    for label, scratch_key, ft_key, color in conditions_pairs:
        vals = [data[scratch_key]["best_val_f1"], data[ft_key]["best_val_f1"]]
        ax.plot(
            ["Scratch", "Finetuned"],
            vals,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=8,
            label=label,
        )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Init Effect", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Channel emb effect (disabled vs dynamic) for each tokenizer × init
    ax = axes[1]
    conditions_pairs = [
        (
            "CWT-Scratch",
            "scratch-cwt-disabled",
            "scratch-cwt-dynamic",
            "#95C8A0",
        ),
        (
            "CWT-Finetuned",
            "finetuned-cwt-disabled",
            "finetuned-cwt-dynamic",
            "#55A868",
        ),
        (
            "RCNN-Scratch",
            "scratch-rcnn-disabled",
            "scratch-rcnn-dynamic",
            "#A0B8D0",
        ),
        (
            "RCNN-Finetuned",
            "finetuned-rcnn-disabled",
            "finetuned-rcnn-dynamic",
            "#4C72B0",
        ),
    ]
    for label, dis_key, dyn_key, color in conditions_pairs:
        vals = [data[dis_key]["best_val_f1"], data[dyn_key]["best_val_f1"]]
        ax.plot(
            ["Disabled", "Dynamic"],
            vals,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=8,
            label=label,
        )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Channel Emb Effect", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Tokenizer effect (CWT vs RCNN) for each channel_emb × init
    ax = axes[2]
    conditions_pairs = [
        (
            "Scratch-Disabled",
            "scratch-cwt-disabled",
            "scratch-rcnn-disabled",
            "#B8B8D0",
        ),
        (
            "Scratch-Dynamic",
            "scratch-cwt-dynamic",
            "scratch-rcnn-dynamic",
            "#95C8A0",
        ),
        (
            "Finetuned-Disabled",
            "finetuned-cwt-disabled",
            "finetuned-rcnn-disabled",
            "#8172B2",
        ),
        (
            "Finetuned-Dynamic",
            "finetuned-cwt-dynamic",
            "finetuned-rcnn-dynamic",
            "#55A868",
        ),
    ]
    for label, cwt_key, rcnn_key, color in conditions_pairs:
        vals = [data[cwt_key]["best_val_f1"], data[rcnn_key]["best_val_f1"]]
        ax.plot(
            ["CWT-CNN", "ResampleCNN"],
            vals,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=8,
            label=label,
        )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Tokenizer Effect", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Interaction Plots: Tokenizer × Channel Emb × Init (exp 022)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "022_interaction_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_f1_curves(data: dict) -> None:
    """Validation F1 learning curves, split by tokenizer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (tok, tok_label) in enumerate(
        [("cwt", "CWT-CNN"), ("rcnn", "ResampleCNN")]
    ):
        ax = axes[ax_i]
        for cond in RUNS:
            if tok not in cond:
                continue
            d = data[cond]
            edf = d["epoch_df"]
            if VAL_F1 not in edf.columns:
                continue
            valid = edf.dropna(subset=[VAL_F1]).sort_values("epoch")
            short = cond.replace(f"-{tok}-", " ")
            label = f"{short} ({d['best_val_f1']:.3f})"
            ax.plot(
                valid["epoch"],
                valid[VAL_F1],
                linewidth=2,
                marker="o",
                markersize=3,
                label=label,
                color=CONDITION_COLORS[cond],
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation F1 (macro)", fontsize=11)
        ax.set_title(tok_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Validation F1 Learning Curves (exp 022)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "022_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_loss_curves(data: dict) -> None:
    """Train/val loss curves in a 2x4 grid."""
    sorted_conds = list(RUNS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axes_flat = axes.flatten()

    for i, cond in enumerate(sorted_conds):
        ax = axes_flat[i]
        d = data[cond]
        edf = d["epoch_df"]
        color = CONDITION_COLORS[cond]

        if TRAIN_LOSS in edf.columns:
            valid_train = edf.dropna(subset=[TRAIN_LOSS]).sort_values("epoch")
            ax.plot(
                valid_train["epoch"],
                valid_train[TRAIN_LOSS],
                color=color,
                linewidth=1.5,
                label="Train",
            )
        if VAL_LOSS in edf.columns:
            valid_val = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
            ax.plot(
                valid_val["epoch"],
                valid_val[VAL_LOSS],
                color=color,
                linewidth=1.5,
                linestyle="--",
                label="Val",
            )

        ax.set_title(CONDITION_LABELS[cond], fontsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Learning Curves — KempSleep Finetuning (exp 022)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "022_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_effect_sizes(data: dict) -> None:
    """Summary bar chart of the three main effects averaged across conditions."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Dynamic effect (averaged across tokenizer × init)
    dynamic_deltas = []
    for tok in ["cwt", "rcnn"]:
        for init in ["scratch", "finetuned"]:
            dyn = data[f"{init}-{tok}-dynamic"]["best_val_f1"]
            dis = data[f"{init}-{tok}-disabled"]["best_val_f1"]
            dynamic_deltas.append((dyn - dis) * 100)

    # Pretraining effect (averaged across tokenizer × channel_emb)
    pretrain_deltas = []
    for tok in ["cwt", "rcnn"]:
        for ch in ["disabled", "dynamic"]:
            ft = data[f"finetuned-{tok}-{ch}"]["best_val_f1"]
            sc = data[f"scratch-{tok}-{ch}"]["best_val_f1"]
            pretrain_deltas.append((ft - sc) * 100)

    # Tokenizer effect (averaged across init × channel_emb)
    tokenizer_deltas = []
    for init in ["scratch", "finetuned"]:
        for ch in ["disabled", "dynamic"]:
            cwt = data[f"{init}-cwt-{ch}"]["best_val_f1"]
            rcnn = data[f"{init}-rcnn-{ch}"]["best_val_f1"]
            tokenizer_deltas.append((cwt - rcnn) * 100)

    effects = [
        (
            "Dynamic\nvs Disabled",
            np.mean(dynamic_deltas),
            np.std(dynamic_deltas),
            "#55A868",
        ),
        (
            "Finetuned\nvs Scratch",
            np.mean(pretrain_deltas),
            np.std(pretrain_deltas),
            "#4C72B0",
        ),
        (
            "CWT-CNN\nvs RCNN",
            np.mean(tokenizer_deltas),
            np.std(tokenizer_deltas),
            "#8172B2",
        ),
    ]

    x = np.arange(len(effects))
    means = [e[1] for e in effects]
    stds = [e[2] for e in effects]
    colors = [e[3] for e in effects]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
        capsize=5,
        error_kw={"linewidth": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in effects], fontsize=11)
    for bar, mean, std in zip(bars, means, stds):
        y_pos = bar.get_height() + std + 0.1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{mean:+.1f}±{std:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("F1 Improvement (pp)", fontsize=11)
    ax.set_title(
        "Average Effect Sizes Across Conditions (exp 022)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = FIGURES_DIR / "022_effect_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 70)
    print("  Experiment 022: KempSleep Finetuning — Full 2×2×2 Analysis")
    print("=" * 70)

    print("\n── Fetching data ──")
    data = fetch_all_data()
    print_summary(data)

    print("\n── Generating plots ──")
    plot_bar_comparison(data)
    plot_grouped_bar(data)
    plot_interaction(data)
    plot_f1_curves(data)
    plot_loss_curves(data)
    plot_effect_sizes(data)

    print("\nDone.")


if __name__ == "__main__":
    main()

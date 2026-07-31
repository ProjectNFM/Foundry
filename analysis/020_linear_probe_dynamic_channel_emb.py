"""Linear probe comparison for dynamic channel embeddings (exp 020).

Compares 4 conditions: {pretrained, random} × {ch-disabled, ch-dynamic}
on 5-class sleep staging with frozen backbone + linear head.

WandB project: foundry_finetuning
Group: KEMP_LINEAR_PROBE_DYNCH

Runs:
  pretrained-ch-disabled:  zmg07ep4
  pretrained-ch-dynamic:   osqqcdrj
  random-ch-disabled:      t54gr0yj
  random-ch-dynamic:       ip8xktxl

Usage:
    uv run python analysis/020_linear_probe_dynamic_channel_emb.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
    fetch_run_summary,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()

RUNS = {
    "pretrained-ch-disabled": "zmg07ep4",
    "pretrained-ch-dynamic": "osqqcdrj",
    "random-ch-disabled": "t54gr0yj",
    "random-ch-dynamic": "ip8xktxl",
}

CONDITION_LABELS = {
    "pretrained-ch-disabled": "Pretrained\nCh=Disabled",
    "pretrained-ch-dynamic": "Pretrained\nCh=Dynamic",
    "random-ch-disabled": "Random\nCh=Disabled",
    "random-ch-dynamic": "Random\nCh=Dynamic",
}

CONDITION_COLORS = {
    "pretrained-ch-disabled": "#8172B2",
    "pretrained-ch-dynamic": "#55A868",
    "random-ch-disabled": "#C44E52",
    "random-ch-dynamic": "#4C72B0",
}

VAL_F1 = "val/sleep_stage_5class_f1"
VAL_ACC = "val/sleep_stage_5class_acc"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"
FIGURES_DIR = figures_dir(__file__)


def fetch_all_data() -> dict[str, dict]:
    """Fetch per-epoch metrics and summary for every run."""
    results = {}
    for cond, run_id in RUNS.items():
        print(f"Fetching {cond} ({run_id})...")

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
    print(f"\n{'=' * 100}")
    print("  Linear Probe: Dynamic Channel Embeddings (exp 020)")
    print(f"{'=' * 100}")

    header = (
        f"{'Condition':<25s}  {'Val F1':>8s}  {'Val Acc':>8s}  "
        f"{'Val Loss':>9s}  {'BF1 Ep':>6s}  {'Max Ep':>6s}  "
        f"{'State':<10s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    for cond in RUNS:
        d = data[cond]
        f1_s = f"{d['best_val_f1']:.4f}" if d["best_val_f1"] else "?"
        acc_s = f"{d['best_val_acc']:.4f}" if d["best_val_acc"] else "?"
        loss_s = f"{d['best_val_loss']:.4f}" if d["best_val_loss"] else "?"
        ep_s = (
            str(d["best_f1_epoch"]) if d["best_f1_epoch"] is not None else "?"
        )
        print(
            f"{cond:<25s}  {f1_s:>8s}  {acc_s:>8s}  "
            f"{loss_s:>9s}  {ep_s:>6s}  {d['max_epoch']:>6}  "
            f"{d['state']:<10s}  {d['run_id']}"
        )

    print()
    print("  Pairwise comparisons (F1):")
    pairs = [
        (
            "pretrained-ch-dynamic",
            "pretrained-ch-disabled",
            "Dynamic vs Disabled (pretrained)",
        ),
        (
            "random-ch-dynamic",
            "random-ch-disabled",
            "Dynamic vs Disabled (random)",
        ),
        (
            "pretrained-ch-dynamic",
            "random-ch-dynamic",
            "Pretrained vs Random (dynamic)",
        ),
        (
            "pretrained-ch-disabled",
            "random-ch-disabled",
            "Pretrained vs Random (disabled)",
        ),
    ]
    for a, b, desc in pairs:
        f1_a = data[a]["best_val_f1"]
        f1_b = data[b]["best_val_f1"]
        if f1_a and f1_b:
            diff = (f1_a - f1_b) * 100
            print(f"    {desc}: {diff:+.1f} pp F1")
    print()


def plot_bar_comparison(data: dict) -> None:
    """Grouped bar chart: F1 and accuracy for all 4 conditions."""
    conds = list(RUNS.keys())
    f1_vals = [data[c]["best_val_f1"] for c in conds]
    acc_vals = [data[c]["best_val_acc"] for c in conds]
    colors = [CONDITION_COLORS[c] for c in conds]
    x_labels = [CONDITION_LABELS[c] for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    bars = ax.bar(
        range(len(conds)),
        f1_vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=9)
    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Macro F1")
    ax.set_title("Best Validation F1 (5-class)")
    ax.set_ylim(0, max(f1_vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(
        range(len(conds)),
        acc_vals,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(x_labels, fontsize=9)
    for bar, val in zip(bars, acc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Validation Accuracy")
    ax.set_ylim(0, max(acc_vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Linear Probe: Dynamic Channel Embeddings (exp 020)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "020_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_f1_curves(data: dict) -> None:
    """Overlay val F1 curves for all 4 conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in RUNS:
        d = data[cond]
        edf = d["epoch_df"]
        if VAL_F1 not in edf.columns:
            continue
        valid = edf.dropna(subset=[VAL_F1]).sort_values("epoch")
        label = f"{cond} (best={d['best_val_f1']:.3f})"
        ax.plot(
            valid["epoch"],
            valid[VAL_F1],
            color=CONDITION_COLORS[cond],
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=label,
        )

        if d["best_f1_epoch"] is not None:
            best_row = valid[valid["epoch"] == d["best_f1_epoch"]]
            if not best_row.empty:
                ax.plot(
                    d["best_f1_epoch"],
                    best_row[VAL_F1].values[0],
                    marker="*",
                    markersize=14,
                    color=CONDITION_COLORS[cond],
                    zorder=5,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation F1 (macro)", fontsize=12)
    ax.set_title(
        "Validation F1 — Linear Probe (exp 020)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIGURES_DIR / "020_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_loss_curves(data: dict) -> None:
    """Train/val loss curves, one subplot per condition."""
    conds = list(RUNS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

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

        ax.set_title(f"{CONDITION_LABELS[cond]}", fontsize=10)
        ax.set_xlabel("Epoch")
        if i % 2 == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Learning Curves — Linear Probe (exp 020)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "020_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def plot_delta_bar(data: dict) -> None:
    """Bar chart showing the F1 advantage of dynamic over disabled."""
    fig, ax = plt.subplots(figsize=(8, 5))

    comparisons = [
        ("Pretrained", "pretrained-ch-dynamic", "pretrained-ch-disabled"),
        ("Random", "random-ch-dynamic", "random-ch-disabled"),
    ]
    deltas = []
    labels = []
    colors = ["#55A868", "#4C72B0"]

    for label, dyn, dis in comparisons:
        delta = (data[dyn]["best_val_f1"] - data[dis]["best_val_f1"]) * 100
        deltas.append(delta)
        labels.append(label)

    bars = ax.bar(
        range(len(deltas)),
        deltas,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, fontsize=12)
    for bar, val in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"+{val:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_ylabel("F1 Improvement (percentage points)")
    ax.set_title(
        "Dynamic Channel Embedding Advantage\n(F1 improvement over ch-disabled baseline)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim(0, max(deltas) * 1.4)

    plt.tight_layout()
    out = FIGURES_DIR / "020_dynamic_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


def main():
    print("Fetching runs for linear probe comparison (exp 020)...")
    data = fetch_all_data()
    print_summary(data)
    plot_bar_comparison(data)
    plot_f1_curves(data)
    plot_loss_curves(data)
    plot_delta_bar(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

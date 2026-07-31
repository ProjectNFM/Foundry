"""CWT-CNN dynamic channel embedding pretraining + linear probe analysis (exp 021).

Covers all three phases of experiment 021:
  Phase 1: Pretraining loss comparison (CWT-CNN: disabled vs dynamic)
  Phase 2: Linear probing comparison (4 conditions: {pretrained, random} × {disabled, dynamic})
  Phase 3: Embedding visualization (handled by 021_cwt_dynamic_channel_emb_viz.py)

Also includes cross-tokenizer comparison with experiment 018 (ResampleCNN)
and experiment 020 (ResampleCNN linear probe).

WandB projects:
  foundry_pretraining — group PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB (exp 021)
  foundry_pretraining — group PRETRAIN_DYNAMIC_CHANNEL_EMB (exp 018, reference)
  foundry_finetuning  — group KEMP_LINEAR_PROBE_DYNCH (exp 020 + 021 linear probe)

Runs:
  Pretraining (exp 021, CWT-CNN):
    ch-disabled:  v6yoko4h
    ch-dynamic:   i069k3tx

  Pretraining (exp 018, ResampleCNN, reference):
    ch-disabled:  zmxyua36
    ch-dynamic:   hggeonah

  Linear probe (exp 021, CWT-CNN):
    pretrained-ch-disabled:  dzkfguc3
    pretrained-ch-dynamic:   l3eafwx5
    random-ch-disabled:      3pnhsc9j
    random-ch-dynamic:       fpso1m3b

  Linear probe (exp 020, ResampleCNN, reference):
    pretrained-ch-disabled:  zmg07ep4
    pretrained-ch-dynamic:   osqqcdrj
    random-ch-disabled:      t54gr0yj
    random-ch-dynamic:       ip8xktxl

Usage:
    uv run python analysis/021_cwt_dynamic_channel_emb.py
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

WANDB_ENTITY = default_entity()
FIGURES_DIR = figures_dir(__file__)

# ── Pretraining runs ──────────────────────────────────────────────────────

PRETRAIN_PROJECT = "foundry_pretraining"

PRETRAIN_021 = {
    "CWT-disabled": "v6yoko4h",
    "CWT-dynamic": "i069k3tx",
}

PRETRAIN_018 = {
    "RCNN-disabled": "zmxyua36",
    "RCNN-dynamic": "hggeonah",
}

PRETRAIN_COLORS = {
    "CWT-disabled": "#8172B2",
    "CWT-dynamic": "#55A868",
    "RCNN-disabled": "#C44E52",
    "RCNN-dynamic": "#4C72B0",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

# ── Linear probe runs ────────────────────────────────────────────────────

LP_PROJECT = "foundry_finetuning"

LP_021 = {
    "CWT-pretrained-disabled": "dzkfguc3",
    "CWT-pretrained-dynamic": "l3eafwx5",
    "CWT-random-disabled": "3pnhsc9j",
    "CWT-random-dynamic": "fpso1m3b",
}

LP_020 = {
    "RCNN-pretrained-disabled": "zmg07ep4",
    "RCNN-pretrained-dynamic": "osqqcdrj",
    "RCNN-random-disabled": "t54gr0yj",
    "RCNN-random-dynamic": "ip8xktxl",
}

VAL_F1 = "val/sleep_stage_5class_f1"
VAL_ACC = "val/sleep_stage_5class_acc"

LP_COLORS = {
    "CWT-pretrained-disabled": "#8172B2",
    "CWT-pretrained-dynamic": "#55A868",
    "CWT-random-disabled": "#DD8452",
    "CWT-random-dynamic": "#4C72B0",
    "RCNN-pretrained-disabled": "#C44E52",
    "RCNN-pretrained-dynamic": "#937860",
    "RCNN-random-disabled": "#DA8BC3",
    "RCNN-random-dynamic": "#8C8C8C",
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Pretraining
# ═══════════════════════════════════════════════════════════════════════════


def fetch_pretrain_data() -> dict[str, dict]:
    all_runs = {**PRETRAIN_021, **PRETRAIN_018}
    results = {}
    for cond, run_id in all_runs.items():
        print(f"  Fetching pretraining {cond} ({run_id})...")
        epoch_df = fetch_metric_history(
            run_id,
            [TRAIN_LOSS, VAL_LOSS],
            PRETRAIN_PROJECT,
            WANDB_ENTITY,
            x_axis="epoch",
            aggregate_epoch=True,
        )
        summary = fetch_run_summary(
            run_id,
            PRETRAIN_PROJECT,
            {
                "best_val_loss": (VAL_LOSS, "min"),
                "max_epoch": ("epoch", "max"),
            },
            WANDB_ENTITY,
        )

        best_val_epoch = None
        best_val_loss = None
        train_at_best = None
        if not epoch_df.empty and VAL_LOSS in epoch_df.columns:
            valid = epoch_df.dropna(subset=[VAL_LOSS])
            if not valid.empty:
                best_idx = valid[VAL_LOSS].idxmin()
                best_row = valid.loc[best_idx]
                best_val_epoch = int(best_row["epoch"])
                best_val_loss = float(best_row[VAL_LOSS])
                if TRAIN_LOSS in best_row and not np.isnan(
                    best_row[TRAIN_LOSS]
                ):
                    train_at_best = float(best_row[TRAIN_LOSS])

        results[cond] = {
            "run_id": run_id,
            "state": summary["state"],
            "best_val_loss": best_val_loss or summary["best_val_loss"],
            "train_at_best": train_at_best,
            "gap_at_best": (
                (best_val_loss - train_at_best)
                if best_val_loss and train_at_best
                else None
            ),
            "best_val_epoch": best_val_epoch,
            "max_epoch": summary["max_epoch"],
            "epoch_df": epoch_df,
        }
    return results


def print_pretrain_summary(data: dict) -> None:
    print(f"\n{'=' * 110}")
    print("  Phase 1: Pretraining — CWT-CNN (exp 021) vs ResampleCNN (exp 018)")
    print(f"{'=' * 110}")

    header = (
        f"{'Condition':<22s}  {'Best Val':>10s}  {'Train@BV':>10s}  "
        f"{'Gap':>8s}  {'BV Ep':>6s}  {'Max Ep':>6s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    for cond in [*PRETRAIN_021, *PRETRAIN_018]:
        d = data[cond]
        val_s = f"{d['best_val_loss']:.4f}" if d["best_val_loss"] else "?"
        train_s = f"{d['train_at_best']:.4f}" if d["train_at_best"] else "?"
        gap_s = f"{d['gap_at_best']:.4f}" if d["gap_at_best"] else "?"
        ep_s = (
            str(d["best_val_epoch"]) if d["best_val_epoch"] is not None else "?"
        )
        print(
            f"{cond:<22s}  {val_s:>10s}  {train_s:>10s}  "
            f"{gap_s:>8s}  {ep_s:>6s}  {d['max_epoch']:>6}  {d['run_id']}"
        )

    print("\n  Cross-tokenizer comparison (dynamic improvement):")
    for tok, runs in [("CWT", PRETRAIN_021), ("RCNN", PRETRAIN_018)]:
        keys = list(runs.keys())
        dis_key = [k for k in keys if "disabled" in k][0]
        dyn_key = [k for k in keys if "dynamic" in k][0]
        dis_val = data[dis_key]["best_val_loss"]
        dyn_val = data[dyn_key]["best_val_loss"]
        if dis_val and dyn_val:
            rel = (dyn_val - dis_val) / dis_val * 100
            print(
                f"    {tok}: disabled={dis_val:.4f} → dynamic={dyn_val:.4f}"
                f" ({rel:+.1f}% relative)"
            )


def plot_pretrain_val_overlay(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    all_runs = {**PRETRAIN_021, **PRETRAIN_018}

    for cond in all_runs:
        d = data[cond]
        edf = d["epoch_df"]
        if VAL_LOSS not in edf.columns:
            continue
        valid = edf.dropna(subset=[VAL_LOSS]).sort_values("epoch")
        label = f"{cond} (best={d['best_val_loss']:.4f})"
        ax.plot(
            valid["epoch"],
            valid[VAL_LOSS],
            color=PRETRAIN_COLORS[cond],
            linewidth=2.5,
            marker="o",
            markersize=3,
            label=label,
        )
        if d["best_val_epoch"] is not None:
            best_row = valid[valid["epoch"] == d["best_val_epoch"]]
            if not best_row.empty:
                ax.plot(
                    d["best_val_epoch"],
                    best_row[VAL_LOSS].values[0],
                    marker="*",
                    markersize=14,
                    color=PRETRAIN_COLORS[cond],
                    zorder=5,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(
        "Pretraining Val Loss — CWT-CNN (exp 021) vs ResampleCNN (exp 018)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    out = FIGURES_DIR / "021_pretrain_val_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_pretrain_bar(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    conds = ["CWT-disabled", "CWT-dynamic", "RCNN-disabled", "RCNN-dynamic"]
    vals = [data[c]["best_val_loss"] for c in conds]
    colors = [PRETRAIN_COLORS[c] for c in conds]
    x_labels = [c.replace("-", "\n") for c in conds]

    ax = axes[0]
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
            bar.get_height() + 0.005,
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
    gaps = [data[c]["gap_at_best"] or 0 for c in conds]
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
    ax.set_title("Train–Val Gap at Best Val Epoch")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.suptitle(
        "Pretraining: CWT-CNN (exp 021) vs ResampleCNN (exp 018)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "021_pretrain_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Linear Probing
# ═══════════════════════════════════════════════════════════════════════════


def fetch_lp_data() -> dict[str, dict]:
    all_runs = {**LP_021, **LP_020}
    results = {}
    for cond, run_id in all_runs.items():
        print(f"  Fetching LP {cond} ({run_id})...")
        epoch_df = fetch_metric_history(
            run_id,
            [TRAIN_LOSS, VAL_LOSS, VAL_F1, VAL_ACC],
            LP_PROJECT,
            WANDB_ENTITY,
            x_axis="epoch",
            aggregate_epoch=True,
        )
        summary = fetch_run_summary(
            run_id,
            LP_PROJECT,
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


def print_lp_summary(data: dict) -> None:
    print(f"\n{'=' * 115}")
    print(
        "  Phase 2: Linear Probe — CWT-CNN (exp 021) vs ResampleCNN (exp 020)"
    )
    print(f"{'=' * 115}")

    header = (
        f"{'Condition':<30s}  {'Val F1':>8s}  {'Val Acc':>8s}  "
        f"{'Val Loss':>9s}  {'BF1 Ep':>6s}  {'Max Ep':>6s}  {'Run ID'}"
    )
    print(header)
    print("-" * len(header))

    all_conds = [*LP_021, *LP_020]
    for cond in all_conds:
        d = data[cond]
        f1_s = f"{d['best_val_f1']:.4f}" if d["best_val_f1"] else "?"
        acc_s = f"{d['best_val_acc']:.4f}" if d["best_val_acc"] else "?"
        loss_s = f"{d['best_val_loss']:.4f}" if d["best_val_loss"] else "?"
        ep_s = (
            str(d["best_f1_epoch"]) if d["best_f1_epoch"] is not None else "?"
        )
        print(
            f"{cond:<30s}  {f1_s:>8s}  {acc_s:>8s}  "
            f"{loss_s:>9s}  {ep_s:>6s}  {d['max_epoch']:>6}  {d['run_id']}"
        )

    print("\n  Pairwise F1 comparisons:")
    comparisons = [
        (
            "CWT-pretrained-disabled",
            "CWT-pretrained-dynamic",
            "CWT: disabled vs dynamic (pretrained)",
        ),
        (
            "CWT-random-disabled",
            "CWT-random-dynamic",
            "CWT: disabled vs dynamic (random)",
        ),
        (
            "RCNN-pretrained-disabled",
            "RCNN-pretrained-dynamic",
            "RCNN: disabled vs dynamic (pretrained)",
        ),
        (
            "RCNN-random-disabled",
            "RCNN-random-dynamic",
            "RCNN: disabled vs dynamic (random)",
        ),
        (
            "CWT-pretrained-disabled",
            "RCNN-pretrained-disabled",
            "Pretrained disabled: CWT vs RCNN",
        ),
        (
            "CWT-pretrained-dynamic",
            "RCNN-pretrained-dynamic",
            "Pretrained dynamic: CWT vs RCNN",
        ),
        (
            "CWT-random-disabled",
            "RCNN-random-disabled",
            "Random disabled: CWT vs RCNN",
        ),
        (
            "CWT-random-dynamic",
            "RCNN-random-dynamic",
            "Random dynamic: CWT vs RCNN",
        ),
    ]
    for a, b, desc in comparisons:
        f1_a = data[a]["best_val_f1"]
        f1_b = data[b]["best_val_f1"]
        if f1_a and f1_b:
            diff = (f1_a - f1_b) * 100
            print(f"    {desc}: {diff:+.1f} pp F1")


def plot_lp_bar_cross_tokenizer(data: dict) -> None:
    """Grouped bar chart: CWT-CNN vs ResampleCNN linear probe F1, 8 conditions."""
    fig, ax = plt.subplots(figsize=(16, 6.5))

    groups = [
        (
            "Pretrained\nDisabled",
            "CWT-pretrained-disabled",
            "RCNN-pretrained-disabled",
        ),
        (
            "Pretrained\nDynamic",
            "CWT-pretrained-dynamic",
            "RCNN-pretrained-dynamic",
        ),
        ("Random\nDisabled", "CWT-random-disabled", "RCNN-random-disabled"),
        ("Random\nDynamic", "CWT-random-dynamic", "RCNN-random-dynamic"),
    ]

    x = np.arange(len(groups))
    width = 0.35

    cwt_vals = [data[g[1]]["best_val_f1"] for g in groups]
    rcnn_vals = [data[g[2]]["best_val_f1"] for g in groups]

    bars1 = ax.bar(
        x - width / 2,
        cwt_vals,
        width,
        label="CWT-CNN",
        color="#55A868",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        rcnn_vals,
        width,
        label="ResampleCNN",
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.003,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title(
        "Linear Probe F1: CWT-CNN (exp 021) vs ResampleCNN (exp 020)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(cwt_vals + rcnn_vals) * 1.15)

    plt.tight_layout()
    out = FIGURES_DIR / "021_lp_cross_tokenizer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_lp_f1_curves(data: dict) -> None:
    """F1 learning curves for all 8 linear probe conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (title, runs) in enumerate(
        [
            ("CWT-CNN (exp 021)", LP_021),
            ("ResampleCNN (exp 020)", LP_020),
        ]
    ):
        ax = axes[ax_i]
        for cond in runs:
            d = data[cond]
            edf = d["epoch_df"]
            if VAL_F1 not in edf.columns:
                continue
            valid = edf.dropna(subset=[VAL_F1]).sort_values("epoch")
            short_label = cond.split("-", 1)[1]
            label = f"{short_label} (best={d['best_val_f1']:.3f})"
            ax.plot(
                valid["epoch"],
                valid[VAL_F1],
                linewidth=2.5,
                marker="o",
                markersize=3,
                label=label,
                color=LP_COLORS[cond],
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation F1 (macro)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Linear Probe Val F1 Learning Curves",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "021_lp_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_lp_dynamic_advantage(data: dict) -> None:
    """Bar chart: dynamic vs disabled F1 advantage for each tokenizer × init."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    comparisons = [
        (
            "CWT\nPretrained",
            "CWT-pretrained-dynamic",
            "CWT-pretrained-disabled",
        ),
        ("CWT\nRandom", "CWT-random-dynamic", "CWT-random-disabled"),
        (
            "RCNN\nPretrained",
            "RCNN-pretrained-dynamic",
            "RCNN-pretrained-disabled",
        ),
        ("RCNN\nRandom", "RCNN-random-dynamic", "RCNN-random-disabled"),
    ]

    deltas = []
    labels = []
    colors = ["#55A868", "#8FBC8F", "#4C72B0", "#7BA3D0"]
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
        width=0.55,
    )
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("F1 Improvement (percentage points)", fontsize=11)
    ax.set_title(
        "Dynamic vs Disabled Channel Embedding\n(Linear Probe F1 Advantage)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = FIGURES_DIR / "021_lp_dynamic_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_lp_pretrain_advantage(data: dict) -> None:
    """Bar chart: pretrained vs random F1 advantage for each tokenizer × channel mode."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    comparisons = [
        ("CWT\nDisabled", "CWT-pretrained-disabled", "CWT-random-disabled"),
        ("CWT\nDynamic", "CWT-pretrained-dynamic", "CWT-random-dynamic"),
        ("RCNN\nDisabled", "RCNN-pretrained-disabled", "RCNN-random-disabled"),
        ("RCNN\nDynamic", "RCNN-pretrained-dynamic", "RCNN-random-dynamic"),
    ]

    deltas = []
    labels = []
    colors = ["#8172B2", "#55A868", "#C44E52", "#4C72B0"]
    for label, pre, rnd in comparisons:
        delta = (data[pre]["best_val_f1"] - data[rnd]["best_val_f1"]) * 100
        deltas.append(delta)
        labels.append(label)

    bars = ax.bar(
        range(len(deltas)),
        deltas,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.55,
    )
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + 0.2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"+{val:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("F1 Improvement (percentage points)", fontsize=11)
    ax.set_title(
        "Pretraining Advantage\n(Pretrained vs Random Linear Probe F1)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylim(0, max(deltas) * 1.4)

    plt.tight_layout()
    out = FIGURES_DIR / "021_lp_pretrain_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print(
        "  Experiment 021: CWT-CNN Dynamic Channel Embeddings — Full Analysis"
    )
    print("=" * 70)

    print("\n── Phase 1: Pretraining ──")
    pretrain_data = fetch_pretrain_data()
    print_pretrain_summary(pretrain_data)
    plot_pretrain_val_overlay(pretrain_data)
    plot_pretrain_bar(pretrain_data)

    print("\n── Phase 2: Linear Probing ──")
    lp_data = fetch_lp_data()
    print_lp_summary(lp_data)
    plot_lp_bar_cross_tokenizer(lp_data)
    plot_lp_f1_curves(lp_data)
    plot_lp_dynamic_advantage(lp_data)
    plot_lp_pretrain_advantage(lp_data)

    print("\n── Phase 3: Embedding Visualization ──")
    print(
        "  Run separately: uv run python analysis/021_cwt_dynamic_channel_emb_viz.py"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

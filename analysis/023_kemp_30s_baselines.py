"""KempSleep 30s-epoch from-scratch baselines (exp 023).

Compares 5 model conditions × 3 folds × 2 dataset sizes (10% and 100% train).
Also contrasts 30s results against 2s results from exp 022.

WandB project: foundry_finetuning
Group: KEMP_30S_BASELINES

Dimensions analysed:
  1. 30s vs 2s window length (exp 023 vs exp 022, fold 0)
  2. Tokenizer comparison (CWT-CNN vs ResampleCNN)
  3. Channel embedding strategy (disabled vs dynamic)
  4. Data scaling (10% → 100% training data)

Usage:
    uv run python analysis/023_kemp_30s_baselines.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()
FIGURES_DIR = figures_dir(__file__)

VAL_F1 = "val/sleep_stage_5class_f1"
VAL_ACC = "val/sleep_stage_5class_acc"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"

# --- Experiment 023 runs (30s, from scratch) ---
RUNS_023 = {
    # Small dataset (10% train)
    "eegnet_smol_f0": "e1va37uj",
    "eegnet_smol_f1": "m0gecgmy",
    "eegnet_smol_f2": "m53rrd5l",
    "cwt_dis_smol_f0": "jn0pjmtb",
    "cwt_dis_smol_f1": "3fvmumdt",
    "cwt_dis_smol_f2": "vwhc52ff",
    "cwt_dyn_smol_f0": "o6l4cv5d",
    "cwt_dyn_smol_f1": "xjhkx13o",
    "cwt_dyn_smol_f2": "hyhkgsic",
    "rcnn_dis_smol_f0": "l5yl7v99",
    "rcnn_dis_smol_f1": "j7939koc",
    "rcnn_dis_smol_f2": "cuhrejqv",
    "rcnn_dyn_smol_f0": "egnt4itq",
    "rcnn_dyn_smol_f1": "wx44epel",
    "rcnn_dyn_smol_f2": "gevk12ti",
    # Full dataset (100% train)
    "eegnet_full_f0": "9x4w789b",
    "eegnet_full_f1": "wxa14ec1",
    "eegnet_full_f2": "7un6237q",
    "cwt_dis_full_f0": "9m98r3we",
    "cwt_dis_full_f1": "bwjwtoq5",
    "cwt_dis_full_f2": "aotuuq3s",
    "cwt_dyn_full_f0": "852zgx76",
    "cwt_dyn_full_f1": "th3g8zdv",
    "cwt_dyn_full_f2": "m4l9b5o4",
    "rcnn_dis_full_f0": "wzmcyafl",
    "rcnn_dis_full_f1": "tjz6nfp6",
    "rcnn_dis_full_f2": "bcr6otbd",
    "rcnn_dyn_full_f0": "axnnllx6",
    "rcnn_dyn_full_f1": "kxi0u259",
    "rcnn_dyn_full_f2": "q0cz800r",
}

# --- Experiment 022 runs (2s, from scratch, fold 0 only) ---
RUNS_022_SCRATCH = {
    "scratch-cwt-disabled": "g3mfdwj6",
    "scratch-cwt-dynamic": "pew03xnz",
    "scratch-rcnn-disabled": "x130d6jj",
    "scratch-rcnn-dynamic": "lhutmecj",
}

CONDITIONS = ["eegnet", "cwt_dis", "cwt_dyn", "rcnn_dis", "rcnn_dyn"]
CONDITION_LABELS = {
    "eegnet": "EEGNet",
    "cwt_dis": "POYO\nCWT-CNN\nDisabled",
    "cwt_dyn": "POYO\nCWT-CNN\nDynamic",
    "rcnn_dis": "POYO\nRCNN\nDisabled",
    "rcnn_dyn": "POYO\nRCNN\nDynamic",
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

FOLDS = [0, 1, 2]
SIZES = ["smol", "full"]


def fetch_all_data() -> dict[str, dict]:
    """Fetch best val F1 for every run via metric history."""
    results = {}
    for key, run_id in RUNS_023.items():
        print(f"  Fetching {key} ({run_id})...")
        try:
            epoch_df = fetch_metric_history(
                run_id,
                [VAL_F1, VAL_ACC, VAL_LOSS],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                aggregate_epoch=True,
            )
            best_f1 = best_acc = best_loss = best_ep = max_ep = None
            if not epoch_df.empty and VAL_F1 in epoch_df.columns:
                valid = epoch_df.dropna(subset=[VAL_F1])
                if not valid.empty:
                    best_idx = valid[VAL_F1].idxmax()
                    best_row = valid.loc[best_idx]
                    best_f1 = float(best_row[VAL_F1])
                    best_acc = (
                        float(best_row[VAL_ACC])
                        if VAL_ACC in valid.columns
                        else None
                    )
                    best_loss = (
                        float(best_row[VAL_LOSS])
                        if VAL_LOSS in valid.columns
                        else None
                    )
                    best_ep = int(best_row["epoch"])
                    max_ep = int(valid["epoch"].max())
            results[key] = {
                "run_id": run_id,
                "best_f1": best_f1,
                "best_acc": best_acc,
                "best_loss": best_loss,
                "best_ep": best_ep,
                "max_ep": max_ep,
                "epoch_df": epoch_df,
            }
        except Exception as e:
            print(f"    WARNING: {e}")
            results[key] = {"run_id": run_id, "best_f1": None, "epoch_df": None}
    return results


def fetch_022_data() -> dict[str, float]:
    """Fetch exp 022 scratch fold-0 F1 for 2s vs 30s comparison."""
    results = {}
    for key, run_id in RUNS_022_SCRATCH.items():
        print(f"  Fetching 022 {key} ({run_id})...")
        try:
            epoch_df = fetch_metric_history(
                run_id,
                [VAL_F1],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                aggregate_epoch=True,
            )
            if not epoch_df.empty and VAL_F1 in epoch_df.columns:
                valid = epoch_df.dropna(subset=[VAL_F1])
                if not valid.empty:
                    results[key] = float(valid[VAL_F1].max())
        except Exception as e:
            print(f"    WARNING: {e}")
    return results


def get_condition_stats(
    data: dict, condition: str, size: str
) -> tuple[float, float, list[float]]:
    """Return (mean, std, values) for a condition across folds."""
    vals = []
    for f in FOLDS:
        key = f"{condition}_{size}_f{f}"
        if key in data and data[key]["best_f1"] is not None:
            vals.append(data[key]["best_f1"])
    if not vals:
        return 0.0, 0.0, []
    return float(np.mean(vals)), float(np.std(vals)), vals


def print_summary(data: dict, data_022: dict) -> None:
    """Print comprehensive summary tables."""
    print(f"\n{'=' * 100}")
    print("  Experiment 023: KempSleep 30s-Epoch From-Scratch Baselines")
    print(f"{'=' * 100}")

    for size, size_label in [("smol", "10% Train"), ("full", "100% Train")]:
        print(f"\n── {size_label} ──")
        print(
            f"{'Condition':<20s}  {'Mean F1':>8s}  {'Std':>6s}  "
            f"{'Fold 0':>8s}  {'Fold 1':>8s}  {'Fold 2':>8s}"
        )
        print("-" * 70)
        for cond in CONDITIONS:
            mean, std, vals = get_condition_stats(data, cond, size)
            fold_strs = [f"{v:.4f}" if v else "  N/A " for v in vals]
            while len(fold_strs) < 3:
                fold_strs.append("  N/A ")
            print(
                f"{CONDITION_SHORT[cond]:<20s}  {mean:.4f}  {std:.4f}  "
                f"{'  '.join(fold_strs)}"
            )

    print("\n── 30s vs 2s Comparison (fold 0 only) ──")
    mapping_2s_to_023 = {
        "scratch-cwt-disabled": "cwt_dis",
        "scratch-cwt-dynamic": "cwt_dyn",
        "scratch-rcnn-disabled": "rcnn_dis",
        "scratch-rcnn-dynamic": "rcnn_dyn",
    }
    print(f"{'Condition':<25s}  {'2s F1':>8s}  {'30s F1':>8s}  {'Δ (pp)':>8s}")
    print("-" * 55)
    for key_2s, cond_023 in mapping_2s_to_023.items():
        f1_2s = data_022.get(key_2s, 0)
        key_30s = f"{cond_023}_full_f0"
        f1_30s = (
            data[key_30s]["best_f1"]
            if key_30s in data and data[key_30s]["best_f1"]
            else 0
        )
        delta = (f1_30s - f1_2s) * 100
        print(f"{key_2s:<25s}  {f1_2s:.4f}  {f1_30s:.4f}  {delta:+.1f}")

    print("\n── Data Scaling (10% → 100%) ──")
    print(
        f"{'Condition':<20s}  {'10% F1':>8s}  {'100% F1':>8s}  {'Δ (pp)':>8s}  {'Relative':>8s}"
    )
    print("-" * 65)
    for cond in CONDITIONS:
        mean_smol, _, _ = get_condition_stats(data, cond, "smol")
        mean_full, _, _ = get_condition_stats(data, cond, "full")
        delta = (mean_full - mean_smol) * 100
        rel = delta / (mean_smol * 100) * 100 if mean_smol > 0 else 0
        print(
            f"{CONDITION_SHORT[cond]:<20s}  {mean_smol:.4f}  {mean_full:.4f}  "
            f"{delta:+.1f}    {rel:+.1f}%"
        )


def plot_main_results(data: dict) -> None:
    """Grouped bar chart: all conditions × dataset sizes with fold error bars."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(CONDITIONS))
    width = 0.35

    means_smol = []
    stds_smol = []
    means_full = []
    stds_full = []

    for cond in CONDITIONS:
        m, s, _ = get_condition_stats(data, cond, "smol")
        means_smol.append(m)
        stds_smol.append(s)
        m, s, _ = get_condition_stats(data, cond, "full")
        means_full.append(m)
        stds_full.append(s)

    bars1 = ax.bar(
        x - width / 2,
        means_smol,
        width,
        yerr=stds_smol,
        label="10% Train",
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )
    bars2 = ax.bar(
        x + width / 2,
        means_full,
        width,
        yerr=stds_full,
        label="100% Train",
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )

    for bars, means in [(bars1, means_smol), (bars2, means_full)]:
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=11)
    ax.set_title(
        "KempSleep 30s Baselines: All Conditions (exp 023)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0.5, 0.80)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "023_main_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_30s_vs_2s(data: dict, data_022: dict) -> None:
    """Bar chart comparing 30s (exp 023) vs 2s (exp 022) on fold 0."""
    fig, ax = plt.subplots(figsize=(12, 7))

    mapping = {
        "CWT-CNN\nDisabled": ("scratch-cwt-disabled", "cwt_dis_full_f0"),
        "CWT-CNN\nDynamic": ("scratch-cwt-dynamic", "cwt_dyn_full_f0"),
        "RCNN\nDisabled": ("scratch-rcnn-disabled", "rcnn_dis_full_f0"),
        "RCNN\nDynamic": ("scratch-rcnn-dynamic", "rcnn_dyn_full_f0"),
    }

    labels = list(mapping.keys())
    x = np.arange(len(labels))
    width = 0.35

    f1_2s = []
    f1_30s = []
    for label, (key_2s, key_30s) in mapping.items():
        f1_2s.append(data_022.get(key_2s, 0))
        f1_30s.append(
            data[key_30s]["best_f1"] if data[key_30s]["best_f1"] else 0
        )

    bars1 = ax.bar(
        x - width / 2,
        f1_2s,
        width,
        label="2s windows (exp 022)",
        color="#C44E52",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        f1_30s,
        width,
        label="30s epochs (exp 023)",
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    for bars, vals in [(bars1, f1_2s), (bars2, f1_30s)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    for i, (v2, v30) in enumerate(zip(f1_2s, f1_30s)):
        delta = (v30 - v2) * 100
        mid_y = max(v2, v30) + 0.035
        ax.annotate(
            f"+{delta:.1f} pp",
            xy=(i, mid_y),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2E4057",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title(
        "Impact of Window Length: 30s vs 2s (fold 0, from scratch)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0.5, 0.82)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "023_30s_vs_2s.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_data_scaling(data: dict) -> None:
    """Slope chart showing how each model scales from 10% to 100% data."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for cond in CONDITIONS:
        mean_smol, std_smol, _ = get_condition_stats(data, cond, "smol")
        mean_full, std_full, _ = get_condition_stats(data, cond, "full")

        ax.plot(
            [0, 1],
            [mean_smol, mean_full],
            "o-",
            color=CONDITION_COLORS[cond],
            linewidth=2.5,
            markersize=10,
            label=f"{CONDITION_SHORT[cond]} (+{(mean_full - mean_smol) * 100:.1f} pp)",
        )
        ax.fill_between(
            [0, 1],
            [mean_smol - std_smol, mean_full - std_full],
            [mean_smol + std_smol, mean_full + std_full],
            color=CONDITION_COLORS[cond],
            alpha=0.15,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["10% Train (~4 subjects)", "100% Train (~39 subjects)"], fontsize=11
    )
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=11)
    ax.set_title(
        "Data Scaling: How Models Benefit from 10× More Training Data",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()
    out = FIGURES_DIR / "023_data_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_tokenizer_channel_emb(data: dict) -> None:
    """2×2 comparison: tokenizer × channel_emb for full dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Tokenizer effect (CWT vs RCNN) for each channel emb mode
    ax = axes[0]
    for ch_mode, ch_label, color_cwt, color_rcnn in [
        ("dis", "Disabled", "#4C72B0", "#C44E52"),
        ("dyn", "Dynamic", "#55A868", "#8172B2"),
    ]:
        cwt_mean, cwt_std, _ = get_condition_stats(
            data, f"cwt_{ch_mode}", "full"
        )
        rcnn_mean, rcnn_std, _ = get_condition_stats(
            data, f"rcnn_{ch_mode}", "full"
        )

        x_pos = 0 if ch_mode == "dis" else 1
        offset = 0.15
        ax.bar(
            x_pos - offset,
            cwt_mean,
            0.25,
            yerr=cwt_std,
            color=color_cwt,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            label=f"CWT-CNN ({ch_label})"
            if ch_mode == "dis"
            else f"CWT-CNN ({ch_label})",
        )
        ax.bar(
            x_pos + offset,
            rcnn_mean,
            0.25,
            yerr=rcnn_std,
            color=color_rcnn,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            label=f"RCNN ({ch_label})",
        )
        delta = (cwt_mean - rcnn_mean) * 100
        ax.text(
            x_pos,
            max(cwt_mean, rcnn_mean) + cwt_std + 0.012,
            f"Δ={delta:+.1f}pp",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ch. Emb: Disabled", "Ch. Emb: Dynamic"], fontsize=10)
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=11)
    ax.set_title(
        "Tokenizer Effect\n(CWT-CNN vs ResampleCNN)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.6, 0.78)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Channel emb effect (disabled vs dynamic) for each tokenizer
    ax = axes[1]
    for tok, tok_label, color_dis, color_dyn in [
        ("cwt", "CWT-CNN", "#4C72B0", "#55A868"),
        ("rcnn", "RCNN", "#C44E52", "#8172B2"),
    ]:
        dis_mean, dis_std, _ = get_condition_stats(data, f"{tok}_dis", "full")
        dyn_mean, dyn_std, _ = get_condition_stats(data, f"{tok}_dyn", "full")

        x_pos = 0 if tok == "cwt" else 1
        offset = 0.15
        ax.bar(
            x_pos - offset,
            dis_mean,
            0.25,
            yerr=dis_std,
            color=color_dis,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            label=f"Disabled ({tok_label})",
        )
        ax.bar(
            x_pos + offset,
            dyn_mean,
            0.25,
            yerr=dyn_std,
            color=color_dyn,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            label=f"Dynamic ({tok_label})",
        )
        delta = (dyn_mean - dis_mean) * 100
        ax.text(
            x_pos,
            max(dis_mean, dyn_mean) + dis_std + 0.012,
            f"Δ={delta:+.1f}pp",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CWT-CNN", "ResampleCNN"], fontsize=10)
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=11)
    ax.set_title(
        "Channel Embedding Effect\n(Disabled vs Dynamic)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.6, 0.78)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Tokenizer × Channel Embedding — Full Dataset, 30s (exp 023)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "023_tokenizer_channel_emb.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_fold_variance(data: dict) -> None:
    """Strip plot showing individual fold results for each condition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (size, size_label) in enumerate(
        [("smol", "10% Train"), ("full", "100% Train")]
    ):
        ax = axes[ax_i]
        for i, cond in enumerate(CONDITIONS):
            _, _, vals = get_condition_stats(data, cond, size)
            if vals:
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
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
            [CONDITION_SHORT[c] for c in CONDITIONS], fontsize=9, rotation=15
        )
        ax.set_ylabel("Macro F1", fontsize=11)
        ax.set_title(size_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if size == "smol":
            ax.set_ylim(0.5, 0.75)
        else:
            ax.set_ylim(0.6, 0.78)

    plt.suptitle(
        "Cross-Fold Variance: Individual Fold Results (exp 023)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "023_fold_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_f1_curves(data: dict) -> None:
    """Validation F1 learning curves for all conditions (full dataset)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (size, size_label) in enumerate(
        [("smol", "10% Train"), ("full", "100% Train")]
    ):
        ax = axes[ax_i]
        for cond in CONDITIONS:
            fold_dfs = []
            for f in FOLDS:
                key = f"{cond}_{size}_f{f}"
                if key in data and data[key].get("epoch_df") is not None:
                    edf = data[key]["epoch_df"]
                    if not edf.empty and VAL_F1 in edf.columns:
                        valid = edf.dropna(subset=[VAL_F1]).sort_values("epoch")
                        if not valid.empty:
                            fold_dfs.append(valid[["epoch", VAL_F1]])

            if not fold_dfs:
                continue

            # Plot fold 0 as representative
            best_f1_mean, _, _ = get_condition_stats(data, cond, size)
            df = fold_dfs[0]
            ax.plot(
                df["epoch"],
                df[VAL_F1],
                linewidth=2,
                color=CONDITION_COLORS[cond],
                label=f"{CONDITION_SHORT[cond]} ({best_f1_mean:.3f})",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation F1 (macro)", fontsize=11)
        ax.set_title(size_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Validation F1 Learning Curves — Fold 0 (exp 023)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "023_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 70)
    print("  Experiment 023: KempSleep 30s-Epoch From-Scratch Baselines")
    print("=" * 70)

    print("\n── Fetching exp 023 data ──")
    data = fetch_all_data()

    print("\n── Fetching exp 022 data (2s comparison) ──")
    data_022 = fetch_022_data()

    print_summary(data, data_022)

    print("\n── Generating plots ──")
    plot_main_results(data)
    plot_30s_vs_2s(data, data_022)
    plot_data_scaling(data)
    plot_tokenizer_channel_emb(data)
    plot_fold_variance(data)
    plot_f1_curves(data)

    print("\nDone.")


if __name__ == "__main__":
    main()

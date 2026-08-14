"""Analysis: Channel Encoder Leak Fix Impact.

Compares 4 pretraining configurations that ablate two information leak fixes
(channel encoder masking + signal zeroing) and two tokenizers (CWT-CNN vs
ResampleCNN), all at the B2 data scale (3 datasets, ~37k ch·h).

Pretraining runs (WandB group CHANNEL_LEAK_FIX):
  - pretrain_leak_baseline:       both leaks present, CWT-CNN
  - pretrain_leak_fixed:          ch-encoder fix only, CWT-CNN
  - pretrain_all_fixed_cwt:       both fixes, CWT-CNN
  - pretrain_all_fixed_resample:  both fixes, ResampleCNN

Downstream evaluation: 3 tasks x 2 modes (finetune + linear probe) x 3 folds.

Usage:
    uv run python analysis/037_channel_encoder_leak_fix_impact.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PRETRAIN_PROJECT = "foundry_pretraining"
DOWNSTREAM_PROJECT = "foundry_finetuning"

# ── Pretraining run definitions ──────────────────────────────────────────────

PRETRAIN_RUNS = {
    "baseline": {
        "name": "pretrain_leak_baseline",
        "label": "Baseline (no fixes)",
        "ch_fix": False,
        "sig_fix": False,
        "tokenizer": "CWT-CNN",
    },
    "ch_fix": {
        "name": "pretrain_leak_fixed",
        "label": "Ch-encoder fix only",
        "ch_fix": True,
        "sig_fix": False,
        "tokenizer": "CWT-CNN",
    },
    "both_cwt": {
        "name": "pretrain_all_fixed_cwt",
        "label": "Both fixes (CWT-CNN)",
        "ch_fix": True,
        "sig_fix": True,
        "tokenizer": "CWT-CNN",
    },
    "both_resample": {
        "name": "pretrain_all_fixed_resample",
        "label": "Both fixes (ResampleCNN)",
        "ch_fix": True,
        "sig_fix": True,
        "tokenizer": "ResampleCNN",
    },
}

PRETRAIN_GROUP = "CHANNEL_LEAK_FIX"

# ── Downstream group / naming conventions ────────────────────────────────────

DOWNSTREAM_GROUPS = {
    ("Kemp Sleep", "finetune"): "KEMP_FT_DATA_SCALING",
    ("Kemp Sleep", "linear_probe"): "KEMP_LP_DATA_SCALING",
    ("PhysioNet MI", "finetune"): "PHYSIONET_FT_DATA_SCALING",
    ("PhysioNet MI", "linear_probe"): "PHYSIONET_LP_DATA_SCALING",
    ("Brain Invaders P300", "finetune"): "BI_P300_FT_DATA_SCALING",
    ("Brain Invaders P300", "linear_probe"): "BI_P300_LP_DATA_SCALING",
}

DOWNSTREAM_RUN_PREFIXES = {
    ("Kemp Sleep", "finetune"): "kemp_ft_",
    ("Kemp Sleep", "linear_probe"): "kemp_lp_",
    ("PhysioNet MI", "finetune"): "physionet_ft_",
    ("PhysioNet MI", "linear_probe"): "physionet_lp_",
    ("Brain Invaders P300", "finetune"): "bi_ft_",
    ("Brain Invaders P300", "linear_probe"): "bi_lp_",
}

METRIC_KEYS = {
    "Kemp Sleep": "val/sleep_stage_5class_f1",
    "PhysioNet MI": "val/motor_imagery_binary_f1",
    "Brain Invaders P300": "val/p300_binary_f1",
}

TASKS = ["Kemp Sleep", "PhysioNet MI", "Brain Invaders P300"]

RUN_ORDER = ["baseline", "ch_fix", "both_cwt", "both_resample"]

RUN_COLORS = {
    "baseline": "#d62728",
    "ch_fix": "#ff7f0e",
    "both_cwt": "#2ca02c",
    "both_resample": "#1f77b4",
}


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_pretrain_losses(api: wandb.Api) -> pd.DataFrame:
    """Fetch val/loss and train/loss curves for all pretraining runs."""
    entity = api.default_entity
    records = []

    for run_id, info in PRETRAIN_RUNS.items():
        print(f"  Fetching pretrain {run_id}: {info['name']}...")
        runs = api.runs(
            f"{entity}/{PRETRAIN_PROJECT}",
            filters={"group": PRETRAIN_GROUP, "display_name": info["name"]},
        )
        run_list = list(runs)
        if not run_list:
            print(f"    [WARN] No run found for {info['name']}")
            continue

        run = run_list[0]

        val_history = run.history(keys=["val/loss"], samples=50000, pandas=True)
        train_history = run.history(
            keys=["train/loss"], samples=50000, pandas=True
        )

        for _, row in val_history.iterrows():
            records.append({
                "run_id": run_id,
                "run_name": info["name"],
                "label": info["label"],
                "step": row.get("_step"),
                "val_loss": row.get("val/loss"),
                "train_loss": None,
            })

        for _, row in train_history.iterrows():
            records.append({
                "run_id": run_id,
                "run_name": info["name"],
                "label": info["label"],
                "step": row.get("_step"),
                "val_loss": None,
                "train_loss": row.get("train/loss"),
            })

        best_val = (
            val_history["val/loss"].dropna().min()
            if "val/loss" in val_history.columns and len(val_history) > 0
            else None
        )
        total_steps = run.summary.get("_step", None)
        state = run.state
        if best_val is not None:
            final_val = val_history["val/loss"].dropna().iloc[-1]
            print(
                f"    state={state}, steps={total_steps}, "
                f"best_val_loss={best_val:.4f}, final_val_loss={final_val:.4f}"
            )
        else:
            print(f"    state={state}, steps={total_steps}, no val/loss data")

    return pd.DataFrame(records)


def fetch_downstream_results(api: wandb.Api) -> pd.DataFrame:
    """Fetch downstream finetuning and linear probe results."""
    entity = api.default_entity
    records = []

    for (task, mode), group in DOWNSTREAM_GROUPS.items():
        metric_key = METRIC_KEYS[task]
        prefix = DOWNSTREAM_RUN_PREFIXES[(task, mode)]
        print(f"\n  Fetching {task} / {mode} (group={group})...")

        runs = api.runs(
            f"{entity}/{DOWNSTREAM_PROJECT}", filters={"group": group}
        )
        run_list = list(runs)
        print(f"    Found {len(run_list)} runs in group")

        for run in run_list:
            pretrain_run_id = None
            for pid, pinfo in PRETRAIN_RUNS.items():
                if pinfo["name"] in run.name:
                    pretrain_run_id = pid
                    break

            if pretrain_run_id is None:
                continue

            fold = None
            for i in range(3):
                if f"fold{i}" in run.name:
                    fold = i
                    break
            if fold is None:
                continue

            history = run.history(keys=[metric_key], samples=50000, pandas=True)
            if metric_key in history.columns:
                vals = history[metric_key].dropna()
                best_f1 = float(vals.max()) if len(vals) > 0 else None
                best_epoch = int(vals.idxmax()) if len(vals) > 0 else None
                num_epochs = len(vals)
            else:
                best_f1 = None
                best_epoch = None
                num_epochs = 0

            records.append({
                "task": task,
                "mode": mode,
                "pretrain_run_id": pretrain_run_id,
                "pretrain_label": PRETRAIN_RUNS[pretrain_run_id]["label"],
                "tokenizer": PRETRAIN_RUNS[pretrain_run_id]["tokenizer"],
                "fold": fold,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "num_epochs": num_epochs,
                "run_name": run.name,
                "state": run.state,
            })

    return pd.DataFrame(records)


def summarize_downstream(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of best F1 across folds per (task, mode, pretrain_run_id)."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["task", "mode", "pretrain_run_id", "pretrain_label", "tokenizer"])[
            "best_f1"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_pretrain_loss_curves(pretrain_df: pd.DataFrame) -> Path:
    """Overlay val/loss curves for all 4 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Val loss
    ax = axes[0]
    for rid in RUN_ORDER:
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(subset=["val_loss"])
        if subset.empty:
            continue
        subset = subset.sort_values("step")
        ax.plot(
            subset["step"],
            subset["val_loss"],
            label=PRETRAIN_RUNS[rid]["label"],
            color=RUN_COLORS[rid],
            linewidth=1.8,
        )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Val Loss (MAE reconstruction)", fontsize=11)
    ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    # Train loss
    ax = axes[1]
    for rid in RUN_ORDER:
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(subset=["train_loss"])
        if subset.empty:
            continue
        subset = subset.sort_values("step")
        ax.plot(
            subset["step"],
            subset["train_loss"],
            label=PRETRAIN_RUNS[rid]["label"],
            color=RUN_COLORS[rid],
            linewidth=1.2,
            alpha=0.7,
        )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Train Loss (MAE reconstruction)", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Pretraining Loss — Channel Encoder Leak Fix Ablation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "037_pretrain_loss_curves.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_pretrain_final_loss_bar(pretrain_df: pd.DataFrame) -> Path:
    """Bar chart of best val/loss per run with % change annotations."""
    best_losses = {}
    for rid in RUN_ORDER:
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(subset=["val_loss"])
        if not subset.empty:
            best_losses[rid] = subset["val_loss"].min()

    if not best_losses:
        return None

    run_ids = [r for r in RUN_ORDER if r in best_losses]
    losses = [best_losses[r] for r in run_ids]
    colors = [RUN_COLORS[r] for r in run_ids]
    labels = [PRETRAIN_RUNS[r]["label"] for r in run_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(run_ids)), losses, color=colors, edgecolor="white", linewidth=0.5)

    baseline_loss = best_losses.get("baseline")
    for bar, loss, rid in zip(bars, losses, run_ids):
        annotation = f"{loss:.4f}"
        if baseline_loss is not None and rid != "baseline":
            pct_change = (loss - baseline_loss) / baseline_loss * 100
            annotation += f"\n({pct_change:+.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Best Val Loss", fontsize=11)
    ax.set_title(
        "Best Pretraining Val Loss — Leak Fix Ablation\n(higher = harder task after fix)",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "037_pretrain_final_loss_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_downstream_comparison(summary_df: pd.DataFrame) -> Path:
    """2x3 grid: finetune (top) and linear probe (bottom) for each task."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for mode_idx, mode in enumerate(["finetune", "linear_probe"]):
        mode_label = "Finetuning" if mode == "finetune" else "Linear Probe"
        sub = summary_df[summary_df["mode"] == mode]

        for col_idx, task in enumerate(TASKS):
            ax = axes[mode_idx, col_idx]
            task_data = sub[sub["task"] == task]

            available = [r for r in RUN_ORDER if r in task_data["pretrain_run_id"].values]
            if not available:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            means = []
            stds = []
            colors = []
            for rid in available:
                row = task_data[task_data["pretrain_run_id"] == rid]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(row["std"].values[0] if not np.isnan(row["std"].values[0]) else 0)
                else:
                    means.append(0)
                    stds.append(0)
                colors.append(RUN_COLORS[rid])

            x = np.arange(len(available))
            bars = ax.bar(
                x, means, 0.6, yerr=stds, capsize=4,
                color=colors, edgecolor="white", linewidth=0.5,
                error_kw=dict(lw=1),
            )

            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [PRETRAIN_RUNS[r]["label"] for r in available],
                fontsize=7, rotation=20, ha="right",
            )
            if col_idx == 0:
                ax.set_ylabel(f"{mode_label}\nBest Val F1", fontsize=10)
            if mode_idx == 0:
                ax.set_title(task, fontsize=12, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

            if means:
                valid = [m for m in means if m > 0]
                if valid:
                    ax.set_ylim(min(valid) - 0.04, max(valid) + 0.04)

    fig.suptitle(
        "Downstream Transfer — Leak Fix Impact\nFinetuning (top) · Linear Probe (bottom)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "037_downstream_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_ablation_deltas(summary_df: pd.DataFrame) -> Path:
    """Bar chart showing delta-F1 from baseline for each fix combination."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, mode in enumerate(["finetune", "linear_probe"]):
        mode_label = "Finetuning" if mode == "finetune" else "Linear Probe"
        ax = axes[ax_idx]
        sub = summary_df[summary_df["mode"] == mode]

        x_pos = np.arange(len(TASKS))
        width = 0.22
        comparison_runs = ["ch_fix", "both_cwt", "both_resample"]

        for i, rid in enumerate(comparison_runs):
            deltas = []
            for task in TASKS:
                baseline_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "baseline")]
                run_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == rid)]
                if not baseline_row.empty and not run_row.empty:
                    delta = run_row["mean"].values[0] - baseline_row["mean"].values[0]
                    deltas.append(delta)
                else:
                    deltas.append(0)

            offset = (i - 1) * width
            bars = ax.bar(
                x_pos + offset, deltas, width,
                color=RUN_COLORS[rid], label=PRETRAIN_RUNS[rid]["label"],
                edgecolor="white", linewidth=0.5,
            )
            for bar, d in zip(bars, deltas):
                if d != 0:
                    va = "bottom" if d >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if d >= 0 else -0.002),
                        f"{d:+.3f}",
                        ha="center", va=va, fontsize=8, fontweight="bold",
                    )

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(TASKS, fontsize=9)
        ax.set_ylabel("ΔF1 vs Baseline (no fixes)", fontsize=10)
        ax.set_title(mode_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Downstream ΔF1 vs Baseline — Ablation of Leak Fixes",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "037_ablation_deltas.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_tokenizer_comparison(summary_df: pd.DataFrame) -> Path:
    """Direct CWT-CNN vs ResampleCNN comparison (both with all fixes)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, mode in enumerate(["finetune", "linear_probe"]):
        mode_label = "Finetuning" if mode == "finetune" else "Linear Probe"
        ax = axes[ax_idx]
        sub = summary_df[summary_df["mode"] == mode]

        x_pos = np.arange(len(TASKS))
        width = 0.3

        for i, rid in enumerate(["both_cwt", "both_resample"]):
            means = []
            stds = []
            for task in TASKS:
                row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == rid)]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(row["std"].values[0] if not np.isnan(row["std"].values[0]) else 0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x_pos + offset, means, width, yerr=stds, capsize=4,
                color=RUN_COLORS[rid], label=PRETRAIN_RUNS[rid]["label"],
                edgecolor="white", linewidth=0.5, error_kw=dict(lw=1),
            )
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(TASKS, fontsize=9)
        ax.set_ylabel("Best Val F1", fontsize=10)
        ax.set_title(mode_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        all_means = []
        for rid in ["both_cwt", "both_resample"]:
            for task in TASKS:
                row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == rid)]
                if not row.empty:
                    all_means.append(row["mean"].values[0])
        if all_means:
            ax.set_ylim(min(all_means) - 0.04, max(all_means) + 0.04)

    fig.suptitle(
        "Tokenizer Comparison at B2 Scale (Both Fixes Applied)\nCWT-CNN vs ResampleCNN",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "037_tokenizer_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


# ── Printing ─────────────────────────────────────────────────────────────────


def print_pretrain_summary(pretrain_df: pd.DataFrame) -> None:
    """Print pretraining loss summary table."""
    print("\n" + "=" * 90)
    print("  PRETRAINING SUMMARY")
    print("=" * 90)
    header = f"  {'Run':<16} {'Label':<28} {'Tokenizer':<12} {'Best Val Loss':>14} {'Final Val Loss':>15} {'Δ vs Baseline':>14}"
    print(header)
    print(f"  {'-'*16} {'-'*28} {'-'*12} {'-'*14} {'-'*15} {'-'*14}")

    baseline_best = None
    for rid in RUN_ORDER:
        info = PRETRAIN_RUNS[rid]
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(subset=["val_loss"])
        if subset.empty:
            print(f"  {rid:<16} {info['label']:<28} {info['tokenizer']:<12} {'N/A':>14} {'N/A':>15} {'N/A':>14}")
            continue
        subset = subset.sort_values("step")
        best_loss = subset["val_loss"].min()
        final_loss = subset["val_loss"].iloc[-1]

        if rid == "baseline":
            baseline_best = best_loss
            delta_str = f"{'—':>14}"
        elif baseline_best is not None:
            pct = (best_loss - baseline_best) / baseline_best * 100
            delta_str = f"{pct:>+13.1f}%"
        else:
            delta_str = f"{'N/A':>14}"

        print(
            f"  {rid:<16} {info['label']:<28} {info['tokenizer']:<12} "
            f"{best_loss:>14.4f} {final_loss:>15.4f} {delta_str}"
        )


def print_downstream_summary(summary_df: pd.DataFrame) -> None:
    """Print downstream results summary."""
    for mode in ["finetune", "linear_probe"]:
        mode_label = "FINETUNING" if mode == "finetune" else "LINEAR PROBE"
        sub = summary_df[summary_df["mode"] == mode]

        print(f"\n\n{'='*90}")
        print(f"  DOWNSTREAM {mode_label}")
        print(f"{'='*90}")

        for task in TASKS:
            task_data = sub[sub["task"] == task]
            print(f"\n  {task}:")
            print(f"  {'Run':<16} {'Label':<28} {'Mean F1':>8} {'± Std':>8} {'N':>3}")
            print(f"  {'-'*16} {'-'*28} {'-'*8} {'-'*8} {'-'*3}")

            baseline_f1 = None
            for rid in RUN_ORDER:
                row = task_data[task_data["pretrain_run_id"] == rid]
                if not row.empty:
                    mean_f1 = row["mean"].values[0]
                    std_f1 = row["std"].values[0] if not np.isnan(row["std"].values[0]) else 0
                    n = int(row["count"].values[0])
                    if rid == "baseline":
                        baseline_f1 = mean_f1
                    print(
                        f"  {rid:<16} {PRETRAIN_RUNS[rid]['label']:<28} "
                        f"{mean_f1:>8.3f} {std_f1:>8.3f} {n:>3}"
                    )
                else:
                    print(
                        f"  {rid:<16} {PRETRAIN_RUNS[rid]['label']:<28} "
                        f"{'N/A':>8} {'':>8} {'':>3}"
                    )

            if baseline_f1 is not None:
                print(f"\n  Deltas vs baseline ({baseline_f1:.3f}):")
                for rid in ["ch_fix", "both_cwt", "both_resample"]:
                    row = task_data[task_data["pretrain_run_id"] == rid]
                    if not row.empty:
                        delta = row["mean"].values[0] - baseline_f1
                        print(f"    {PRETRAIN_RUNS[rid]['label']:<28} ΔF1 = {delta:+.3f}")


def print_key_comparisons(summary_df: pd.DataFrame) -> None:
    """Print structured comparison summaries for the 3 key questions."""
    print(f"\n\n{'#'*90}")
    print("  KEY COMPARISONS")
    print(f"{'#'*90}")

    ft = summary_df[summary_df["mode"] == "finetune"]
    lp = summary_df[summary_df["mode"] == "linear_probe"]

    print("\n  1. LEAK FIX IMPACT (baseline → both_cwt)")
    print("  " + "-" * 60)
    for task in TASKS:
        for mode, sub, label in [("FT", ft, "finetune"), ("LP", lp, "linear_probe")]:
            b_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "baseline")]
            f_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "both_cwt")]
            if not b_row.empty and not f_row.empty:
                b_val = b_row["mean"].values[0]
                f_val = f_row["mean"].values[0]
                print(f"    {task} ({mode}): {b_val:.3f} → {f_val:.3f} (Δ = {f_val - b_val:+.3f})")

    print("\n  2. SIGNAL ZEROING INCREMENTAL IMPACT (ch_fix → both_cwt)")
    print("  " + "-" * 60)
    for task in TASKS:
        for mode, sub, label in [("FT", ft, "finetune"), ("LP", lp, "linear_probe")]:
            c_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "ch_fix")]
            f_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "both_cwt")]
            if not c_row.empty and not f_row.empty:
                c_val = c_row["mean"].values[0]
                f_val = f_row["mean"].values[0]
                print(f"    {task} ({mode}): {c_val:.3f} → {f_val:.3f} (Δ = {f_val - c_val:+.3f})")

    print("\n  3. TOKENIZER COMPARISON (both_cwt vs both_resample)")
    print("  " + "-" * 60)
    for task in TASKS:
        for mode, sub, label in [("FT", ft, "finetune"), ("LP", lp, "linear_probe")]:
            c_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "both_cwt")]
            r_row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == "both_resample")]
            if not c_row.empty and not r_row.empty:
                c_val = c_row["mean"].values[0]
                r_val = r_row["mean"].values[0]
                diff = r_val - c_val
                winner = "ResampleCNN" if diff > 0 else "CWT-CNN" if diff < 0 else "tie"
                print(
                    f"    {task} ({mode}): CWT={c_val:.3f}, Resample={r_val:.3f} "
                    f"(Δ = {diff:+.3f}, {winner})"
                )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    api = wandb.Api()
    print(f"WandB entity: {api.default_entity}")
    print(f"Pretrain project: {PRETRAIN_PROJECT}")
    print(f"Downstream project: {DOWNSTREAM_PROJECT}")

    # 1. Fetch pretraining data
    print("\n" + "=" * 75)
    print("  FETCHING PRETRAINING DATA")
    print("=" * 75)
    pretrain_df = fetch_pretrain_losses(api)
    print_pretrain_summary(pretrain_df)

    # 2. Fetch downstream data
    print("\n\n" + "=" * 75)
    print("  FETCHING DOWNSTREAM DATA")
    print("=" * 75)
    downstream_df = fetch_downstream_results(api)
    valid = downstream_df[downstream_df["best_f1"].notna()]
    print(f"\n  Total downstream runs: {len(downstream_df)}, with metrics: {len(valid)}")

    if not valid.empty:
        print("\n  Per-fold detail:")
        for _, row in valid.sort_values(
            ["task", "mode", "pretrain_run_id", "fold"]
        ).iterrows():
            print(
                f"    {row['task']:<22} {row['mode']:<14} {row['pretrain_run_id']:<16} "
                f"fold{row['fold']}  F1={row['best_f1']:.4f}  "
                f"(best @ epoch {row['best_epoch']}, {row['num_epochs']} epochs, state={row['state']})"
            )

    # 3. Summarize and print
    summary_df = summarize_downstream(valid)
    print_downstream_summary(summary_df)
    print_key_comparisons(summary_df)

    # 4. Generate figures
    print("\n\n" + "=" * 75)
    print("  GENERATING FIGURES")
    print("=" * 75)

    generated = []

    if not pretrain_df.empty:
        generated.append(plot_pretrain_loss_curves(pretrain_df))
        p = plot_pretrain_final_loss_bar(pretrain_df)
        if p:
            generated.append(p)

    if not summary_df.empty:
        generated.append(plot_downstream_comparison(summary_df))
        generated.append(plot_ablation_deltas(summary_df))
        generated.append(plot_tokenizer_comparison(summary_df))

    print(f"\nGenerated {len(generated)} figures:")
    for p in generated:
        print(f"  {p}")
    print("\nDone!")


if __name__ == "__main__":
    main()

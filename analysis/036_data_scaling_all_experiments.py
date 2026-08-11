"""Comprehensive analysis: All 5 data-scaling experiments.

Fetches pretraining (val/loss) and downstream (finetuning + linear probe) results
from WandB for all 12 pretraining runs across 5 experiment groups:
  - Volume Scaling (A1, A2, A3)
  - Diversity Scaling (B1, B2, B3)
  - Controls (C1, C2)
  - Paradigm Diversity (D1, D2, D3)
  - Maximum Data (E1)

Each downstream run uses CWT-CNN + dynamic channel_emb only, evaluated on
3 tasks (Kemp Sleep, PhysioNet MI, Brain Invaders P300) with 3-fold CV.

Usage:
    uv run python analysis/036_data_scaling_all_experiments.py
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

# ── All pretraining runs ──────────────────────────────────────────────────────

PRETRAIN_RUNS = {
    "A1": {
        "name": "pretrain_A1_klinzing_small",
        "group": "DATA_SCALING_VOLUME",
        "label": "A1: Klinzing small (28 rec)",
        "eff_data": 2338,
        "n_datasets": 1,
        "experiment": "volume",
    },
    "A2": {
        "name": "pretrain_A2_klinzing_full",
        "group": "DATA_SCALING_VOLUME",
        "label": "A2: Klinzing full (256 rec)",
        "eff_data": 19484,
        "n_datasets": 1,
        "experiment": "volume",
    },
    "A3": {
        "name": "pretrain_A3_shirazi_only",
        "group": "DATA_SCALING_VOLUME",
        "label": "A3: Shirazi (1342 rec, 129ch)",
        "eff_data": 15163,
        "n_datasets": 1,
        "experiment": "volume",
    },
    "B1": {
        "name": "pretrain_B1_two_dataset",
        "group": "DATA_SCALING_DIVERSITY",
        "label": "B1: Klinzing+Shirazi (2ds)",
        "eff_data": 34647,
        "n_datasets": 2,
        "experiment": "diversity",
    },
    "B2": {
        "name": "pretrain_B2_three_dataset",
        "group": "DATA_SCALING_DIVERSITY",
        "label": "B2: +Pavlov (3ds)",
        "eff_data": 37134,
        "n_datasets": 3,
        "experiment": "diversity",
    },
    "B3": {
        "name": "pretrain_B3_four_dataset",
        "group": "DATA_SCALING_DIVERSITY",
        "label": "B3: +Getzmann (4ds)",
        "eff_data": 48001,
        "n_datasets": 4,
        "experiment": "diversity",
    },
    "C1": {
        "name": "pretrain_C1_headband_only",
        "group": "DATA_SCALING_CONTROLS",
        "label": "C1: Headband only (~6ch)",
        "eff_data": 7292,
        "n_datasets": 1,
        "experiment": "controls",
    },
    "C2": {
        "name": "pretrain_C2_volume_matched",
        "group": "DATA_SCALING_CONTROLS",
        "label": "C2: 3ds vol-matched",
        "eff_data": 19580,
        "n_datasets": 3,
        "experiment": "controls",
    },
    "D1": {
        "name": "pretrain_D1_kochi_only",
        "group": "DATA_SCALING_PARADIGM",
        "label": "D1: Kochi only",
        "eff_data": 2565,
        "n_datasets": 1,
        "experiment": "paradigm",
    },
    "D2": {
        "name": "pretrain_D2_klinzing_kochi",
        "group": "DATA_SCALING_PARADIGM",
        "label": "D2: Klinzing+Kochi",
        "eff_data": 22049,
        "n_datasets": 2,
        "experiment": "paradigm",
    },
    "D3": {
        "name": "pretrain_D3_klinzing_shirazi_kochi",
        "group": "DATA_SCALING_PARADIGM",
        "label": "D3: Klinzing+Shirazi+Kochi",
        "eff_data": 37211,
        "n_datasets": 3,
        "experiment": "paradigm",
    },
    "E1": {
        "name": "pretrain_E1_all_datasets",
        "group": "DATA_SCALING_MAXDATA",
        "label": "E1: All 5 datasets",
        "eff_data": 50566,
        "n_datasets": 5,
        "experiment": "maxdata",
    },
}

# ── Downstream WandB groups ──────────────────────────────────────────────────

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

BEST_BASELINE = {
    "Kemp Sleep": ("POYO CWT-CNN (dynamic)", 0.730),
    "PhysioNet MI": ("EEGNet", 0.887),
    "Brain Invaders P300": ("EEGNet", 0.386),
}

BEST_POYO_BASELINE = {
    "Kemp Sleep": ("CWT-CNN / dynamic", 0.730),
    "PhysioNet MI": ("CWT-CNN / disabled", 0.884),
    "Brain Invaders P300": ("CWT-CNN / dynamic", 0.364),
}

# Colors for the run IDs
RUN_COLORS = {
    "A1": "#1f77b4",
    "A2": "#ff7f0e",
    "A3": "#2ca02c",
    "B1": "#d62728",
    "B2": "#9467bd",
    "B3": "#8c564b",
    "C1": "#e377c2",
    "C2": "#7f7f7f",
    "D1": "#bcbd22",
    "D2": "#17becf",
    "D3": "#aec7e8",
    "E1": "#ff9896",
}

EXPERIMENT_COLORS = {
    "volume": "#4C72B0",
    "diversity": "#DD8452",
    "controls": "#55A868",
    "paradigm": "#C44E52",
    "maxdata": "#8172B2",
}


# ── Data fetching ─────────────────────────────────────────────────────────────


def fetch_pretrain_losses(api: wandb.Api) -> pd.DataFrame:
    """Fetch val/loss curves for all pretraining runs."""
    entity = api.default_entity
    records = []

    for run_id, info in PRETRAIN_RUNS.items():
        print(
            f"  Fetching pretrain {run_id}: {info['name']} (group={info['group']})..."
        )
        runs = api.runs(
            f"{entity}/{PRETRAIN_PROJECT}",
            filters={"group": info["group"], "display_name": info["name"]},
        )
        run_list = list(runs)
        if not run_list:
            print(f"    [WARN] No run found for {info['name']}")
            continue

        run = run_list[0]

        # val/loss and train/loss are logged on different steps, fetch separately
        val_history = run.history(keys=["val/loss"], samples=50000, pandas=True)
        train_history = run.history(
            keys=["train/loss"], samples=50000, pandas=True
        )

        for _, row in val_history.iterrows():
            records.append(
                {
                    "run_id": run_id,
                    "run_name": info["name"],
                    "label": info["label"],
                    "experiment": info["experiment"],
                    "step": row.get("_step"),
                    "val_loss": row.get("val/loss"),
                    "train_loss": None,
                }
            )

        for _, row in train_history.iterrows():
            records.append(
                {
                    "run_id": run_id,
                    "run_name": info["name"],
                    "label": info["label"],
                    "experiment": info["experiment"],
                    "step": row.get("_step"),
                    "val_loss": None,
                    "train_loss": row.get("train/loss"),
                }
            )

        best_val = (
            val_history["val/loss"].dropna().min()
            if "val/loss" in val_history.columns and len(val_history) > 0
            else None
        )
        total_steps = run.summary.get("_step", None)
        if best_val is not None:
            final_val = val_history["val/loss"].dropna().iloc[-1]
            print(
                f"    state={run.state}, steps={total_steps}, best_val_loss={best_val:.4f}, final_val_loss={final_val:.4f}"
            )
        else:
            print(
                f"    state={run.state}, steps={total_steps}, no val/loss data"
            )

    return pd.DataFrame(records)


def fetch_downstream_results(api: wandb.Api) -> pd.DataFrame:
    """Fetch downstream finetuning and linear probe results for all runs."""
    entity = api.default_entity
    records = []

    for (task, mode), group in DOWNSTREAM_GROUPS.items():
        metric_key = METRIC_KEYS[task]
        _ = DOWNSTREAM_RUN_PREFIXES[(task, mode)]
        print(f"\n  Fetching {task} / {mode} (group={group})...")

        runs = api.runs(
            f"{entity}/{DOWNSTREAM_PROJECT}", filters={"group": group}
        )
        run_list = list(runs)
        print(f"    Found {len(run_list)} runs")

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

            records.append(
                {
                    "task": task,
                    "mode": mode,
                    "pretrain_run_id": pretrain_run_id,
                    "pretrain_label": PRETRAIN_RUNS[pretrain_run_id]["label"],
                    "experiment": PRETRAIN_RUNS[pretrain_run_id]["experiment"],
                    "eff_data": PRETRAIN_RUNS[pretrain_run_id]["eff_data"],
                    "n_datasets": PRETRAIN_RUNS[pretrain_run_id]["n_datasets"],
                    "fold": fold,
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "num_epochs": num_epochs,
                    "run_name": run.name,
                    "state": run.state,
                }
            )

    return pd.DataFrame(records)


def summarize_downstream(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std of best F1 across folds per (task, mode, pretrain_run_id)."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(
            [
                "task",
                "mode",
                "pretrain_run_id",
                "pretrain_label",
                "experiment",
                "eff_data",
                "n_datasets",
            ]
        )["best_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_pretrain_loss_curves(pretrain_df: pd.DataFrame) -> list[Path]:
    """Plot pretraining val/loss curves grouped by experiment."""
    paths = []
    experiments = {
        "volume": ("Volume Scaling", ["A1", "A2", "A3"]),
        "diversity": ("Diversity Scaling", ["B1", "B2", "B3"]),
        "controls": ("Controls", ["C1", "C2"]),
        "paradigm": ("Paradigm Diversity", ["D1", "D2", "D3"]),
        "maxdata": ("Maximum Data", ["E1"]),
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for ax_idx, (exp_key, (exp_title, run_ids)) in enumerate(
        experiments.items()
    ):
        ax = axes[ax_idx]
        for rid in run_ids:
            subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(
                subset=["val_loss"]
            )
            if subset.empty:
                continue
            subset = subset.sort_values("step")
            ax.plot(
                subset["step"],
                subset["val_loss"],
                label=rid,
                color=RUN_COLORS[rid],
                linewidth=1.5,
            )

        ax.set_title(exp_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        if ax_idx == 0:
            ax.set_ylabel("Val Loss (MAE reconstruction)")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Pretraining Validation Loss Curves — All Experiments",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "036_pretrain_loss_curves_all.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    paths.append(out)

    # Also a combined overlay plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for rid in PRETRAIN_RUNS:
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(
            subset=["val_loss"]
        )
        if subset.empty:
            continue
        subset = subset.sort_values("step")
        ax.plot(
            subset["step"],
            subset["val_loss"],
            label=f"{rid}: {PRETRAIN_RUNS[rid]['label'][:30]}",
            color=RUN_COLORS[rid],
            linewidth=1.2,
        )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Val Loss (MAE reconstruction)", fontsize=11)
    ax.set_title(
        "Pretraining Validation Loss — All Runs Overlaid",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "036_pretrain_loss_curves_overlay.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    paths.append(out)

    return paths


def plot_pretrain_final_loss_bar(pretrain_df: pd.DataFrame) -> Path:
    """Bar chart of final val/loss for each pretraining run."""
    final_losses = {}
    for rid in PRETRAIN_RUNS:
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(
            subset=["val_loss"]
        )
        if not subset.empty:
            final_losses[rid] = subset.sort_values("step")["val_loss"].iloc[-1]

    if not final_losses:
        return None

    run_ids = list(final_losses.keys())
    losses = [final_losses[r] for r in run_ids]
    colors = [RUN_COLORS[r] for r in run_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(
        range(len(run_ids)),
        losses,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(
        [
            f"{r}\n({PRETRAIN_RUNS[r]['eff_data'] // 1000}k ch·h)"
            for r in run_ids
        ],
        fontsize=8,
    )
    ax.set_ylabel("Final Val Loss", fontsize=11)
    ax.set_title(
        "Pretraining Final Validation Loss — All Runs\n(lower = better reconstruction)",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "036_pretrain_final_loss_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_downstream_grand_comparison(
    summary_df: pd.DataFrame, mode: str
) -> Path:
    """Grouped bar chart: all pretrain runs vs baseline for a given mode (finetune or linear_probe)."""
    mode_label = "Finetuning" if mode == "finetune" else "Linear Probe"
    sub = summary_df[summary_df["mode"] == mode].copy()
    if sub.empty:
        return None

    ordered_runs = [
        r for r in PRETRAIN_RUNS if r in sub["pretrain_run_id"].values
    ]
    if not ordered_runs:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        task_data = sub[sub["task"] == task]

        means = []
        stds = []
        colors = []
        labels = []
        for rid in ordered_runs:
            row = task_data[task_data["pretrain_run_id"] == rid]
            if not row.empty:
                means.append(row["mean"].values[0])
                stds.append(
                    row["std"].values[0]
                    if not np.isnan(row["std"].values[0])
                    else 0
                )
            else:
                means.append(0)
                stds.append(0)
            colors.append(RUN_COLORS[rid])
            labels.append(rid)

        x = np.arange(len(ordered_runs))
        bars = ax.bar(
            x,
            means,
            0.7,
            yerr=stds,
            capsize=3,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            error_kw=dict(lw=1),
        )

        if mode == "finetune":
            _, baseline_val = BEST_BASELINE[task]
            ax.axhline(
                y=baseline_val,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Best baseline ({baseline_val:.3f})",
            )
            _, poyo_val = BEST_POYO_BASELINE[task]
            ax.axhline(
                y=poyo_val,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label=f"Best POYO baseline ({poyo_val:.3f})",
            )

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    rotation=45,
                )

        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        if ax_idx == 0:
            ax.set_ylabel("Best Val F1 (mean across 3 folds)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="lower right")

        if means:
            valid_means = [m for m in means if m > 0]
            if valid_means:
                ymin = min(valid_means) - 0.05
                ymax = max(valid_means) + 0.04
                if mode == "finetune":
                    ymin = min(ymin, baseline_val - 0.03)
                    ymax = max(ymax, baseline_val + 0.03)
                ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"Downstream {mode_label} — All Pretraining Configurations",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"036_downstream_{mode}_grand_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_delta_heatmap(summary_df: pd.DataFrame) -> Path:
    """Heatmap: finetuning delta vs best baseline for all runs x tasks."""
    sub = summary_df[summary_df["mode"] == "finetune"].copy()
    if sub.empty:
        return None

    ordered_runs = [
        r for r in PRETRAIN_RUNS if r in sub["pretrain_run_id"].values
    ]
    if not ordered_runs:
        return None

    fig, ax = plt.subplots(figsize=(8, max(5, len(ordered_runs) * 0.45)))
    delta_matrix = np.full((len(ordered_runs), len(TASKS)), np.nan)

    for j, task in enumerate(TASKS):
        _, baseline_val = BEST_BASELINE[task]
        for i, rid in enumerate(ordered_runs):
            row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == rid)]
            if not row.empty and not np.isnan(row["mean"].values[0]):
                delta_matrix[i, j] = row["mean"].values[0] - baseline_val

    vmax = (
        max(0.03, np.nanmax(np.abs(delta_matrix)))
        if not np.all(np.isnan(delta_matrix))
        else 0.03
    )
    im = ax.imshow(
        delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto"
    )

    for i in range(len(ordered_runs)):
        for j in range(len(TASKS)):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="black" if abs(val) < vmax * 0.7 else "white",
                )

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=10)
    ax.set_yticks(range(len(ordered_runs)))
    ax.set_yticklabels(
        [
            f"{r} ({PRETRAIN_RUNS[r]['eff_data'] // 1000}k)"
            for r in ordered_runs
        ],
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="ΔF1 vs best baseline")

    ax.set_title(
        "Finetuning Transfer Gain vs Best Baseline\n(all pretraining configs)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "036_downstream_finetune_delta_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_scaling_curves(summary_df: pd.DataFrame) -> Path:
    """Plot F1 vs effective data (ch·h) for finetuning, colored by experiment."""
    sub = summary_df[summary_df["mode"] == "finetune"].copy()
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        task_data = sub[sub["task"] == task]

        for exp_name, exp_color in EXPERIMENT_COLORS.items():
            exp_data = task_data[
                task_data["experiment"] == exp_name
            ].sort_values("eff_data")
            if exp_data.empty:
                continue
            ax.errorbar(
                exp_data["eff_data"],
                exp_data["mean"],
                yerr=exp_data["std"].fillna(0),
                marker="o",
                color=exp_color,
                linewidth=1.5,
                capsize=4,
                label=exp_name.capitalize(),
                markersize=6,
            )
            for _, row in exp_data.iterrows():
                ax.annotate(
                    row["pretrain_run_id"],
                    (row["eff_data"], row["mean"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=exp_color,
                )

        _, baseline_val = BEST_BASELINE[task]
        ax.axhline(
            y=baseline_val,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Best baseline ({baseline_val:.3f})",
        )

        ax.set_xlabel("Effective Data (ch·h)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Best Val F1 (finetuning)", fontsize=10)
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Downstream Finetuning F1 vs Effective Pretraining Data",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "036_downstream_f1_vs_effective_data.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_diversity_vs_volume(summary_df: pd.DataFrame) -> Path:
    """Side-by-side comparison: Volume runs vs Diversity runs vs Controls."""
    sub = summary_df[summary_df["mode"] == "finetune"].copy()
    if sub.empty:
        return None

    key_comparisons = {
        "A2 vs C2\n(volume vs diversity\nat ~19.5k ch·h)": ["A2", "C2"],
        "B1 vs A2+A3\n(2ds combined vs\nsingle sources)": ["A2", "A3", "B1"],
        "B2 vs C2\n(3ds full vs\n3ds vol-matched)": ["C2", "B2"],
        "B3 vs E1\n(4ds vs 5ds)": ["B3", "E1"],
    }

    fig, axes = plt.subplots(
        len(key_comparisons), 3, figsize=(18, 4 * len(key_comparisons))
    )

    for row_idx, (comp_title, run_ids) in enumerate(key_comparisons.items()):
        for col_idx, task in enumerate(TASKS):
            ax = (
                axes[row_idx, col_idx]
                if len(key_comparisons) > 1
                else axes[col_idx]
            )
            task_data = sub[sub["task"] == task]

            means = []
            stds = []
            colors = []
            x_labels = []
            for rid in run_ids:
                row = task_data[task_data["pretrain_run_id"] == rid]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(
                        row["std"].values[0]
                        if not np.isnan(row["std"].values[0])
                        else 0
                    )
                else:
                    means.append(0)
                    stds.append(0)
                colors.append(RUN_COLORS[rid])
                x_labels.append(
                    f"{rid}\n({PRETRAIN_RUNS[rid]['eff_data'] // 1000}k)"
                )

            x = np.arange(len(run_ids))
            bars = ax.bar(
                x,
                means,
                0.5,
                yerr=stds,
                capsize=4,
                color=colors,
                edgecolor="white",
                linewidth=0.5,
                error_kw=dict(lw=1),
            )

            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            _, baseline_val = BEST_BASELINE[task]
            ax.axhline(
                y=baseline_val,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"Baseline ({baseline_val:.3f})",
            )

            if col_idx == 0:
                ax.set_ylabel(comp_title, fontsize=9, fontweight="bold")
            if row_idx == 0:
                ax.set_title(task, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if means:
                valid = [m for m in means if m > 0]
                if valid:
                    ax.set_ylim(
                        min(min(valid), baseline_val) - 0.03,
                        max(max(valid), baseline_val) + 0.03,
                    )
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Key Comparisons: Diversity vs Volume vs Controls",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "036_key_comparisons_diversity_vs_volume.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_linear_probe_heatmap(summary_df: pd.DataFrame) -> Path:
    """Heatmap: linear probe F1 (absolute) for all runs x tasks."""
    sub = summary_df[summary_df["mode"] == "linear_probe"].copy()
    if sub.empty:
        return None

    ordered_runs = [
        r for r in PRETRAIN_RUNS if r in sub["pretrain_run_id"].values
    ]
    if not ordered_runs:
        return None

    fig, ax = plt.subplots(figsize=(8, max(5, len(ordered_runs) * 0.45)))
    f1_matrix = np.full((len(ordered_runs), len(TASKS)), np.nan)

    for j, task in enumerate(TASKS):
        for i, rid in enumerate(ordered_runs):
            row = sub[(sub["task"] == task) & (sub["pretrain_run_id"] == rid)]
            if not row.empty and not np.isnan(row["mean"].values[0]):
                f1_matrix[i, j] = row["mean"].values[0]

    im = ax.imshow(
        f1_matrix,
        cmap="YlOrRd",
        aspect="auto",
        vmin=np.nanmin(f1_matrix) if not np.all(np.isnan(f1_matrix)) else 0,
        vmax=np.nanmax(f1_matrix) if not np.all(np.isnan(f1_matrix)) else 1,
    )

    for i in range(len(ordered_runs)):
        for j in range(len(TASKS)):
            val = f1_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white"
                    if val > (np.nanmax(f1_matrix) + np.nanmin(f1_matrix)) / 2
                    else "black",
                )

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=10)
    ax.set_yticks(range(len(ordered_runs)))
    ax.set_yticklabels(
        [
            f"{r} ({PRETRAIN_RUNS[r]['eff_data'] // 1000}k)"
            for r in ordered_runs
        ],
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Linear Probe F1")

    ax.set_title(
        "Linear Probe F1 — Representation Quality\n(frozen backbone, all pretraining configs)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "036_downstream_linear_probe_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def plot_per_experiment_downstream(
    summary_df: pd.DataFrame, experiment: str, run_ids: list[str], title: str
) -> Path:
    """Per-experiment downstream comparison: finetune + linear probe side-by-side."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for mode_idx, mode in enumerate(["finetune", "linear_probe"]):
        mode_label = "Finetuning" if mode == "finetune" else "Linear Probe"
        sub = summary_df[(summary_df["mode"] == mode)].copy()

        for col_idx, task in enumerate(TASKS):
            ax = axes[mode_idx, col_idx]
            task_data = sub[sub["task"] == task]

            available_runs = [
                r for r in run_ids if r in task_data["pretrain_run_id"].values
            ]
            if not available_runs:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            means = []
            stds = []
            colors = []
            for rid in available_runs:
                row = task_data[task_data["pretrain_run_id"] == rid]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(
                        row["std"].values[0]
                        if not np.isnan(row["std"].values[0])
                        else 0
                    )
                else:
                    means.append(0)
                    stds.append(0)
                colors.append(RUN_COLORS[rid])

            x = np.arange(len(available_runs))
            bars = ax.bar(
                x,
                means,
                0.6,
                yerr=stds,
                capsize=4,
                color=colors,
                edgecolor="white",
                linewidth=0.5,
                error_kw=dict(lw=1),
            )

            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            if mode == "finetune":
                _, baseline_val = BEST_BASELINE[task]
                ax.axhline(
                    y=baseline_val,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Best baseline ({baseline_val:.3f})",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"{r}\n({PRETRAIN_RUNS[r]['eff_data'] // 1000}k)"
                    for r in available_runs
                ],
                fontsize=9,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{mode_label}\nBest Val F1", fontsize=10)
            if mode_idx == 0:
                ax.set_title(task, fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=7, loc="lower right")

            if means:
                valid = [m for m in means if m > 0]
                if valid:
                    ymin = min(valid) - 0.04
                    ymax = max(valid) + 0.04
                    if mode == "finetune":
                        ymin = min(ymin, baseline_val - 0.02)
                        ymax = max(ymax, baseline_val + 0.02)
                    ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"{title}\nFinetuning (top) and Linear Probe (bottom)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    slug = experiment.replace(" ", "_").lower()
    out = FIGURES_DIR / f"036_{slug}_downstream.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


# ── Printing ──────────────────────────────────────────────────────────────────


def print_pretrain_summary(pretrain_df: pd.DataFrame) -> None:
    """Print pretrain final loss summary."""
    print("\n" + "=" * 85)
    print("  PRETRAINING SUMMARY — Final Validation Loss")
    print("=" * 85)
    print(
        f"  {'Run':<5} {'Label':<35} {'Eff Data':>10} {'Final Val Loss':>15} {'Best Val Loss':>15}"
    )
    print(f"  {'-' * 5} {'-' * 35} {'-' * 10} {'-' * 15} {'-' * 15}")

    for rid in PRETRAIN_RUNS:
        info = PRETRAIN_RUNS[rid]
        subset = pretrain_df[pretrain_df["run_id"] == rid].dropna(
            subset=["val_loss"]
        )
        if subset.empty:
            print(
                f"  {rid:<5} {info['label']:<35} {info['eff_data']:>10,} {'N/A':>15} {'N/A':>15}"
            )
            continue
        subset = subset.sort_values("step")
        final_loss = subset["val_loss"].iloc[-1]
        best_loss = subset["val_loss"].min()
        print(
            f"  {rid:<5} {info['label']:<35} {info['eff_data']:>10,} {final_loss:>15.4f} {best_loss:>15.4f}"
        )


def print_downstream_summary(summary_df: pd.DataFrame) -> None:
    """Print comprehensive downstream summary tables."""
    for mode in ["finetune", "linear_probe"]:
        mode_label = "FINETUNING" if mode == "finetune" else "LINEAR PROBE"
        sub = summary_df[summary_df["mode"] == mode]

        print(f"\n\n{'=' * 100}")
        print(f"  DOWNSTREAM {mode_label} — All Runs")
        print(f"{'=' * 100}")

        for task in TASKS:
            task_data = sub[sub["task"] == task].sort_values(
                "mean", ascending=False
            )
            if task_data.empty:
                continue

            baseline_name, baseline_val = BEST_BASELINE[task]
            print(
                f"\n  {task} (baseline: {baseline_name} = {baseline_val:.3f})"
            )
            print(
                f"  {'Run':<5} {'Label':<35} {'Mean F1':>8} {'± Std':>8} {'N':>3} {'Δ Baseline':>12}"
            )
            print(
                f"  {'-' * 5} {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 3} {'-' * 12}"
            )

            for _, row in task_data.iterrows():
                rid = row["pretrain_run_id"]
                label = row["pretrain_label"][:35]
                mean_f1 = row["mean"]
                std_f1 = row["std"] if not np.isnan(row["std"]) else 0
                n = int(row["count"])
                delta = mean_f1 - baseline_val if mode == "finetune" else None
                delta_str = (
                    f"{delta:>+12.3f}" if delta is not None else f"{'':>12}"
                )
                print(
                    f"  {rid:<5} {label:<35} {mean_f1:>8.3f} {std_f1:>8.3f} {n:>3} {delta_str}"
                )


def print_per_experiment_summary(summary_df: pd.DataFrame) -> None:
    """Print experiment-specific summaries for each of the 5 experiments."""
    experiments = {
        "volume": {
            "title": "VOLUME SCALING (A1, A2, A3)",
            "runs": ["A1", "A2", "A3"],
            "comparisons": [
                ("A1 → A2", "10x volume increase, same source"),
                (
                    "A2 vs A3",
                    "Similar effective data, Shirazi has 129ch vs Klinzing ~10ch",
                ),
            ],
        },
        "diversity": {
            "title": "DIVERSITY SCALING (B1, B2, B3)",
            "runs": ["B1", "B2", "B3"],
            "comparisons": [
                ("B1 → B2", "Adding Pavlov (+2,488 ch·h, low-density)"),
                ("B2 → B3", "Adding Getzmann (+10,867 ch·h, 64ch resting)"),
            ],
        },
        "controls": {
            "title": "CONTROLS (C1, C2)",
            "runs": ["C1", "C2"],
            "comparisons": [
                ("C1 vs A2", "Headband-only vs full Klinzing"),
                (
                    "C2 vs A2",
                    "3-source vol-matched vs single-source at ~19.5k ch·h",
                ),
            ],
        },
        "paradigm": {
            "title": "PARADIGM DIVERSITY (D1, D2, D3)",
            "runs": ["D1", "D2", "D3"],
            "comparisons": [
                ("D2 vs A2", "Klinzing+Kochi vs Klinzing alone"),
                ("D3 vs B1", "Klinzing+Shirazi+Kochi vs Klinzing+Shirazi"),
            ],
        },
        "maxdata": {
            "title": "MAXIMUM DATA (E1)",
            "runs": ["E1"],
            "comparisons": [
                ("E1 vs B3", "5 datasets vs 4 datasets"),
            ],
        },
    }

    ft = summary_df[summary_df["mode"] == "finetune"]

    for exp_key, exp_info in experiments.items():
        print(f"\n\n{'#' * 90}")
        print(f"  {exp_info['title']}")
        print(f"{'#' * 90}")

        for task in TASKS:
            _, baseline_val = BEST_BASELINE[task]
            print(f"\n  {task}:")
            for rid in exp_info["runs"]:
                row = ft[(ft["task"] == task) & (ft["pretrain_run_id"] == rid)]
                if not row.empty:
                    m = row["mean"].values[0]
                    s = (
                        row["std"].values[0]
                        if not np.isnan(row["std"].values[0])
                        else 0
                    )
                    d = m - baseline_val
                    print(
                        f"    {rid}: {m:.3f} ± {s:.3f} (Δ baseline: {d:+.3f})"
                    )
                else:
                    print(f"    {rid}: N/A")

        print("\n  Key comparisons:")
        for comp_label, comp_desc in exp_info["comparisons"]:
            print(f"    {comp_label}: {comp_desc}")
            run_ids = [
                r.strip() for r in comp_label.replace("→", "vs").split("vs")
            ]
            for task in TASKS:
                vals = {}
                for rid in run_ids:
                    row = ft[
                        (ft["task"] == task) & (ft["pretrain_run_id"] == rid)
                    ]
                    if not row.empty:
                        vals[rid] = row["mean"].values[0]
                if len(vals) == 2:
                    rids = list(vals.keys())
                    diff = vals[rids[1]] - vals[rids[0]]
                    print(
                        f"      {task}: {rids[0]}={vals[rids[0]]:.3f}, {rids[1]}={vals[rids[1]]:.3f}, Δ={diff:+.3f}"
                    )


# ── Main ──────────────────────────────────────────────────────────────────────


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
    print(
        f"\n  Total downstream runs: {len(downstream_df)}, with metrics: {len(valid)}"
    )

    # Print per-fold detail
    if not valid.empty:
        print("\n  Per-fold detail:")
        for _, row in valid.sort_values(
            ["task", "mode", "pretrain_run_id", "fold"]
        ).iterrows():
            print(
                f"    {row['task']:<22} {row['mode']:<14} {row['pretrain_run_id']:<4} "
                f"fold{row['fold']}  F1={row['best_f1']:.4f}  (best @ epoch {row['best_epoch']}, "
                f"{row['num_epochs']} epochs, state={row['state']})"
            )

    # 3. Summarize
    summary_df = summarize_downstream(valid)
    print_downstream_summary(summary_df)
    print_per_experiment_summary(summary_df)

    # 4. Generate all figures
    print("\n\n" + "=" * 75)
    print("  GENERATING FIGURES")
    print("=" * 75)

    generated = []

    # Pretraining loss curves
    if not pretrain_df.empty:
        generated.extend(plot_pretrain_loss_curves(pretrain_df))
        p = plot_pretrain_final_loss_bar(pretrain_df)
        if p:
            generated.append(p)

    # Grand downstream comparisons
    if not summary_df.empty:
        for mode in ["finetune", "linear_probe"]:
            p = plot_downstream_grand_comparison(summary_df, mode)
            if p:
                generated.append(p)

        # Delta heatmap
        p = plot_delta_heatmap(summary_df)
        if p:
            generated.append(p)

        # Scaling curves
        p = plot_scaling_curves(summary_df)
        if p:
            generated.append(p)

        # Key comparisons
        p = plot_diversity_vs_volume(summary_df)
        if p:
            generated.append(p)

        # Linear probe heatmap
        p = plot_linear_probe_heatmap(summary_df)
        if p:
            generated.append(p)

        # Per-experiment detailed plots
        per_exp = {
            "volume": (["A1", "A2", "A3"], "Volume Scaling: A1 → A2 → A3"),
            "diversity": (
                ["B1", "B2", "B3"],
                "Diversity Scaling: B1 → B2 → B3",
            ),
            "controls": (
                ["C1", "C2"],
                "Controls: C1 (Headband) & C2 (Vol-Matched)",
            ),
            "paradigm": (
                ["D1", "D2", "D3"],
                "Paradigm Diversity: D1 → D2 → D3",
            ),
            "maxdata": (["E1"], "Maximum Data: E1 (All 5 Datasets)"),
        }
        for exp_key, (run_ids, title) in per_exp.items():
            p = plot_per_experiment_downstream(
                summary_df, exp_key, run_ids, title
            )
            if p:
                generated.append(p)

    print(f"\n\nGenerated {len(generated)} figures:")
    for p in generated:
        print(f"  {p}")
    print("\nDone!")


if __name__ == "__main__":
    main()

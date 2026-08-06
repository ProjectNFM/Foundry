"""PhysioNet Motor Imagery from-scratch baselines (exp 025).

Compares 5 model conditions × 3 folds for binary MI classification.

WandB project: foundry_finetuning
Group: PHYSIONET_MI_BASELINES

Conditions:
  - EEGNet
  - POYO CWT-CNN, channel_emb disabled
  - POYO CWT-CNN, channel_emb dynamic
  - POYO ResampleCNN, channel_emb disabled
  - POYO ResampleCNN, channel_emb dynamic

Usage:
    uv run python analysis/025_physionet_mi_baselines.py
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    fetch_metric_history,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()
WANDB_GROUP = "PHYSIONET_MI_BASELINES"
FIGURES_DIR = figures_dir(__file__)

VAL_F1 = "val/motor_imagery_binary_f1"
VAL_ACC = "val/motor_imagery_binary_acc"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"

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


def parse_run_name(name: str) -> tuple[str, int] | None:
    """Parse a run name into (condition_key, fold_number).

    Expected patterns:
      physionet_mi_eegnet_fold0
      physionet_mi_per_channel_cwt_cnn_ch-disabled_fold1
      physionet_mi_per_channel_resample_cnn_ch-dynamic_fold2
    """
    if "eegnet" in name:
        for f in FOLDS:
            if f"fold{f}" in name:
                return "eegnet", f
        return None

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

    for f in FOLDS:
        if f"fold{f}" in name:
            return f"{tok}_{ch}", f

    return None


def fetch_group_runs() -> dict[str, str]:
    """Fetch all run IDs from the WandB group, keyed by condition_fold."""
    api = wandb.Api()
    entity = WANDB_ENTITY or api.default_entity
    runs = api.runs(
        f"{entity}/{WANDB_PROJECT}",
        filters={"group": WANDB_GROUP, "state": "finished"},
    )

    run_map = {}
    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"  WARNING: Could not parse run name: {run.name} ({run.id})")
            continue
        cond, fold = parsed
        key = f"{cond}_f{fold}"
        run_map[key] = run.id
        print(f"  Mapped: {run.name} -> {key} ({run.id})")

    return run_map


def fetch_all_data(run_map: dict[str, str]) -> dict[str, dict]:
    """Fetch best val F1 for every run via metric history."""
    results = {}
    for key, run_id in run_map.items():
        print(f"  Fetching {key} ({run_id})...")
        try:
            epoch_df = fetch_metric_history(
                run_id,
                [VAL_F1, VAL_LOSS],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                aggregate_epoch=True,
            )
            best_f1 = best_loss = best_ep = max_ep = None
            if not epoch_df.empty and VAL_F1 in epoch_df.columns:
                valid = epoch_df.dropna(subset=[VAL_F1])
                if not valid.empty:
                    best_idx = valid[VAL_F1].idxmax()
                    best_row = valid.loc[best_idx]
                    best_f1 = float(best_row[VAL_F1])
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
                "best_loss": best_loss,
                "best_ep": best_ep,
                "max_ep": max_ep,
                "epoch_df": epoch_df,
            }
        except Exception as e:
            print(f"    WARNING: {e}")
            results[key] = {"run_id": run_id, "best_f1": None, "epoch_df": None}
    return results


def get_condition_stats(
    data: dict, condition: str
) -> tuple[float, float, list[float]]:
    """Return (mean, std, values) for a condition across folds."""
    vals = []
    for f in FOLDS:
        key = f"{condition}_f{f}"
        if key in data and data[key]["best_f1"] is not None:
            vals.append(data[key]["best_f1"])
    if not vals:
        return 0.0, 0.0, []
    return float(np.mean(vals)), float(np.std(vals)), vals


def print_summary(data: dict) -> None:
    """Print comprehensive summary table."""
    print(f"\n{'=' * 90}")
    print("  PhysioNet Motor Imagery From-Scratch Baselines")
    print(f"{'=' * 90}")

    print(
        f"\n{'Condition':<20s}  {'Mean F1':>8s}  {'Std':>6s}  "
        f"{'Fold 0':>8s}  {'Fold 1':>8s}  {'Fold 2':>8s}  "
        f"{'Best Ep (f0)':>12s}"
    )
    print("-" * 85)
    for cond in CONDITIONS:
        mean, std, vals = get_condition_stats(data, cond)
        fold_strs = [f"{v:.4f}" if v else "  N/A " for v in vals]
        while len(fold_strs) < 3:
            fold_strs.append("  N/A ")
        key_f0 = f"{cond}_f0"
        best_ep = data.get(key_f0, {}).get("best_ep", "N/A")
        print(
            f"{CONDITION_SHORT[cond]:<20s}  {mean:.4f}  {std:.4f}  "
            f"{'  '.join(fold_strs)}  {str(best_ep):>12s}"
        )

    print("\n── Tokenizer Comparison (CWT-CNN vs ResampleCNN) ──")
    print(
        f"{'Channel Emb':<15s}  {'CWT F1':>8s}  {'RCNN F1':>8s}  {'Δ (pp)':>8s}"
    )
    print("-" * 45)
    for ch, ch_label in [("dis", "Disabled"), ("dyn", "Dynamic")]:
        cwt_mean, _, _ = get_condition_stats(data, f"cwt_{ch}")
        rcnn_mean, _, _ = get_condition_stats(data, f"rcnn_{ch}")
        delta = (cwt_mean - rcnn_mean) * 100
        print(f"{ch_label:<15s}  {cwt_mean:.4f}  {rcnn_mean:.4f}  {delta:+.1f}")

    print("\n── Channel Embedding Effect (Disabled vs Dynamic) ──")
    print(
        f"{'Tokenizer':<15s}  {'Disabled F1':>11s}  {'Dynamic F1':>10s}  {'Δ (pp)':>8s}"
    )
    print("-" * 50)
    for tok, tok_label in [("cwt", "CWT-CNN"), ("rcnn", "ResampleCNN")]:
        dis_mean, _, _ = get_condition_stats(data, f"{tok}_dis")
        dyn_mean, _, _ = get_condition_stats(data, f"{tok}_dyn")
        delta = (dyn_mean - dis_mean) * 100
        print(
            f"{tok_label:<15s}  {dis_mean:.4f}     {dyn_mean:.4f}    {delta:+.1f}"
        )


def plot_main_results(data: dict) -> None:
    """Bar chart: all conditions with fold error bars."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(CONDITIONS))
    width = 0.6

    means = []
    stds = []
    for cond in CONDITIONS:
        m, s, _ = get_condition_stats(data, cond)
        means.append(m)
        stds.append(s)

    bars = ax.bar(
        x,
        means,
        width,
        yerr=stds,
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
        error_kw={"linewidth": 1.5},
    )

    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Macro F1 (3-fold mean ± std)", fontsize=11)
    ax.set_title(
        "PhysioNet Motor Imagery Baselines: All Conditions",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(
        0.5, color="gray", linestyle="--", alpha=0.5, label="Chance level"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    y_vals = [m for m in means if m > 0]
    if y_vals:
        ax.set_ylim(max(0, min(y_vals) - 0.1), max(y_vals) + 0.06)

    plt.tight_layout()
    out = FIGURES_DIR / "025_physionet_mi_main_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_f1_curves(data: dict) -> None:
    """Validation F1 learning curves for all conditions (fold 0)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for cond in CONDITIONS:
        key = f"{cond}_f0"
        if key not in data or data[key].get("epoch_df") is None:
            continue
        edf = data[key]["epoch_df"]
        if edf.empty or VAL_F1 not in edf.columns:
            continue
        valid = edf.dropna(subset=[VAL_F1]).sort_values("epoch")
        if valid.empty:
            continue

        mean_f1, _, _ = get_condition_stats(data, cond)
        ax.plot(
            valid["epoch"],
            valid[VAL_F1],
            linewidth=2,
            color=CONDITION_COLORS[cond],
            label=f"{CONDITION_SHORT[cond]} (best={mean_f1:.3f})",
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation F1 (binary)", fontsize=11)
    ax.set_title(
        "PhysioNet MI: Validation F1 Learning Curves (fold 0)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "025_physionet_mi_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_fold_variance(data: dict) -> None:
    """Strip plot showing individual fold results for each condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cond in enumerate(CONDITIONS):
        _, _, vals = get_condition_stats(data, cond)
        if vals:
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                [i + j for j in jitter],
                vals,
                color=CONDITION_COLORS[cond],
                s=120,
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
        [CONDITION_SHORT[c] for c in CONDITIONS], fontsize=10, rotation=15
    )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title(
        "PhysioNet MI: Cross-Fold Variance",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "025_physionet_mi_fold_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 70)
    print("  PhysioNet Motor Imagery From-Scratch Baselines Analysis")
    print("=" * 70)

    print("\n── Fetching runs from WandB group ──")
    run_map = fetch_group_runs()
    print(f"\n  Found {len(run_map)} runs in group '{WANDB_GROUP}'")

    if not run_map:
        print("  ERROR: No runs found. Check group name and entity.")
        return

    expected = {f"{c}_f{f}" for c in CONDITIONS for f in FOLDS}
    missing = expected - set(run_map.keys())
    if missing:
        print(f"  WARNING: Missing runs for: {sorted(missing)}")

    print("\n── Fetching metric histories ──")
    data = fetch_all_data(run_map)

    print_summary(data)

    print("\n── Generating plots ──")
    plot_main_results(data)
    plot_f1_curves(data)
    plot_fold_variance(data)

    print("\nDone.")


if __name__ == "__main__":
    main()

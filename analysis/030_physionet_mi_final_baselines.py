"""PhysioNet Motor Imagery Final Baselines — All Conditions × 3 Folds.

Cross-architecture comparison of EEGNet and 4 POYO conditions (CWT-CNN/ResampleCNN
× disabled/dynamic channel embedding) with HP-tuned settings on 3 intersubject folds.

WandB project: foundry_finetuning
Groups:
  - PHYSIONET_MI_POYO_BASELINES_3FOLD  (12 runs: 4 conditions × 3 folds, SLURM 10282387)
  - PHYSIONET_MI_EEGNET_FINAL_3FOLD    (3 runs: 1 condition × 3 folds, SLURM 10282876)

Usage:
    uv run python analysis/030_physionet_mi_final_baselines.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from analysis._wandb_utils import (
    default_entity,
    fetch_metric_history,
    figures_dir,
)

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()
POYO_GROUP = "PHYSIONET_MI_POYO_BASELINES_3FOLD"
EEGNET_GROUP = "PHYSIONET_MI_EEGNET_FINAL_3FOLD"
FIGURES_DIR = figures_dir(__file__)

VAL_F1 = "val/motor_imagery_binary_f1"
VAL_ACC = "val/motor_imagery_binary_acc"
VAL_AUROC = "val/motor_imagery_binary_auroc"
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


def parse_poyo_run_name(name: str) -> tuple[str, int] | None:
    """Parse a POYO run name into (condition_key, fold_number).

    Expected patterns:
      physionet_mi_per_channel_cwt_cnn_ch-disabled_fold0
      physionet_mi_per_channel_cwt_cnn_ch-dynamic_fold1
      physionet_mi_per_channel_resample_cnn_ch-disabled_fold2
      physionet_mi_per_channel_resample_cnn_ch-dynamic_fold0
    """
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


def parse_eegnet_run_name(name: str) -> tuple[str, int] | None:
    """Parse an EEGNet run name into ("eegnet", fold_number)."""
    if "eegnet" not in name.lower():
        return None
    for f in FOLDS:
        if f"fold{f}" in name:
            return "eegnet", f
    return None


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


def fetch_group_runs(group: str, parser) -> dict[str, dict]:
    """Fetch all runs from a WandB group, return dict keyed by condition_foldN."""
    api = wandb.Api()
    entity = WANDB_ENTITY or api.default_entity
    runs = api.runs(
        f"{entity}/{WANDB_PROJECT}",
        filters={"group": group},
    )

    run_map = {}
    for run in runs:
        parsed = parser(run.name)
        if parsed is None:
            print(f"  WARNING: Could not parse run name: {run.name} ({run.id})")
            continue
        cond, fold = parsed
        key = f"{cond}_f{fold}"

        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "val_f1": unwrap(run.summary.get(VAL_F1), "max"),
            "val_acc": unwrap(run.summary.get(VAL_ACC), "max"),
            "val_auroc": unwrap(run.summary.get(VAL_AUROC), "max"),
            "val_loss": unwrap(run.summary.get(VAL_LOSS), "min"),
            "epoch": run.summary.get("epoch", 0),
        }
        run_map[key] = run_data
        print(f"  Mapped: {run.name} -> {key} ({run.id}, state={run.state})")

    return run_map


def fetch_f1_curves(run_map: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Fetch epoch-level val F1 curves for fold 0 of each condition."""
    curves = {}
    for key, info in run_map.items():
        if not key.endswith("_f0"):
            continue
        try:
            df = fetch_metric_history(
                info["run_id"],
                [VAL_F1, VAL_LOSS],
                WANDB_PROJECT,
                WANDB_ENTITY,
                x_axis="epoch",
                aggregate_epoch=True,
            )
            curves[key] = df
        except Exception as e:
            print(f"    WARNING fetching curve for {key}: {e}")
    return curves


def get_condition_stats(
    data: dict, condition: str
) -> tuple[float, float, list[float]]:
    vals = []
    for f in FOLDS:
        key = f"{condition}_f{f}"
        if key in data and data[key].get("val_f1", 0) > 0:
            vals.append(data[key]["val_f1"])
    if not vals:
        return 0.0, 0.0, []
    return float(np.mean(vals)), float(np.std(vals)), vals


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 100}")
    print(
        "  PhysioNet Motor Imagery Final Baselines — All Conditions × 3 Folds"
    )
    print(f"{'=' * 100}")

    # Per-fold table
    header = (
        f"{'Condition':<16s}  {'Mean F1':>8s}  {'Std':>6s}  "
        f"{'Fold 0':>8s}  {'Fold 1':>8s}  {'Fold 2':>8s}  "
        f"{'Mean Acc':>8s}  {'Mean AUROC':>10s}  {'Epochs (f0)':>11s}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for cond in CONDITIONS:
        mean_f1, std_f1, fold_vals = get_condition_stats(data, cond)
        fold_strs = []
        for f in FOLDS:
            key = f"{cond}_f{f}"
            if key in data and data[key].get("val_f1", 0) > 0:
                fold_strs.append(f"{data[key]['val_f1']:.4f}")
            else:
                fold_strs.append("  N/A ")

        # Mean acc and auroc
        accs = [
            data[f"{cond}_f{f}"]["val_acc"]
            for f in FOLDS
            if f"{cond}_f{f}" in data
            and data[f"{cond}_f{f}"].get("val_acc", 0) > 0
        ]
        aurocs = [
            data[f"{cond}_f{f}"]["val_auroc"]
            for f in FOLDS
            if f"{cond}_f{f}" in data
            and data[f"{cond}_f{f}"].get("val_auroc", 0) > 0
        ]
        mean_acc = np.mean(accs) if accs else 0.0
        mean_auroc = np.mean(aurocs) if aurocs else 0.0

        key_f0 = f"{cond}_f0"
        epochs = data.get(key_f0, {}).get("epoch", "N/A")

        print(
            f"{CONDITION_SHORT[cond]:<16s}  {mean_f1:.4f}  {std_f1:.4f}  "
            f"{'  '.join(fold_strs)}  "
            f"{mean_acc:.4f}  {mean_auroc:>10.4f}  {str(epochs):>11s}"
        )

    # Tokenizer comparison (POYO only)
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

    # Channel embedding effect
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

    # POYO vs EEGNet
    eeg_mean, eeg_std, _ = get_condition_stats(data, "eegnet")
    best_poyo_cond = max(
        [c for c in CONDITIONS if c != "eegnet"],
        key=lambda c: get_condition_stats(data, c)[0],
    )
    best_poyo_mean, _, _ = get_condition_stats(data, best_poyo_cond)

    print("\n── POYO vs EEGNet ──")
    print(
        f"  Best POYO condition:  {CONDITION_SHORT[best_poyo_cond]} "
        f"({best_poyo_mean:.4f} mean F1)"
    )
    print(f"  EEGNet:               {eeg_mean:.4f} ± {eeg_std:.4f}")
    delta = (best_poyo_mean - eeg_mean) * 100
    print(f"  Δ (best POYO − EEGNet): {delta:+.1f} pp")


def plot_main_results(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(CONDITIONS))
    width = 0.6

    means, stds = [], []
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
        "PhysioNet MI Final Baselines: All Conditions × 3 Folds",
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
    out = FIGURES_DIR / "030_physionet_mi_final_main_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_f1_curves(data: dict, curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for cond in CONDITIONS:
        key = f"{cond}_f0"
        if key not in curves or curves[key].empty:
            continue
        edf = curves[key]
        if VAL_F1 not in edf.columns:
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
            label=f"{CONDITION_SHORT[cond]} (mean={mean_f1:.3f})",
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation F1 (binary)", fontsize=11)
    ax.set_title(
        "PhysioNet MI Final Baselines: Val F1 Learning Curves (fold 0)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "030_physionet_mi_final_f1_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_fold_variance(data: dict) -> None:
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
        [CONDITION_SHORT[c] for c in CONDITIONS],
        fontsize=10,
        rotation=15,
    )
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title(
        "PhysioNet MI Final Baselines: Cross-Fold Variance",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "030_physionet_mi_final_fold_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 70)
    print("  PhysioNet MI Final Baselines Analysis")
    print("=" * 70)

    # Fetch POYO runs
    print(f"\n── Fetching POYO runs from group '{POYO_GROUP}' ──")
    poyo_data = fetch_group_runs(POYO_GROUP, parse_poyo_run_name)
    print(f"  Found {len(poyo_data)} POYO runs")

    # Fetch EEGNet runs
    print(f"\n── Fetching EEGNet runs from group '{EEGNET_GROUP}' ──")
    eegnet_data = fetch_group_runs(EEGNET_GROUP, parse_eegnet_run_name)
    print(f"  Found {len(eegnet_data)} EEGNet runs")

    # Merge
    data = {**poyo_data, **eegnet_data}

    # Check for missing runs
    expected = {f"{c}_f{f}" for c in CONDITIONS for f in FOLDS}
    missing = expected - set(data.keys())
    if missing:
        print(f"\n  WARNING: Missing runs for: {sorted(missing)}")

    not_finished = {
        k: v for k, v in data.items() if v.get("state") != "finished"
    }
    if not_finished:
        print("\n  WARNING: Non-finished runs:")
        for k, v in not_finished.items():
            print(f"    {k}: state={v['state']}")

    # Summary table
    print_summary(data)

    # Fetch F1 curves for fold 0
    print("\n── Fetching F1 learning curves (fold 0) ──")
    curves = fetch_f1_curves(data)

    # Generate plots
    print("\n── Generating plots ──")
    plot_main_results(data)
    plot_f1_curves(data, curves)
    plot_fold_variance(data)

    print("\nDone.")


if __name__ == "__main__":
    main()

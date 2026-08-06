"""PhysioNet Motor Imagery HP Search Analysis.

Fetches EEGNet runs from WandB group PHYSIONET_MI_HP_SEARCH_EEGNET (50 runs)
and the 4 earlier runs from PHYSIONET_MI_HP_SEARCH. Also reports on
the 12 crashed POYO runs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "poyo-eeg"
PROJECT = "foundry_finetuning"
EEGNET_GROUP = "PHYSIONET_MI_HP_SEARCH_EEGNET"
MAIN_GROUP = "PHYSIONET_MI_HP_SEARCH"

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_eegnet_name_v2(name: str) -> dict:
    """Parse the newer naming: physionet_mi_hp_eegnet_lr{}_wd{}_cw-{}_F1-{}"""
    m = re.match(
        r"physionet_mi_hp_eegnet_lr([\d.]+)_wd([\d.]+)_cw-(\w+)_F1-(\d+)", name
    )
    if not m:
        return {}
    return {
        "model": "EEGNet",
        "lr": float(m.group(1)),
        "weight_decay": float(m.group(2)),
        "class_weights": m.group(3),
        "F1": int(m.group(4)),
    }


def parse_eegnet_name_v1(name: str) -> dict:
    """Parse older naming: physionet_mi_hp_eegnet_lr{}_wd{}_F1-{}"""
    m = re.match(r"physionet_mi_hp_eegnet_lr([\d.]+)_wd([\d.]+)_F1-(\d+)", name)
    if not m:
        return {}
    return {
        "model": "EEGNet",
        "lr": float(m.group(1)),
        "weight_decay": float(m.group(2)),
        "class_weights": "unknown",
        "F1": int(m.group(3)),
    }


def unwrap(val, key="max"):
    if hasattr(val, "get"):
        return float(val.get(key, 0.0))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def fetch_all_runs():
    api = wandb.Api()

    records = []

    # Main group (50 EEGNet runs)
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": EEGNET_GROUP})
    for run in runs:
        if run.state != "finished":
            continue
        params = parse_eegnet_name_v2(run.name)
        if not params:
            params = parse_eegnet_name_v1(run.name)
        if not params:
            continue
        params["run_id"] = run.id
        params["run_name"] = run.name
        params["group"] = EEGNET_GROUP
        params["val_f1"] = unwrap(
            run.summary.get("val/motor_imagery_binary_f1"), "max"
        )
        params["val_auroc"] = unwrap(
            run.summary.get("val/motor_imagery_binary_auroc"), "max"
        )
        params["val_acc"] = unwrap(
            run.summary.get("val/motor_imagery_binary_acc"), "max"
        )
        params["val_loss"] = unwrap(
            run.summary.get("val/motor_imagery_binary_loss"), "min"
        )
        params["val_precision"] = unwrap(
            run.summary.get("val/motor_imagery_binary_precision"), "max"
        )
        params["val_recall"] = unwrap(
            run.summary.get("val/motor_imagery_binary_recall"), "max"
        )
        params["epoch"] = run.summary.get("epoch", 0)
        records.append(params)

    # Crashed POYO runs (for reporting)
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": MAIN_GROUP})
    poyo_crashed = sum(1 for r in runs if r.state == "failed")

    return pd.DataFrame(records), poyo_crashed


def print_summary(df: pd.DataFrame, poyo_crashed: int):
    print("\n" + "=" * 90)
    print("PHYSIONET MI HP SEARCH — RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n  POYO CWT-CNN: ALL {poyo_crashed} runs CRASHED")
    print(
        "    Error: RuntimeError: Trying to resize storage that is not resizable"
    )
    print("    (collation failure in DataLoader — variable tensor sizes)")

    print(f"\n{'─' * 90}")
    print(f"  EEGNet ({len(df)} finished runs)")
    print(f"{'─' * 90}")

    cols = [
        "run_name",
        "lr",
        "weight_decay",
        "class_weights",
        "F1",
        "val_f1",
        "val_auroc",
        "val_acc",
        "epoch",
    ]
    sorted_df = df.sort_values("val_f1", ascending=False)
    print(sorted_df[cols].head(20).to_string(index=False, float_format="%.4f"))
    if len(df) > 20:
        print(f"  ... ({len(df) - 20} more rows)")

    # Overall best
    print(f"\n{'=' * 90}")
    print("TOP 10 OVERALL (by val F1):")
    print("=" * 90)
    top10 = sorted_df.head(10)
    print(
        top10[
            [
                "run_name",
                "val_f1",
                "val_auroc",
                "val_acc",
                "val_precision",
                "val_recall",
                "epoch",
            ]
        ].to_string(index=False, float_format="%.4f")
    )

    # Baseline comparison
    baseline_f1 = 0.873
    best_f1 = sorted_df.iloc[0]["val_f1"]
    print(f"\n  Baseline EEGNet F1: {baseline_f1:.3f}")
    print(f"  Best HP-tuned F1:   {best_f1:.3f}")
    print(
        f"  Improvement:        {best_f1 - baseline_f1:+.3f} ({(best_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%)"
    )

    # Effect of hyperparameters
    print(f"\n{'=' * 90}")
    print("MEAN VAL F1 BY HYPERPARAMETER:")
    print("=" * 90)

    print("\n  By Learning Rate:")
    print(
        df.groupby("lr")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    print("\n  By Weight Decay:")
    print(
        df.groupby("weight_decay")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    print("\n  By Class Weights:")
    print(
        df.groupby("class_weights")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    print("\n  By F1 (spatial filters):")
    print(
        df.groupby("F1")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    # Interaction: lr × weight_decay
    print("\n  LR × Weight Decay (mean F1):")
    pivot = df.pivot_table(
        values="val_f1", index="lr", columns="weight_decay", aggfunc="mean"
    )
    print(pivot.to_string(float_format="%.4f"))


def plot_hp_heatmap(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LR × Weight Decay
    pivot = df.pivot_table(
        values="val_f1", index="lr", columns="weight_decay", aggfunc="mean"
    )
    im = axes[0].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f"{v:.0e}" for v in pivot.index])
    axes[0].set_xlabel("Weight Decay")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("EEGNet: Val F1 by LR × Weight Decay")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            axes[0].text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=9
            )
    plt.colorbar(im, ax=axes[0], label="Val F1")

    # LR × F1 filters
    pivot2 = df.pivot_table(
        values="val_f1", index="lr", columns="F1", aggfunc="mean"
    )
    im2 = axes[1].imshow(pivot2.values, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(pivot2.columns)))
    axes[1].set_xticklabels(pivot2.columns)
    axes[1].set_yticks(range(len(pivot2.index)))
    axes[1].set_yticklabels([f"{v:.0e}" for v in pivot2.index])
    axes[1].set_xlabel("F1 (spatial filters)")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("EEGNet: Val F1 by LR × F1")
    for i in range(len(pivot2.index)):
        for j in range(len(pivot2.columns)):
            val = pivot2.values[i, j]
            axes[1].text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=9
            )
    plt.colorbar(im2, ax=axes[1], label="Val F1")

    plt.tight_layout()
    path = FIGURES_DIR / "027_physionet_mi_hp_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_class_weights_effect(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    cw_groups = df.groupby(["lr", "class_weights"])["val_f1"].mean().unstack()
    if "auto" in cw_groups.columns and "none" in cw_groups.columns:
        x = range(len(cw_groups.index))
        ax.bar(
            [i - 0.2 for i in x],
            cw_groups["none"],
            0.35,
            label="No class weights",
            color="#2196F3",
        )
        ax.bar(
            [i + 0.2 for i in x],
            cw_groups["auto"],
            0.35,
            label="Auto class weights",
            color="#FF9800",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v:.0e}" for v in cw_groups.index])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Mean Val F1")
        ax.set_title("PhysioNet MI EEGNet: Class Weights Effect")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / "027_physionet_mi_hp_class_weights.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_top_configs(df: pd.DataFrame):
    top10 = df.sort_values("val_f1", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(top10))
    bars = ax.barh(y_pos, top10["val_f1"], color="#4CAF50", alpha=0.8)

    ax.axvline(
        x=0.873,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Baseline (0.873)",
    )
    ax.set_yticks(y_pos)
    short_names = [
        n.replace("physionet_mi_hp_eegnet_", "") for n in top10["run_name"]
    ]
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Val F1")
    ax.set_title("PhysioNet MI: Top 10 EEGNet Configs vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    for i, (bar, val) in enumerate(zip(bars, top10["val_f1"])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "027_physionet_mi_hp_top_configs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    df, poyo_crashed = fetch_all_runs()
    print(f"Fetched {len(df)} EEGNet runs from group {EEGNET_GROUP}")
    print(f"POYO crashed runs: {poyo_crashed}")
    print_summary(df, poyo_crashed)
    plot_hp_heatmap(df)
    plot_class_weights_effect(df)
    plot_top_configs(df)

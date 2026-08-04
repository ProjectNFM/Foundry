"""PhysioNet Motor Imagery HP Search — POYO CWT-CNN Results.

Fetches 24 POYO CWT-CNN runs from WandB group PHYSIONET_MI_HP_SEARCH_POYO
(SLURM job array 10273554_[0-23]). Compares against best EEGNet result (0.924 F1).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "poyo-eeg"
PROJECT = "foundry_finetuning"
POYO_GROUP = "PHYSIONET_MI_HP_SEARCH_POYO"
EEGNET_BEST_F1 = 0.924

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_poyo_name(name: str) -> dict:
    """Parse: physionet_mi_hp_poyo_lr{}_bs{}_cw-{}_dim{}"""
    m = re.match(
        r"physionet_mi_hp_poyo_lr([\d.]+)_bs(\d+)_cw-(\w+)_dim(\d+)", name
    )
    if not m:
        return {}
    return {
        "model": "POYO CWT-CNN",
        "lr": float(m.group(1)),
        "batch_size": int(m.group(2)),
        "class_weights": m.group(3),
        "embed_dim": int(m.group(4)),
    }


def unwrap(val, key="max"):
    if hasattr(val, "get"):
        return float(val.get(key, 0.0))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def fetch_poyo_runs():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": POYO_GROUP})

    records = []
    for run in runs:
        params = parse_poyo_name(run.name)
        if not params:
            continue
        params["run_id"] = run.id
        params["run_name"] = run.name
        params["state"] = run.state
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

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("PHYSIONET MI HP SEARCH — POYO CWT-CNN RESULTS")
    print("=" * 100)

    finished = df[df["state"] == "finished"]
    crashed = df[df["state"] != "finished"]
    converged = finished[finished["val_f1"] > 0.70]

    print(f"\n  Total runs: {len(df)}")
    print(f"  Finished:   {len(finished)}")
    print(f"  Crashed:    {len(crashed)}")
    print(f"  Converged (val F1 > 0.70): {len(converged)}/{len(finished)}")

    print(f"\n{'─' * 100}")
    print("ALL RUNS SORTED BY VAL F1:")
    print(f"{'─' * 100}")

    cols = [
        "run_name",
        "lr",
        "batch_size",
        "class_weights",
        "embed_dim",
        "val_f1",
        "val_auroc",
        "val_acc",
        "epoch",
        "state",
    ]
    sorted_df = finished.sort_values("val_f1", ascending=False)
    print(sorted_df[cols].to_string(index=False, float_format="%.4f"))

    print(f"\n{'=' * 100}")
    print("TOP 5 CONFIGS:")
    print("=" * 100)
    top5 = sorted_df.head(5)
    print(
        top5[
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

    # Comparison with EEGNet
    best_f1 = sorted_df.iloc[0]["val_f1"] if len(sorted_df) > 0 else 0.0
    print(f"\n{'=' * 100}")
    print("COMPARISON WITH EEGNET:")
    print("=" * 100)
    print(f"  Best EEGNet val F1:      {EEGNET_BEST_F1:.3f}")
    print(f"  Best POYO CWT-CNN val F1: {best_f1:.3f}")
    delta = best_f1 - EEGNET_BEST_F1
    print(
        f"  Difference:               {delta:+.3f} ({delta / EEGNET_BEST_F1 * 100:+.1f}%)"
    )

    # Effect of hyperparameters (converged runs only)
    if len(converged) > 0:
        print(f"\n{'=' * 100}")
        print("MEAN VAL F1 BY HYPERPARAMETER (converged runs only):")
        print("=" * 100)

        print("\n  By Learning Rate:")
        print(
            converged.groupby("lr")["val_f1"]
            .agg(["count", "mean", "std", "max"])
            .to_string(float_format="%.4f")
        )

        print("\n  By Batch Size:")
        print(
            converged.groupby("batch_size")["val_f1"]
            .agg(["count", "mean", "std", "max"])
            .to_string(float_format="%.4f")
        )

        print("\n  By Class Weights:")
        print(
            converged.groupby("class_weights")["val_f1"]
            .agg(["count", "mean", "std", "max"])
            .to_string(float_format="%.4f")
        )

        print("\n  By Embed Dim:")
        print(
            converged.groupby("embed_dim")["val_f1"]
            .agg(["count", "mean", "std", "max"])
            .to_string(float_format="%.4f")
        )

    # Convergence analysis
    print(f"\n{'=' * 100}")
    print("CONVERGENCE ANALYSIS:")
    print("=" * 100)
    print("\n  By LR — fraction of runs that converged:")
    for lr_val in sorted(df["lr"].unique()):
        lr_runs = finished[finished["lr"] == lr_val]
        lr_converged = lr_runs[lr_runs["val_f1"] > 0.70]
        print(
            f"    lr={lr_val:.4f}: {len(lr_converged)}/{len(lr_runs)} converged"
        )

    print("\n  By Embed Dim — fraction that converged:")
    for dim in sorted(df["embed_dim"].unique()):
        dim_runs = finished[finished["embed_dim"] == dim]
        dim_converged = dim_runs[dim_runs["val_f1"] > 0.70]
        print(f"    dim={dim}: {len(dim_converged)}/{len(dim_runs)} converged")


def plot_hp_heatmap(df: pd.DataFrame):
    finished = df[df["state"] == "finished"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LR × Batch Size
    pivot = finished.pivot_table(
        values="val_f1", index="lr", columns="batch_size", aggfunc="mean"
    )
    im = axes[0].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f"{v:.0e}" for v in pivot.index])
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("POYO CWT-CNN: Val F1 by LR × Batch Size")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            axes[0].text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=9
            )
    plt.colorbar(im, ax=axes[0], label="Val F1")

    # LR × Embed Dim
    pivot2 = finished.pivot_table(
        values="val_f1", index="lr", columns="embed_dim", aggfunc="mean"
    )
    im2 = axes[1].imshow(pivot2.values, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(pivot2.columns)))
    axes[1].set_xticklabels(pivot2.columns)
    axes[1].set_yticks(range(len(pivot2.index)))
    axes[1].set_yticklabels([f"{v:.0e}" for v in pivot2.index])
    axes[1].set_xlabel("Embed Dim")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("POYO CWT-CNN: Val F1 by LR × Embed Dim")
    for i in range(len(pivot2.index)):
        for j in range(len(pivot2.columns)):
            val = pivot2.values[i, j]
            axes[1].text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=9
            )
    plt.colorbar(im2, ax=axes[1], label="Val F1")

    plt.tight_layout()
    path = FIGURES_DIR / "028_physionet_mi_poyo_hp_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_convergence(df: pd.DataFrame):
    finished = df[df["state"] == "finished"].sort_values(
        "val_f1", ascending=False
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(finished))
    colors = [
        "#4CAF50" if f1 > 0.90 else "#FFC107" if f1 > 0.70 else "#F44336"
        for f1 in finished["val_f1"]
    ]
    bars = ax.barh(y_pos, finished["val_f1"], color=colors, alpha=0.8)

    ax.axvline(
        x=EEGNET_BEST_F1,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Best EEGNet ({EEGNET_BEST_F1:.3f})",
    )
    ax.axvline(
        x=0.662,
        color="gray",
        linestyle=":",
        linewidth=1,
        label="Majority baseline (0.662)",
    )

    ax.set_yticks(y_pos)
    short_names = [
        n.replace("physionet_mi_hp_poyo_", "") for n in finished["run_name"]
    ]
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Val F1")
    ax.set_title("PhysioNet MI: POYO CWT-CNN HP Search (24 runs)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0.5, 1.0)

    for i, (bar, val) in enumerate(zip(bars, finished["val_f1"])):
        if val > 0.70:
            ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    path = FIGURES_DIR / "028_physionet_mi_poyo_hp_all_configs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_lr_effect(df: pd.DataFrame):
    finished = df[df["state"] == "finished"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot by LR
    lr_groups = [
        finished[finished["lr"] == lr]["val_f1"].values
        for lr in sorted(finished["lr"].unique())
    ]
    _ = axes[0].boxplot(
        lr_groups,
        labels=[f"{lr:.0e}" for lr in sorted(finished["lr"].unique())],
    )
    axes[0].axhline(
        y=EEGNET_BEST_F1,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"EEGNet best ({EEGNET_BEST_F1})",
    )
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Val F1")
    axes[0].set_title("Val F1 Distribution by Learning Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Scatter: embed_dim effect (colored by LR)
    colors_map = {1e-4: "#2196F3", 5e-4: "#FF9800", 1e-3: "#F44336"}
    for lr_val in sorted(finished["lr"].unique()):
        subset = finished[finished["lr"] == lr_val]
        axes[1].scatter(
            subset["embed_dim"],
            subset["val_f1"],
            c=colors_map.get(lr_val, "gray"),
            s=60,
            alpha=0.7,
            label=f"lr={lr_val:.0e}",
        )
    axes[1].axhline(y=EEGNET_BEST_F1, color="blue", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Embed Dim")
    axes[1].set_ylabel("Val F1")
    axes[1].set_title("Val F1 by Embed Dim (colored by LR)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "028_physionet_mi_poyo_hp_lr_effect.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    df = fetch_poyo_runs()
    print(f"Fetched {len(df)} POYO CWT-CNN runs from group {POYO_GROUP}")
    print_summary(df)
    plot_hp_heatmap(df)
    plot_convergence(df)
    plot_lr_effect(df)

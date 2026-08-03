"""Brain Invaders P300 HP Search Analysis.

Fetches all 36 runs from WandB group BI_P300_HP_SEARCH and compares
POYO CWT-CNN and EEGNet configurations across lr, smoothing, and
model-specific parameters.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY = "poyo-eeg"
PROJECT = "foundry_finetuning"
GROUP = "BI_P300_HP_SEARCH"

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_poyo_name(name: str) -> dict:
    m = re.match(r"bi_p300_hp_poyo_lr([\d.]+)_sm([\d.]+)_dim(\d+)", name)
    if not m:
        return {}
    return {
        "model": "POYO CWT-CNN",
        "lr": float(m.group(1)),
        "smoothing": float(m.group(2)),
        "embed_dim": int(m.group(3)),
    }


def parse_eegnet_name(name: str) -> dict:
    m = re.match(r"bi_p300_hp_eegnet_lr([\d.]+)_sm([\d.]+)_F1-(\d+)", name)
    if not m:
        return {}
    return {
        "model": "EEGNet",
        "lr": float(m.group(1)),
        "smoothing": float(m.group(2)),
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
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP})

    records = []
    for run in runs:
        if "poyo" in run.name:
            params = parse_poyo_name(run.name)
        elif "eegnet" in run.name:
            params = parse_eegnet_name(run.name)
        else:
            continue

        params["run_id"] = run.id
        params["run_name"] = run.name
        params["state"] = run.state
        params["val_f1"] = unwrap(run.summary.get("val/p300_binary_f1"), "max")
        params["val_auroc"] = unwrap(
            run.summary.get("val/p300_binary_auroc"), "max"
        )
        params["val_acc"] = unwrap(
            run.summary.get("val/p300_binary_acc"), "max"
        )
        params["val_loss"] = unwrap(
            run.summary.get("val/p300_binary_loss"), "min"
        )
        params["val_precision"] = unwrap(
            run.summary.get("val/p300_binary_precision"), "max"
        )
        params["val_recall"] = unwrap(
            run.summary.get("val/p300_binary_recall"), "max"
        )
        params["epoch"] = run.summary.get("epoch", 0)
        records.append(params)

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("BRAIN INVADERS P300 HP SEARCH — RESULTS SUMMARY")
    print("=" * 90)

    for model_name in ["POYO CWT-CNN", "EEGNet"]:
        sub = df[df["model"] == model_name].sort_values(
            "val_f1", ascending=False
        )
        print(f"\n{'─' * 90}")
        print(f"  {model_name} ({len(sub)} runs)")
        print(f"{'─' * 90}")

        if model_name == "POYO CWT-CNN":
            cols = [
                "run_name",
                "lr",
                "smoothing",
                "embed_dim",
                "val_f1",
                "val_auroc",
                "val_precision",
                "val_recall",
                "epoch",
            ]
        else:
            cols = [
                "run_name",
                "lr",
                "smoothing",
                "F1",
                "val_f1",
                "val_auroc",
                "val_precision",
                "val_recall",
                "epoch",
            ]

        print(sub[cols].to_string(index=False, float_format="%.4f"))

    # Overall best
    print(f"\n{'=' * 90}")
    print("TOP 5 OVERALL (by val F1):")
    print("=" * 90)
    top5 = df.sort_values("val_f1", ascending=False).head(5)
    print(
        top5[
            [
                "run_name",
                "model",
                "val_f1",
                "val_auroc",
                "val_precision",
                "val_recall",
                "epoch",
            ]
        ].to_string(index=False, float_format="%.4f")
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

    print("\n  By Smoothing:")
    print(
        df.groupby("smoothing")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    print("\n  By Model:")
    print(
        df.groupby("model")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    # POYO-specific
    poyo = df[df["model"] == "POYO CWT-CNN"]
    print("\n  POYO by embed_dim:")
    print(
        poyo.groupby("embed_dim")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )

    # EEGNet-specific
    eegnet = df[df["model"] == "EEGNet"]
    print("\n  EEGNet by F1 (spatial filters):")
    print(
        eegnet.groupby("F1")["val_f1"]
        .agg(["mean", "std", "max"])
        .to_string(float_format="%.4f")
    )


def plot_hp_heatmap(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_name in zip(axes, ["POYO CWT-CNN", "EEGNet"]):
        sub = df[df["model"] == model_name]

        if model_name == "POYO CWT-CNN":
            pivot = sub.pivot_table(
                values="val_f1", index="lr", columns="embed_dim", aggfunc="mean"
            )
            title = "POYO CWT-CNN: Val F1 by LR × Embed Dim"
        else:
            pivot = sub.pivot_table(
                values="val_f1", index="lr", columns="F1", aggfunc="mean"
            )
            title = "EEGNet: Val F1 by LR × F1 filters"

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0e}" for v in pivot.index])
        ax.set_xlabel(pivot.columns.name)
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(
                    j, i, f"{val:.3f}", ha="center", va="center", fontsize=9
                )

        plt.colorbar(im, ax=ax, label="Val F1")

    plt.tight_layout()
    path = FIGURES_DIR / "026_bi_p300_hp_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_smoothing_effect(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model_name in zip(axes, ["POYO CWT-CNN", "EEGNet"]):
        sub = df[df["model"] == model_name]
        pivot = sub.pivot_table(
            values="val_f1", index="smoothing", columns="lr", aggfunc="mean"
        )

        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], "o-", label=f"lr={col:.0e}")

        ax.set_xlabel("Class Weight Smoothing")
        ax.set_ylabel("Val F1")
        ax.set_title(f"{model_name}: Smoothing Effect")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "026_bi_p300_hp_smoothing.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_best_comparison(df: pd.DataFrame):
    best_poyo = (
        df[df["model"] == "POYO CWT-CNN"]
        .sort_values("val_f1", ascending=False)
        .iloc[0]
    )
    best_eegnet = (
        df[df["model"] == "EEGNet"]
        .sort_values("val_f1", ascending=False)
        .iloc[0]
    )

    metrics = ["val_f1", "val_auroc", "val_precision", "val_recall"]
    labels = ["F1", "AUROC", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35

    poyo_vals = [best_poyo[m] for m in metrics]
    eegnet_vals = [best_eegnet[m] for m in metrics]

    bars1 = ax.bar(
        x - width / 2, poyo_vals, width, label="POYO (best)", color="#2196F3"
    )
    bars2 = ax.bar(
        x + width / 2,
        eegnet_vals,
        width,
        label="EEGNet (best)",
        color="#FF9800",
    )

    ax.set_ylabel("Score")
    ax.set_title("Brain Invaders P300: Best Config Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    path = FIGURES_DIR / "026_bi_p300_hp_best_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    df = fetch_all_runs()
    print(f"Fetched {len(df)} runs from group {GROUP}")
    print_summary(df)
    plot_hp_heatmap(df)
    plot_smoothing_effect(df)
    plot_best_comparison(df)

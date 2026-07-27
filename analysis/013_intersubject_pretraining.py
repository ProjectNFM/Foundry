"""Intersubject pretraining — session embedding generalization diagnostic.

Compares train-val loss gap between intersubject pretraining (exp 013) and
intrasession pretraining (exp 005) for both ResampleCNN and CWT-CNN tokenizers.

WandB project: foundry_pretraining
Groups:
  - PRETRAIN_TOKENIZER_INTERSUBJECT  (exp 013, intersubject split)
  - PRETRAIN_TOKENIZER_SWEEP         (exp 005, intrasession split)

Runs:
  Intersubject ResampleCNN: znqri8rf
  Intersubject CWT-CNN:     65nbol38
  Intrasession ResampleCNN: vup5m7er
  Intrasession CWT-CNN:     wlmobz7y

Usage:
    uv run python analysis/013_intersubject_pretraining.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

WANDB_PROJECT = "foundry_pretraining"
WANDB_ENTITY = default_entity()

RUNS = {
    ("ResampleCNN", "intersubject"): "znqri8rf",
    ("CWT-CNN", "intersubject"): "65nbol38",
    ("ResampleCNN", "intrasession"): "vup5m7er",
    ("CWT-CNN", "intrasession"): "wlmobz7y",
}

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)

TOKENIZER_COLORS = {"ResampleCNN": "#4C72B0", "CWT-CNN": "#55A868"}
SPLIT_STYLES = {"intrasession": "-", "intersubject": "--"}


def _fetch_run(run_id: str, api: wandb.Api):
    entity = WANDB_ENTITY
    path = (
        f"{entity}/{WANDB_PROJECT}/{run_id}"
        if entity
        else f"{WANDB_PROJECT}/{run_id}"
    )
    return api.run(path)


def _per_epoch_metrics(run) -> pd.DataFrame:
    """Extract per-epoch average train loss and val loss from scan_history."""
    all_rows = list(run.scan_history())

    train_rows = [
        r for r in all_rows if TRAIN_LOSS in r and r[TRAIN_LOSS] is not None
    ]
    val_rows = [
        r for r in all_rows if VAL_LOSS in r and r[VAL_LOSS] is not None
    ]

    val_df = pd.DataFrame(
        [
            {"epoch": int(r.get("epoch", 0)), "val_loss": r[VAL_LOSS]}
            for r in val_rows
        ]
    )

    train_df = pd.DataFrame(
        [
            {"epoch": int(r.get("epoch", 0)), "train_loss": r[TRAIN_LOSS]}
            for r in train_rows
        ]
    )
    train_agg = train_df.groupby("epoch")["train_loss"].mean().reset_index()

    if val_df.empty:
        return train_agg
    merged = val_df.merge(train_agg, on="epoch", how="outer").sort_values(
        "epoch"
    )
    merged["gap"] = merged["val_loss"] - merged["train_loss"]
    return merged


def fetch_all_data(api: wandb.Api) -> dict[tuple[str, str], dict]:
    """Fetch summary + per-epoch data for all runs."""
    results = {}
    for (tok, split), run_id in RUNS.items():
        print(f"Fetching {tok} / {split} ({run_id})...")
        run = _fetch_run(run_id, api)
        s = run.summary

        epoch_df = _per_epoch_metrics(run)

        best_val_epoch = None
        best_val_loss = None
        train_at_best_val = None
        gap_at_best_val = None
        if not epoch_df.empty and "val_loss" in epoch_df.columns:
            valid = epoch_df.dropna(subset=["val_loss"])
            if not valid.empty:
                best_idx = valid["val_loss"].idxmin()
                best_row = valid.loc[best_idx]
                best_val_epoch = int(best_row["epoch"])
                best_val_loss = best_row["val_loss"]
                train_at_best_val = best_row.get("train_loss")
                if train_at_best_val is not None:
                    gap_at_best_val = best_val_loss - train_at_best_val

        results[(tok, split)] = {
            "run_id": run_id,
            "state": run.state,
            "best_val_loss": best_val_loss
            or unwrap_summary_value(s.get(VAL_LOSS), "min"),
            "best_train_loss": unwrap_summary_value(s.get(TRAIN_LOSS), "min"),
            "train_at_best_val": train_at_best_val,
            "gap_at_best_val": gap_at_best_val,
            "best_val_epoch": best_val_epoch,
            "max_epoch": unwrap_summary_value(s.get("epoch"), "max"),
            "epoch_df": epoch_df,
        }
    return results


def print_summary(data: dict) -> None:
    print(f"\n{'=' * 90}")
    print("  Intersubject vs Intrasession Pretraining — Summary")
    print(f"{'=' * 90}")

    for tok in ["ResampleCNN", "CWT-CNN"]:
        print(f"\n--- {tok} ---")
        for split in ["intrasession", "intersubject"]:
            d = data.get((tok, split))
            if d is None:
                continue
            gap_s = (
                f"{d['gap_at_best_val']:.4f}"
                if d["gap_at_best_val"] is not None
                else "?"
            )
            train_s = (
                f"{d['train_at_best_val']:.4f}"
                if d["train_at_best_val"] is not None
                else "?"
            )
            print(
                f"  {split:<14s}  "
                f"val={d['best_val_loss']:.4f}  "
                f"train@best_val={train_s}  "
                f"gap={gap_s}  "
                f"best_epoch={d['best_val_epoch']}  "
                f"max_epoch={d['max_epoch']}  "
                f"state={d['state']}  "
                f"(run={d['run_id']})"
            )

    print(f"\n{'=' * 90}")
    print("  Gap comparison (intersubject vs intrasession)")
    print(f"{'=' * 90}")
    for tok in ["ResampleCNN", "CWT-CNN"]:
        intra = data.get((tok, "intrasession"))
        inter = data.get((tok, "intersubject"))
        if (
            intra
            and inter
            and intra["gap_at_best_val"] is not None
            and inter["gap_at_best_val"] is not None
        ):
            ratio = (
                inter["gap_at_best_val"] / intra["gap_at_best_val"]
                if intra["gap_at_best_val"] != 0
                else float("inf")
            )
            print(
                f"  {tok:<14s}  "
                f"intra_gap={intra['gap_at_best_val']:.4f}  "
                f"inter_gap={inter['gap_at_best_val']:.4f}  "
                f"ratio={ratio:.1f}x"
            )


def plot_gap_comparison(data: dict) -> None:
    """Bar chart: train-val gap and best val loss, intersubject vs intrasession."""
    tokenizers = ["ResampleCNN", "CWT-CNN"]
    splits = ["intrasession", "intersubject"]
    split_colors = {"intrasession": "#4C72B0", "intersubject": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    x = np.arange(len(tokenizers))
    width = 0.35

    # Panel 1: Train-Val Gap
    ax = axes[0]
    for i, split in enumerate(splits):
        gaps = []
        for tok in tokenizers:
            d = data.get((tok, split))
            gaps.append(
                d["gap_at_best_val"]
                if d and d["gap_at_best_val"] is not None
                else 0
            )
        bars = ax.bar(
            x + (i - 0.5) * width,
            gaps,
            width,
            label=split.capitalize(),
            color=split_colors[split],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, gaps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tokenizers)
    ax.set_ylabel("Val Loss − Train Loss (at best val epoch)")
    ax.set_title("Train-Val Loss Gap")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Best Val Loss
    ax = axes[1]
    for i, split in enumerate(splits):
        vals = []
        for tok in tokenizers:
            d = data.get((tok, split))
            vals.append(d["best_val_loss"] if d else 0)
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=split.capitalize(),
            color=split_colors[split],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tokenizers)
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Val Loss")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Intersubject vs Intrasession Pretraining",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "013_gap_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_learning_curves(data: dict) -> None:
    """Per-epoch train/val loss curves for all 4 runs in a 2x2 grid."""
    tokenizers = ["ResampleCNN", "CWT-CNN"]
    splits = ["intrasession", "intersubject"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey="row")

    for col, tok in enumerate(tokenizers):
        for row, split in enumerate(splits):
            ax = axes[row, col]
            d = data.get((tok, split))
            if d is None or d["epoch_df"].empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            edf = d["epoch_df"]
            color = TOKENIZER_COLORS[tok]

            if "train_loss" in edf.columns:
                valid_train = edf.dropna(subset=["train_loss"])
                ax.plot(
                    valid_train["epoch"],
                    valid_train["train_loss"],
                    color=color,
                    linewidth=2,
                    label="Train",
                )
            if "val_loss" in edf.columns:
                valid_val = edf.dropna(subset=["val_loss"])
                ax.plot(
                    valid_val["epoch"],
                    valid_val["val_loss"],
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label="Val",
                )
            if "train_loss" in edf.columns and "val_loss" in edf.columns:
                both = edf.dropna(subset=["train_loss", "val_loss"])
                if not both.empty:
                    ax.fill_between(
                        both["epoch"],
                        both["train_loss"],
                        both["val_loss"],
                        alpha=0.15,
                        color=color,
                    )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            label = f"{tok} — {split}"
            gap_s = (
                f"gap={d['gap_at_best_val']:.3f}"
                if d["gap_at_best_val"] is not None
                else ""
            )
            ax.set_title(f"{label}\n(val={d['best_val_loss']:.4f}, {gap_s})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Pretraining Loss Curves — Intersubject vs Intrasession",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "013_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_gap_evolution(data: dict) -> None:
    """Per-epoch train-val gap evolution for both tokenizers."""
    tokenizers = ["ResampleCNN", "CWT-CNN"]
    splits = ["intrasession", "intersubject"]
    split_styles = {"intrasession": "-", "intersubject": "--"}
    split_colors = {"intrasession": "#4C72B0", "intersubject": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, tok in enumerate(tokenizers):
        ax = axes[i]
        for split in splits:
            d = data.get((tok, split))
            if d is None or d["epoch_df"].empty:
                continue
            edf = d["epoch_df"]
            if "gap" not in edf.columns:
                continue
            valid = edf.dropna(subset=["gap"])
            if valid.empty:
                continue
            ax.plot(
                valid["epoch"],
                valid["gap"],
                color=split_colors[split],
                linewidth=2,
                linestyle=split_styles[split],
                marker="o",
                markersize=6,
                label=f"{split} (gap={valid['gap'].iloc[-1]:.3f})",
            )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss − Train Loss")
        ax.set_title(f"{tok}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Train-Val Gap Evolution by Epoch",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "013_gap_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()
    print("Fetching runs for intersubject pretraining comparison...")
    data = fetch_all_data(api)
    print_summary(data)
    plot_gap_comparison(data)
    plot_learning_curves(data)
    plot_gap_evolution(data)


if __name__ == "__main__":
    main()

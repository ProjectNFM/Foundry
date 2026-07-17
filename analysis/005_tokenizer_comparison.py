"""Tokenizer architecture comparison for masked pretraining.

WandB project: foundry_pretraining
Group: PRETRAIN_TOKENIZER_SWEEP

Runs (name -> run_id):
  pretrain_tokenizer_per_channel_resample_cnn: vup5m7er
  pretrain_tokenizer_per_channel_cwt_cnn: wlmobz7y
  pretrain_tokenizer_per_channel_per_timepoint_linear: 092n6bv1
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_PROJECT = "foundry_pretraining"
WANDB_ENTITY = None  # uses default entity

RUNS = {
    "ResampleCNN": "vup5m7er",
    "CWT-CNN": "wlmobz7y",
    "PerTimestepLinear": "092n6bv1",
}

VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"
VAL_RECON_MSE = "val/masked_reconstruction_recon_mse"

FIGURES_DIR = Path(__file__).parent / "figures"


def make_run_path(run_id: str) -> str:
    if WANDB_ENTITY:
        return f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
    return f"{WANDB_PROJECT}/{run_id}"


def _unwrap(val, key="min"):
    """WandB summary may return a SummarySubDict like {'min': 0.12}."""
    try:
        return float(val[key])
    except (TypeError, KeyError, IndexError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def fetch_run_summary(run_id: str) -> dict:
    api = wandb.Api()
    run = api.run(make_run_path(run_id))
    return {
        "best_val_loss": _unwrap(run.summary.get(VAL_LOSS), "min"),
        "best_val_recon_mse": _unwrap(run.summary.get(VAL_RECON_MSE), "min"),
        "best_train_loss": _unwrap(run.summary.get(TRAIN_LOSS), "min"),
        "epoch": _unwrap(run.summary.get("epoch")),
        "global_step": _unwrap(run.summary.get("trainer/global_step")),
        "runtime_s": _unwrap(run.summary.get("_runtime")),
        "state": run.state,
    }


def fetch_metric_history(run_id: str, metric: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(make_run_path(run_id))
    history = run.history(keys=[metric, "_step"], pandas=True)
    history = history.dropna(subset=[metric])
    return history[["_step", metric]].reset_index(drop=True)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Collect summaries ---
    rows = []
    for tokenizer, run_id in RUNS.items():
        print(f"Fetching {tokenizer} ({run_id})...")
        s = fetch_run_summary(run_id)
        rows.append(
            {
                "Tokenizer": tokenizer,
                "Run ID": run_id,
                "State": s["state"],
                "Best Val Loss": s["best_val_loss"],
                "Best Val Recon MSE": s["best_val_recon_mse"],
                "Best Train Loss": s["best_train_loss"],
                "Epoch": s["epoch"],
                "Global Step": s["global_step"],
                "Runtime (s)": s["runtime_s"],
            }
        )

    df = pd.DataFrame(rows)

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("PRETRAIN TOKENIZER SWEEP — Summary")
    print("=" * 100)
    print(df.to_string(index=False))
    print()

    for _, row in df.iterrows():
        print(
            f"  {row['Tokenizer']:<22}  "
            f"Val Loss={row['Best Val Loss']:.4f}  "
            f"Val MSE={row['Best Val Recon MSE']:.4f}  "
            f"Train Loss={row['Best Train Loss']:.4f}  "
            f"Epochs={row['Epoch']}  "
            f"Runtime={row['Runtime (s)']:.0f}s"
        )

    # --- Figure 1: Val loss bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    tokenizer_names = list(RUNS.keys())
    x = np.arange(len(tokenizer_names))
    val_losses = [
        df[df["Tokenizer"] == t]["Best Val Loss"].values[0]
        for t in tokenizer_names
    ]

    bars = ax.bar(
        x,
        val_losses,
        color=["#4C72B0", "#55A868", "#C44E52"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, val_losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tokenizer_names)
    ax.set_ylabel("Best Validation Loss (MSE)")
    ax.set_title("Masked Pretraining — Tokenizer Comparison (Best Val Loss)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(val_losses) * 1.15)
    plt.tight_layout()

    out_path = FIGURES_DIR / "005_tokenizer_comparison_val_loss.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()

    # --- Figure 2: Val loss learning curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "ResampleCNN": "#4C72B0",
        "CWT-CNN": "#55A868",
        "PerTimestepLinear": "#C44E52",
    }

    for tokenizer, run_id in RUNS.items():
        h = fetch_metric_history(run_id, VAL_LOSS)
        if not h.empty:
            ax.plot(
                h["_step"],
                h[VAL_LOSS],
                color=colors[tokenizer],
                linewidth=2,
                label=f"{tokenizer} (best={h[VAL_LOSS].min():.4f})",
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Masked Pretraining — Val Loss Learning Curves by Tokenizer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "005_tokenizer_comparison_learning_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()

    # --- Figure 3: Train loss learning curves ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for tokenizer, run_id in RUNS.items():
        h = fetch_metric_history(run_id, TRAIN_LOSS)
        if not h.empty:
            ax.plot(
                h["_step"],
                h[TRAIN_LOSS],
                color=colors[tokenizer],
                linewidth=1.5,
                alpha=0.8,
                label=f"{tokenizer} (best={h[TRAIN_LOSS].min():.4f})",
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Masked Pretraining — Train Loss Learning Curves by Tokenizer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "005_tokenizer_comparison_train_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

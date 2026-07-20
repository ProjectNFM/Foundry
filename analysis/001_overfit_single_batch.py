"""Fetch and plot metrics for the single-batch overfit experiment.

Usage:
    uv run python analysis/001_overfit_single_batch.py

Requires WANDB_API_KEY (and optionally WANDB_ENTITY) in environment or .env.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis._wandb_utils import (
    default_entity,
    fetch_metric_history,
    figures_dir,
    get_run,
)

WANDB_PROJECT = "foundry_pretraining"
RUN_ID = "gii7gvev"

FIGURES_DIR = figures_dir(__file__)


def main():
    entity = default_entity()
    run = get_run(RUN_ID, WANDB_PROJECT, entity)

    print(f"Run: {run.name}  (id={run.id})")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")
    print()

    train_loss = fetch_metric_history(
        RUN_ID, "train/loss", WANDB_PROJECT, entity
    )
    val_loss = fetch_metric_history(RUN_ID, "val/loss", WANDB_PROJECT, entity)

    print("=== Training Loss ===")
    print(f"  Start (epoch 0):   {train_loss['train/loss'].iloc[0]:.4f}")
    print(f"  End (last epoch):  {train_loss['train/loss'].iloc[-1]:.4f}")
    print(f"  Min:               {train_loss['train/loss'].min():.4f}")
    print()

    print("=== Validation Loss ===")
    print(f"  Start (epoch 0):   {val_loss['val/loss'].iloc[0]:.4f}")
    print(f"  Best:              {val_loss['val/loss'].min():.4f}")
    print(f"  End (last epoch):  {val_loss['val/loss'].iloc[-1]:.4f}")
    print()

    best_val_idx = val_loss["val/loss"].idxmin()
    best_val_step = val_loss.loc[best_val_idx, "_step"]
    best_val_value = val_loss.loc[best_val_idx, "val/loss"]
    print(f"  Best val loss {best_val_value:.4f} at step {int(best_val_step)}")
    print()

    summary = run.summary
    summary_keys = [k for k in summary.keys() if "loss" in k.lower()]
    if summary_keys:
        print("=== WandB Summary (loss keys) ===")
        for k in sorted(summary_keys):
            print(f"  {k}: {summary[k]}")
        print()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        train_loss["_step"],
        train_loss["train/loss"],
        label="Train Loss",
        linewidth=1.5,
    )
    ax.plot(
        val_loss["_step"], val_loss["val/loss"], label="Val Loss", linewidth=1.5
    )
    ax.axvline(
        best_val_step,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Best val @ step {int(best_val_step)}",
    )
    ax.set_xlabel("Step (epoch)")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Single-Batch Overfit: Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "001_overfit_single_batch_loss.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

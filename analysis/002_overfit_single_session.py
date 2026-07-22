"""Fetch and plot metrics for the single-session overfit experiment.

Usage:
    uv run python analysis/002_overfit_single_session.py

Update RUN_ID below after launching the experiment.
Requires WANDB_API_KEY (and optionally WANDB_ENTITY) in environment or .env.
"""

import sys

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
RUN_ID = "1x9sf5ar"

FIGURES_DIR = figures_dir(__file__)


def main():
    if RUN_ID == "TODO":
        print(
            "ERROR: Set RUN_ID at the top of this script before running.",
            file=sys.stderr,
        )
        sys.exit(1)

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
    print(f"  Start:  {train_loss['train/loss'].iloc[0]:.4f}")
    print(f"  End:    {train_loss['train/loss'].iloc[-1]:.4f}")
    print(f"  Min:    {train_loss['train/loss'].min():.4f}")
    print()

    print("=== Validation Loss ===")
    print(f"  Start:  {val_loss['val/loss'].iloc[0]:.4f}")
    print(f"  Best:   {val_loss['val/loss'].min():.4f}")
    print(f"  End:    {val_loss['val/loss'].iloc[-1]:.4f}")
    print()

    best_val_idx = val_loss["val/loss"].idxmin()
    best_val_step = val_loss.loc[best_val_idx, "_step"]
    best_val_value = val_loss.loc[best_val_idx, "val/loss"]
    print(f"  Best val loss {best_val_value:.4f} at step {int(best_val_step)}")

    val_start = val_loss["val/loss"].iloc[0]
    val_end = val_loss["val/loss"].iloc[-1]
    if val_end > val_start * 1.05:
        print("  WARNING: val loss diverged (end > start * 1.05)")
    elif val_end < val_start * 0.95:
        print("  Val loss decreased meaningfully (end < start * 0.95)")
    else:
        print("  Val loss roughly stable")
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
    ax.set_title("Single-Session Overfit: Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "002_overfit_single_session_loss.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

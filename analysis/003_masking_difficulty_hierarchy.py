"""Compare training loss across masking strategies to establish difficulty hierarchy.

Runs:
  - FullChannelMaskingFixed (j0i9jacr): ChannelMasking, mask_ratio=0.5
  - FullTimeMasking10 (ax19kghy): TemporalBlockMasking, block_size=10, mask_ratio=0.5
  - TestingFull (xcqs9lt5): RandomTokenMasking, mask_ratio=0.5
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_PROJECT = "foundry_pretraining"
WANDB_ENTITY = None  # uses default entity

RUNS = {
    "ChannelMasking": "j0i9jacr",
    "TemporalBlockMasking": "ax19kghy",
    "RandomTokenMasking": "xcqs9lt5",
}

METRIC = "train/loss"
FIGURES_DIR = Path(__file__).parent / "figures"


def fetch_training_loss(run_id: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    history = run.history(keys=[METRIC, "trainer/global_step"], pandas=True)
    history = history.dropna(subset=[METRIC])
    return history[["trainer/global_step", METRIC]].reset_index(drop=True)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for name, run_id in RUNS.items():
        print(f"Fetching {name} ({run_id})...")
        dfs[name] = fetch_training_loss(run_id)

    # Find common step range (intersection of all runs)
    max_min_step = max(df["trainer/global_step"].min() for df in dfs.values())
    min_max_step = min(df["trainer/global_step"].max() for df in dfs.values())
    print(f"\nCommon step range: [{max_min_step}, {min_max_step}]")

    # Filter to common steps
    for name in dfs:
        df = dfs[name]
        dfs[name] = df[
            (df["trainer/global_step"] >= max_min_step)
            & (df["trainer/global_step"] <= min_max_step)
        ]

    # Print summary table
    print(
        f"\n{'Masking Strategy':<25} {'Mean Loss':>12} {'Final Loss':>12} {'Steps':>8}"
    )
    print("-" * 60)
    for name, df in dfs.items():
        mean_loss = df[METRIC].mean()
        final_loss = df[METRIC].iloc[-1] if len(df) > 0 else float("nan")
        n_steps = len(df)
        print(f"{name:<25} {mean_loss:>12.6f} {final_loss:>12.6f} {n_steps:>8}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in dfs.items():
        ax.plot(df["trainer/global_step"], df[METRIC], label=name, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Masking Strategy Difficulty Hierarchy - Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "003_masking_difficulty_hierarchy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

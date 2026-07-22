"""Compare training loss across masking strategies to establish difficulty hierarchy.

Runs:
  - FullChannelMaskingFixed (j0i9jacr): ChannelMasking, mask_ratio=0.5
  - FullTimeMasking10 (ax19kghy): TemporalBlockMasking, block_size=10, mask_ratio=0.5
  - TestingFull (xcqs9lt5): RandomTokenMasking, mask_ratio=0.5
"""

import matplotlib.pyplot as plt

from analysis._wandb_utils import fetch_metric_history, figures_dir

WANDB_PROJECT = "foundry_pretraining"

RUNS = {
    "ChannelMasking": "j0i9jacr",
    "TemporalBlockMasking": "ax19kghy",
    "RandomTokenMasking": "xcqs9lt5",
}

METRIC = "train/loss"
FIGURES_DIR = figures_dir(__file__)


def main():
    dfs = {}
    for name, run_id in RUNS.items():
        print(f"Fetching {name} ({run_id})...")
        dfs[name] = fetch_metric_history(
            run_id, METRIC, WANDB_PROJECT, x_axis="trainer/global_step"
        )

    max_min_step = max(df["trainer/global_step"].min() for df in dfs.values())
    min_max_step = min(df["trainer/global_step"].max() for df in dfs.values())
    print(f"\nCommon step range: [{max_min_step}, {min_max_step}]")

    for name in dfs:
        df = dfs[name]
        dfs[name] = df[
            (df["trainer/global_step"] >= max_min_step)
            & (df["trainer/global_step"] <= min_max_step)
        ]

    print(
        f"\n{'Masking Strategy':<25} {'Mean Loss':>12} {'Final Loss':>12} {'Steps':>8}"
    )
    print("-" * 60)
    for name, df in dfs.items():
        mean_loss = df[METRIC].mean()
        final_loss = df[METRIC].iloc[-1] if len(df) > 0 else float("nan")
        n_steps = len(df)
        print(f"{name:<25} {mean_loss:>12.6f} {final_loss:>12.6f} {n_steps:>8}")

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

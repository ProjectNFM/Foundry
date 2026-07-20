"""Compare training loss: baseline (no channel emb in decoder) vs with channel emb.

Runs:
  - FullTimeMasking10 (ax19kghy): Baseline, TemporalBlockMasking block_size=10
  - TimeMasking_ChannelEmbDecoder (qgohh6dc): + channel identity in decoder queries
"""

import matplotlib.pyplot as plt

from analysis._wandb_utils import fetch_metric_history, figures_dir

WANDB_PROJECT = "foundry_pretraining"

RUNS = {
    "Baseline (no ch_emb)": "ax19kghy",
    "With ch_emb in decoder": "qgohh6dc",
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

    print(f"\n{'Run':<30} {'Mean Loss':>12} {'Final Loss':>12} {'Steps':>8}")
    print("-" * 65)
    for name, df in dfs.items():
        mean_loss = df[METRIC].mean()
        final_loss = df[METRIC].iloc[-1] if len(df) > 0 else float("nan")
        n_steps = len(df)
        print(f"{name:<30} {mean_loss:>12.6f} {final_loss:>12.6f} {n_steps:>8}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in dfs.items():
        ax.plot(df["trainer/global_step"], df[METRIC], label=name, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        "Channel Identity in Decoder — TemporalBlockMasking Training Loss"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "004_channel_identity_decoder.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

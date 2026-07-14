"""Kemp Sleep EDF tokenizer baseline: compare 3 tokenizers × 3 CV folds.

WandB project: foundry_finetuning
Group: KEMP_SLEEP_TOKENIZER_BASELINE

Runs (name -> run_id):
  ResampleCNN:
    fold0: mj5b3gsu  fold1: vzfktdlv  fold2: eay0303t
  CWT-CNN:
    fold0: tm7jvvvs  fold1: 7snau2mc  fold2: 3c19d512
  PerTimestepLinear:
    fold0: 182pkp6v  fold1: hb4n732s  fold2: oi7v77lc
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = None  # uses default entity

RUNS = {
    "ResampleCNN": {
        "fold0": "mj5b3gsu",
        "fold1": "vzfktdlv",
        "fold2": "eay0303t",
    },
    "CWT-CNN": {
        "fold0": "tm7jvvvs",
        "fold1": "7snau2mc",
        "fold2": "3c19d512",
    },
    "PerTimestepLinear": {
        "fold0": "182pkp6v",
        "fold1": "hb4n732s",
        "fold2": "oi7v77lc",
    },
}

VAL_F1 = "val/sleep_stage_5class_f1"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"

FIGURES_DIR = Path(__file__).parent / "figures"


def make_run_path(run_id: str) -> str:
    if WANDB_ENTITY:
        return f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
    return f"{WANDB_PROJECT}/{run_id}"


def _unwrap(val, key="max"):
    """WandB summary may return a SummarySubDict like {'max': 0.57}."""
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
        "best_val_f1": _unwrap(run.summary.get(VAL_F1), "max"),
        "best_val_loss": _unwrap(run.summary.get(VAL_LOSS), "min"),
        "final_train_loss": _unwrap(run.summary.get(TRAIN_LOSS), "min"),
        "best_epoch": _unwrap(run.summary.get("epoch"), "max"),
        "state": run.state,
    }


def fetch_metric_history(run_id: str, metric: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(make_run_path(run_id))
    history = run.history(keys=[metric, "epoch"], pandas=True)
    history = history.dropna(subset=[metric])
    return history[["epoch", metric]].reset_index(drop=True)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Collect summaries ---
    rows = []
    for tokenizer, folds in RUNS.items():
        for fold_name, run_id in folds.items():
            print(f"Fetching {tokenizer} {fold_name} ({run_id})...")
            s = fetch_run_summary(run_id)
            rows.append(
                {
                    "Tokenizer": tokenizer,
                    "Fold": fold_name,
                    "Run ID": run_id,
                    "State": s["state"],
                    "Best Val F1": s["best_val_f1"],
                    "Best Val Loss": s["best_val_loss"],
                    "Final Train Loss": s["final_train_loss"],
                    "Best Epoch": s["best_epoch"],
                }
            )

    df = pd.DataFrame(rows)

    # --- Per-run table ---
    print("\n" + "=" * 90)
    print("Per-run results")
    print("=" * 90)
    print(df.to_string(index=False))

    # --- Aggregate table (mean ± std across folds) ---
    agg = (
        df.groupby("Tokenizer")[
            ["Best Val F1", "Best Val Loss", "Final Train Loss"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n" + "=" * 90)
    print("Aggregated across folds (mean ± std)")
    print("=" * 90)
    for tokenizer in RUNS:
        f1_mean = agg.loc[tokenizer, ("Best Val F1", "mean")]
        f1_std = agg.loc[tokenizer, ("Best Val F1", "std")]
        vl_mean = agg.loc[tokenizer, ("Best Val Loss", "mean")]
        vl_std = agg.loc[tokenizer, ("Best Val Loss", "std")]
        tl_mean = agg.loc[tokenizer, ("Final Train Loss", "mean")]
        tl_std = agg.loc[tokenizer, ("Final Train Loss", "std")]
        print(
            f"  {tokenizer:<22}  "
            f"F1={f1_mean:.4f}±{f1_std:.4f}  "
            f"Val Loss={vl_mean:.4f}±{vl_std:.4f}  "
            f"Train Loss={tl_mean:.4f}±{tl_std:.4f}"
        )

    # --- Figure 1: Val F1 bar chart with per-fold dots ---
    fig, ax = plt.subplots(figsize=(8, 5))
    tokenizer_names = list(RUNS.keys())
    x = np.arange(len(tokenizer_names))
    means = [agg.loc[t, ("Best Val F1", "mean")] for t in tokenizer_names]
    stds = [agg.loc[t, ("Best Val F1", "std")] for t in tokenizer_names]

    _ = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=["#4C72B0", "#55A868", "#C44E52"],
        alpha=0.8,
    )
    for i, t in enumerate(tokenizer_names):
        fold_vals = df[df["Tokenizer"] == t]["Best Val F1"].values
        ax.scatter(
            [i] * len(fold_vals), fold_vals, color="black", zorder=5, s=30
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tokenizer_names)
    ax.set_ylabel("Best Validation F1 (5-class)")
    ax.set_title("Kemp Sleep — Tokenizer Comparison (inter-subject CV)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "006_kemp_sleep_tokenizer_val_f1.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()

    # --- Figure 2: Val F1 learning curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "ResampleCNN": "#4C72B0",
        "CWT-CNN": "#55A868",
        "PerTimestepLinear": "#C44E52",
    }

    for tokenizer, folds in RUNS.items():
        all_histories = []
        for fold_name, run_id in folds.items():
            h = fetch_metric_history(run_id, VAL_F1)
            if not h.empty:
                all_histories.append(h.set_index("epoch")[VAL_F1])
                ax.plot(
                    h["epoch"],
                    h[VAL_F1],
                    color=colors[tokenizer],
                    alpha=0.25,
                    linewidth=0.8,
                )

        if all_histories:
            combined = pd.concat(all_histories, axis=1)
            mean_curve = combined.mean(axis=1)
            ax.plot(
                mean_curve.index,
                mean_curve.values,
                color=colors[tokenizer],
                linewidth=2,
                label=tokenizer,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 (5-class)")
    ax.set_title("Kemp Sleep — Val F1 Learning Curves by Tokenizer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "006_kemp_sleep_tokenizer_learning_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

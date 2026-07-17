"""Pretraining loss vs downstream performance: finetuned vs from-scratch.

Compares finetuned ResampleCNN and CWT-CNN (from exp 005 pretrained checkpoints)
against the from-scratch baselines from exp 006 on Kemp Sleep 5-class staging.

WandB project: foundry_finetuning
Groups: KEMP_FINETUNE_FROM_PRETRAIN (finetuned), KEMP_SLEEP_TOKENIZER_BASELINE (scratch)

Usage:
    uv run python analysis/007_pretraining_loss_vs_downstream.py

Fill in the FINETUNE_RUNS dict below with wandb run IDs once the runs complete.
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = None

# --- From-scratch baselines (experiment 006) ---
SCRATCH_RUNS = {
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
}

# --- Finetuned runs (experiment 007) — fill in after runs complete ---
FINETUNE_RUNS = {
    "ResampleCNN": {
        "fold0": "FILL_IN",
        "fold1": "FILL_IN",
        "fold2": "FILL_IN",
    },
    "CWT-CNN": {
        "fold0": "FILL_IN",
        "fold1": "FILL_IN",
        "fold2": "FILL_IN",
    },
}

PRETRAIN_VAL_LOSS = {
    "ResampleCNN": 0.1201,
    "CWT-CNN": 0.0364,
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


def _has_valid_run_ids(runs: dict) -> bool:
    return all(
        rid != "FILL_IN" for folds in runs.values() for rid in folds.values()
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not _has_valid_run_ids(FINETUNE_RUNS):
        print(
            "WARNING: FINETUNE_RUNS contains placeholder IDs ('FILL_IN').\n"
            "Update the run IDs in this script after the finetuning runs "
            "complete, then re-run.\n"
            "Proceeding with from-scratch baselines only.\n"
        )

    rows = []

    for tokenizer, folds in SCRATCH_RUNS.items():
        for fold_name, run_id in folds.items():
            print(f"Fetching scratch {tokenizer} {fold_name} ({run_id})...")
            s = fetch_run_summary(run_id)
            rows.append(
                {
                    "Tokenizer": tokenizer,
                    "Condition": "From scratch",
                    "Fold": fold_name,
                    "Run ID": run_id,
                    "Best Val F1": s["best_val_f1"],
                    "Best Val Loss": s["best_val_loss"],
                    "Final Train Loss": s["final_train_loss"],
                    "Best Epoch": s["best_epoch"],
                    "Pretrain Val Loss": None,
                }
            )

    if _has_valid_run_ids(FINETUNE_RUNS):
        for tokenizer, folds in FINETUNE_RUNS.items():
            for fold_name, run_id in folds.items():
                print(
                    f"Fetching finetuned {tokenizer} {fold_name} ({run_id})..."
                )
                s = fetch_run_summary(run_id)
                rows.append(
                    {
                        "Tokenizer": tokenizer,
                        "Condition": "Finetuned",
                        "Fold": fold_name,
                        "Run ID": run_id,
                        "Best Val F1": s["best_val_f1"],
                        "Best Val Loss": s["best_val_loss"],
                        "Final Train Loss": s["final_train_loss"],
                        "Best Epoch": s["best_epoch"],
                        "Pretrain Val Loss": PRETRAIN_VAL_LOSS[tokenizer],
                    }
                )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("Per-run results")
    print("=" * 100)
    print(df.to_string(index=False))

    agg = (
        df.groupby(["Tokenizer", "Condition"])["Best Val F1"]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\n" + "=" * 100)
    print("Aggregated F1 across folds (mean ± std)")
    print("=" * 100)
    print(agg.to_string())

    # --- Figure 1: Grouped bar chart (scratch vs finetuned) ---
    if _has_valid_run_ids(FINETUNE_RUNS):
        _plot_comparison(df, agg)
        _plot_pretrain_loss_vs_downstream_f1(df)

    # --- Figure 2: Learning curves ---
    _plot_learning_curves(df)


def _plot_comparison(df: pd.DataFrame, agg: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    tokenizers = ["ResampleCNN", "CWT-CNN"]
    conditions = ["From scratch", "Finetuned"]
    x = np.arange(len(tokenizers))
    width = 0.35
    colors = {"From scratch": "#4C72B0", "Finetuned": "#DD8452"}

    for i, cond in enumerate(conditions):
        means = []
        stds = []
        for tok in tokenizers:
            if (tok, cond) in agg.index:
                means.append(agg.loc[(tok, cond), "mean"])
                stds.append(agg.loc[(tok, cond), "std"])
            else:
                means.append(0)
                stds.append(0)

        _ = ax.bar(
            x + i * width - width / 2,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=colors[cond],
            alpha=0.8,
            label=cond,
        )

        for j, tok in enumerate(tokenizers):
            vals = df[(df["Tokenizer"] == tok) & (df["Condition"] == cond)][
                "Best Val F1"
            ].values
            ax.scatter(
                [x[j] + i * width - width / 2] * len(vals),
                vals,
                color="black",
                s=25,
                zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tokenizers)
    ax.set_ylabel("Best Validation F1 (5-class)")
    ax.set_title("Kemp Sleep — Finetuned vs From Scratch")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "007_finetune_vs_scratch_f1.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def _plot_pretrain_loss_vs_downstream_f1(df: pd.DataFrame):
    """Scatter plot: pretraining val loss (x) vs downstream F1 (y)."""
    ft = df[df["Condition"] == "Finetuned"].copy()
    if ft.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"ResampleCNN": "#4C72B0", "CWT-CNN": "#55A868"}

    for tok in ["ResampleCNN", "CWT-CNN"]:
        subset = ft[ft["Tokenizer"] == tok]
        ax.scatter(
            subset["Pretrain Val Loss"],
            subset["Best Val F1"],
            color=colors[tok],
            s=60,
            label=tok,
            zorder=5,
        )
        mean_f1 = subset["Best Val F1"].mean()
        ax.axhline(mean_f1, color=colors[tok], linestyle="--", alpha=0.4)

    ax.set_xlabel("Pretraining Val Loss (lower = better)")
    ax.set_ylabel("Downstream Val F1 (higher = better)")
    ax.set_title("Pretraining Loss vs Downstream Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()

    out = FIGURES_DIR / "007_pretrain_loss_vs_downstream_f1.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def _plot_learning_curves(df: pd.DataFrame):
    """Val F1 learning curves for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        ("ResampleCNN", "From scratch"): ("#4C72B0", "-"),
        ("ResampleCNN", "Finetuned"): ("#4C72B0", "--"),
        ("CWT-CNN", "From scratch"): ("#55A868", "-"),
        ("CWT-CNN", "Finetuned"): ("#55A868", "--"),
    }

    for (tok, cond), (color, ls) in styles.items():
        sub = df[(df["Tokenizer"] == tok) & (df["Condition"] == cond)]
        if sub.empty:
            continue

        all_histories = []
        for _, row in sub.iterrows():
            rid = row["Run ID"]
            if rid == "FILL_IN":
                continue
            h = fetch_metric_history(rid, VAL_F1)
            if not h.empty:
                all_histories.append(h.set_index("epoch")[VAL_F1])
                ax.plot(
                    h["epoch"],
                    h[VAL_F1],
                    color=color,
                    alpha=0.15,
                    linewidth=0.6,
                    linestyle=ls,
                )

        if all_histories:
            combined = pd.concat(all_histories, axis=1)
            mean_curve = combined.mean(axis=1)
            ax.plot(
                mean_curve.index,
                mean_curve.values,
                color=color,
                linewidth=2,
                linestyle=ls,
                label=f"{tok} ({cond})",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 (5-class)")
    ax.set_title("Kemp Sleep — Val F1 Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "007_learning_curves.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()

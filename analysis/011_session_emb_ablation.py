"""Session embedding ablation: no-session-emb vs with-session-emb.

Fetches runs from KEMP_SESSION_EMB_ABLATION (no session emb, LR sweep),
and compares against baselines from previous experiments that ran on the
same fold (KEMP_SCRATCH_HP_SEARCH and KEMP_FINETUNE_HP_SEARCH from exp 009).

Key analysis: compare val F1 and train-val loss gap with and without session
embeddings to quantify how much session embeddings contribute to
subject-level overfitting.

WandB project: foundry_finetuning

Usage:
    uv run python analysis/011_session_emb_ablation.py
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

WANDB_PROJECT = "foundry_finetuning"
WANDB_ENTITY = default_entity()

ABLATION_GROUP = "KEMP_SESSION_EMB_ABLATION"

# Baseline groups from previous experiments (same fold 0)
BASELINE_GROUPS = {
    "Scratch (with sess, exp 009)": "KEMP_SCRATCH_HP_SEARCH",
    "Pretrained (with sess, exp 009)": "KEMP_FINETUNE_HP_SEARCH",
}

VAL_F1 = "val/sleep_stage_5class_f1"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"

FIGURES_DIR = figures_dir(__file__)


def _fetch_group_runs(group: str, api: wandb.Api) -> list:
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    return list(api.runs(path, filters={"group": group}))


def _extract_from_run(run) -> dict:
    config = run.config
    hp = config.get("hyperparameters", {})
    model_cfg = config.get("model", {})
    run_cfg = config.get("run", {})

    is_pretrained = run_cfg.get("pretrained_checkpoint") is not None
    no_session_emb = model_cfg.get("disable_session_emb", False)

    s = run.summary
    train_loss = unwrap_summary_value(s.get(TRAIN_LOSS), "min")
    val_loss = unwrap_summary_value(s.get(VAL_LOSS), "min")
    val_f1 = unwrap_summary_value(s.get(VAL_F1), "max")

    gap = (
        val_loss - train_loss
        if isinstance(train_loss, float) and isinstance(val_loss, float)
        else None
    )

    return {
        "LR": hp.get("learning_rate"),
        "Fold": hp.get("fold_number", 0),
        "Init": "Pretrained" if is_pretrained else "Scratch",
        "Session Emb": "Disabled" if no_session_emb else "Enabled",
        "Run ID": run.id,
        "Run Name": run.name,
        "best_val_f1": val_f1,
        "best_val_loss": val_loss,
        "final_train_loss": train_loss,
        "train_val_gap": gap,
        "best_epoch": unwrap_summary_value(s.get("epoch"), "max"),
    }


def fetch_ablation_runs(api: wandb.Api) -> pd.DataFrame:
    """Fetch runs from the session embedding ablation group."""
    rows = []
    runs = _fetch_group_runs(ABLATION_GROUP, api)
    if not runs:
        print(f"  No runs found for group '{ABLATION_GROUP}'")
        return pd.DataFrame(rows)

    print(f"  Found {len(runs)} runs in {ABLATION_GROUP}")
    for run in runs:
        if run.state != "finished":
            print(f"    Skipping {run.id} (state={run.state})")
            continue
        rows.append(_extract_from_run(run))

    return pd.DataFrame(rows)


def fetch_baseline_runs(api: wandb.Api, fold: int = 0) -> pd.DataFrame:
    """Fetch baseline runs from previous experiment groups, filtered to same fold."""
    rows = []
    for label, group in BASELINE_GROUPS.items():
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs found for baseline group '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for baseline: {label} ({group})")

        for run in runs:
            if run.state != "finished":
                continue
            row = _extract_from_run(run)
            if row["Fold"] != fold:
                continue
            row["Source"] = label
            rows.append(row)

    return pd.DataFrame(rows)


def get_best_baselines(baseline_df: pd.DataFrame) -> dict:
    """Extract best val F1 per init condition from baseline runs."""
    best = {}
    for init in ["Scratch", "Pretrained"]:
        sub = baseline_df[baseline_df["Init"] == init]
        if sub.empty:
            continue
        best_row = sub.loc[sub["best_val_f1"].idxmax()]
        best[init] = {
            "val_f1": best_row["best_val_f1"],
            "lr": best_row["LR"],
            "run_id": best_row["Run ID"],
            "train_val_gap": best_row["train_val_gap"],
        }
    return best


def print_results(ablation_df: pd.DataFrame, baselines: dict) -> None:
    print(f"\n{'=' * 70}")
    print("  Session Embedding Ablation — LR Sweep (fold 0, no session emb)")
    print(f"{'=' * 70}")

    for init in ["Scratch", "Pretrained"]:
        sub = ablation_df[ablation_df["Init"] == init]
        if sub.empty:
            continue
        print(f"\n--- {init} (session emb disabled) ---")
        for _, row in sub.sort_values(
            "best_val_f1", ascending=False
        ).iterrows():
            gap_str = (
                f"{row['train_val_gap']:+.4f}" if row["train_val_gap"] else "?"
            )
            print(
                f"  lr={row['LR']:.0e}  "
                f"train_loss={row['final_train_loss']:.4f}  "
                f"val_loss={row['best_val_loss']:.4f}  "
                f"gap={gap_str}  "
                f"val_f1={row['best_val_f1']:.4f}  "
                f"({row['Run ID']})"
            )

    print(f"\n{'=' * 70}")
    print("  Baselines from previous experiments (fold 0, with session emb)")
    print(f"{'=' * 70}")
    for init, info in baselines.items():
        print(
            f"  {init:>10} best: val_f1={info['val_f1']:.4f}  "
            f"lr={info['lr']:.0e}  gap={info['train_val_gap']:+.4f}  "
            f"({info['run_id']})"
        )

    print(f"\n{'=' * 70}")
    print("  Comparison: best no-session-emb vs best with-session-emb")
    print(f"{'=' * 70}")
    for init in ["Scratch", "Pretrained"]:
        sub = ablation_df[ablation_df["Init"] == init]
        if sub.empty or init not in baselines:
            continue
        best_ablation = sub.loc[sub["best_val_f1"].idxmax()]
        baseline_f1 = baselines[init]["val_f1"]
        delta = best_ablation["best_val_f1"] - baseline_f1
        print(
            f"  {init:>10}: no_sess={best_ablation['best_val_f1']:.4f} "
            f"vs with_sess={baseline_f1:.4f}  "
            f"Δ={delta:+.4f} ({delta * 100:+.1f} pp)"
        )


def plot_train_val_gap(
    ablation_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> None:
    """Bar chart of train-val gap for each condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    gaps = []
    colors = []

    for init, color_no_sess, color_with_sess in [
        ("Scratch", "#4C72B0", "#A0C0E8"),
        ("Pretrained", "#DD8452", "#E8A0A0"),
    ]:
        # Best no-session-emb
        sub = ablation_df[ablation_df["Init"] == init]
        if not sub.empty:
            best = sub.loc[sub["best_val_f1"].idxmax()]
            if best["train_val_gap"] is not None:
                labels.append(f"{init}\nNo sess emb\n(lr={best['LR']:.0e})")
                gaps.append(best["train_val_gap"])
                colors.append(color_no_sess)

        # Best with-session-emb (from baselines)
        base_sub = baseline_df[baseline_df["Init"] == init]
        if not base_sub.empty:
            best_base = base_sub.loc[base_sub["best_val_f1"].idxmax()]
            if best_base["train_val_gap"] is not None:
                labels.append(
                    f"{init}\nWith sess emb\n(lr={best_base['LR']:.0e})"
                )
                gaps.append(best_base["train_val_gap"])
                colors.append(color_with_sess)

    x = np.arange(len(labels))
    bars = ax.bar(
        x, gaps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )

    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Val Loss − Train Loss (gap)")
    ax.set_title("Train-Val Loss Gap — Session Emb Ablation (fold 0)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "011_train_val_gap.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_comparison_bar(ablation_df: pd.DataFrame, baselines: dict) -> None:
    """Bar chart comparing best F1 across all conditions."""
    labels = []
    vals = []
    colors = []

    # Baselines (with session emb, from previous experiments)
    if "Scratch" in baselines:
        info = baselines["Scratch"]
        labels.append(f"Exp 009\nScratch\n(with sess, lr={info['lr']:.0e})")
        vals.append(info["val_f1"])
        colors.append("#A0C0E8")

    if "Pretrained" in baselines:
        info = baselines["Pretrained"]
        labels.append(f"Exp 009\nPretrained\n(with sess, lr={info['lr']:.0e})")
        vals.append(info["val_f1"])
        colors.append("#E8A0A0")

    # Ablation runs (no session emb)
    for init, color in [("Scratch", "#4C72B0"), ("Pretrained", "#DD8452")]:
        sub = ablation_df[ablation_df["Init"] == init]
        if sub.empty:
            continue
        best = sub.loc[sub["best_val_f1"].idxmax()]
        labels.append(f"Exp 011\n{init}\n(no sess, lr={best['LR']:.0e})")
        vals.append(best["best_val_f1"])
        colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Val F1 (5-class)")
    ax.set_title("Session Embedding Ablation — Val F1 Comparison (fold 0)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(vals) - 0.05))
    plt.tight_layout()

    out = FIGURES_DIR / "011_session_emb_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_lr_sweep(ablation_df: pd.DataFrame, baselines: dict) -> None:
    """Line plot showing val F1 across LR sweep, with baseline reference lines."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for init, color, marker in [
        ("Scratch", "#4C72B0", "o"),
        ("Pretrained", "#DD8452", "s"),
    ]:
        sub = ablation_df[ablation_df["Init"] == init].sort_values("LR")
        if sub.empty:
            continue
        ax.plot(
            range(len(sub)),
            sub["best_val_f1"],
            marker=marker,
            color=color,
            label=f"{init} (no session emb)",
            linewidth=1.5,
            markersize=8,
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in sub["LR"]])

        if init in baselines:
            ax.axhline(
                baselines[init]["val_f1"],
                color=color,
                linestyle="--",
                alpha=0.6,
                label=f"{init} baseline (with sess, exp 009)",
            )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Val F1")
    ax.set_title("LR Sweep — No Session Emb vs Baselines (fold 0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "011_lr_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def main():
    api = wandb.Api()

    print("Fetching session embedding ablation runs...")
    ablation_df = fetch_ablation_runs(api)

    if ablation_df.empty:
        print("\nNo completed ablation runs found. Launch Phase 1 first.")
        return

    print(f"\nTotal ablation runs: {len(ablation_df)}")

    print("\nFetching baseline runs from previous experiments (fold 0)...")
    baseline_df = fetch_baseline_runs(api, fold=0)
    baselines = get_best_baselines(baseline_df)

    if not baselines:
        print("WARNING: Could not fetch baselines from previous experiments.")
        print("  Comparison will be incomplete.")

    print_results(ablation_df, baselines)
    plot_train_val_gap(ablation_df, baseline_df)
    plot_comparison_bar(ablation_df, baselines)
    plot_lr_sweep(ablation_df, baselines)


if __name__ == "__main__":
    main()

"""Session embedding ablation: no-session-emb vs with-session-emb.

Fetches runs from KEMP_SESSION_EMB_ABLATION (no session emb, LR sweep),
KEMP_SESSION_EMB_ABLATION_CONTROLS (with session emb baselines), and
KEMP_SESSION_EMB_ABLATION_VALIDATION (3-fold validation, if launched).

Key analysis: compare train-val loss gap with and without session embeddings
to quantify how much session embeddings contribute to subject-level overfitting.

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
CONTROLS_GROUP = "KEMP_SESSION_EMB_ABLATION_CONTROLS"
VALIDATION_GROUP = "KEMP_SESSION_EMB_ABLATION_VALIDATION"

EXP_009_BEST_SCRATCH = 0.5629
EXP_009_BEST_PRETRAINED = 0.5425

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


def fetch_all_runs(api: wandb.Api) -> pd.DataFrame:
    rows = []
    for label, group in [
        ("Ablation", ABLATION_GROUP),
        ("Controls", CONTROLS_GROUP),
    ]:
        runs = _fetch_group_runs(group, api)
        if not runs:
            print(f"  No runs found for group '{group}' — skipping.")
            continue
        print(f"  Found {len(runs)} runs for {label} ({group})")

        for run in runs:
            if run.state != "finished":
                print(f"    Skipping {run.id} (state={run.state})")
                continue
            rows.append(_extract_from_run(run))

    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print("  Session Embedding Ablation — All Runs (fold 0)")
    print(f"{'=' * 70}")

    for session_mode in ["Disabled", "Enabled"]:
        sub = df[df["Session Emb"] == session_mode]
        if sub.empty:
            continue
        print(f"\n--- Session Emb: {session_mode} ---")
        for _, row in sub.sort_values(
            ["Init", "best_val_f1"], ascending=[True, False]
        ).iterrows():
            gap_str = (
                f"{row['train_val_gap']:+.4f}" if row["train_val_gap"] else "?"
            )
            print(
                f"  {row['Init']:>10}  lr={row['LR']:.0e}  "
                f"train_loss={row['final_train_loss']:.4f}  "
                f"val_loss={row['best_val_loss']:.4f}  "
                f"gap={gap_str}  "
                f"val_f1={row['best_val_f1']:.4f}  "
                f"({row['Run ID']})"
            )

    print(f"\n{'=' * 70}")
    print("  Reference baselines (fold 0)")
    print(f"{'=' * 70}")
    print(
        f"  Exp 009 scratch (with session emb):    {EXP_009_BEST_SCRATCH:.4f}"
    )
    print(
        f"  Exp 009 pretrained (with session emb):  {EXP_009_BEST_PRETRAINED:.4f}"
    )


def plot_train_val_gap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, init in zip(axes, ["Scratch", "Pretrained"]):
        sub = df[df["Init"] == init].copy()
        if sub.empty:
            ax.set_title(f"{init} — no data")
            continue

        for session_mode, color, marker in [
            ("Disabled", "#4C72B0", "o"),
            ("Enabled", "#DD8452", "s"),
        ]:
            mode_df = sub[sub["Session Emb"] == session_mode].sort_values("LR")
            if mode_df.empty:
                continue
            ax.plot(
                range(len(mode_df)),
                mode_df["train_val_gap"],
                marker=marker,
                color=color,
                label=f"Session emb {session_mode.lower()}",
                linewidth=1.5,
                markersize=8,
            )
            ax.set_xticks(range(len(mode_df)))
            ax.set_xticklabels([f"{lr:.0e}" for lr in mode_df["LR"]])

        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Val Loss − Train Loss")
        ax.set_title(f"{init} — Train-Val Loss Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "011_train_val_gap.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.close()


def plot_comparison_bar(df: pd.DataFrame) -> None:
    labels = []
    vals = []
    colors = []

    labels.append("Exp 009\nScratch\n(with sess)")
    vals.append(EXP_009_BEST_SCRATCH)
    colors.append("#A0C0E8")

    labels.append("Exp 009\nPretrained\n(with sess)")
    vals.append(EXP_009_BEST_PRETRAINED)
    colors.append("#E8A0A0")

    for init in ["Scratch", "Pretrained"]:
        sub = df[(df["Init"] == init) & (df["Session Emb"] == "Disabled")]
        if sub.empty:
            continue
        best = sub.loc[sub["best_val_f1"].idxmax()]
        labels.append(f"Exp 011\n{init}\n(no sess, lr={best['LR']:.0e})")
        vals.append(best["best_val_f1"])
        colors.append("#4C72B0" if init == "Scratch" else "#DD8452")

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


def main():
    api = wandb.Api()

    print("Fetching session embedding ablation runs...")
    df = fetch_all_runs(api)

    if df.empty:
        print("\nNo completed runs found. Launch Phase 1 first.")
        return

    print(f"\nTotal completed runs: {len(df)}")
    print_results(df)
    plot_train_val_gap(df)
    plot_comparison_bar(df)

    print("\nChecking for validation runs...")
    val_runs = _fetch_group_runs(VALIDATION_GROUP, api)
    finished = [r for r in val_runs if r.state == "finished"]
    if finished:
        rows = [_extract_from_run(r) for r in finished]
        val_df = pd.DataFrame(rows)
        agg = (
            val_df.groupby(["Init", "Session Emb"])["best_val_f1"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(f"\n{'=' * 60}")
        print("  Phase 2 — 3-fold validation (mean ± std)")
        print(f"{'=' * 60}")
        print(agg.to_string())
    else:
        print("  No validation runs yet (Phase 2 not launched).")


if __name__ == "__main__":
    main()

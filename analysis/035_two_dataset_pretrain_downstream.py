"""Phase 2 & 3 analysis: Two-dataset pretrain downstream evaluation.

Fetches finetuning (Phase 2) and linear probing (Phase 3) results from WandB
for the two-dataset pretraining experiment. Compares 4 POYO variants
(2 tokenizers × 2 channel_emb modes) across 3 downstream tasks against
the best from-scratch baseline per task.

For each run, the BEST (max) validation F1 across all epochs is used,
corresponding to the model checkpoint that would be selected by early stopping.

Usage:
    uv run python analysis/035_two_dataset_pretrain_downstream.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PROJECT = "foundry_finetuning"

PHASE2_GROUPS = {
    "Kemp Sleep": "KEMP_FINETUNE_FROM_2DS_PRETRAIN",
    "PhysioNet MI": "PHYSIONET_MI_FINETUNE_FROM_2DS_PRETRAIN",
    "Brain Invaders P300": "BI_P300_FINETUNE_FROM_2DS_PRETRAIN",
}

PHASE3_GROUPS = {
    "Kemp Sleep": "KEMP_LINEAR_PROBE_FROM_2DS_PRETRAIN",
    "PhysioNet MI": "PHYSIONET_MI_LINEAR_PROBE_FROM_2DS_PRETRAIN",
    "Brain Invaders P300": "BI_P300_LINEAR_PROBE_FROM_2DS_PRETRAIN",
}

METRIC_KEYS = {
    "Kemp Sleep": "val/sleep_stage_5class_f1",
    "PhysioNet MI": "val/motor_imagery_binary_f1",
    "Brain Invaders P300": "val/p300_binary_f1",
}

# Best from-scratch baseline per task (across ALL models including EEGNet)
BEST_BASELINE = {
    "Kemp Sleep": ("POYO CWT-CNN (dynamic)", 0.730),
    "PhysioNet MI": ("EEGNet", 0.887),
    "Brain Invaders P300": ("EEGNet", 0.386),
}

# Best POYO-only from-scratch baseline per task
BEST_POYO_BASELINE = {
    "Kemp Sleep": ("CWT-CNN / dynamic", 0.730),
    "PhysioNet MI": ("CWT-CNN / disabled", 0.884),
    "Brain Invaders P300": ("CWT-CNN / dynamic", 0.364),
}

TOKENIZER_MAP = {
    "per_channel_cwt_cnn": "CWT-CNN",
    "per_channel_resample_cnn": "ResampleCNN",
}


def parse_run_variant(run_name: str) -> tuple[str, str, int] | None:
    """Extract (tokenizer, channel_emb, fold) from a run name."""
    for tok_key, tok_label in TOKENIZER_MAP.items():
        if tok_key in run_name:
            tokenizer = tok_label
            break
    else:
        return None

    if "_ch-disabled_" in run_name:
        ch_emb = "disabled"
    elif "_ch-dynamic_" in run_name:
        ch_emb = "dynamic"
    else:
        return None

    for i in range(3):
        if f"fold{i}" in run_name:
            fold = i
            break
    else:
        return None

    return tokenizer, ch_emb, fold


def fetch_group_results(group: str, metric_key: str, api: wandb.Api) -> pd.DataFrame:
    """Fetch best (max) metric across all epochs for each run in a group."""
    entity = api.default_entity
    runs = api.runs(
        f"{entity}/{PROJECT}",
        filters={"group": group},
    )

    records = []
    for run in runs:
        variant = parse_run_variant(run.name)
        if variant is None:
            print(f"  [WARN] Could not parse run name: {run.name}")
            continue

        tokenizer, ch_emb, fold = variant

        # Always fetch full history and take max — don't trust summary
        history = run.history(keys=[metric_key], samples=50000, pandas=True)
        if metric_key in history.columns:
            vals = history[metric_key].dropna()
            best_f1 = float(vals.max()) if len(vals) > 0 else None
            best_epoch = int(vals.idxmax()) if len(vals) > 0 else None
            num_epochs = len(vals)
        else:
            best_f1 = None
            best_epoch = None
            num_epochs = 0

        records.append({
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "tokenizer": tokenizer,
            "channel_emb": ch_emb,
            "fold": fold,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "num_epochs": num_epochs,
        })

    return pd.DataFrame(records)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std of best F1 across folds for each variant."""
    summary = (
        df.groupby(["tokenizer", "channel_emb"])["best_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["variant"] = summary["tokenizer"] + " / " + summary["channel_emb"]
    return summary


def print_phase2_table(task: str, summary: pd.DataFrame) -> None:
    """Print Phase 2 comparison table vs best baseline."""
    best_name, best_val = BEST_BASELINE[task]
    poyo_name, poyo_val = BEST_POYO_BASELINE[task]

    print(f"\n{'='*75}")
    print(f"  {task} — FINETUNING (Phase 2)")
    print(f"  Best overall baseline: {best_name} = {best_val:.3f}")
    print(f"  Best POYO baseline:    {poyo_name} = {poyo_val:.3f}")
    print(f"{'='*75}")
    print(f"  {'Variant':<28} {'Mean F1':>8} {'± Std':>8} {'N':>3}  {'Δ Best':>8} {'Δ POYO':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*3}  {'-'*8} {'-'*8}")

    for _, row in summary.sort_values("mean", ascending=False).iterrows():
        variant = row["variant"]
        mean_f1 = row["mean"]
        std_f1 = row["std"]
        n = int(row["count"])

        delta_best = mean_f1 - best_val
        delta_poyo = mean_f1 - poyo_val

        print(
            f"  {variant:<28} {mean_f1:.3f}    {std_f1:.3f}    {n:>3}"
            f"  {delta_best:>+8.3f} {delta_poyo:>+8.3f}"
        )


def print_phase3_table(task: str, summary: pd.DataFrame) -> None:
    """Print Phase 3 linear probe table (representation quality)."""
    print(f"\n{'='*75}")
    print(f"  {task} — LINEAR PROBE (Phase 3, representation quality)")
    print(f"{'='*75}")
    print(f"  {'Variant':<28} {'Mean F1':>8} {'± Std':>8} {'N':>3}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*3}")

    for _, row in summary.sort_values("mean", ascending=False).iterrows():
        variant = row["variant"]
        mean_f1 = row["mean"]
        std_f1 = row["std"]
        n = int(row["count"])
        print(f"  {variant:<28} {mean_f1:.3f}    {std_f1:.3f}    {n:>3}")


def make_comparison_figure(
    all_results: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    """Generate grouped bar chart: pretrained finetuning vs best baseline."""
    tasks = list(METRIC_KEYS.keys())
    variants = [
        "CWT-CNN / disabled",
        "CWT-CNN / dynamic",
        "ResampleCNN / disabled",
        "ResampleCNN / dynamic",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        _, best_val = BEST_BASELINE[task]

        summary = all_results[task].get("Phase 2: Finetuning")
        if summary is None or summary.empty:
            continue

        x = np.arange(len(variants))
        means = []
        stds = []
        for variant in variants:
            row = summary[summary["variant"] == variant]
            if not row.empty:
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(
            x, means, 0.6,
            yerr=stds, capsize=4,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw=dict(lw=1.2),
        )

        ax.axhline(y=best_val, color="black", linestyle="--", linewidth=1.5,
                   label=f"Best baseline ({best_val:.3f})")

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace(" / ", "\n") for v in variants], fontsize=8
        )
        ax.set_ylabel("Best Val F1 (max across epochs)" if ax_idx == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", fontsize=8)

        ymin = min(means) - 0.05 if means else 0
        ymax = max(max(means), best_val) + 0.04
        ax.set_ylim(ymin, ymax)

    fig.suptitle(
        "Phase 2: Pretrained Finetuning vs Best Baseline (max F1 per fold)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "035_phase2_finetuning_vs_best_baseline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
    return out


def make_linear_probe_figure(
    all_results: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    """Bar chart showing linear probe representation quality."""
    tasks = list(METRIC_KEYS.keys())
    variants = [
        "CWT-CNN / disabled",
        "CWT-CNN / dynamic",
        "ResampleCNN / disabled",
        "ResampleCNN / dynamic",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        summary = all_results[task].get("Phase 3: Linear Probe")
        if summary is None or summary.empty:
            continue

        x = np.arange(len(variants))
        means = []
        stds = []
        for variant in variants:
            row = summary[summary["variant"] == variant]
            if not row.empty:
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(
            x, means, 0.6,
            yerr=stds, capsize=4,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw=dict(lw=1.2),
        )

        # Reference: best finetuned result for this task (shows gap to full finetuning)
        ft_summary = all_results[task].get("Phase 2: Finetuning")
        if ft_summary is not None and not ft_summary.empty:
            ft_best = ft_summary["mean"].max()
            ax.axhline(y=ft_best, color="black", linestyle="--", linewidth=1.2,
                       label=f"Best finetuned ({ft_best:.3f})")

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace(" / ", "\n") for v in variants], fontsize=8
        )
        ax.set_ylabel("Best Val F1 (frozen backbone)" if ax_idx == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, max(means) + 0.08 if means else 1.0)

    fig.suptitle(
        "Phase 3: Linear Probe — Pretrained Representation Quality",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "035_phase3_linear_probe_representation_quality.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
    return out


def make_delta_heatmap(
    all_results: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    """Heatmap: Phase 2 delta vs best baseline per task."""
    tasks = list(METRIC_KEYS.keys())
    variants = [
        "CWT-CNN / disabled",
        "CWT-CNN / dynamic",
        "ResampleCNN / disabled",
        "ResampleCNN / dynamic",
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    delta_matrix = np.full((len(variants), len(tasks)), np.nan)

    for j, task in enumerate(tasks):
        _, best_val = BEST_BASELINE[task]
        summary = all_results[task].get("Phase 2: Finetuning")
        if summary is None or summary.empty:
            continue
        for i, variant in enumerate(variants):
            row = summary[summary["variant"] == variant]
            if not row.empty:
                m = row["mean"].values[0]
                if not np.isnan(m):
                    delta_matrix[i, j] = m - best_val

    vmax = max(0.03, np.nanmax(np.abs(delta_matrix)))
    im = ax.imshow(
        delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto",
    )

    for i in range(len(variants)):
        for j in range(len(tasks)):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:+.3f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if abs(val) < vmax * 0.7 else "white",
                )

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label="ΔF1 vs best baseline")

    ax.set_title(
        "Finetuning Transfer Gain vs Best Baseline\n(max F1 per fold, mean across folds)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "035_phase2_delta_vs_best_baseline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def main():
    api = wandb.Api()
    print(f"WandB entity: {api.default_entity}")
    print(f"Project: {PROJECT}")
    print(f"\nNOTE: For each run, we take the MAX val F1 across all epochs")
    print(f"      (best epoch per fold, as selected by early stopping).")

    all_results: dict[str, dict[str, pd.DataFrame]] = {}

    # Phase 2: Finetuning
    print("\n" + "=" * 75)
    print("  PHASE 2: DOWNSTREAM FINETUNING")
    print("=" * 75)

    for task, group in PHASE2_GROUPS.items():
        metric = METRIC_KEYS[task]
        print(f"\nFetching {task} (group={group})...")
        df = fetch_group_results(group, metric, api)
        valid = df[df["best_f1"].notna()]
        print(f"  Found {len(df)} runs, {len(valid)} with metrics")

        if valid.empty:
            all_results.setdefault(task, {})["Phase 2: Finetuning"] = pd.DataFrame()
            continue

        # Show per-fold detail
        print(f"\n  Per-fold best F1:")
        for _, row in valid.sort_values(["tokenizer", "channel_emb", "fold"]).iterrows():
            print(
                f"    {row['tokenizer']:<14} {row['channel_emb']:<9} fold{row['fold']}"
                f"  F1={row['best_f1']:.4f}  (best @ step {row['best_epoch']}, "
                f"{row['num_epochs']} logged epochs, state={row['state']})"
            )

        summary = summarize_results(valid)
        print_phase2_table(task, summary)
        all_results.setdefault(task, {})["Phase 2: Finetuning"] = summary

    # Phase 3: Linear Probing
    print("\n\n" + "=" * 75)
    print("  PHASE 3: LINEAR PROBING (representation quality)")
    print("=" * 75)

    for task, group in PHASE3_GROUPS.items():
        metric = METRIC_KEYS[task]
        print(f"\nFetching {task} (group={group})...")
        df = fetch_group_results(group, metric, api)
        valid = df[df["best_f1"].notna()]
        print(f"  Found {len(df)} runs, {len(valid)} with metrics")

        if valid.empty:
            all_results.setdefault(task, {})["Phase 3: Linear Probe"] = pd.DataFrame()
            continue

        print(f"\n  Per-fold best F1:")
        for _, row in valid.sort_values(["tokenizer", "channel_emb", "fold"]).iterrows():
            print(
                f"    {row['tokenizer']:<14} {row['channel_emb']:<9} fold{row['fold']}"
                f"  F1={row['best_f1']:.4f}  (best @ step {row['best_epoch']}, "
                f"{row['num_epochs']} logged epochs, state={row['state']})"
            )

        summary = summarize_results(valid)
        print_phase3_table(task, summary)
        all_results.setdefault(task, {})["Phase 3: Linear Probe"] = summary

    # Final summary
    print("\n\n" + "=" * 75)
    print("  FINAL SUMMARY")
    print("=" * 75)

    print("\n  Phase 2 — Finetuning (vs best overall baseline per task):")
    print(f"  {'Task':<22} {'Best Pretrained Variant':<28} {'F1':>6} {'Δ Best Baseline':>16}")
    print(f"  {'-'*22} {'-'*28} {'-'*6} {'-'*16}")
    for task in METRIC_KEYS:
        summary = all_results.get(task, {}).get("Phase 2: Finetuning")
        if summary is None or summary.empty:
            continue
        best_row = summary.loc[summary["mean"].idxmax()]
        _, baseline_val = BEST_BASELINE[task]
        delta = best_row["mean"] - baseline_val
        print(
            f"  {task:<22} {best_row['variant']:<28} {best_row['mean']:.3f}"
            f" {delta:>+16.3f}"
        )

    print("\n  Phase 3 — Linear Probe (representation quality, best variant per task):")
    print(f"  {'Task':<22} {'Best Variant':<28} {'F1':>6}")
    print(f"  {'-'*22} {'-'*28} {'-'*6}")
    for task in METRIC_KEYS:
        summary = all_results.get(task, {}).get("Phase 3: Linear Probe")
        if summary is None or summary.empty:
            continue
        best_row = summary.loc[summary["mean"].idxmax()]
        print(f"  {task:<22} {best_row['variant']:<28} {best_row['mean']:.3f}")

    # Generate figures
    print("\n\nGenerating figures...")
    make_comparison_figure(all_results)
    make_linear_probe_figure(all_results)
    make_delta_heatmap(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()

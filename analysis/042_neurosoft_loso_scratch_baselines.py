"""Analyze NeuroSoft 8-band LOSO from-scratch baselines for minipigs and monkeys.

Fetches all runs from the NEUROSOFT_8B_LOSO_SCRATCH_BASELINES group, extracts
per-subject best validation F1, and produces summary tables and comparison figures.

Usage:
    uv run python analysis/042_neurosoft_loso_scratch_baselines.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PROJECT = "auditory_decoding"
GROUP = "NEUROSOFT_8B_LOSO_SCRATCH_BASELINES"
METRIC = "val/neurosoft_acoustic_stim_8band_f1"


def classify_run(run) -> dict | None:
    """Extract species and held-out subject from a LOSO scratch run."""
    name = run.name.lower()

    if "minipig" in name:
        species = "minipigs"
    elif "monkey" in name:
        species = "monkeys"
    else:
        return None

    match = re.search(r"(sub-\d+)", name)
    subject = match.group(1) if match else None

    return {"species": species, "subject": subject}


def fetch_best_f1(run) -> float | None:
    """Get best validation F1 from a run's summary or history."""
    summary_val = run.summary.get(METRIC)
    if summary_val is not None and isinstance(summary_val, (int, float)):
        return float(summary_val)

    history = run.history(keys=[METRIC], samples=50_000, pandas=True)
    scores = history.get(METRIC, pd.Series(dtype=float)).dropna()
    if scores.empty:
        return None
    return float(scores.max())


def fetch_runs(api: wandb.Api) -> pd.DataFrame:
    """Fetch all LOSO scratch baseline runs."""
    entity = api.default_entity
    records = []

    runs = api.runs(f"{entity}/{PROJECT}", filters={"group": GROUP})
    for run in runs:
        if run.state not in ("finished", "running"):
            continue
        info = classify_run(run)
        if info is None:
            continue
        best_f1 = fetch_best_f1(run)
        if best_f1 is not None:
            records.append(
                {
                    **info,
                    "best_f1": best_f1,
                    "run_name": run.name,
                    "run_id": run.id,
                    "state": run.state,
                }
            )

    return pd.DataFrame(records)


def plot_loso_per_subject(df: pd.DataFrame) -> Path:
    """Side-by-side bar charts of per-subject LOSO F1 for both species."""
    species_list = ["minipigs", "monkeys"]
    n_species = sum(1 for s in species_list if not df[df["species"] == s].empty)
    if n_species == 0:
        return FIGURES_DIR / "042_neurosoft_loso_scratch_subjects.png"

    fig, axes = plt.subplots(1, n_species, figsize=(7 * n_species, 5), squeeze=False)
    ax_idx = 0

    for species in species_list:
        subset = df[df["species"] == species].sort_values("subject")
        if subset.empty:
            continue
        ax = axes[0, ax_idx]
        ax_idx += 1

        x = np.arange(len(subset))
        ax.bar(
            x, subset["best_f1"].values,
            color="#7f8c8d", edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        mean_f1 = subset["best_f1"].mean()
        ax.axhline(
            mean_f1, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Mean = {mean_f1:.3f}",
        )
        chance = 1.0 / 8
        ax.axhline(
            chance, color="#95a5a6", linestyle=":", linewidth=1.2,
            label=f"Chance = {chance:.3f}",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(subset["subject"].values, fontsize=9, rotation=45, ha="right")
        ax.set_xlabel("Held-out Subject")
        ax.set_ylabel("Best Validation F1")
        ax.set_title(f"{species.capitalize()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ymax = max(subset["best_f1"].max() * 1.3 + 0.05, 0.3)
        ax.set_ylim(0, min(1.0, ymax))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "NeuroSoft 8-Band LOSO: From-Scratch Baselines",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "042_neurosoft_loso_scratch_subjects.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(api: wandb.Api, df: pd.DataFrame) -> Path:
    """Training curves (F1 over validation steps) for all LOSO runs."""
    entity = api.default_entity
    species_list = ["minipigs", "monkeys"]
    n_species = sum(1 for s in species_list if not df[df["species"] == s].empty)
    if n_species == 0:
        return FIGURES_DIR / "042_neurosoft_loso_scratch_curves.png"

    fig, axes = plt.subplots(1, n_species, figsize=(7 * n_species, 5), squeeze=False)
    ax_idx = 0

    cmap_minipigs = plt.cm.Set2
    cmap_monkeys = plt.cm.Set1

    for species in species_list:
        subset = df[df["species"] == species].sort_values("subject")
        if subset.empty:
            continue
        ax = axes[0, ax_idx]
        cmap = cmap_minipigs if species == "minipigs" else cmap_monkeys
        ax_idx += 1

        for i, (_, row) in enumerate(subset.iterrows()):
            try:
                run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
                history = run.history(keys=[METRIC], samples=50_000, pandas=True)
                scores = history[METRIC].dropna()
                if scores.empty:
                    continue
                ax.plot(
                    range(len(scores)), scores.values,
                    color=cmap(i / max(len(subset) - 1, 1)),
                    alpha=0.7, linewidth=1.2,
                    label=row["subject"],
                )
            except Exception:
                continue

        ax.set_xlabel("Validation Step")
        ax.set_ylabel("Validation F1")
        ax.set_title(f"{species.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="lower right", ncol=2)

    fig.suptitle(
        "NeuroSoft 8-Band LOSO Scratch: Validation F1 Training Curves",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "042_neurosoft_loso_scratch_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    api = wandb.Api()

    print("=" * 70)
    print("NeuroSoft 8-Band LOSO: From-Scratch Baselines")
    print("=" * 70)

    print(f"\nFetching runs from group: {GROUP}")
    df = fetch_runs(api)

    if df.empty:
        print("[ERROR] No runs found. Check group name and WandB project.")
        return

    print(f"Found {len(df)} completed runs.\n")

    for species in ["minipigs", "monkeys"]:
        subset = df[df["species"] == species].sort_values("subject")
        if subset.empty:
            print(f"\n[WARN] No {species} runs found.")
            continue

        print(f"\n{'─' * 70}")
        print(f"  {species.upper()} — Per-Subject Results")
        print(f"{'─' * 70}")
        print(
            subset[["subject", "best_f1", "run_name", "state"]]
            .to_string(index=False)
        )

        mean_f1 = subset["best_f1"].mean()
        std_f1 = subset["best_f1"].std()
        min_f1 = subset["best_f1"].min()
        max_f1 = subset["best_f1"].max()

        print(f"\n  Subjects:  {len(subset)}")
        print(f"  Mean F1:   {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Min F1:    {min_f1:.4f}")
        print(f"  Max F1:    {max_f1:.4f}")
        print(f"  Chance:    {1/8:.4f}")
        print(f"  All above chance: {'Yes' if min_f1 > 1/8 else 'No'}")

    # --- Cross-species summary ---
    print(f"\n{'=' * 70}")
    print("CROSS-SPECIES SUMMARY")
    print(f"{'=' * 70}")
    summary = (
        df.groupby("species")["best_f1"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary["mean±std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )
    print(summary[["species", "mean±std", "min", "max", "count"]].to_string(index=False))

    # --- Figures ---
    print("\nGenerating figures...")
    fig_subj = plot_loso_per_subject(df)
    print(f"Saved: {fig_subj}")

    fig_curves = plot_training_curves(api, df)
    print(f"Saved: {fig_curves}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()

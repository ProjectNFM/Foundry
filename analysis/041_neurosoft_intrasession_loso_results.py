"""Analyze NeuroSoft 8-band intrasession baselines vs pretrained transfer, plus
LOSO scratch-vs-transfer comparison.

Compares three intrasession conditions (scratch, Kochi-only, Kochi+B2) across
both species and three block folds, then compares LOSO scratch baselines with
available LOSO pretrained transfer runs across both species.

Usage:
    uv run python analysis/041_neurosoft_intrasession_loso_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PROJECT = "auditory_decoding"
METRIC = "val/neurosoft_acoustic_stim_8band_f1"

INTRASESSION_GROUPS = {
    "scratch": "NEUROSOFT_8B_INTRASESSION_SCRATCH_BASELINES",
    "transfer": {
        "minipigs": "NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS",
        "monkeys": "NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS",
    },
}

LOSO_SCRATCH_GROUP = "NEUROSOFT_8B_LOSO_SCRATCH_BASELINES"
LOSO_TRANSFER_GROUPS = {
    "minipigs": "NEUROSOFT_8B_LOSO_PRETRAIN_TRANSFER_MINIPIGS",
    "monkeys": "NEUROSOFT_8B_LOSO_PRETRAIN_TRANSFER_MONKEYS",
}


def classify_intrasession_run(run) -> dict | None:
    """Extract condition, species, and fold from an intrasession run."""
    name = run.name.lower()

    if "minipig" in name:
        species = "minipigs"
    elif "monkey" in name:
        species = "monkeys"
    else:
        return None

    if "kochi_b2_fixed" in name:
        condition = "Kochi + B2"
    elif "kochi_fixed" in name:
        condition = "Kochi-only"
    elif "scratch" in name:
        condition = "Scratch"
    else:
        return None

    fold = None
    for i in range(10):
        if f"fold{i}" in name or f"fold_{i}" in name:
            fold = i
            break

    return {"condition": condition, "species": species, "fold": fold}


def classify_loso_run(run, default_condition: str | None = None) -> dict | None:
    """Extract condition, species, and subject from a LOSO run."""
    import re

    name = run.name.lower()

    if "kochi_b2_fixed" in name:
        condition = "Kochi + B2"
    elif "kochi_fixed" in name:
        condition = "Kochi-only"
    elif "scratch" in name:
        condition = "Scratch"
    elif default_condition:
        condition = default_condition
    else:
        return None

    if "minipig" in name:
        species = "minipigs"
    elif "monkey" in name:
        species = "monkeys"
    else:
        return None

    match = re.search(r"(sub-\d+)", name)
    subject = match.group(1) if match else None

    return {"condition": condition, "species": species, "subject": subject}


def fetch_best_f1(run) -> float | None:
    """Get best validation F1 from a run's history."""
    history = run.history(keys=[METRIC], samples=50_000, pandas=True)
    scores = history.get(METRIC, pd.Series(dtype=float)).dropna()
    if scores.empty:
        return None
    return float(scores.max())


def fetch_intrasession_runs(api: wandb.Api) -> pd.DataFrame:
    """Fetch all intrasession runs (scratch + transfer)."""
    entity = api.default_entity
    records = []

    scratch_group = INTRASESSION_GROUPS["scratch"]
    scratch_runs = api.runs(
        f"{entity}/{PROJECT}", filters={"group": scratch_group}
    )
    for run in scratch_runs:
        if run.state not in ("finished", "running"):
            continue
        info = classify_intrasession_run(run)
        if info is None:
            continue
        info["condition"] = "Scratch"
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

    for species, group in INTRASESSION_GROUPS["transfer"].items():
        transfer_runs = api.runs(
            f"{entity}/{PROJECT}", filters={"group": group}
        )
        for run in transfer_runs:
            if run.state not in ("finished", "running"):
                continue
            info = classify_intrasession_run(run)
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


def fetch_loso_runs(api: wandb.Api) -> pd.DataFrame:
    """Fetch all LOSO runs: scratch baselines + pretrained transfer."""
    entity = api.default_entity
    records = []

    scratch_runs = api.runs(
        f"{entity}/{PROJECT}", filters={"group": LOSO_SCRATCH_GROUP}
    )
    for run in scratch_runs:
        if run.state not in ("finished", "running"):
            continue
        info = classify_loso_run(run, default_condition="Scratch")
        if info is None:
            continue
        best_f1 = fetch_best_f1(run)
        if best_f1 is not None:
            records.append(
                {**info, "best_f1": best_f1, "run_name": run.name,
                 "run_id": run.id, "state": run.state}
            )

    for species, group in LOSO_TRANSFER_GROUPS.items():
        transfer_runs = api.runs(
            f"{entity}/{PROJECT}", filters={"group": group}
        )
        for run in transfer_runs:
            if run.state not in ("finished", "running"):
                continue
            info = classify_loso_run(run)
            if info is None:
                continue
            best_f1 = fetch_best_f1(run)
            if best_f1 is not None:
                records.append(
                    {**info, "best_f1": best_f1, "run_name": run.name,
                     "run_id": run.id, "state": run.state}
                )

    return pd.DataFrame(records)


def plot_intrasession_comparison(df: pd.DataFrame) -> Path:
    """Bar chart comparing intrasession conditions per species."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    condition_order = ["Scratch", "Kochi-only", "Kochi + B2"]
    colors = {"Scratch": "#7f8c8d", "Kochi-only": "#2980b9", "Kochi + B2": "#27ae60"}

    for ax, species in zip(axes, ["minipigs", "monkeys"]):
        subset = df[df["species"] == species]
        means = []
        stds = []
        labels = []
        for cond in condition_order:
            cond_data = subset[subset["condition"] == cond]["best_f1"]
            if cond_data.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(cond_data.mean())
                stds.append(cond_data.std())
            labels.append(cond)

        x = np.arange(len(labels))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5, color=[colors[c] for c in labels],
            edgecolor="black", linewidth=0.5, alpha=0.85,
        )

        for bar, fold_data in zip(bars, [subset[subset["condition"] == c]["best_f1"] for c in condition_order]):
            if not fold_data.empty:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(fold_data))
                ax.scatter(
                    bar.get_x() + bar.get_width() / 2 + jitter,
                    fold_data.values, color="black", s=25, zorder=5, alpha=0.7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f"{species.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Best Validation F1" if species == "minipigs" else "")
        ax.set_ylim(0, min(1.0, max(means) * 1.4 + 0.05) if means and max(means) > 0 else 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "NeuroSoft 8-Band Intrasession: Scratch vs Pretrained Transfer",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "041_neurosoft_intrasession_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loso_comparison(df: pd.DataFrame) -> dict[str, Path]:
    """Per-species grouped bar chart comparing LOSO scratch vs transfer per subject."""
    saved = {}
    if df.empty:
        return saved

    condition_order = ["Scratch", "Kochi-only", "Kochi + B2"]
    colors = {"Scratch": "#7f8c8d", "Kochi-only": "#2980b9", "Kochi + B2": "#27ae60"}

    for species in ["minipigs", "monkeys"]:
        sp_df = df[df["species"] == species]
        if sp_df.empty:
            continue

        subjects = sorted(sp_df["subject"].dropna().unique())
        conditions_present = [c for c in condition_order if c in sp_df["condition"].values]
        n_cond = len(conditions_present)
        if n_cond == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.5), 5))
        bar_width = 0.8 / n_cond
        x = np.arange(len(subjects))

        for i, cond in enumerate(conditions_present):
            vals = []
            for subj in subjects:
                row = sp_df[(sp_df["subject"] == subj) & (sp_df["condition"] == cond)]
                vals.append(row["best_f1"].values[0] if not row.empty else 0)
            offset = (i - (n_cond - 1) / 2) * bar_width
            ax.bar(
                x + offset, vals, bar_width * 0.9,
                label=cond, color=colors[cond],
                edgecolor="black", linewidth=0.5, alpha=0.85,
            )

        chance = 1.0 / 8
        ax.axhline(chance, color="#95a5a6", linestyle=":", linewidth=1.2,
                    label=f"Chance = {chance:.3f}")

        for cond in conditions_present:
            cond_vals = sp_df[sp_df["condition"] == cond]["best_f1"]
            if not cond_vals.empty:
                ax.axhline(
                    cond_vals.mean(), color=colors[cond], linestyle="--",
                    linewidth=1.2, alpha=0.6,
                    label=f"{cond} mean = {cond_vals.mean():.3f}",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=10)
        ax.set_xlabel("Held-out Subject")
        ax.set_ylabel("Best Validation F1")
        ax.set_title(
            f"NeuroSoft 8-Band LOSO: Scratch vs Transfer — {species.capitalize()}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ymax = max(sp_df["best_f1"].max() * 1.3 + 0.05, 0.3)
        ax.set_ylim(0, min(1.0, ymax))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        path = FIGURES_DIR / f"041_neurosoft_loso_comparison_{species}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved[species] = path

    return saved


def plot_training_curves(api: wandb.Api, intrasession_df: pd.DataFrame) -> Path | None:
    """Training curves (F1 over epochs) for intrasession runs."""
    if intrasession_df.empty:
        return None

    entity = api.default_entity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"Scratch": "#7f8c8d", "Kochi-only": "#2980b9", "Kochi + B2": "#27ae60"}
    linestyles = {0: "-", 1: "--", 2: ":"}

    for ax, species in zip(axes, ["minipigs", "monkeys"]):
        subset = intrasession_df[intrasession_df["species"] == species]
        for _, row in subset.iterrows():
            try:
                run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
                history = run.history(keys=[METRIC], samples=50_000, pandas=True)
                scores = history[METRIC].dropna()
                if scores.empty:
                    continue
                fold_label = f" (fold {row['fold']})" if row["fold"] is not None else ""
                ax.plot(
                    range(len(scores)), scores.values,
                    color=colors.get(row["condition"], "#333"),
                    linestyle=linestyles.get(row["fold"], "-"),
                    alpha=0.7, linewidth=1.2,
                    label=f"{row['condition']}{fold_label}",
                )
            except Exception:
                continue

        ax.set_xlabel("Validation Step")
        ax.set_ylabel("Validation F1" if species == "minipigs" else "")
        ax.set_title(f"{species.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")

    fig.suptitle(
        "NeuroSoft 8-Band: Validation F1 Training Curves",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIGURES_DIR / "041_neurosoft_training_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    api = wandb.Api()

    print("=" * 70)
    print("NeuroSoft 8-Band: Intrasession + LOSO Results")
    print("=" * 70)

    # --- Intrasession ---
    print("\nFetching intrasession runs...")
    intra_df = fetch_intrasession_runs(api)

    if intra_df.empty:
        print("[WARN] No intrasession runs found.")
    else:
        print(f"\nFound {len(intra_df)} intrasession runs.\n")
        print("Individual runs:")
        print(
            intra_df[["species", "condition", "fold", "best_f1", "run_name", "state"]]
            .sort_values(["species", "condition", "fold"])
            .to_string(index=False)
        )

        print("\n" + "-" * 70)
        print("INTRASESSION SUMMARY (mean ± std over folds)")
        print("-" * 70)
        summary = (
            intra_df.groupby(["species", "condition"])["best_f1"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["mean±std"] = summary.apply(
            lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r["std"]) else f"{r['mean']:.4f}",
            axis=1,
        )
        print(summary[["species", "condition", "mean±std", "count"]].to_string(index=False))

        fig_path = plot_intrasession_comparison(intra_df)
        print(f"\nSaved: {fig_path}")

        curves_path = plot_training_curves(api, intra_df)
        if curves_path:
            print(f"Saved: {curves_path}")

    # --- LOSO ---
    print("\n" + "=" * 70)
    print("Fetching LOSO runs (scratch + transfer)...")
    loso_df = fetch_loso_runs(api)

    if loso_df.empty:
        print("[WARN] No LOSO runs found.")
    else:
        print(f"\nFound {len(loso_df)} LOSO runs.\n")
        print("All LOSO runs:")
        print(
            loso_df[["species", "condition", "subject", "best_f1", "run_name", "state"]]
            .sort_values(["species", "condition", "subject"])
            .to_string(index=False)
        )

        print("\n" + "-" * 70)
        print("LOSO SUMMARY (mean ± std per species × condition)")
        print("-" * 70)
        loso_summary = (
            loso_df.groupby(["species", "condition"])["best_f1"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        loso_summary["mean±std"] = loso_summary.apply(
            lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r["std"]) else f"{r['mean']:.4f}",
            axis=1,
        )
        print(loso_summary[["species", "condition", "mean±std", "count"]].to_string(index=False))

        # Per-subject paired comparison where both scratch and transfer exist
        print("\n" + "-" * 70)
        print("LOSO PER-SUBJECT COMPARISON (subjects with both scratch and transfer)")
        print("-" * 70)
        for species in ["minipigs", "monkeys"]:
            sp_df = loso_df[loso_df["species"] == species]
            scratch = sp_df[sp_df["condition"] == "Scratch"].set_index("subject")["best_f1"]
            for cond in ["Kochi-only", "Kochi + B2"]:
                transfer = sp_df[sp_df["condition"] == cond].set_index("subject")["best_f1"]
                shared = sorted(set(scratch.index) & set(transfer.index))
                if not shared:
                    continue
                print(f"\n  {species.capitalize()} — Scratch vs {cond} ({len(shared)} subjects):")
                print(f"  {'Subject':<10} {'Scratch':>10} {cond:>12} {'Delta':>10}")
                for subj in shared:
                    s_val = scratch[subj]
                    t_val = transfer[subj]
                    delta = t_val - s_val
                    print(f"  {subj:<10} {s_val:>10.4f} {t_val:>12.4f} {delta:>+10.4f}")
                s_mean = scratch.loc[shared].mean()
                t_mean = transfer.loc[shared].mean()
                print(f"  {'Mean':<10} {s_mean:>10.4f} {t_mean:>12.4f} {t_mean - s_mean:>+10.4f}")

        fig_paths = plot_loso_comparison(loso_df)
        for species, path in fig_paths.items():
            print(f"\nSaved: {path}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

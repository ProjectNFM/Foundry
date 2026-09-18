"""Analyze the reference checkpoints against recipe-matched scratch.

The immutable compiled matrices define every W&B run. This script hash-audits
those matrices, fetches exact run IDs through ``wandb.Api()``, verifies run
identity and completion, replaces the legacy scratch mapping with the new
recipe-matched scratch mapping, and performs the preregistered whole-subject
simultaneous bootstrap across the five source checkpoints.

It deliberately imports no code from the main :mod:`foundry` package.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _architecture_transfer_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_THRESHOLD,
    SOURCE_STEPS,
    _read_jsonl,
    _sha256,
    _source_step,
    _target_treatment,
    aggregate_and_cache,
    audit_coverage,
    fetch_or_load,
    make_pairs,
    print_principal_tables,
)


ROOT = Path(__file__).resolve().parents[1]
STEM = "20260918-MS-recipe-matched-scratch-control"
SPECIES_LABEL = "minipig"
EXPECTED_TRANSFER_CELLS = 600
EXPECTED_SCRATCH_CELLS = 120
EXPECTED_SUBJECTS = 7
PARENT_MATRIX = (
    ROOT
    / "launch"
    / "architecture_transfer"
    / "reference-backbone-minipigs.jsonl"
)
SCRATCH_MATRIX = (
    ROOT
    / "launch"
    / "recipe_matched_scratch"
    / "recipe-matched-scratch-minipigs.jsonl"
)
CSV_DIR = ROOT / "analysis" / "csv"
FIGURE_DIR = ROOT / "analysis" / "figures"
CLASS_NAMES = [
    "low_bass",
    "mid_bass",
    "low_mids",
    "midrange",
    "high_mids",
    "low_treble",
    "mid_treble",
    "high_treble",
]
COLORS = {
    "transfer": "#3569a8",
    "new_scratch": "#222222",
    "old_scratch": "#9a9a9a",
    "positive": "#26734d",
    "negative": "#b4473d",
}


def _matrix_rows(path: Path, matrix: str) -> list[dict[str, Any]]:
    lock_path = path.with_suffix(".lock.json")
    lock = json.loads(lock_path.read_text())
    digest = _sha256(path)
    if digest != lock["output_sha256"]:
        raise RuntimeError(f"Hash mismatch for {path}")
    cells = _read_jsonl(path)
    if len(cells) != int(lock["output_count"]):
        raise RuntimeError(f"Count mismatch for {path}")
    rows = []
    for cell in cells:
        rows.append(
            {
                "matrix": matrix,
                "matrix_path": str(path.relative_to(ROOT)),
                "matrix_sha256": digest,
                "recipe_sha256": lock["recipe_sha256"],
                "registry_sha256": lock["registry_sha256"],
                "cell_id": cell["cell_id"],
                "run_id": cell["wandb_run_id"],
                "expected_run_name": cell["run_name"],
                "expected_group": cell["wandb_group"],
                "condition": cell["condition_id"],
                "species": cell["species"],
                "subject": cell["target_subject"],
                "recording": cell["target_recording"],
                "target_seed": int(cell["target_finetuning_seed"]),
                "target_fraction": float(cell["target_fraction"]),
                "source_step": _source_step(cell),
                "source_condition": (
                    (cell.get("source_condition") or {}).get(
                        "source_condition", "scratch"
                    )
                    if isinstance(cell.get("source_condition"), dict)
                    else str(cell.get("source_condition") or "scratch")
                ),
                "source_model_seed": cell.get("source_model_seed", np.nan),
                "source_selection_seed": cell.get(
                    "source_selection_seed", np.nan
                ),
                "checkpoint_set_id": cell.get("checkpoint_set_id", ""),
                "checkpoint_id": cell.get("checkpoint_id", ""),
                "checkpoint_manifest_hash": cell.get(
                    "checkpoint_manifest_hash", ""
                ),
                "checkpoint_sha256": cell.get("checkpoint_sha256", ""),
                "matched_scratch_id": cell.get("matched_scratch_id", ""),
                "scratch_family": cell.get("scratch_family", ""),
                "transfer_regime": cell.get("transfer_regime", "scratch"),
                "target_treatment": _target_treatment(cell),
            }
        )
    return rows


def load_design() -> pd.DataFrame:
    """Load exact parent/new inventories and remap transfer to new scratch."""
    parent = pd.DataFrame(_matrix_rows(PARENT_MATRIX, "reference-backbone"))
    parent = parent[
        parent.condition.isin(["transfer_reference", "scratch_reference"])
    ].copy()
    parent.loc[parent.condition == "scratch_reference", "condition"] = (
        "scratch_legacy"
    )
    new = pd.DataFrame(_matrix_rows(SCRATCH_MATRIX, "recipe-matched-scratch"))
    if set(new.condition) != {"scratch_recipe_matched"}:
        raise RuntimeError("Unexpected recipe-matched scratch condition")
    if (
        len(parent[parent.condition == "transfer_reference"])
        != EXPECTED_TRANSFER_CELLS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TRANSFER_CELLS} completed reference transfer cells"
        )
    if (
        len(parent[parent.condition == "scratch_legacy"])
        != EXPECTED_SCRATCH_CELLS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_SCRATCH_CELLS} legacy scratch cells"
        )
    if len(new) != EXPECTED_SCRATCH_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_SCRATCH_CELLS} recipe-matched scratch cells"
        )

    new_lookup = {
        (row.recording, int(row.target_seed)): row.cell_id
        for row in new.itertuples(index=False)
    }
    if len(new_lookup) != EXPECTED_SCRATCH_CELLS:
        raise RuntimeError("Recipe-matched scratch pairing keys are not unique")
    transfer = parent.condition == "transfer_reference"
    parent.loc[transfer, "matched_scratch_id"] = [
        new_lookup[(row.recording, int(row.target_seed))]
        for row in parent[transfer].itertuples(index=False)
    ]
    parent.loc[transfer, "scratch_family"] = "reference-recipe-matched"

    design = pd.concat([parent, new], ignore_index=True)
    for column in ("cell_id", "run_id"):
        if design[column].duplicated().any():
            raise RuntimeError(f"Duplicate {column} in analysis design")
    steps = set(
        design.loc[design.condition == "transfer_reference", "source_step"]
        .dropna()
        .astype(int)
    )
    if steps != set(SOURCE_STEPS):
        raise RuntimeError(f"Incomplete checkpoint inventory: {steps}")
    return design.sort_values(["matrix", "condition", "cell_id"]).reset_index(
        drop=True
    )


def _subject_bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    boot = values[rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))].mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def simultaneous_intervals(subject: pd.DataFrame) -> pd.DataFrame:
    """Max-t simultaneous 95% intervals across the five checkpoint effects."""
    pivot = subject.pivot(
        index="subject", columns="source_step", values="effect"
    )
    pivot = pivot.reindex(columns=list(SOURCE_STEPS))
    if (
        pivot.shape != (EXPECTED_SUBJECTS, len(SOURCE_STEPS))
        or pivot.isna().any().any()
    ):
        raise RuntimeError(
            f"Expected complete {EXPECTED_SUBJECTS}x{len(SOURCE_STEPS)} "
            f"subject matrix, got {pivot.shape}"
        )
    values = pivot.to_numpy(float)
    estimates = values.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    boot = values[indices].mean(axis=1)
    standard_errors = boot.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0):
        raise RuntimeError("Degenerate bootstrap standard error")
    max_t = np.max(
        np.abs((boot - estimates[None, :]) / standard_errors[None, :]),
        axis=1,
    )
    critical = float(np.quantile(max_t, 0.95))
    rows = []
    for index, step in enumerate(SOURCE_STEPS):
        low = float(estimates[index] - critical * standard_errors[index])
        high = float(estimates[index] + critical * standard_errors[index])
        rows.append(
            {
                "source_step": step,
                "mean": float(estimates[index]),
                "pointwise_ci_low": float(np.quantile(boot[:, index], 0.025)),
                "pointwise_ci_high": float(np.quantile(boot[:, index], 0.975)),
                "simultaneous_ci_low": low,
                "simultaneous_ci_high": high,
                "max_t_critical": critical,
                "subjects_positive": int((values[:, index] > 0).sum()),
                "subjects": len(values),
                "familywise_significant_improvement": bool(low > 0),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(CSV_DIR / f"{STEM}_simultaneous_f1_summary.csv", index=False)
    return result


def scratch_sensitivity(
    endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["subject", "recording", "target_seed", "test_f1", "run_id"]
    old = endpoints[endpoints.condition == "scratch_legacy"][columns].rename(
        columns={"test_f1": "legacy_f1", "run_id": "legacy_run_id"}
    )
    new = endpoints[endpoints.condition == "scratch_recipe_matched"][
        columns
    ].rename(columns={"test_f1": "recipe_f1", "run_id": "recipe_run_id"})
    paired = new.merge(
        old,
        on=["subject", "recording", "target_seed"],
        validate="one_to_one",
    )
    paired["recipe_minus_legacy"] = paired.recipe_f1 - paired.legacy_f1
    recording = paired.groupby(["subject", "recording"], as_index=False).agg(
        recipe_f1=("recipe_f1", "mean"),
        legacy_f1=("legacy_f1", "mean"),
        recipe_minus_legacy=("recipe_minus_legacy", "mean"),
    )
    subject = recording.groupby("subject", as_index=False).mean(
        numeric_only=True
    )
    mean, low, high = _subject_bootstrap(subject.recipe_minus_legacy.to_numpy())
    summary = pd.DataFrame(
        [
            {
                "contrast": "recipe_matched_minus_legacy_scratch",
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "subjects_positive": int(
                    (subject.recipe_minus_legacy > 0).sum()
                ),
                "subjects": len(subject),
            }
        ]
    )
    paired.to_csv(CSV_DIR / f"{STEM}_scratch_pairing_seed.csv", index=False)
    recording.to_csv(
        CSV_DIR / f"{STEM}_scratch_pairing_recording.csv", index=False
    )
    subject.to_csv(CSV_DIR / f"{STEM}_scratch_pairing_subject.csv", index=False)
    summary.to_csv(CSV_DIR / f"{STEM}_scratch_recipe_summary.csv", index=False)
    return subject, summary


def _save(fig: plt.Figure, suffix: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{STEM}_{suffix}.png", dpi=190, bbox_inches="tight"
    )
    plt.close(fig)


def plot_recentered_main(
    subject: pd.DataFrame, simultaneous: pd.DataFrame
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(SOURCE_STEPS))
    for _, frame in subject.groupby("subject"):
        frame = frame.set_index("source_step").reindex(SOURCE_STEPS)
        ax.plot(
            positions,
            100 * frame.effect,
            color="0.72",
            marker="o",
            ms=3,
            lw=0.9,
            alpha=0.75,
        )
    means = 100 * simultaneous["mean"].to_numpy()
    low = 100 * simultaneous.simultaneous_ci_low.to_numpy()
    high = 100 * simultaneous.simultaneous_ci_high.to_numpy()
    ax.errorbar(
        positions,
        means,
        yerr=np.vstack([means - low, high - means]),
        color=COLORS["transfer"],
        marker="o",
        lw=2.3,
        capsize=5,
        label="Equal-subject mean ± simultaneous 95% CI",
        zorder=5,
    )
    ax.axhline(0, color="black", lw=1, ls="--", label="Recipe-matched scratch")
    ax.set_xticks(positions, [f"{step:,}" for step in SOURCE_STEPS])
    ax.set_xlabel("Source-pretraining optimizer step")
    ax.set_ylabel("Transfer − recipe-matched scratch F1 (pp)")
    ax.set_title(
        "Reference transfer trajectory recentered to recipe-matched scratch"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "recentered_transfer_effects")


def plot_absolute(
    performance_subject: pd.DataFrame, scratch_subject: pd.DataFrame
) -> None:
    rows = []
    for label, column in (
        ("Legacy scratch", "legacy_f1"),
        ("Recipe-matched scratch", "recipe_f1"),
    ):
        values = scratch_subject[column].to_numpy(float)
        mean, low, high = _subject_bootstrap(values)
        rows.append({"label": label, "mean": mean, "low": low, "high": high})
    for step, frame in performance_subject.groupby("source_step"):
        values = frame.test_f1.to_numpy(float)
        mean, low, high = _subject_bootstrap(values)
        rows.append(
            {
                "label": f"Transfer {int(step):,}",
                "mean": mean,
                "low": low,
                "high": high,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(CSV_DIR / f"{STEM}_absolute_f1_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.1))
    x = np.arange(len(summary))
    colors = [COLORS["old_scratch"], COLORS["new_scratch"]] + [
        COLORS["transfer"]
    ] * len(SOURCE_STEPS)
    ax.errorbar(
        x,
        100 * summary["mean"],
        yerr=np.vstack(
            [
                100 * (summary["mean"] - summary["low"]),
                100 * (summary["high"] - summary["mean"]),
            ]
        ),
        fmt="none",
        ecolor="0.25",
        capsize=4,
        zorder=3,
    )
    ax.scatter(x, 100 * summary["mean"], c=colors, s=70, zorder=4)
    ax.set_xticks(x, summary.label, rotation=25, ha="right")
    ax.set_ylabel("Test supported macro-F1 (%)")
    ax.set_title(
        f"Absolute {SPECIES_LABEL} performance under both scratch controls"
    )
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "absolute_performance")


def plot_scratch_recipe(subject: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].scatter(
        100 * subject.legacy_f1,
        100 * subject.recipe_f1,
        color=COLORS["transfer"],
        s=60,
    )
    bounds = 100 * np.array(
        [
            subject[["legacy_f1", "recipe_f1"]].min().min(),
            subject[["legacy_f1", "recipe_f1"]].max().max(),
        ]
    )
    axes[0].plot(bounds, bounds, color="0.4", ls="--", lw=1)
    for row in subject.itertuples(index=False):
        axes[0].annotate(
            row.subject,
            (100 * row.legacy_f1, 100 * row.recipe_f1),
            fontsize=8,
        )
    axes[0].set_xlabel("Legacy scratch F1 (%)")
    axes[0].set_ylabel("Recipe-matched scratch F1 (%)")
    axes[0].set_title("Subject-level scratch comparison")
    effects = 100 * subject.recipe_minus_legacy.to_numpy()
    axes[1].bar(
        subject.subject,
        effects,
        color=np.where(effects >= 0, COLORS["positive"], COLORS["negative"]),
    )
    mean = 100 * float(summary.iloc[0]["mean"])
    low = 100 * float(summary.iloc[0]["ci_low"])
    high = 100 * float(summary.iloc[0]["ci_high"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(
        mean,
        color=COLORS["transfer"],
        lw=2,
        label=f"Mean {mean:+.2f} pp [{low:+.2f}, {high:+.2f}]",
    )
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("Recipe-matched − legacy scratch F1 (pp)")
    axes[1].set_title("Effect of matching the optimizer recipe")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "scratch_recipe_sensitivity")


def plot_subject_heatmap(subject: pd.DataFrame) -> None:
    pivot = subject.pivot(
        index="subject", columns="source_step", values="effect"
    )
    pivot = 100 * pivot.reindex(columns=SOURCE_STEPS)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    limit = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Transfer effect (pp)"},
        ax=ax,
    )
    ax.set_xlabel("Source-pretraining step")
    ax.set_ylabel("Excluded target subject")
    ax.set_title("Subject heterogeneity relative to recipe-matched scratch")
    _save(fig, "subject_effect_heatmap")


def plot_recording_heatmap(recording: pd.DataFrame) -> None:
    pivot = recording.pivot(
        index="recording", columns="source_step", values="effect"
    )
    pivot = 100 * pivot.reindex(columns=SOURCE_STEPS)
    fig, ax = plt.subplots(figsize=(8.2, 12.5))
    limit = max(1.0, float(np.nanquantile(np.abs(pivot.to_numpy()), 0.98)))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        yticklabels=True,
        cbar_kws={"label": "Transfer effect (pp)"},
        ax=ax,
    )
    ax.set_xlabel("Source-pretraining step")
    ax.set_ylabel("Target recording")
    ax.set_title("Recording-level transfer effects")
    ax.tick_params(axis="y", labelsize=6)
    _save(fig, "recording_effect_heatmap")


def plot_seed_dispersion(paired: pd.DataFrame) -> None:
    recording = paired.groupby(
        ["subject", "recording", "target_seed", "source_step"], as_index=False
    ).effect.mean()
    seed_subject = recording.groupby(
        ["subject", "target_seed", "source_step"], as_index=False
    ).effect.mean()
    seed_subject["effect_pp"] = 100 * seed_subject.effect
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    sns.boxplot(
        data=seed_subject,
        x="source_step",
        y="effect_pp",
        hue="target_seed",
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Source-pretraining step")
    ax.set_ylabel("Transfer − recipe-matched scratch F1 (pp)")
    ax.set_title("Target-seed dispersion across subjects")
    ax.legend(title="Target seed", frameon=False)
    _save(fig, "target_seed_dispersion")


def plot_efficiency(tables: dict[str, pd.DataFrame]) -> None:
    subject = tables["efficiency_subject"]
    subject = subject[np.isclose(subject.threshold, PRIMARY_THRESHOLD)].copy()
    summary = tables["efficiency_summary"]
    summary = summary[summary.metric == "steps_saved"].sort_values(
        "source_step"
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    positions = np.arange(len(SOURCE_STEPS))
    for _, frame in subject.groupby("subject"):
        frame = frame.set_index("source_step").reindex(SOURCE_STEPS)
        ax.plot(
            positions,
            frame.steps_saved,
            color="0.72",
            marker="o",
            ms=3,
            lw=0.9,
        )
    means = summary["mean"].to_numpy()
    ax.errorbar(
        positions,
        means,
        yerr=np.vstack([means - summary.ci_low, summary.ci_high - means]),
        color=COLORS["transfer"],
        marker="o",
        lw=2.2,
        capsize=4,
        label="Equal-subject mean ± pointwise 95% CI",
    )
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xticks(positions, [f"{step:,}" for step in SOURCE_STEPS])
    ax.set_xlabel("Source-pretraining step")
    ax.set_ylabel("Optimizer steps saved to 90% scratch quality")
    ax.set_title("Optimization efficiency relative to recipe-matched scratch")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "efficiency_trajectory")


def plot_dynamics(tables: dict[str, pd.DataFrame]) -> None:
    dynamics = tables["dynamics"]
    recording = dynamics.groupby(
        ["subject", "recording", "source_step", "kind", "budget_fraction"],
        as_index=False,
    ).normalized_progress.mean()
    subject = recording.groupby(
        ["subject", "source_step", "kind", "budget_fraction"], as_index=False
    ).normalized_progress.mean()
    mean = subject.groupby(
        ["source_step", "kind", "budget_fraction"], as_index=False
    ).normalized_progress.mean()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharex=True, sharey=True)
    for ax, step in zip(axes, SOURCE_STEPS):
        view = mean[mean.source_step == step]
        for kind, color, label in (
            ("transfer", COLORS["transfer"], "Transfer"),
            ("scratch", COLORS["new_scratch"], "Recipe-matched scratch"),
        ):
            line = view[view.kind == kind]
            ax.plot(
                line.budget_fraction,
                line.normalized_progress,
                color=color,
                lw=2,
                label=label,
            )
        ax.axhline(PRIMARY_THRESHOLD, color="0.6", lw=0.8, ls="--")
        ax.set_title(f"{step:,} steps")
        ax.set_xlabel("Budget fraction")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative-best F1 / scratch peak")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Normalized downstream validation dynamics", y=1.03)
    _save(fig, "normalized_training_dynamics")


def plot_classwise(paired: pd.DataFrame) -> None:
    rows = []
    for row in paired.itertuples(index=False):
        transfer = np.asarray(json.loads(row.class_f1), dtype=float)
        scratch = np.asarray(json.loads(row.scratch_class_f1), dtype=float)
        if transfer.shape != (8,) or scratch.shape != (8,):
            continue
        for index, name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "subject": row.subject,
                    "recording": row.recording,
                    "target_seed": row.target_seed,
                    "source_step": int(row.source_step),
                    "class": name,
                    "effect": transfer[index] - scratch[index],
                }
            )
    frame = pd.DataFrame(rows)
    recording = frame.groupby(
        ["subject", "recording", "source_step", "class"], as_index=False
    ).effect.mean()
    subject = recording.groupby(
        ["subject", "source_step", "class"], as_index=False
    ).effect.mean()
    mean = subject.groupby(
        ["source_step", "class"], as_index=False
    ).effect.mean()
    mean.to_csv(CSV_DIR / f"{STEM}_classwise_subject_mean.csv", index=False)
    pivot = mean.pivot(index="class", columns="source_step", values="effect")
    pivot = 100 * pivot.reindex(index=CLASS_NAMES, columns=SOURCE_STEPS)
    fig, ax = plt.subplots(figsize=(7.5, 5.1))
    limit = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Classwise F1 effect (pp)"},
        ax=ax,
    )
    ax.set_xlabel("Source-pretraining step")
    ax.set_ylabel("Supported class")
    ax.set_title("Classwise transfer effects")
    _save(fig, "classwise_effects")


def print_summary(simultaneous: pd.DataFrame, scratch: pd.DataFrame) -> None:
    display = simultaneous.copy()
    for column in (
        "mean",
        "pointwise_ci_low",
        "pointwise_ci_high",
        "simultaneous_ci_low",
        "simultaneous_ci_high",
    ):
        display[column] *= 100
    print("\nConfirmatory simultaneous performance table (percentage points)")
    print(
        display[
            [
                "source_step",
                "mean",
                "pointwise_ci_low",
                "pointwise_ci_high",
                "simultaneous_ci_low",
                "simultaneous_ci_high",
                "subjects_positive",
                "familywise_significant_improvement",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nScratch-recipe sensitivity (percentage points)")
    scratch_display = scratch.copy()
    for column in ("mean", "ci_low", "ci_high"):
        scratch_display[column] *= 100
    print(
        scratch_display.to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "poyo-eeg")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    design = load_design()
    if args.skip_fetch:
        endpoint_path = CSV_DIR / f"{STEM}_run_endpoints.csv"
        history_path = CSV_DIR / f"{STEM}_validation_histories.csv"
        if not endpoint_path.exists() or not history_path.exists():
            raise SystemExit("--skip-fetch requested but caches are missing")
        endpoints = pd.read_csv(endpoint_path)
        histories = pd.read_csv(history_path)
    else:
        endpoints, histories = fetch_or_load(
            design, STEM, args.entity, args.workers, args.refresh
        )
    endpoints = endpoints[endpoints.run_id.isin(design.run_id)].copy()
    # Raw caches legitimately retain the design columns from the experiment
    # that first fetched a run. Overlay this experiment's audited design so the
    # parent transfer rows pair to recipe-matched scratch, not legacy scratch.
    design_columns = [column for column in design.columns if column != "run_id"]
    endpoints = endpoints.drop(
        columns=[column for column in design_columns if column in endpoints]
    ).merge(design, on="run_id", how="left", validate="one_to_one")
    histories = histories[histories.run_id.isin(design.run_id)].copy()
    endpoints.to_csv(CSV_DIR / f"{STEM}_run_endpoints.csv", index=False)
    histories.to_csv(CSV_DIR / f"{STEM}_validation_histories.csv", index=False)
    audit_coverage(design, endpoints, histories, STEM)

    paired, efficiency, dynamics = make_pairs(endpoints, histories)
    tables = aggregate_and_cache(paired, efficiency, dynamics, STEM)
    simultaneous = simultaneous_intervals(tables["paired_performance_subject"])
    scratch_subject, scratch_summary = scratch_sensitivity(endpoints)
    print_principal_tables(tables)
    print_summary(simultaneous, scratch_summary)

    if not args.skip_figures:
        sns.set_theme(style="whitegrid", context="notebook")
        plot_recentered_main(tables["paired_performance_subject"], simultaneous)
        plot_absolute(tables["paired_performance_subject"], scratch_subject)
        plot_scratch_recipe(scratch_subject, scratch_summary)
        plot_subject_heatmap(tables["paired_performance_subject"])
        plot_recording_heatmap(tables["paired_performance_recording"])
        plot_seed_dispersion(paired)
        plot_efficiency(tables)
        plot_dynamics(tables)
        plot_classwise(paired)
        print(f"\nFigures written under {FIGURE_DIR}")


if __name__ == "__main__":
    main()

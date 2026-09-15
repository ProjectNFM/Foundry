#!/usr/bin/env python3
"""Analyze the definitive Phase 4E adapter-bias evaluation artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir


STEM = "20260915-MS-adapter-bias-perturbation"
PROJECT = "neurosoft_supervised_pretraining"
FIXED_STEPS = (100, 300, 1_000, 3_000, 10_000)
DERANGEMENTS = tuple(f"derangement_{index}" for index in range(5))
METRICS = (
    "cross_entropy",
    "pooled_supported_f1",
    "recording_mean_supported_f1",
)


def artifact_path(entity: str | None, project: str, artifact: str) -> str:
    prefix = f"{entity}/" if entity else ""
    return f"{prefix}{project}/{artifact}"


def fetch_results(
    artifact: str,
    project: str,
    entity: str | None,
    api: Any | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch the definitive CSV from a W&B artifact via ``wandb.Api``."""
    api = api or wandb.Api()
    resolved = api.artifact(artifact_path(entity, project, artifact))
    with tempfile.TemporaryDirectory(prefix=f"{STEM}-") as temp_dir:
        directory = Path(resolved.download(root=temp_dir))
        candidates = sorted(directory.rglob("results.csv"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one results.csv in artifact, found {len(candidates)}"
            )
        frame = pd.read_csv(candidates[0])
    return frame, str(resolved.qualified_name)


def add_intact_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "species",
        "source_run_name",
        "checkpoint_kind",
        "checkpoint_step",
        "checkpoint_sha256",
    ]
    intact = (
        frame.loc[frame["condition"] == "intact", keys + list(METRICS)]
        .rename(columns={metric: f"intact_{metric}" for metric in METRICS})
        .copy()
    )
    if intact.duplicated(keys).any():
        raise RuntimeError("Duplicate intact reference rows")
    merged = frame.merge(intact, on=keys, how="left", validate="many_to_one")
    if merged[[f"intact_{metric}" for metric in METRICS]].isna().any().any():
        raise RuntimeError("At least one condition lacks an intact reference")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"intact_{metric}"]
    return merged


def average_derangements(frame: pd.DataFrame) -> pd.DataFrame:
    deranged = frame[frame["condition"].isin(DERANGEMENTS)].copy()
    keys = [
        "species",
        "source_run_name",
        "excluded_target_subject",
        "source_selection_seed",
        "source_model_seed",
        "checkpoint_kind",
        "checkpoint_step",
        "checkpoint_global_step",
    ]
    counts = deranged.groupby(keys, dropna=False).size()
    if not (counts == 5).all():
        raise RuntimeError(
            "Every source/checkpoint must contain five derangements"
        )
    columns = [f"delta_{metric}" for metric in METRICS]
    return deranged.groupby(keys, as_index=False, dropna=False)[columns].mean()


def subject_checkpoint_means(derangement_means: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "species",
        "excluded_target_subject",
        "checkpoint_kind",
        "checkpoint_step",
    ]
    columns = ["checkpoint_global_step"] + [
        f"delta_{metric}" for metric in METRICS
    ]
    return derangement_means.groupby(keys, as_index=False, dropna=False)[
        columns
    ].mean()


def subject_slopes(
    subject_means: pd.DataFrame,
    metric: str,
    fixed_steps: Sequence[int] = FIXED_STEPS,
) -> pd.DataFrame:
    """Fit one fixed-checkpoint log-step slope per excluded target subject."""
    fixed = subject_means[
        (subject_means["checkpoint_kind"] == "milestone")
        & subject_means["checkpoint_step"].isin(fixed_steps)
    ]
    rows: list[dict[str, Any]] = []
    value_column = f"delta_{metric}"
    for (species, subject), group in fixed.groupby(
        ["species", "excluded_target_subject"]
    ):
        group = group.sort_values("checkpoint_step")
        if tuple(group["checkpoint_step"].astype(int)) != tuple(fixed_steps):
            raise RuntimeError(
                f"{species}/{subject} does not have the requested fixed "
                f"checkpoints {list(fixed_steps)}"
            )
        slope, intercept = np.polyfit(
            np.log(group["checkpoint_step"].to_numpy(float)),
            group[value_column].to_numpy(float),
            1,
        )
        rows.append(
            {
                "species": species,
                "excluded_target_subject": subject,
                "metric": metric,
                "slope": slope,
                "intercept": intercept,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_species_slopes(
    slopes: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap excluded target subjects and summarize mean subject slopes."""
    rng = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (species, metric), group in slopes.groupby(["species", "metric"]):
        values = group["slope"].to_numpy(float)
        if values.size < 2:
            raise RuntimeError(
                f"Need at least two subjects for {species}/{metric}"
            )
        bootstrap = np.empty(replicates, dtype=float)
        for index in range(replicates):
            bootstrap[index] = rng.choice(
                values, size=len(values), replace=True
            ).mean()
        draws.extend(
            {
                "species": species,
                "metric": metric,
                "replicate": index,
                "slope": value,
            }
            for index, value in enumerate(bootstrap)
        )
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        summaries.append(
            {
                "species": species,
                "metric": metric,
                "subject_count": len(values),
                "mean_slope": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "strictly_positive": bool(low > 0),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(draws)


def secondary_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["condition"].isin(("zero", "mean"))]
    keys = [
        "species",
        "excluded_target_subject",
        "checkpoint_kind",
        "checkpoint_step",
        "condition",
    ]
    columns = [f"delta_{metric}" for metric in METRICS]
    return selected.groupby(keys, as_index=False, dropna=False)[columns].mean()


def bias_condition_subject_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Average loss effects over mappings, then seeds, within each subject."""
    selected = frame[
        frame["condition"].isin(("intact", "zero", "mean", *DERANGEMENTS))
    ].copy()
    selected["bias_condition"] = selected["condition"].where(
        ~selected["condition"].isin(DERANGEMENTS), "derangement"
    )
    source_keys = [
        "species",
        "source_run_name",
        "excluded_target_subject",
        "checkpoint_kind",
        "checkpoint_step",
        "bias_condition",
    ]
    source_means = selected.groupby(
        source_keys, as_index=False, dropna=False
    )["delta_cross_entropy"].mean()
    subject_keys = [
        "species",
        "excluded_target_subject",
        "checkpoint_kind",
        "checkpoint_step",
        "bias_condition",
    ]
    return source_means.groupby(
        subject_keys, as_index=False, dropna=False
    )["delta_cross_entropy"].mean()


def plot_bias_condition_comparison(
    subject_means: pd.DataFrame, destination: Path
) -> None:
    """Compare the cross-entropy effect of each bias intervention."""
    species_order = [
        species
        for species in ("minipigs", "monkeys")
        if species in set(subject_means["species"])
    ]
    if not species_order:
        raise RuntimeError("No species are available to plot")
    condition_styles = {
        "intact": ("Intact (reference)", "#333333", "o"),
        "zero": ("Zero bias", "#d62728", "s"),
        "mean": ("Mean bias", "#2ca02c", "^"),
        "derangement": ("Deranged bias (mean of 5)", "#1f77b4", "D"),
    }
    fig, axes = plt.subplots(
        1,
        len(species_order),
        figsize=(5 * len(species_order), 4),
        sharey=True,
        squeeze=False,
    )
    for axis, species in zip(axes[0], species_order):
        fixed = subject_means[
            (subject_means["species"] == species)
            & (subject_means["checkpoint_kind"] == "milestone")
        ]
        for condition, (label, color, marker) in condition_styles.items():
            subset = fixed[fixed["bias_condition"] == condition]
            summary = subset.groupby("checkpoint_step")[
                "delta_cross_entropy"
            ].agg(["mean", "sem"])
            axis.errorbar(
                summary.index,
                summary["mean"],
                yerr=1.96 * summary["sem"].fillna(0),
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
                capsize=3,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(species.capitalize())
        axis.set_xlabel("Fixed checkpoint step")
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Condition − intact cross-entropy")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_metric_curves(
    subject_means: pd.DataFrame,
    metric: str,
    destination: Path,
    ylabel: str,
) -> None:
    """Plot subject trajectories and their mean for each available species."""
    species_order = [
        species
        for species in ("minipigs", "monkeys")
        if species in set(subject_means["species"])
    ]
    if not species_order:
        raise RuntimeError("No species are available to plot")
    fig, axes = plt.subplots(
        1,
        len(species_order),
        figsize=(5 * len(species_order), 4),
        sharey=True,
        squeeze=False,
    )
    value_column = f"delta_{metric}"
    for axis, species in zip(axes[0], species_order):
        subset = subject_means[subject_means["species"] == species]
        fixed = subset[subset["checkpoint_kind"] == "milestone"]
        for _, subject in fixed.groupby("excluded_target_subject"):
            subject = subject.sort_values("checkpoint_step")
            axis.plot(
                subject["checkpoint_step"],
                subject[value_column],
                color="0.7",
                linewidth=1,
                alpha=0.8,
            )
        summary = fixed.groupby("checkpoint_step")[value_column].agg(
            ["mean", "sem"]
        )
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=1.96 * summary["sem"],
            marker="o",
            color="#1f77b4",
            linewidth=2.2,
            label="mean across exclusions ± 1.96 SEM",
        )
        best = subset[subset["checkpoint_kind"] == "best"]
        if not best.empty:
            axis.scatter(
                best["checkpoint_global_step"],
                best[value_column],
                marker="x",
                alpha=0.7,
                label="loss-selected best (reference)",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(species.capitalize())
        axis.set_xlabel("Fixed checkpoint step")
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=f"{STEM}-results:latest")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--entity", default=default_entity())
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument(
        "--fixed-steps",
        default=",".join(str(step) for step in FIXED_STEPS),
        help="Fixed checkpoint steps used for slopes (at least three)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    fixed_steps = tuple(int(item) for item in args.fixed_steps.split(","))
    if len(fixed_steps) < 3 or any(
        step not in FIXED_STEPS for step in fixed_steps
    ):
        raise ValueError(
            f"fixed-steps must contain at least three values from {FIXED_STEPS}"
        )
    if tuple(sorted(set(fixed_steps))) != fixed_steps:
        raise ValueError("fixed-steps must be unique and increasing")
    results, qualified_artifact = fetch_results(
        args.artifact, args.project, args.entity
    )
    deltas = add_intact_deltas(results)
    derangement_means = average_derangements(deltas)
    subject_means = subject_checkpoint_means(derangement_means)
    slopes = pd.concat(
        [
            subject_slopes(subject_means, metric, fixed_steps)
            for metric in METRICS
        ],
        ignore_index=True,
    )
    bootstrap_summary, bootstrap_draws = bootstrap_species_slopes(
        slopes, args.bootstrap_replicates, args.bootstrap_seed
    )
    contrasts = secondary_contrasts(deltas)
    bias_condition_means = bias_condition_subject_means(deltas)

    output_csv = csv_dir(__file__)
    deltas.to_csv(output_csv / f"{STEM}_condition_deltas.csv", index=False)
    derangement_means.to_csv(
        output_csv / f"{STEM}_derangement_means.csv", index=False
    )
    subject_means.to_csv(
        output_csv / f"{STEM}_subject_checkpoint_means.csv", index=False
    )
    slopes.to_csv(output_csv / f"{STEM}_subject_slopes.csv", index=False)
    bootstrap_summary.to_csv(
        output_csv / f"{STEM}_bootstrap_summary.csv", index=False
    )
    bootstrap_draws.to_csv(
        output_csv / f"{STEM}_bootstrap_draws.csv", index=False
    )
    contrasts.to_csv(
        output_csv / f"{STEM}_zero_mean_contrasts.csv", index=False
    )
    bias_condition_means.to_csv(
        output_csv / f"{STEM}_bias_condition_subject_means.csv", index=False
    )
    figure_dir = figures_dir(__file__)
    plot_bias_condition_comparison(
        bias_condition_means,
        figure_dir / f"{STEM}_bias_condition_loss_comparison.png",
    )
    plot_metric_curves(
        subject_means,
        "cross_entropy",
        figure_dir / f"{STEM}_cross_entropy_penalty.png",
        "Perturbed − intact cross-entropy",
    )
    plot_metric_curves(
        subject_means,
        "pooled_supported_f1",
        figure_dir / f"{STEM}_pooled_supported_f1_change.png",
        "Perturbed − intact pooled supported F1",
    )
    plot_metric_curves(
        subject_means,
        "recording_mean_supported_f1",
        figure_dir / f"{STEM}_recording_mean_supported_f1_change.png",
        "Perturbed − intact recording-mean supported F1",
    )

    primary = bootstrap_summary[bootstrap_summary["metric"] == "cross_entropy"]
    print(f"Artifact: {qualified_artifact}")
    print(
        "\nPrimary cross-entropy log-step slopes (bootstrap unit: excluded subject)"
    )
    print(primary.to_string(index=False))
    print(
        "\nSupported-F1 slopes (recording_mean is aligned with source-session "
        "checkpoint reporting)"
    )
    print(
        bootstrap_summary[
            bootstrap_summary["metric"].isin(
                ("pooled_supported_f1", "recording_mean_supported_f1")
            )
        ].to_string(index=False)
    )
    print("\nZero/mean secondary contrasts")
    print(
        contrasts.groupby(["species", "condition"], as_index=False)[
            [f"delta_{metric}" for metric in METRICS]
        ]
        .mean()
        .to_string(index=False)
    )
    print("\nBias-condition cross-entropy deltas at fixed checkpoints")
    print(
        bias_condition_means[
            bias_condition_means["checkpoint_kind"] == "milestone"
        ]
        .groupby(
            ["species", "bias_condition", "checkpoint_step"], as_index=False
        )["delta_cross_entropy"]
        .mean()
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

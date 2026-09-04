"""Analyze normalized scratch baselines against Phase-1 raw EEGNet.

Fetches the two global-normalization production pools and the read-only Phase-1
raw EEGNet reference from Weights & Biases.  The script selects one completed
test result per recording/fraction/seed cell, writes reproducible CSV tables,
and generates subject-balanced learning and data-efficiency figures.

Usage:
    uv run python analysis/20260901-MS-scratch-baselines-normalization.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# This analysis performs only small tabular reductions.  Keeping BLAS
# single-threaded avoids exhausting restricted login-node process limits.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GOMAXPROCS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir, unwrap_summary_value


PREFIX = "20260901-MS-scratch-baselines-normalization"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SEEDS = (42, 43, 44)

# These immutable W&B groups are the run identifiers for this experiment.
GROUPS = {
    "raw_eegnet": {
        "label": "EEGNet raw (Phase 1)",
        "model": "EEGNet",
        "normalization": "raw",
        "minipigs": "PHASE1_EEGNET_MINIPIGS",
        "monkeys": "PHASE1_EEGNET_MONKEYS",
    },
    "global_eegnet": {
        "label": "EEGNet train-global z-score",
        "model": "EEGNet",
        "normalization": "train-global z-score",
        "minipigs": "NORM_GLOBAL_EEGNET_MINIPIGS_PROD_OFFLINE_16_20260902",
        "monkeys": "NORM_GLOBAL_EEGNET_MONKEYS_PROD_OFFLINE_16_20260902",
    },
    "global_conv_bigru": {
        "label": "Conv--BiGRU train-global z-score",
        "model": "Conv--BiGRU",
        "normalization": "train-global z-score",
        "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
        "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
    },
}

COLORS = {
    "raw_eegnet": "#777777",
    "global_eegnet": "#0072b2",
    "global_conv_bigru": "#d55e00",
}


def nested_or_flat(config: dict[str, Any], *keys: str) -> Any:
    """Read nested or dotted configuration fields from W&B config data."""
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def neurosoft_field(config: dict[str, Any], name: str) -> Any:
    value = nested_or_flat(config, "neurosoft", name)
    return value if value is not None else config.get(f"neurosoft/{name}")


def recording_id(config: dict[str, Any], run_name: str) -> str | None:
    value = neurosoft_field(config, "recording_id")
    if value:
        return str(value)
    ids = nested_or_flat(config, "data", "dataset_kwargs", "recording_ids")
    if isinstance(ids, list) and ids:
        return str(ids[0])
    match = re.search(r"(sub-\d+_ses-\d+_task-AcousStim_acq-[A-Za-z]+(?:anest)?_desc-raw)", run_name)
    return match.group(1) if match else None


def subject_from(recording: str | None, config: dict[str, Any]) -> str | None:
    value = neurosoft_field(config, "subject")
    if value:
        return str(value)
    match = re.search(r"(sub-\d+)", recording or "")
    return match.group(1) if match else None


def fraction_from(config: dict[str, Any], run_name: str) -> float | None:
    for key in ("training_fraction_requested", "training_fraction"):
        value = neurosoft_field(config, key)
        if value is not None:
            return float(value)
    value = nested_or_flat(config, "data", "training_fraction")
    if value is not None:
        return float(value)
    match = re.search(r"_f(0?\.\d+|1\.0+|1)_", run_name)
    return float(match.group(1)) if match else None


def seed_from(config: dict[str, Any], run_name: str) -> int | None:
    for path in (("neurosoft", "model_seed"), ("run", "seed")):
        value = nested_or_flat(config, *path)
        if value is not None:
            return int(value)
    value = config.get("neurosoft/model_seed")
    if value is not None:
        return int(value)
    match = re.search(r"_s(\d+)$", run_name)
    return int(match.group(1)) if match else None


def test_f1_from(summary: dict[str, Any]) -> float | None:
    key = f"test/{TASK}_supported_f1"
    for candidate in (f"{key}.max", key):
        value = summary.get(candidate)
        if value is not None:
            try:
                return float(unwrap_summary_value(value, "max"))
            except (TypeError, ValueError):
                pass
    return None


def collect_runs(entity: str | None) -> pd.DataFrame:
    """Fetch declared groups and retain only planned model/fraction/seed cells."""
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Could not resolve a W&B entity; set WANDB_ENTITY.")
    path = f"{entity}/{PROJECT}"
    rows: list[dict[str, Any]] = []

    for condition, info in GROUPS.items():
        for species in ("minipigs", "monkeys"):
            group = info[species]
            print(f"Fetching W&B group: {group}", flush=True)
            # ``lazy=False`` asks W&B for config and summary metrics in the
            # paginated query.  Accessing those fields lazily causes one
            # additional request per run, which is prohibitively slow here.
            group_runs = list(
                api.runs(
                    path,
                    filters={"group": group},
                    per_page=500,
                    lazy=False,
                )
            )
            print(f"Fetched {len(group_runs):4d} raw records: {group}")
            for run in group_runs:
                config = dict(run.config or {})
                name = str(run.name or "")
                fraction = fraction_from(config, name)
                seed = seed_from(config, name)
                rec = recording_id(config, name)
                f1 = test_f1_from(dict(run.summary or {}))
                if fraction not in FRACTIONS or seed not in SEEDS or not rec:
                    continue
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": info["label"],
                        "model": info["model"],
                        "normalization": info["normalization"],
                        "wandb_group": group,
                        "species": species,
                        "recording_id": rec,
                        "subject": subject_from(rec, config),
                        "fraction": fraction,
                        "seed": seed,
                        "run_id": run.id,
                        "run_name": name,
                        "state": run.state,
                        "created_at": getattr(run, "created_at", None),
                        "is_retry": "retry" in set(getattr(run, "tags", []) or []),
                        "test_supported_macro_f1": f1,
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No planned runs were resolved from the declared W&B groups.")
    return table.sort_values(["condition", "species", "recording_id", "fraction", "seed", "created_at"])


def canonical_test_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Prefer a completed primary run over a retry for each planned cell."""
    complete = raw[(raw.state == "finished") & raw.test_supported_macro_f1.notna()].copy()
    if complete.empty:
        raise RuntimeError("No completed runs with test supported macro-F1 were found.")
    complete["is_retry"] = complete.is_retry.fillna(False).astype(bool)
    return (
        complete.sort_values(
            ["condition", "species", "recording_id", "fraction", "seed", "is_retry", "created_at", "run_id"]
        )
        .drop_duplicates(["condition", "species", "recording_id", "fraction", "seed"], keep="first")
        .reset_index(drop=True)
    )


def subject_balanced_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Average seed → recording → subject → species at every condition/fraction."""
    session = (
        runs.groupby(["condition", "condition_label", "species", "subject", "recording_id", "fraction"], as_index=False)
        .test_supported_macro_f1.mean()
    )
    subject = (
        session.groupby(["condition", "condition_label", "species", "subject", "fraction"], as_index=False)
        .test_supported_macro_f1.mean()
    )
    return (
        subject.groupby(["condition", "condition_label", "species", "fraction"], as_index=False)
        .agg(n_subjects=("subject", "nunique"), mean_test_f1=("test_supported_macro_f1", "mean"), sd_test_f1=("test_supported_macro_f1", "std"))
        .sort_values(["species", "fraction", "condition"])
    )


def cumulative_to_80(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute condition-specific 80%-of-own-full-data target attainment."""
    session = (
        runs.groupby(["condition", "condition_label", "species", "recording_id", "fraction"], as_index=False)
        .test_supported_macro_f1.mean()
    )
    rows: list[dict[str, Any]] = []
    for (condition, label, species, rec), data in session.groupby(["condition", "condition_label", "species", "recording_id"]):
        full = data[np.isclose(data.fraction, 1.0)]
        if full.empty:
            continue
        target = 0.8 * float(full.test_supported_macro_f1.iloc[0])
        reached = next((f for f in FRACTIONS if not data[np.isclose(data.fraction, f)].empty and float(data[np.isclose(data.fraction, f)].test_supported_macro_f1.iloc[0]) >= target), None)
        rows.append({"condition": condition, "condition_label": label, "species": species, "recording_id": rec, "target_f1": target, "reached_fraction": reached})
    targets = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (condition, label, species), data in targets.groupby(["condition", "condition_label", "species"]):
        for fraction in FRACTIONS:
            n = len(data)
            reached = int(data.reached_fraction.le(fraction).sum())
            summary_rows.append({"condition": condition, "condition_label": label, "species": species, "fraction": fraction, "n_sessions": n, "n_reached_80pct": reached, "share_reached_80pct": reached / n})
    return targets, pd.DataFrame(summary_rows)


def performance_qualified_cumulative(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute target attainment on a shared, absolute-performance set.

    A recording qualifies when its mean full-data F1 pooled over all three
    conditions is at least the median pooled full-data F1 for its species. The
    same qualified recordings are therefore used in every condition, avoiding
    a condition-specific denominator that could itself induce an apparent
    data-efficiency advantage.
    """
    session = (
        runs.groupby(
            ["condition", "species", "recording_id", "fraction"],
            as_index=False,
        )
        .test_supported_macro_f1.mean()
    )
    full = session[np.isclose(session.fraction, 1.0)].pivot(
        index=["species", "recording_id"],
        columns="condition",
        values="test_supported_macro_f1",
    )
    if full.empty or not set(GROUPS).issubset(full.columns):
        raise RuntimeError("Missing full-data results needed for performance qualification.")
    full = full[list(GROUPS)].copy()
    full["pooled_full_data_f1"] = full.mean(axis=1)
    full["species_median_pooled_full_data_f1"] = full.groupby(level="species")[
        "pooled_full_data_f1"
    ].transform("median")
    full["qualified"] = (
        full.pooled_full_data_f1 >= full.species_median_pooled_full_data_f1
    )
    eligibility = full.reset_index()
    qualified_keys = eligibility.loc[
        eligibility.qualified, ["species", "recording_id"]
    ]
    qualified_runs = runs.merge(qualified_keys, on=["species", "recording_id"])
    targets, cumulative = cumulative_to_80(qualified_runs)
    return eligibility, targets, cumulative


def paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair contrasts by species, recording, fraction, and initialization seed."""
    index = ["species", "recording_id", "subject", "fraction", "seed"]
    wide = runs.pivot_table(index=index, columns="condition", values="test_supported_macro_f1", aggfunc="first").reset_index()
    rows: list[pd.DataFrame] = []
    for comparison, lhs, rhs in (
        ("global EEGNet − raw EEGNet", "global_eegnet", "raw_eegnet"),
        ("global Conv--BiGRU − global EEGNet", "global_conv_bigru", "global_eegnet"),
    ):
        if lhs not in wide or rhs not in wide:
            continue
        part = wide[index + [lhs, rhs]].dropna().copy()
        part["comparison"] = comparison
        part["lhs_f1"] = part[lhs]
        part["rhs_f1"] = part[rhs]
        part["delta_f1"] = part[lhs] - part[rhs]
        rows.append(part[index + ["comparison", "lhs_f1", "rhs_f1", "delta_f1"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return contrasts
    return (
        contrasts.groupby(["comparison", "species", "fraction"], as_index=False)
        .agg(n_paired_cells=("delta_f1", "count"), mean_delta_f1=("delta_f1", "mean"), sd_delta_f1=("delta_f1", "std"))
        .sort_values(["comparison", "species", "fraction"])
    )


def plot_learning_curves(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        for condition, data in summary[summary.species.eq(species)].groupby("condition"):
            data = data.sort_values("fraction")
            axis.errorbar(data.fraction * 100, data.mean_test_f1, yerr=data.sd_test_f1.fillna(0), marker="o", capsize=3, color=COLORS[condition], label=data.condition_label.iloc[0])
        axis.set(title=species.title(), xlabel="Training data (%)", ylabel="Subject-balanced test supported macro-F1")
        axis.set_xticks(np.array(FRACTIONS) * 100)
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Scratch-baseline learning curves (mean ± SD across subjects)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_data_efficiency(
    cumulative: pd.DataFrame,
    output: Path,
    *,
    title: str = "Condition-specific data efficiency",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        for condition, data in cumulative[cumulative.species.eq(species)].groupby("condition"):
            data = data.sort_values("fraction")
            axis.plot(
                data.fraction * 100,
                data.share_reached_80pct * 100,
                marker="o",
                color=COLORS[condition],
                label=data.condition_label.iloc[0],
            )
        axis.set(title=species.title(), xlabel="Training data (%)", ylabel="Sessions reaching 80% of own full-data F1 (%)", ylim=(-2, 102))
        axis.set_xticks(np.array(FRACTIONS) * 100)
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_contrasts(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        for comparison, data in summary[summary.species.eq(species)].groupby("comparison"):
            data = data.sort_values("fraction")
            axis.errorbar(data.fraction * 100, data.mean_delta_f1, yerr=data.sd_delta_f1.fillna(0), marker="o", capsize=3, label=comparison)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=species.title(), xlabel="Training data (%)", ylabel="Paired test supported macro-F1 difference")
        axis.set_xticks(np.array(FRACTIONS) * 100)
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Paired contrasts: positive values favor the named left condition")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw = collect_runs(default_entity())
    canonical = canonical_test_runs(raw)
    balanced = subject_balanced_summary(canonical)
    targets, cumulative = cumulative_to_80(canonical)
    qualification, qualified_targets, qualified_cumulative = (
        performance_qualified_cumulative(canonical)
    )
    contrasts = paired_contrasts(canonical)
    contrast_summary = paired_summary(contrasts)
    csv_root, figure_root = csv_dir(__file__), figures_dir(__file__)
    outputs = {
        "raw W&B records": (raw, csv_root / f"{PREFIX}_raw_runs.csv"),
        "canonical test results": (canonical, csv_root / f"{PREFIX}_canonical_test_results.csv"),
        "subject-balanced summary": (balanced, csv_root / f"{PREFIX}_subject_balanced.csv"),
        "80% targets": (targets, csv_root / f"{PREFIX}_data_to_80_targets.csv"),
        "80% cumulative summary": (cumulative, csv_root / f"{PREFIX}_cumulative_data_to_80.csv"),
        "performance-qualification table": (
            qualification,
            csv_root / f"{PREFIX}_performance_qualification.csv",
        ),
        "qualified 80% targets": (
            qualified_targets,
            csv_root / f"{PREFIX}_qualified_data_to_80_targets.csv",
        ),
        "qualified 80% cumulative summary": (
            qualified_cumulative,
            csv_root / f"{PREFIX}_qualified_cumulative_data_to_80.csv",
        ),
        "paired cells": (contrasts, csv_root / f"{PREFIX}_paired_contrasts.csv"),
        "paired summary": (contrast_summary, csv_root / f"{PREFIX}_paired_summary.csv"),
    }
    for label, (table, path) in outputs.items():
        table.to_csv(path, index=False)
        print(f"Wrote {label}: {path} ({len(table)} rows)")
    plots = {
        "learning curves": (plot_learning_curves, balanced),
        "data efficiency": (plot_data_efficiency, cumulative),
        "performance-qualified data efficiency": (
            lambda data, path: plot_data_efficiency(
                data,
                path,
                title="Data efficiency among above-median full-data recordings",
            ),
            qualified_cumulative,
        ),
        "paired contrasts": (plot_paired_contrasts, contrast_summary),
    }
    for suffix, (plotter, table) in plots.items():
        path = figure_root / f"{PREFIX}_{suffix.replace(' ', '_')}.png"
        plotter(table, path)
        print(f"Wrote {suffix} figure: {path}")
    print("\nSubject-balanced test supported macro-F1:")
    print(balanced.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCumulative sessions reaching 80% of condition-specific full-data F1:")
    print(cumulative.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPerformance-qualified cumulative attainment (shared recording set):")
    print(qualified_cumulative.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired test supported macro-F1 contrasts:")
    print(contrast_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

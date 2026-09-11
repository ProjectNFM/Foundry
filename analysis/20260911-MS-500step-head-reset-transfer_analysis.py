"""Analyze the completed Phase 4C 500-step head-reset transfer experiment.

The script audits the immutable Phase 4C cell lists against W&B, adds the
exact Phase 4B step-500/head-reuse and Phase-2 scratch controls, and computes
the preregistered subject-balanced F1 and stable time-to-90%-of-peak endpoints.

Run with::

    WANDB_ENTITY=poyo-eeg uv run python analysis/20260911-MS-500step-head-reset-transfer_analysis.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir, unwrap_summary_value


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
VALIDATION_F1 = f"val/{TASK}_supported_f1"
PREFIX = "20260911-MS-500step-head-reset-transfer"
SPECIES = ("minipigs", "monkeys")
SEEDS = {42, 43, 44}
UNIT = ["species", "subject", "recording", "target_seed"]
CONDITION_ORDER = [
    "scratch",
    "head_reuse_full",
    "head_reset_full",
    "frozen_representation",
    "frozen_random_control",
]
CONDITION_LABELS = {
    "scratch": "Scratch",
    "head_reuse_full": "500-step\nhead reuse",
    "head_reset_full": "500-step\nhead reset",
    "frozen_representation": "500-step\nfrozen repr.",
    "frozen_random_control": "Frozen\nrandom",
}

NEW_CELL_LISTS = {
    species: ROOT / "launch/phase4c" / f"phase4c-step500-head-reset-{species}.jsonl"
    for species in SPECIES
}
EARLY_CELL_LISTS = {
    species: ROOT / "launch/phase4a" / f"phase4a-early-checkpoint-full-ft-lr1p5e3-{species}.jsonl"
    for species in SPECIES
}
NEW_GROUPS = {
    "head_reset_full": {
        "minipigs": "PHASE4C_STEP500_HEAD_RESET_FULL_FT_MINIPIGS",
        "monkeys": "PHASE4C_STEP500_HEAD_RESET_FULL_FT_MONKEYS",
    },
    "frozen_representation": {
        "minipigs": "PHASE4C_STEP500_FROZEN_REPRESENTATION_MINIPIGS",
        "monkeys": "PHASE4C_STEP500_FROZEN_REPRESENTATION_MONKEYS",
    },
    "frozen_random_control": {
        "minipigs": "PHASE4C_FROZEN_RANDOM_MINIPIGS",
        "monkeys": "PHASE4C_FROZEN_RANDOM_MONKEYS",
    },
}
EARLY_GROUPS = {
    "minipigs": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MINIPIGS",
    "monkeys": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MONKEYS",
}
SCRATCH_GROUPS = {
    "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
}
EXPECTED_NEW = {"minipigs": 360, "monkeys": 117}
EXPECTED_PER_CONDITION = {"minipigs": 120, "monkeys": 39}
EXPECTED_HEAD_REUSE = {"minipigs": 120, "monkeys": 39}
EXPECTED_SCRATCH = {"minipigs": 120, "monkeys": 39}
SUMMARY_METRICS = [
    "test_supported_f1",
    "best_val_supported_f1",
    "best_step",
    "best_windows",
    "best_flops",
    "best_wall_time_s",
]


def nested(config: dict[str, Any], *keys: str) -> Any:
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def scalar(summary: dict[str, Any], key: str) -> float | None:
    for candidate in (f"{key}.max", key):
        value = summary.get(candidate)
        if value is None:
            continue
        try:
            result = float(unwrap_summary_value(value, "max"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def metric_values(summary: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in SUMMARY_METRICS:
        key = (
            f"test/{TASK}_supported_f1"
            if metric == "test_supported_f1"
            else f"val/{TASK}_supported_f1"
            if metric == "best_val_supported_f1"
            else f"compute/{metric}"
        )
        result[metric] = scalar(summary, key)
    return result


def load_cells(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for species, path in paths.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("species") != species:
                raise RuntimeError(f"{path}: cell species mismatch")
            cell_id = str(row["cell_id"])
            if cell_id in cells:
                raise RuntimeError(f"Duplicate cell ID: {cell_id}")
            cells[cell_id] = row
    return cells


def selected_head_reuse_cells() -> dict[str, dict[str, Any]]:
    """Select the exact Phase 4B step-500, source-(42,42) cells."""
    all_cells = load_cells(EARLY_CELL_LISTS)
    selected = {
        cell_id: row
        for cell_id, row in all_cells.items()
        if str(row["checkpoint_id"]).endswith("step500")
        and row["source_selection_seed"] == 42
        and row["source_model_seed"] == 42
    }
    counts = {
        species: sum(row["species"] == species for row in selected.values())
        for species in SPECIES
    }
    if counts != EXPECTED_HEAD_REUSE:
        raise RuntimeError(f"Unexpected Phase 4B step-500 cell counts: {counts}")
    return selected


def expected_observation(config: dict[str, Any]) -> dict[str, Any]:
    recording_ids = nested(config, "data", "dataset_kwargs", "recording_ids")
    return {
        "cell_id": nested(config, "run", "cell_id"),
        "recording": str(recording_ids[0])
        if isinstance(recording_ids, list) and recording_ids
        else None,
        "target_fraction": as_float(nested(config, "data", "training_fraction")),
        "target_seed": nested(config, "run", "seed"),
        "checkpoint_id": nested(config, "run", "checkpoint_id"),
        "checkpoint_manifest": nested(
            config, "run", "pretrained_checkpoint_manifest"
        ),
        "checkpoint_manifest_hash": nested(
            config, "run", "pretrained_checkpoint_manifest_hash"
        ),
        "checkpoint_sha256": nested(config, "run", "pretrained_checkpoint_sha256"),
        "source_selection_seed": nested(config, "run", "source_selection_seed"),
        "source_model_seed": nested(config, "run", "source_model_seed"),
        "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
    }


def audit_declared_run(
    run: Any,
    expected: dict[str, Any],
    species: str,
    condition: str,
) -> None:
    config = dict(run.config or {})
    observed = expected_observation(config)
    expected_observed = {
        "cell_id": expected["cell_id"],
        "recording": expected["target_recording"],
        "target_fraction": float(expected["target_fraction"]),
        "target_seed": expected["target_finetuning_seed"],
        "checkpoint_id": expected.get("checkpoint_id"),
        "checkpoint_manifest": expected.get("checkpoint_manifest"),
        "checkpoint_manifest_hash": expected.get("checkpoint_manifest_hash"),
        "checkpoint_sha256": expected.get("checkpoint_sha256"),
        "source_selection_seed": expected.get("source_selection_seed"),
        "source_model_seed": expected.get("source_model_seed"),
        "transfer_regime": expected.get("transfer_regime"),
    }
    if condition == "head_reuse_full":
        expected_observed["transfer_regime"] = "full_finetuning"
    if observed != expected_observed:
        raise RuntimeError(
            f"W&B provenance mismatch for {species}/{condition}/{run.id}:\n"
            f"observed={observed}\nexpected={expected_observed}"
        )
    if run.state != "finished":
        raise RuntimeError(
            f"Declared {species}/{condition} run {run.id} is {run.state}, not finished"
        )


def run_record(
    run: Any,
    *,
    expected: dict[str, Any],
    species: str,
    condition: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": condition,
        "species": species,
        "subject": expected["target_subject"],
        "recording": expected["target_recording"],
        "target_seed": int(expected["target_finetuning_seed"]),
        "run_id": str(run.id),
        "run_name": str(run.name or ""),
        "state": str(run.state),
        "cell_id": expected.get("cell_id"),
        "wandb_group": str(run.group or ""),
        "source_selection_seed": expected.get("source_selection_seed"),
        "source_model_seed": expected.get("source_model_seed"),
        "transfer_regime": expected.get("transfer_regime"),
        "target_fraction": expected.get("target_fraction"),
    }
    row.update(metric_values(dict(run.summary or {})))
    return row


def fetch_declared_group(
    api: Any,
    entity: str,
    group: str,
    species: str,
    condition: str,
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    runs = list(
        api.runs(
            f"{entity}/{PROJECT}",
            filters={"group": group},
            per_page=500,
            lazy=False,
        )
    )
    by_id = {str(run.id): run for run in runs}
    expected = {
        str(row["wandb_run_id"]): row
        for row in cells.values()
        if row["species"] == species
        and (
            condition != "head_reset_full"
            or row["transfer_regime"] == "full_finetuning_reset_router"
        )
        and (
            condition != "frozen_representation"
            or row["transfer_regime"] == "frozen_representation"
        )
        and (
            condition != "frozen_random_control"
            or row["transfer_regime"] == "frozen_random_control"
        )
    }
    if not set(expected).issubset(by_id):
        missing = sorted(set(expected) - set(by_id))
        raise RuntimeError(
            f"{species}/{condition}: group {group} has {len(runs)} runs; "
            f"missing={missing[:5]}"
        )
    rows = []
    for run_id, row in expected.items():
        run = by_id[run_id]
        audit_declared_run(run, row, species, condition)
        record = run_record(run, expected=row, species=species, condition=condition)
        if any(record[key] is None for key in SUMMARY_METRICS):
            raise RuntimeError(f"{run_id}: missing required summary metric")
        rows.append(record)
    return rows


def fetch_head_reuse(
    api: Any, entity: str, cells: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, group in EARLY_GROUPS.items():
        rows.extend(fetch_declared_group(api, entity, group, species, "head_reuse_full", cells))
    return rows


def fetch_scratch(
    api: Any,
    entity: str,
    target_cells: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    targets = {
        species: {
            (row["target_subject"], row["target_recording"], seed)
            for row in target_cells.values()
            if row["species"] == species
            for seed in SEEDS
        }
        for species in SPECIES
    }
    rows: list[dict[str, Any]] = []
    for species, group in SCRATCH_GROUPS.items():
        runs = list(api.runs(f"{entity}/{PROJECT}", filters={"group": group}, per_page=500, lazy=False))
        for run in runs:
            config, summary = dict(run.config or {}), dict(run.summary or {})
            recording_ids = nested(config, "data", "dataset_kwargs", "recording_ids")
            recording = str(recording_ids[0]) if isinstance(recording_ids, list) and recording_ids else None
            seed = nested(config, "run", "seed")
            fraction = as_float(nested(config, "data", "training_fraction"))
            subject = next(
                (row["target_subject"] for row in target_cells.values()
                 if row["species"] == species and row["target_recording"] == recording),
                None,
            )
            key = (subject, recording, seed)
            if run.state != "finished" or key not in targets[species] or fraction != 1.0:
                continue
            values = metric_values(summary)
            if any(value is None for value in values.values()):
                continue
            rows.append({
                "condition": "scratch", "species": species, "subject": subject,
                "recording": recording, "target_seed": int(seed), "run_id": str(run.id),
                "run_name": str(run.name or ""), "state": str(run.state), "cell_id": None,
                "wandb_group": group, "source_selection_seed": None,
                "source_model_seed": None, "transfer_regime": None,
                "target_fraction": fraction, **values,
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(UNIT, keep=False).any():
        raise RuntimeError("Scratch controls are empty or duplicated by paired unit")
    missing_rows = []
    for species in SPECIES:
        expected = targets[species]
        actual = set(
            map(
                tuple,
                frame.loc[frame.species.eq(species), ["subject", "recording", "target_seed"]].to_numpy(),
            )
        )
        for subject, recording, seed in sorted(expected - actual):
            missing_rows.append({
                "species": species, "subject": subject, "recording": recording,
                "target_seed": seed, "reason": "no finished 100%-data scratch control found",
            })
    missing = pd.DataFrame(missing_rows)
    print(f"Scratch available by species: {frame.groupby('species').size().to_dict()}; declared: {EXPECTED_SCRATCH}")
    if not missing.empty:
        print("Missing scratch units:\n" + missing.to_string(index=False))
    return rows, missing


def stable_time_to_90_percent_peak(history: pd.DataFrame) -> dict[str, Any]:
    values = history[["optimizer_step", VALIDATION_F1]].dropna().copy()
    values = values.sort_values("optimizer_step").drop_duplicates("optimizer_step", keep="last")
    if len(values) < 3:
        raise ValueError("Need at least three validation evaluations")
    values["smoothed_f1"] = values[VALIDATION_F1].rolling(3, min_periods=3).median()
    values = values.dropna(subset=["smoothed_f1"]).reset_index(drop=True)
    threshold = 0.90 * float(values["smoothed_f1"].max())
    stable = values["smoothed_f1"].ge(threshold)
    crossings = stable & stable.shift(-1, fill_value=False) & stable.shift(-2, fill_value=False)
    if crossings.any():
        row = values.loc[crossings.idxmax()]
        return {
            "stable_step": float(row["optimizer_step"]), "censored": False,
            "peak_smoothed_f1": float(values["smoothed_f1"].max()),
            "final_validation_step": float(values["optimizer_step"].iloc[-1]),
        }
    return {
        "stable_step": float(values["optimizer_step"].iloc[-1]), "censored": True,
        "peak_smoothed_f1": float(values["smoothed_f1"].max()),
        "final_validation_step": float(values["optimizer_step"].iloc[-1]),
    }


def fetch_history(api: Any, entity: str, row: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
    history = run.history(keys=["_step", "epoch", "trainer/global_step", VALIDATION_F1], samples=10_000, pandas=True)
    required = ["_step", "trainer/global_step", VALIDATION_F1]
    if any(key not in history.columns for key in required):
        raise RuntimeError(f"{row['run_id']}: missing convergence history fields")
    history = history.rename(columns={"trainer/global_step": "optimizer_step"})
    history = history.dropna(subset=["optimizer_step", VALIDATION_F1]).copy()
    endpoint = stable_time_to_90_percent_peak(history)
    endpoint_row = {
        "run_id": row["run_id"], "condition": row["condition"], "species": row["species"],
        "subject": row["subject"], "recording": row["recording"], "target_seed": row["target_seed"],
        **endpoint, "n_validations": len(history),
    }
    curve = history[["_step", "epoch", "optimizer_step", VALIDATION_F1]].copy()
    for key in ("condition", "species", "subject", "recording", "target_seed", "run_id"):
        curve[key] = row[key]
    return endpoint_row, curve


def fetch_all_histories(api: Any, entity: str, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_history, api, entity, row.to_dict()): row["run_id"] for _, row in runs.iterrows()}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                endpoint, curve = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch validation history for {futures[future]}") from exc
            endpoints.append(endpoint)
            curves.append(curve)
            if index % 100 == 0:
                print(f"Fetched {index}/{len(futures)} validation histories", flush=True)
    return pd.DataFrame(endpoints), pd.concat(curves, ignore_index=True)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    bootstrap = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def subject_level(runs: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    merged = runs.merge(
        endpoints[["run_id", "stable_step", "censored", "peak_smoothed_f1", "final_validation_step", "n_validations"]],
        on="run_id", validate="one_to_one",
    )
    return merged.groupby(["species", "condition", "subject"], as_index=False).agg(
        test_supported_f1=("test_supported_f1", "mean"),
        best_val_supported_f1=("best_val_supported_f1", "mean"),
        stable_step=("stable_step", "mean"),
        censored_fraction=("censored", "mean"),
        n_units=("run_id", "size"),
    )


def paired_subject_effects(subjects: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["test_supported_f1", "stable_step"]
    scratch = subjects[subjects.condition.eq("scratch")][["species", "subject"] + value_cols].rename(
        columns={key: f"scratch_{key}" for key in value_cols}
    )
    paired = subjects[~subjects.condition.eq("scratch")].merge(
        scratch, on=["species", "subject"], validate="many_to_one"
    )
    paired["test_f1_gain"] = paired["test_supported_f1"] - paired["scratch_test_supported_f1"]
    paired["stable_step_saved"] = paired["scratch_stable_step"] - paired["stable_step"]
    return paired


def summarize_absolute(subjects: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260911)
    records = []
    for (species, condition), data in subjects.groupby(["species", "condition"], sort=False):
        values = data["test_supported_f1"].to_numpy(float)
        low, high = bootstrap_ci(values, rng)
        records.append({
            "species": species, "condition": condition, "n_subjects": len(values),
            "mean": values.mean(), "ci_low": low, "ci_high": high,
            "censored_fraction": data["censored_fraction"].mean(),
        })
    return pd.DataFrame(records)


def summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260912)
    records = []
    for (species, condition), data in paired.groupby(["species", "condition"], sort=False):
        for metric in ("test_f1_gain", "stable_step_saved"):
            values = data[metric].to_numpy(float)
            low, high = bootstrap_ci(values, rng)
            records.append({
                "species": species, "condition": condition, "metric": metric,
                "n_subjects": len(values), "mean": values.mean(),
                "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(records)


def plot_absolute(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, species in zip(axes, SPECIES, strict=True):
        data = summary[summary.species.eq(species)].set_index("condition").reindex(CONDITION_ORDER).reset_index()
        x = np.arange(len(data))
        axis.errorbar(x, data["mean"], yerr=[data["mean"] - data["ci_low"], data["ci_high"] - data["mean"]], fmt="o", capsize=4, linewidth=1.8)
        axis.set(xticks=x, xticklabels=[CONDITION_LABELS[c] for c in data.condition], title=species.title(), ylabel="Subject-balanced test supported macro-F1")
        axis.grid(axis="y", alpha=.25)
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle("Phase 4C target-test performance (95% subject bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_paired(summary: pd.DataFrame, output: Path, metric: str, ylabel: str, title: str) -> None:
    data = summary[summary.metric.eq(metric)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, species in zip(axes, SPECIES, strict=True):
        subset = data[data.species.eq(species)].set_index("condition").reindex(CONDITION_ORDER[1:]).reset_index()
        x = np.arange(len(subset))
        axis.errorbar(x, subset["mean"], yerr=[subset["mean"] - subset["ci_low"], subset["ci_high"] - subset["mean"]], fmt="o", capsize=4, linewidth=1.8)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(xticks=x, xticklabels=[CONDITION_LABELS[c] for c in subset.condition], title=species.title(), ylabel=ylabel)
        axis.grid(axis="y", alpha=.25)
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(curves: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, species in zip(axes, SPECIES, strict=True):
        species_curves = curves[curves.species.eq(species)]
        for condition in CONDITION_ORDER:
            subset = species_curves[species_curves.condition.eq(condition)]
            if subset.empty:
                continue
            normalized = []
            for _, run in subset.groupby("run_id"):
                run = run.sort_values("optimizer_step")
                x = np.linspace(0, 1, len(run))
                normalized.append(np.interp(np.linspace(0, 1, 51), x, run[VALIDATION_F1].to_numpy(float)))
            axis.plot(np.linspace(0, 1, 51), np.nanmedian(np.asarray(normalized), axis=0), linewidth=2, label=CONDITION_LABELS[condition].replace("\n", " "))
        axis.set(title=species.title(), xlabel="Relative training progress", ylabel="Median validation supported macro-F1")
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Validation learning curves (run-normalized progress)")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    api = wandb.Api(timeout=120)
    entity = default_entity() or getattr(api, "default_entity", None)
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY, e.g. WANDB_ENTITY=poyo-eeg")
    new_cells = load_cells(NEW_CELL_LISTS)
    new_counts = {species: sum(row["species"] == species for row in new_cells.values()) for species in SPECIES}
    if new_counts != EXPECTED_NEW:
        raise RuntimeError(f"Phase 4C compiled cell counts do not match 3 conditions x 3 seeds: {new_counts}")
    raw_rows: list[dict[str, Any]] = []
    for condition, groups in NEW_GROUPS.items():
        for species, group in groups.items():
            raw_rows.extend(fetch_declared_group(api, entity, group, species, condition, new_cells))
    raw_rows.extend(fetch_head_reuse(api, entity, selected_head_reuse_cells()))
    scratch_rows, missing_scratch = fetch_scratch(api, entity, new_cells)
    raw_rows.extend(scratch_rows)
    runs = pd.DataFrame(raw_rows)
    expected_counts = {"head_reset_full": EXPECTED_PER_CONDITION, "frozen_representation": EXPECTED_PER_CONDITION, "frozen_random_control": EXPECTED_PER_CONDITION, "head_reuse_full": EXPECTED_HEAD_REUSE}
    print("=== W&B completion audit ===")
    for condition, expected_by_species in expected_counts.items():
        data = runs[runs.condition.eq(condition)]
        counts = data.groupby(["species", "state"]).size().unstack(fill_value=0)
        print(f"{condition}:\n{counts.to_string()}")
        for species, expected in expected_by_species.items():
            actual = int(data.species.eq(species).sum())
            if actual != expected:
                raise RuntimeError(f"{condition}/{species}: {actual}, expected {expected}")
        if not data.state.eq("finished").all():
            raise RuntimeError(f"{condition}: non-finished run found")
    print(f"New Phase 4C runs: {sum(runs.condition.isin(NEW_GROUPS))}/477 finished")
    print(f"Reused Phase 4B head-reuse runs: {len(runs[runs.condition.eq('head_reuse_full')])}/159 finished")
    print(f"Scratch controls: {len(scratch_rows)}/159 discovered as finished 100%-data controls")

    endpoints, curves = fetch_all_histories(api, entity, runs)
    paired_units = set(map(tuple, runs[UNIT].to_numpy()))
    for condition in CONDITION_ORDER:
        paired_units &= set(map(tuple, runs.loc[runs.condition.eq(condition), UNIT].to_numpy()))
    common = pd.DataFrame(sorted(paired_units), columns=UNIT)
    if common.groupby("species").size().to_dict() != {"minipigs": 120, "monkeys": 38}:
        raise RuntimeError("Unexpected exact common paired population")
    paired_runs = runs.merge(common, on=UNIT, how="inner", validate="many_to_one")
    subjects = subject_level(paired_runs, endpoints)
    absolute = summarize_absolute(subjects)
    paired = paired_subject_effects(subjects)
    paired_summary = summarize_paired(paired)
    out_csv, out_figures = csv_dir(__file__), figures_dir(__file__)
    for name, frame in {
        "run_audit": runs, "missing_scratch_units": missing_scratch,
        "validation_endpoints": endpoints, "validation_history": curves,
        "common_paired_units": common, "subject_level_metrics": subjects,
        "paired_subject_effects": paired, "absolute_f1_summary": absolute,
        "paired_effect_summary": paired_summary,
    }.items():
        frame.to_csv(out_csv / f"{PREFIX}_{name}.csv", index=False)
    plot_absolute(absolute, out_figures / f"{PREFIX}_absolute_test_f1.png")
    plot_paired(paired_summary, out_figures / f"{PREFIX}_paired_test_f1_gain.png", "test_f1_gain", "Condition − scratch test supported macro-F1", "Paired subject-balanced F1 effect versus scratch (95% CI)")
    plot_paired(paired_summary, out_figures / f"{PREFIX}_time_to_90_saved.png", "stable_step_saved", "Optimizer steps saved versus scratch", "Stable time-to-90%-of-peak efficiency versus scratch (95% CI)")
    plot_validation_curves(curves, out_figures / f"{PREFIX}_validation_curves.png")

    print("\n=== Exact common paired population ===")
    print(common.groupby("species").size().rename("subject-seed-recording units").to_string())
    print("\n=== Subject-balanced absolute test supported macro-F1 ===")
    print(absolute.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Paired effects versus scratch ===")
    print(paired_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Prespecified gates by species ===")
    f1 = paired_summary[paired_summary.metric.eq("test_f1_gain")]
    time = paired_summary[paired_summary.metric.eq("stable_step_saved")]
    for _, row in f1.iterrows():
        time_row = time[(time.species == row.species) & (time.condition == row.condition)].iloc[0]
        safe = row.ci_low >= -0.01
        earlier = time_row.ci_low > 0
        print(f"{row.species:8s} {row.condition:24s} F1-safe={safe} (lower={row.ci_low:.4f}); earlier-supported={earlier} (saved={time_row['mean']:.1f}, lower={time_row['ci_low']:.1f}); favorable={safe and earlier}")


if __name__ == "__main__":
    main()

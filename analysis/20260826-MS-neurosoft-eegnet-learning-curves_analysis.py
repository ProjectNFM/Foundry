"""Analyze Phase 1 EEGNet learning curves for NeuroSoft supervised pretraining.

Fetches runs from WandB groups ``PHASE1_EEGNET_MINIPIGS`` and
``PHASE1_EEGNET_MONKEYS``, validates against the Phase 0 audit JSON, and
writes per-run metrics, learning curves, data/optimization efficiency tables,
class-count-stratified summaries, subject-balanced aggregates, and integrity
reports.

Usage::

    uv run python analysis/20260826-MS-neurosoft-eegnet-learning-curves_analysis.py
    uv run python analysis/20260826-MS-neurosoft-eegnet-learning-curves_analysis.py --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import (
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

OUTPUT_PREFIX = "20260826-MS-neurosoft-eegnet-learning-curves"
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)

TASK = "neurosoft_acoustic_stim_8band"
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SEEDS = (42, 43, 44)
GROUPS = {
    "minipigs": "PHASE1_EEGNET_MINIPIGS",
    "monkeys": "PHASE1_EEGNET_MONKEYS",
}

TEST_METRICS = (
    "supported_f1",
    "supported_balanced_acc",
    "supported_auroc",
    "supported_precision",
    "supported_recall",
)
VAL_MONITOR = f"val/{TASK}_supported_f1"

COMPUTE_HISTORY_KEYS = (
    "epoch",
    VAL_MONITOR,
    "compute/optimizer_steps",
    "compute/processed_examples",
    "compute/processed_windows",
    "compute/cumulative_flops",
    "compute/elapsed_wall_time_s",
)

RUN_COLUMNS = [
    "run_id",
    "run_name",
    "group",
    "species",
    "subject",
    "session",
    "recording_id",
    "fraction",
    "fraction_realized",
    "model_seed",
    "fraction_seed",
    "num_present_classes",
    "present_classes",
    "absent_classes",
    "manifest_hash",
    "audit_artifact_sha256",
    "runtime_split_hashes",
    "audit_expected_split_hashes",
    "runtime_split_class_counts",
    "audit_expected_split_class_counts",
    "split_type",
    "state",
    "finished",
    "failed",
    "test_f1",
    "test_balanced_acc",
    "test_auroc",
    "test_precision",
    "test_recall",
    "best_val_supported_f1",
    "best_step",
    "best_examples",
    "best_windows",
    "best_wall_time_s",
    "best_flops",
    "best_monitor_value",
]

BEST_COMPUTE_KEYS = {
    "best_step": ("compute/best_step", "last"),
    "best_examples": ("compute/best_examples", "last"),
    "best_windows": ("compute/best_windows", "last"),
    "best_wall_time_s": ("compute/best_wall_time_s", "last"),
    "best_flops": ("compute/best_flops", "last"),
    "best_monitor_value": ("compute/best_monitor_value", "last"),
}

SPECIES_COLORS = {
    "minipigs": "#4c78a8",
    "monkeys": "#e45756",
}


# ---------------------------------------------------------------------------
# Audit loading
# ---------------------------------------------------------------------------


def load_audit(audit_path: Path) -> dict[str, Any]:
    """Load and verify the Phase 0 audit JSON artifact hash."""
    with audit_path.open() as handle:
        audit = json.load(handle)

    expected_hash = audit.get("artifact_sha256")
    if not expected_hash:
        raise ValueError(f"Audit file {audit_path} is missing artifact_sha256")

    payload = {k: v for k, v in audit.items() if k != "artifact_sha256"}
    canonical = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Audit artifact hash mismatch: expected {expected_hash[:16]}..., "
            f"got {actual_hash[:16]}..."
        )
    return audit


def build_audit_tables(
    audit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    """Build expected-cell and recording lookup tables from the audit."""
    seeds = tuple(audit["protocol"]["seeds"])
    expected_rows: list[dict[str, Any]] = []
    recording_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for rec in audit["recordings"]:
        rid = rec["recording_id"]
        recording_lookup[(rec["species"], rid)] = rec
        eligible = bool(rec.get("eligible", False))
        for frac_str, frac_info in rec.get("fraction_availability", {}).items():
            fraction = float(frac_str)
            available = bool(frac_info.get("available", False))
            for seed in seeds:
                expected_rows.append(
                    {
                        "species": rec["species"],
                        "subject": rec["subject"],
                        "recording_id": rid,
                        "fraction": fraction,
                        "fraction_seed": seed,
                        "eligible_recording": eligible,
                        "fraction_available": available,
                        "audit_present_classes": frac_info.get(
                            "present_classes"
                        ),
                        "audit_present_class_count": frac_info.get(
                            "present_class_count"
                        ),
                        "audit_failure_reason": frac_info.get("failure_reason"),
                        "recording_present_class_count": len(
                            rec.get("present_classes", [])
                        ),
                    }
                )

    expected = pd.DataFrame(expected_rows)
    supported = expected[
        expected["eligible_recording"] & expected["fraction_available"]
    ].copy()
    return expected, supported, recording_lookup


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------


def _nested_get(config: dict[str, Any], *keys: str) -> Any:
    """Walk nested dict keys, returning ``None`` when a segment is missing."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def get_neurosoft_field(config: dict[str, Any], field: str) -> Any:
    """Read a ``neurosoft/`` config field from nested or flat WandB config."""
    ns = config.get("neurosoft")
    if isinstance(ns, dict) and field in ns:
        return ns[field]
    flat = config.get(f"neurosoft/{field}")
    if flat is not None:
        return flat
    return None


def parse_recording_id(recording_id: str) -> tuple[str | None, str | None]:
    """Return ``(subject, session)`` parsed from a BIDS-like recording ID."""
    subject_match = re.search(r"(sub-\d+)", recording_id)
    session_match = re.search(r"(ses-\d+)", recording_id)
    subject = subject_match.group(1) if subject_match else None
    session = session_match.group(1) if session_match else None
    return subject, session


def infer_species(config: dict[str, Any], group_species: str) -> str:
    """Infer species from neurosoft metadata or data config."""
    value = get_neurosoft_field(config, "species")
    if value:
        return str(value)
    data_root = str(_nested_get(config, "data", "root") or "")
    if "monkey" in data_root.lower():
        return "monkeys"
    if "minipig" in data_root.lower() or "mp" in data_root.lower():
        return "minipigs"
    return group_species


def infer_recording_id(config: dict[str, Any], run_name: str) -> str | None:
    """Infer recording ID from neurosoft metadata, data config, or run name."""
    for key in ("recording_id",):
        value = get_neurosoft_field(config, key)
        if value:
            return str(value)
    rec_ids = _nested_get(config, "data", "dataset_kwargs", "recording_ids")
    if isinstance(rec_ids, list) and rec_ids:
        return str(rec_ids[0])
    match = re.search(
        r"(sub-\d+_ses-\d+_task-AcousStim_acq-[A-Za-z]+(?:anest)?_desc-raw)",
        run_name or "",
    )
    return match.group(1) if match else None


def infer_fraction(config: dict[str, Any], run_name: str) -> float | None:
    """Infer requested training fraction from config or run name."""
    for key in ("training_fraction_requested", "training_fraction"):
        value = get_neurosoft_field(config, key)
        if value is not None:
            return float(value)
    data_frac = _nested_get(config, "data", "training_fraction")
    if data_frac is not None:
        return float(data_frac)
    match = re.search(r"_f(0?\.\d+|1\.0+|1)_", run_name or "")
    if match:
        return float(match.group(1))
    return None


def infer_seed(config: dict[str, Any], run_name: str, key: str) -> int | None:
    """Infer a seed field from neurosoft metadata, run config, or run name."""
    value = get_neurosoft_field(config, key)
    if value is not None:
        return int(value)
    if key in ("model_seed", "seed"):
        run_seed = _nested_get(config, "run", "seed")
        if run_seed is not None:
            return int(run_seed)
    if key == "fraction_seed":
        frac_seed = _nested_get(config, "data", "training_fraction_seed")
        if frac_seed is not None:
            return int(frac_seed)
    if key in ("model_seed", "seed"):
        match = re.search(r"_s(\d+)$", run_name or "")
        if match:
            return int(match.group(1))
    return None


def extract_summary_metric(
    summary: dict[str, Any], split: str, metric: str
) -> float | None:
    """Extract one supported test/val metric from a WandB run summary."""
    prefix = f"{split}/{TASK}_{metric}"
    for key in (f"{prefix}.max", prefix):
        val = summary.get(key)
        if val is None:
            continue
        unwrapped = unwrap_summary_value(val, "max")
        try:
            return float(unwrapped)
        except (TypeError, ValueError):
            continue
    return None


def extract_best_compute(summary: dict[str, Any]) -> dict[str, float | None]:
    """Extract best-checkpoint compute counters from run summary."""
    result: dict[str, float | None] = {}
    for out_name, (wandb_key, unwrap_key) in BEST_COMPUTE_KEYS.items():
        result[out_name] = unwrap_summary_value(
            summary.get(wandb_key), unwrap_key
        )
        if result[out_name] is not None:
            try:
                result[out_name] = float(result[out_name])
            except (TypeError, ValueError):
                result[out_name] = None
    return result


def run_cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical duplicate-detection key for one run."""
    return (
        row.get("species"),
        row.get("recording_id"),
        row.get("fraction"),
        row.get("model_seed"),
        row.get("fraction_seed"),
    )


def fetch_group_runs(
    api: wandb.Api,
    entity: str | None,
    project: str,
    group: str,
    species: str,
) -> list[dict[str, Any]]:
    """Fetch and normalize all runs in one WandB group."""
    path = f"{entity}/{project}" if entity else project
    rows: list[dict[str, Any]] = []

    try:
        run_iter = iter(api.runs(path, filters={"group": group}))
        while True:
            try:
                run = next(run_iter)
            except StopIteration:
                break
            except ValueError as exc:
                print(f"  WARNING: could not query {path}: {exc}")
                break

            config = dict(run.config or {})
            summary = dict(run.summary or {})
            recording_id = infer_recording_id(config, run.name or "")
            subject_cfg = get_neurosoft_field(config, "subject")
            session_cfg = get_neurosoft_field(config, "session")
            subject_parsed, session_parsed = (
                parse_recording_id(recording_id)
                if recording_id
                else (None, None)
            )

            present_classes = get_neurosoft_field(config, "present_classes")
            absent_classes = get_neurosoft_field(config, "absent_classes")
            num_present = get_neurosoft_field(config, "num_present_classes")
            if num_present is not None:
                num_present = int(num_present)
            elif isinstance(present_classes, (list, tuple)):
                num_present = len(present_classes)

            row: dict[str, Any] = {
                "run_id": run.id,
                "run_name": run.name,
                "group": group,
                "species": infer_species(config, species),
                "subject": subject_cfg or subject_parsed,
                "session": session_cfg or session_parsed,
                "recording_id": recording_id,
                "fraction": infer_fraction(config, run.name or ""),
                "fraction_realized": get_neurosoft_field(
                    config, "training_fraction_realized"
                ),
                "model_seed": infer_seed(config, run.name or "", "model_seed"),
                "fraction_seed": infer_seed(
                    config, run.name or "", "fraction_seed"
                ),
                "num_present_classes": num_present,
                "present_classes": present_classes,
                "absent_classes": absent_classes,
                "manifest_hash": get_neurosoft_field(config, "manifest_hash"),
                "audit_artifact_sha256": get_neurosoft_field(
                    config, "audit_artifact_sha256"
                ),
                "runtime_split_hashes": get_neurosoft_field(
                    config, "runtime_split_hashes"
                ),
                "audit_expected_split_hashes": get_neurosoft_field(
                    config, "audit_expected_split_hashes"
                ),
                "runtime_split_class_counts": get_neurosoft_field(
                    config, "runtime_split_class_counts"
                ),
                "audit_expected_split_class_counts": get_neurosoft_field(
                    config, "audit_expected_split_class_counts"
                ),
                "split_type": get_neurosoft_field(config, "split_type")
                or _nested_get(config, "data", "split_type"),
                "state": run.state,
                "finished": run.state == "finished",
                "failed": run.state in ("failed", "crashed"),
            }

            for metric in TEST_METRICS:
                short = metric.removeprefix("supported_")
                row[f"test_{short}"] = extract_summary_metric(
                    summary, "test", metric
                )

            row["best_val_supported_f1"] = extract_summary_metric(
                summary, "val", "supported_f1"
            )
            row.update(extract_best_compute(summary))
            rows.append(row)
    except ValueError as exc:
        print(f"  WARNING: could not query {path}: {exc}")

    return rows


def fetch_optimization_history(
    api: wandb.Api,
    entity: str | None,
    project: str,
    run_id: str,
) -> pd.DataFrame:
    """Fetch validation monitor and compute counters for optimization-to-80%."""
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(path)
    history = run.history(
        keys=list(COMPUTE_HISTORY_KEYS), samples=10_000, pandas=True
    )
    if history.empty:
        return history
    present = [k for k in COMPUTE_HISTORY_KEYS if k in history.columns]
    history = history[present].dropna(subset=[VAL_MONITOR], how="any")
    if "epoch" in history.columns:
        history = history.sort_values("epoch").reset_index(drop=True)
    return history


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------


def compute_data_to_80(runs: pd.DataFrame) -> pd.DataFrame:
    """Find the smallest fraction reaching 80% of full-data mean test F1."""
    if runs.empty:
        return pd.DataFrame()
    finished = runs[runs["finished"] & runs["test_f1"].notna()].copy()
    rows: list[dict[str, Any]] = []

    for (species, recording_id), session_df in finished.groupby(
        ["species", "recording_id"], dropna=False
    ):
        full = session_df[np.isclose(session_df["fraction"], 1.0)]
        if full.empty:
            continue
        target = 0.8 * float(full["test_f1"].mean())
        subject = session_df["subject"].iloc[0]
        num_classes = session_df["num_present_classes"].iloc[0]

        reached_fraction: float | None = None
        reached_mean_f1: float | None = None
        for fraction in FRACTIONS:
            frac_df = session_df[np.isclose(session_df["fraction"], fraction)]
            if frac_df.empty:
                continue
            mean_f1 = float(frac_df["test_f1"].mean())
            if mean_f1 >= target:
                reached_fraction = fraction
                reached_mean_f1 = mean_f1
                break

        rows.append(
            {
                "species": species,
                "subject": subject,
                "recording_id": recording_id,
                "num_present_classes": num_classes,
                "full_data_mean_test_f1": float(full["test_f1"].mean()),
                "target_f1_80pct": target,
                "data_to_80_fraction": reached_fraction,
                "data_to_80_mean_test_f1": reached_mean_f1,
                "right_censored": reached_fraction is None,
            }
        )

    return pd.DataFrame(rows)


def compute_optimization_to_80(
    runs: pd.DataFrame,
    api: wandb.Api,
    entity: str | None,
    project: str,
    *,
    offline: bool,
) -> pd.DataFrame:
    """First validation event reaching 80% of eventual best supported val F1."""
    if offline or runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    candidates = runs[runs["finished"]].copy()

    for _, row in candidates.iterrows():
        best_val = row.get("best_val_supported_f1")
        if best_val is None or pd.isna(best_val) or best_val <= 0:
            continue
        threshold = 0.8 * float(best_val)

        try:
            history = fetch_optimization_history(
                api, entity, project, row["run_id"]
            )
        except Exception:
            continue
        if history.empty or VAL_MONITOR not in history.columns:
            continue

        hit = history[history[VAL_MONITOR] >= threshold]
        if hit.empty:
            first = None
        else:
            first = hit.iloc[0]

        rows.append(
            {
                "run_id": row["run_id"],
                "species": row["species"],
                "subject": row["subject"],
                "recording_id": row["recording_id"],
                "fraction": row["fraction"],
                "model_seed": row["model_seed"],
                "best_val_supported_f1": float(best_val),
                "opt_to_80_threshold": threshold,
                "opt_to_80_reached": first is not None,
                "opt_to_80_epoch": None
                if first is None
                else first.get("epoch"),
                "opt_to_80_val_supported_f1": None
                if first is None
                else float(first[VAL_MONITOR]),
                "opt_to_80_optimizer_steps": None
                if first is None
                else first.get("compute/optimizer_steps"),
                "opt_to_80_processed_examples": None
                if first is None
                else first.get("compute/processed_examples"),
                "opt_to_80_processed_windows": None
                if first is None
                else first.get("compute/processed_windows"),
                "opt_to_80_cumulative_flops": None
                if first is None
                else first.get("compute/cumulative_flops"),
                "opt_to_80_elapsed_wall_time_s": None
                if first is None
                else first.get("compute/elapsed_wall_time_s"),
            }
        )

    return pd.DataFrame(rows)


def compute_time_compute_to_best(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize verified best-checkpoint compute snapshots."""
    if runs.empty:
        return pd.DataFrame()
    cols = [
        "run_id",
        "species",
        "subject",
        "recording_id",
        "fraction",
        "model_seed",
        "state",
        "best_val_supported_f1",
        "best_step",
        "best_examples",
        "best_windows",
        "best_wall_time_s",
        "best_flops",
        "best_monitor_value",
    ]
    present = [c for c in cols if c in runs.columns]
    return runs[present].copy()


def aggregate_by_class_count(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute separate summaries for 6-, 7-, and 8-class sessions."""
    if runs.empty:
        return pd.DataFrame()
    finished = runs[runs["finished"] & runs["test_f1"].notna()].copy()
    if finished.empty:
        return finished

    rows: list[dict[str, Any]] = []
    for class_count in (6, 7, 8):
        sub = finished[finished["num_present_classes"] == class_count]
        if sub.empty:
            continue
        for species, species_df in sub.groupby("species"):
            rows.append(
                {
                    "num_present_classes": class_count,
                    "species": species,
                    "n_runs": len(species_df),
                    "n_sessions": species_df["recording_id"].nunique(),
                    "test_f1_mean": species_df["test_f1"].mean(),
                    "test_f1_std": species_df["test_f1"].std(ddof=0),
                    "test_balanced_acc_mean": species_df[
                        "test_balanced_acc"
                    ].mean(),
                    "test_auroc_mean": species_df["test_auroc"].mean(),
                }
            )
    return pd.DataFrame(rows)


def subject_balanced_summary(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average seeds → sessions → subjects → species at each fraction."""
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame()
    finished = runs[runs["finished"] & runs["test_f1"].notna()].copy()
    if finished.empty:
        return pd.DataFrame(), pd.DataFrame()

    session_means = (
        finished.groupby(
            ["species", "subject", "recording_id", "fraction"], dropna=False
        )["test_f1"]
        .mean()
        .reset_index(name="session_mean_test_f1")
    )
    subject_means = (
        session_means.groupby(["species", "subject", "fraction"], dropna=False)[
            "session_mean_test_f1"
        ]
        .mean()
        .reset_index(name="subject_mean_test_f1")
    )
    balanced = (
        subject_means.groupby(["species", "fraction"], dropna=False)[
            "subject_mean_test_f1"
        ]
        .agg(
            n_subjects="count",
            test_f1_mean="mean",
            test_f1_std=lambda s: s.std(ddof=0),
        )
        .reset_index()
    )

    unweighted = (
        session_means.groupby(["species", "fraction"], dropna=False)[
            "session_mean_test_f1"
        ]
        .agg(
            n_sessions="count",
            test_f1_mean="mean",
            test_f1_std=lambda s: s.std(ddof=0),
        )
        .reset_index()
    )
    return balanced, unweighted


def build_integrity_table(
    runs: pd.DataFrame,
    expected: pd.DataFrame,
    supported: pd.DataFrame,
    recording_lookup: dict[tuple[str, str], dict[str, Any]],
    audit: dict[str, Any],
) -> pd.DataFrame:
    """Check duplicates, audit hashes, manifest metadata, and coverage."""
    issues: list[dict[str, Any]] = []
    audit_hash = audit["artifact_sha256"]

    if not runs.empty:
        counts = runs.apply(run_cell_key, axis=1).value_counts()
        for key, count in counts.items():
            if count > 1:
                issues.append(
                    {
                        "issue_type": "duplicate_key",
                        "detail": f"{count} runs share cell key {key}",
                        "count": int(count),
                    }
                )

        for _, row in runs.iterrows():
            if not row.get("audit_artifact_sha256"):
                issues.append(
                    {
                        "issue_type": "missing_audit_hash",
                        "detail": f"run {row['run_id']} did not log an audit hash",
                        "count": 1,
                    }
                )
            elif row["audit_artifact_sha256"] != audit_hash:
                issues.append(
                    {
                        "issue_type": "wrong_audit_hash",
                        "detail": (
                            f"run {row['run_id']} logged audit hash "
                            f"{row['audit_artifact_sha256'][:16]}..."
                        ),
                        "count": 1,
                    }
                )

            rec = recording_lookup.get(
                (row.get("species"), row.get("recording_id") or "")
            )
            if rec:
                expected_hashes = rec.get("split_hashes", {})
                logged_expected = row.get("audit_expected_split_hashes")
                runtime_hashes = row.get("runtime_split_hashes")
                if logged_expected != expected_hashes:
                    issues.append(
                        {
                            "issue_type": "audit_expected_split_hash_mismatch",
                            "detail": (
                                f"run {row['run_id']} did not log the audit "
                                "split hashes for its species/recording"
                            ),
                            "count": 1,
                        }
                    )
                if runtime_hashes != expected_hashes:
                    issues.append(
                        {
                            "issue_type": "runtime_split_hash_mismatch",
                            "detail": (
                                f"run {row['run_id']} runtime split hashes do "
                                "not match the Phase 0 audit"
                            ),
                            "count": 1,
                        }
                    )
            if rec and row.get("num_present_classes") is not None:
                expected_count = len(rec.get("present_classes", []))
                if int(row["num_present_classes"]) != expected_count:
                    issues.append(
                        {
                            "issue_type": "manifest_class_mismatch",
                            "detail": (
                                f"run {row['run_id']} num_present_classes="
                                f"{row['num_present_classes']} vs audit "
                                f"{expected_count}"
                            ),
                            "count": 1,
                        }
                    )

            if (
                rec
                and row.get("split_type")
                and row["split_type"] != audit["protocol"].get("split")
            ):
                issues.append(
                    {
                        "issue_type": "split_mismatch",
                        "detail": (
                            f"run {row['run_id']} split_type={row['split_type']}"
                        ),
                        "count": 1,
                    }
                )

            if row.get("failed"):
                issues.append(
                    {
                        "issue_type": "failed_run",
                        "detail": f"run {row['run_id']} state={row['state']}",
                        "count": 1,
                    }
                )

            if (
                row.get("recording_id")
                and row.get("fraction") is not None
                and rec
            ):
                frac_key = f"{float(row['fraction']):.2f}"
                frac_info = rec.get("fraction_availability", {}).get(frac_key)
                if frac_info and not frac_info.get("available", False):
                    issues.append(
                        {
                            "issue_type": "unavailable_cell_run",
                            "detail": (
                                f"run {row['run_id']} targets unavailable "
                                f"cell {row['recording_id']}@{frac_key}"
                            ),
                            "count": 1,
                        }
                    )

    observed_keys = set()
    if not runs.empty:
        for _, row in runs.iterrows():
            if row.get("fraction") is None:
                continue
            observed_keys.add(
                (
                    row.get("species"),
                    row.get("recording_id"),
                    float(row["fraction"]),
                    row.get("fraction_seed") or row.get("model_seed"),
                )
            )

    for _, cell in supported.iterrows():
        key = (
            cell["species"],
            cell["recording_id"],
            float(cell["fraction"]),
            int(cell["fraction_seed"]),
        )
        if key not in observed_keys:
            issues.append(
                {
                    "issue_type": "missing_expected_run",
                    "detail": (
                        f"missing {cell['species']} "
                        f"{cell['recording_id']} f={cell['fraction']} "
                        f"seed={cell['fraction_seed']}"
                    ),
                    "count": 1,
                }
            )

    unavailable_expected = len(expected) - len(supported)
    issues.append(
        {
            "issue_type": "unavailable_cells_in_audit",
            "detail": "expected unavailable or ineligible cells excluded from launch",
            "count": unavailable_expected,
        }
    )

    if not issues:
        issues.append(
            {
                "issue_type": "none",
                "detail": "no integrity issues detected",
                "count": 0,
            }
        )

    return pd.DataFrame(issues)


def empty_runs_frame() -> pd.DataFrame:
    """Return an empty runs frame with the expected schema."""
    return pd.DataFrame(columns=RUN_COLUMNS)


def learning_curve_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std test F1 by species and fraction (seed-level observations)."""
    if runs.empty:
        return pd.DataFrame(
            columns=["species", "fraction", "n_runs", "mean", "std"]
        )
    finished = runs[runs["finished"] & runs["test_f1"].notna()].copy()
    if finished.empty:
        return finished

    summary = (
        finished.groupby(["species", "fraction"], dropna=False)["test_f1"]
        .agg(n_runs="count", mean="mean", std=lambda s: s.std(ddof=0))
        .reset_index()
        .sort_values(["species", "fraction"])
    )
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_learning_curves(
    runs: pd.DataFrame,
    expected: pd.DataFrame,
    curve_summary: pd.DataFrame,
) -> Path:
    """Plot supported test F1 vs fraction for each species."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, species in zip(axes, ("minipigs", "monkeys")):
        sub = curve_summary[curve_summary["species"] == species]
        if not sub.empty:
            ax.errorbar(
                sub["fraction"],
                sub["mean"],
                yerr=sub["std"].fillna(0),
                marker="o",
                capsize=4,
                color=SPECIES_COLORS[species],
                linewidth=2,
                label="finished runs (mean ± SD)",
            )

        failed = runs[
            (runs["species"] == species)
            & runs["failed"]
            & runs["fraction"].notna()
        ]
        for fraction in sorted(failed["fraction"].unique()):
            ax.scatter(
                [fraction],
                [0.02],
                marker="x",
                color="crimson",
                s=60,
                zorder=4,
            )

        unavailable = expected[
            (expected["species"] == species) & ~expected["fraction_available"]
        ]
        unavailable_fracs = sorted(unavailable["fraction"].unique())
        for fraction in unavailable_fracs:
            ax.axvline(
                fraction,
                color="grey",
                linestyle=":",
                alpha=0.35,
                linewidth=1,
            )

        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS)
        ax.set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
        ax.set_xlabel("Training fraction")
        ax.set_title(species.capitalize())
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test supported macro-F1")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=SPECIES_COLORS["minipigs"],
            marker="o",
            label="mean ± SD",
        ),
        plt.Line2D(
            [0],
            [0],
            color="crimson",
            marker="x",
            linestyle="",
            label="failed run",
        ),
        plt.Line2D(
            [0], [0], color="grey", linestyle=":", label="unavailable cell"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Phase 1 EEGNet learning curves — test supported macro-F1",
        fontsize=13,
        y=1.05,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{OUTPUT_PREFIX}_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_primary_summaries(
    balanced: pd.DataFrame,
    unweighted: pd.DataFrame,
    class_summary: pd.DataFrame,
    integrity: pd.DataFrame,
) -> None:
    """Print primary summaries and integrity counts to stdout."""
    print("\n" + "=" * 72)
    print(
        "SUBJECT-BALANCED SPECIES SUMMARY (seeds → sessions → subjects → species)"
    )
    print("=" * 72)
    if balanced.empty:
        print("  (no finished runs)")
    else:
        print(
            balanced.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )

    print("\n" + "=" * 72)
    print("UNWEIGHTED SESSION DISTRIBUTION BY SPECIES")
    print("=" * 72)
    if unweighted.empty:
        print("  (no finished runs)")
    else:
        print(
            unweighted.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )

    print("\n" + "=" * 72)
    print("6 / 7 / 8-CLASS STRATIFIED SUMMARIES")
    print("=" * 72)
    if class_summary.empty:
        print("  (no finished runs)")
    else:
        print(
            class_summary.to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )

    print("\n" + "=" * 72)
    print("INTEGRITY COUNTS")
    print("=" * 72)
    counts = integrity.groupby("issue_type")["count"].sum().sort_index()
    for issue_type, count in counts.items():
        print(f"  {issue_type}: {int(count)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze Phase 1 NeuroSoft EEGNet learning curves from WandB."
    )
    parser.add_argument("--entity", default=default_entity())
    parser.add_argument("--project", default="neurosoft_supervised_pretraining")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=repo_root / "docs/neurosoft-phase0-audit.json",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip WandB API calls (writes empty run tables if no cache).",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch WandB runs, compute tables, and write CSV/figure outputs."""
    args = parse_args()
    audit = load_audit(args.audit_json)
    expected, supported, recording_lookup = build_audit_tables(audit)

    n_supported_cells = (
        supported[["species", "recording_id", "fraction"]]
        .drop_duplicates()
        .shape[0]
    )
    print(
        f"Loaded audit: {len(audit['recordings'])} recordings, "
        f"{n_supported_cells} supported cells, "
        f"{len(supported)} expected runs"
    )

    rows: list[dict[str, Any]] = []
    if args.offline:
        print("\nOffline mode: skipping WandB API fetch.")
    else:
        api = wandb.Api()
        for species, group in GROUPS.items():
            print(f"\nFetching {group}...")
            group_rows = fetch_group_runs(
                api, args.entity, args.project, group, species
            )
            print(f"  {len(group_rows)} runs returned")
            rows.extend(group_rows)

    runs = pd.DataFrame(rows) if rows else empty_runs_frame()
    if not runs.empty:
        runs = runs.sort_values(
            ["species", "recording_id", "fraction", "model_seed"]
        ).reset_index(drop=True)

    runs_path = CSV_DIR / f"{OUTPUT_PREFIX}_per_run_metrics.csv"
    runs.to_csv(runs_path, index=False)
    print(f"\nWrote {runs_path}")

    curve_summary = learning_curve_summary(runs)
    curve_summary.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_learning_curves_summary.csv", index=False
    )

    data_to_80 = compute_data_to_80(runs)
    data_to_80.to_csv(CSV_DIR / f"{OUTPUT_PREFIX}_data_to_80.csv", index=False)

    if args.offline:
        opt_to_80 = pd.DataFrame()
    else:
        print("\nFetching validation histories for optimization-to-80%...")
        opt_to_80 = compute_optimization_to_80(
            runs,
            api,
            args.entity,
            args.project,
            offline=args.offline,
        )
    opt_to_80.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_optimization_to_80.csv", index=False
    )

    time_best = compute_time_compute_to_best(runs)
    time_best.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_time_compute_to_best.csv", index=False
    )

    class_summary = aggregate_by_class_count(runs)
    class_summary.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_class_count_summary.csv", index=False
    )

    balanced, unweighted = subject_balanced_summary(runs)
    balanced.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_subject_balanced_summary.csv", index=False
    )
    unweighted.to_csv(
        CSV_DIR / f"{OUTPUT_PREFIX}_session_unweighted_summary.csv", index=False
    )

    integrity = build_integrity_table(
        runs, expected, supported, recording_lookup, audit
    )
    integrity.to_csv(CSV_DIR / f"{OUTPUT_PREFIX}_integrity.csv", index=False)

    print_primary_summaries(balanced, unweighted, class_summary, integrity)

    if not curve_summary.empty or not runs.empty:
        fig_path = plot_learning_curves(runs, expected, curve_summary)
        print(f"\nSaved figure: {fig_path}")
    else:
        print("\nNo data available for learning-curve figure.")


if __name__ == "__main__":
    main()

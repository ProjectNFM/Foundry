"""Shared audited analysis for the batch-128 architecture-transfer program.

The immutable compiled JSONL matrices define the scientific run inventory.
This module verifies those matrices and their locks, fetches only their exact
W&B run IDs, caches endpoint/history data, constructs treatment-matched pairs,
aggregates seed -> recording -> excluded subject -> species, bootstraps whole
subjects, and renders the common figure suite.

It deliberately has no imports from :mod:`foundry`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import textwrap
import time
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "analysis" / "csv"
FIGURE_DIR = ROOT / "analysis" / "figures"
MATRIX_DIR = ROOT / "launch" / "architecture_transfer"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
TEST_F1 = f"test/{TASK}_supported_f1"
VAL_F1 = f"val/{TASK}_supported_f1"
TEST_CONFUSION = f"test/{TASK}_confusion_counts"
TEST_CLASSES = f"test/{TASK}_confusion_class_names"
SOURCE_STEPS = (100, 300, 1_000, 3_000, 10_000)
PRIMARY_THRESHOLD = 0.90
THRESHOLDS = (0.80, PRIMARY_THRESHOLD, 0.95)
SPECIES = ("minipigs", "monkeys")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_916

CONDITION_LABELS = {
    "transfer_reference": "Reference",
    "transfer_small": "Small",
    "transfer_large": "Large",
    "transfer_bias_free_to_biased": "Bias-free source / ordinary target",
    "transfer_bias_free_matched": "Bias-free source / bias-free target",
    "transfer_shared_to_ordinary": "Shared source / ordinary target",
    "transfer_shared_retained": "Retained shared interface",
}
CONDITION_COLORS = {
    "transfer_reference": "#3b6fb6",
    "transfer_small": "#2a9d8f",
    "transfer_large": "#d1495b",
    "transfer_bias_free_to_biased": "#8f63b8",
    "transfer_bias_free_matched": "#d17c1f",
    "transfer_shared_to_ordinary": "#5b8e7d",
    "transfer_shared_retained": "#c8553d",
}
SUBJECT_COLORS = {
    species: dict(
        zip(
            [f"sub-{index:02d}" for index in range(1, count + 1)],
            sns.color_palette("tab10", count),
        )
    )
    for species, count in (("minipigs", 7), ("monkeys", 5))
}
EXPERIMENTS = {
    "reference": {
        "stem": "20260916-MS-batch128-reference-transfer-replication",
        "matrices": ("reference-backbone",),
        "conditions": ("transfer_reference",),
    },
    "scale": {
        "stem": "20260916-MS-model-scale-transfer",
        "matrices": ("reference-backbone", "small-backbone", "large-backbone"),
        "conditions": (
            "transfer_small",
            "transfer_reference",
            "transfer_large",
        ),
    },
    "bias": {
        "stem": "20260916-MS-bias-free-transfer",
        "matrices": ("reference-backbone", "bias-free"),
        "conditions": (
            "transfer_reference",
            "transfer_bias_free_to_biased",
            "transfer_bias_free_matched",
        ),
    },
    "shared": {
        "stem": "20260916-MS-shared-adapter-transfer",
        "matrices": ("reference-backbone", "shared-adapter"),
        "conditions": (
            "transfer_reference",
            "transfer_shared_to_ordinary",
            "transfer_shared_retained",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source_step(cell: dict[str, Any]) -> float:
    value = cell.get("source_condition") or {}
    if not isinstance(value, dict):
        return np.nan
    return float(value.get("milestone_step", np.nan))


def _target_treatment(cell: dict[str, Any]) -> str:
    labels = cell.get("condition_labels") or {}
    return str(
        labels.get("target_adapter_treatment")
        or labels.get("target_adapter")
        or labels.get("condition")
        or cell.get("condition_id", "")
    )


def load_compiled_design(experiment: str) -> pd.DataFrame:
    """Load and hash-audit the exact immutable matrices for an experiment."""
    spec = EXPERIMENTS[experiment]
    rows: list[dict[str, Any]] = []
    for matrix in spec["matrices"]:
        for species in SPECIES:
            path = MATRIX_DIR / f"{matrix}-{species}.jsonl"
            lock_path = MATRIX_DIR / f"{matrix}-{species}.lock.json"
            lock = json.loads(lock_path.read_text())
            actual_hash = _sha256(path)
            if actual_hash != lock["output_sha256"]:
                raise RuntimeError(
                    f"Hash mismatch for {path}: {actual_hash} != {lock['output_sha256']}"
                )
            cells = _read_jsonl(path)
            if len(cells) != int(lock["output_count"]):
                raise RuntimeError(f"Count mismatch for {path}")
            for cell in cells:
                rows.append(
                    {
                        "matrix": matrix,
                        "matrix_path": str(path.relative_to(ROOT)),
                        "matrix_sha256": actual_hash,
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
                        "source_model_seed": cell.get(
                            "source_model_seed", np.nan
                        ),
                        "source_selection_seed": cell.get(
                            "source_selection_seed", np.nan
                        ),
                        "checkpoint_set_id": cell.get("checkpoint_set_id", ""),
                        "checkpoint_id": cell.get("checkpoint_id", ""),
                        "checkpoint_manifest_hash": cell.get(
                            "checkpoint_manifest_hash", ""
                        ),
                        "checkpoint_sha256": cell.get("checkpoint_sha256", ""),
                        "matched_scratch_id": cell.get(
                            "matched_scratch_id", ""
                        ),
                        "scratch_family": cell.get("scratch_family", ""),
                        "transfer_regime": cell.get(
                            "transfer_regime", "scratch"
                        ),
                        "target_treatment": _target_treatment(cell),
                    }
                )
    frame = pd.DataFrame(rows)
    for column in ("cell_id", "run_id"):
        duplicates = frame[frame[column].duplicated(keep=False)]
        if not duplicates.empty:
            raise RuntimeError(f"Duplicate {column}s in compiled design")
    scratch_ids = set(
        frame.loc[frame.condition.str.startswith("scratch"), "cell_id"]
    )
    transfer = frame[~frame.condition.str.startswith("scratch")]
    missing = sorted(set(transfer.matched_scratch_id) - scratch_ids)
    if missing:
        raise RuntimeError(
            f"Missing compiled scratch comparators: {missing[:3]}"
        )
    if set(transfer.source_step.dropna().astype(int)) != set(SOURCE_STEPS):
        raise RuntimeError("Compiled source checkpoint inventory is incomplete")
    return frame.sort_values(["matrix", "species", "cell_id"]).reset_index(
        drop=True
    )


def _finite_scalar(summary: Any, key: str) -> float:
    value = summary.get(key)
    if isinstance(value, dict):
        for nested in ("max", "min", "last"):
            if nested in value:
                value = value[nested]
                break
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _test_f1(run: Any) -> float:
    for key in (f"{TEST_F1}.max", TEST_F1):
        value = _finite_scalar(run.summary or {}, key)
        if np.isfinite(value):
            return value
    # Public GraphQL summaries occasionally omit an otherwise synced metric.
    with tempfile.TemporaryDirectory(
        prefix="architecture-transfer-summary-"
    ) as tmp:
        payload = json.loads(
            Path(
                run.file("wandb-summary.json")
                .download(root=tmp, replace=True)
                .name
            ).read_text()
        )
    for key in (TEST_F1, f"{TEST_F1}.max"):
        value = _finite_scalar(payload, key)
        if np.isfinite(value):
            return value
    return np.nan


def _class_f1(confusion: Any) -> list[float]:
    matrix = np.asarray(confusion, dtype=float)
    if matrix.shape != (8, 8):
        return [np.nan] * 8
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    denom = support + predicted
    values = np.divide(
        2 * np.diag(matrix),
        denom,
        out=np.full(8, np.nan),
        where=(denom > 0) & (support > 0),
    )
    return values.tolist()


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _fetch_one(
    entity: str, row: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch and validate one exact run, with bounded transient retries."""
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            api = wandb.Api(timeout=120)
            run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
            config = dict(run.config or {})
            summary = run.summary or {}
            history_rows = list(
                run.scan_history(
                    keys=[
                        "trainer/global_step",
                        "epoch",
                        VAL_F1,
                        "compute/optimizer_steps",
                        "compute/processed_windows",
                        "compute/cumulative_flops",
                    ],
                    page_size=10_000,
                )
            )
            history = pd.DataFrame(history_rows)
            required = {"trainer/global_step", "epoch", VAL_F1}
            if not required.issubset(history.columns):
                raise ValueError(
                    f"missing history columns {required - set(history.columns)}"
                )
            history = history.dropna(
                subset=["trainer/global_step", VAL_F1]
            ).copy()
            history["optimizer_step"] = pd.to_numeric(
                history.get("compute/optimizer_steps"), errors="coerce"
            ).fillna(pd.to_numeric(history["trainer/global_step"]) + 1)
            history["val_f1"] = pd.to_numeric(history[VAL_F1], errors="coerce")
            history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
            history["processed_windows"] = pd.to_numeric(
                history.get("compute/processed_windows"), errors="coerce"
            )
            history["cumulative_flops"] = pd.to_numeric(
                history.get("compute/cumulative_flops"), errors="coerce"
            )
            history = (
                history[
                    [
                        "optimizer_step",
                        "epoch",
                        "val_f1",
                        "processed_windows",
                        "cumulative_flops",
                    ]
                ]
                .dropna(subset=["optimizer_step", "val_f1"])
                .sort_values("optimizer_step")
                .drop_duplicates("optimizer_step", keep="last")
            )
            if len(history) < 3:
                raise ValueError("fewer than three validation evaluations")
            steps = history.optimizer_step.to_numpy(dtype=float)
            epochs = history.epoch.to_numpy(dtype=float)
            valid = np.isfinite(epochs)
            increments = np.diff(steps[valid]) / np.diff(epochs[valid])
            increments = increments[np.isfinite(increments) & (increments > 0)]
            if not len(increments):
                raise ValueError("cannot reconstruct optimizer steps per epoch")
            steps_per_epoch = float(np.median(increments))
            max_epochs = int(
                _nested(config, "trainer", "max_epochs", default=200)
            )
            planned_budget = int(round(steps_per_epoch * max_epochs))
            confusion = summary.get(TEST_CONFUSION, [])
            classes = summary.get(TEST_CLASSES, [])
            model_sessions = _nested(
                config, "model", "session_configs", default={}
            )
            channel_count = (
                model_sessions.get(row["recording"], np.nan)
                if isinstance(model_sessions, dict)
                else np.nan
            )
            run_config = _nested(config, "run", default={}) or {}
            record = dict(row)
            record.update(
                {
                    "run_name": str(run.name or ""),
                    "group": str(run.group or ""),
                    "state": str(run.state or ""),
                    "test_f1": _test_f1(run),
                    "test_confusion": json.dumps(confusion),
                    "test_class_names": json.dumps(classes),
                    "class_f1": json.dumps(_class_f1(confusion)),
                    "history_points": len(history),
                    "max_epochs": max_epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "planned_budget": planned_budget,
                    "effective_batch_size": _finite_scalar(
                        summary, "compute/effective_batch_size"
                    ),
                    "flops_per_window": _finite_scalar(
                        summary, "compute/flops_per_window"
                    ),
                    "total_parameters": _finite_scalar(
                        summary, "compute/total_parameters"
                    ),
                    "channel_count": channel_count,
                    "config_cell_id": run_config.get(
                        "cell_id", config.get("cell_id", "")
                    ),
                    "config_checkpoint_set_id": run_config.get(
                        "checkpoint_set_id", config.get("checkpoint_set_id", "")
                    ),
                    "config_checkpoint_id": run_config.get(
                        "checkpoint_id", config.get("checkpoint_id", "")
                    ),
                    "config_manifest_hash": run_config.get(
                        "pretrained_checkpoint_manifest_hash", ""
                    ),
                    "config_checkpoint_sha256": run_config.get(
                        "pretrained_checkpoint_sha256", ""
                    ),
                    "analysis_error": "",
                }
            )
            history.insert(0, "run_id", row["run_id"])
            return record, history
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    record = dict(row)
    record.update(
        {
            "run_name": "",
            "group": "",
            "state": "fetch_error",
            "test_f1": np.nan,
            "test_confusion": "[]",
            "test_class_names": "[]",
            "class_f1": "[]",
            "history_points": 0,
            "max_epochs": np.nan,
            "steps_per_epoch": np.nan,
            "planned_budget": np.nan,
            "effective_batch_size": np.nan,
            "flops_per_window": np.nan,
            "total_parameters": np.nan,
            "channel_count": np.nan,
            "config_cell_id": "",
            "config_checkpoint_set_id": "",
            "config_checkpoint_id": "",
            "config_manifest_hash": "",
            "config_checkpoint_sha256": "",
            "analysis_error": repr(last_error),
        }
    )
    return record, pd.DataFrame()


def fetch_or_load(
    design: pd.DataFrame,
    stem: str,
    entity: str,
    workers: int,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_path = CSV_DIR / f"{stem}_run_endpoints.csv"
    history_path = CSV_DIR / f"{stem}_validation_histories.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    cached_endpoints = (
        pd.read_csv(endpoint_path) if endpoint_path.exists() else pd.DataFrame()
    )
    cached_histories = (
        pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    )
    # Reuse exact-run raw caches made by another entry point.  Each entry point
    # still writes a complete stem-prefixed cache, while the 6,360 unique runs
    # are downloaded only once across the four overlapping experiment views.
    if not refresh:
        endpoint_parts = (
            [cached_endpoints] if not cached_endpoints.empty else []
        )
        history_parts = [cached_histories] if not cached_histories.empty else []
        for candidate in CSV_DIR.glob("20260916-MS-*_run_endpoints.csv"):
            if candidate != endpoint_path:
                endpoint_parts.append(pd.read_csv(candidate))
        for candidate in CSV_DIR.glob("20260916-MS-*_validation_histories.csv"):
            if candidate != history_path:
                history_parts.append(pd.read_csv(candidate))
        if endpoint_parts:
            cached_endpoints = pd.concat(
                endpoint_parts, ignore_index=True
            ).drop_duplicates("run_id", keep="last")
        if history_parts:
            cached_histories = pd.concat(
                history_parts, ignore_index=True
            ).drop_duplicates(["run_id", "optimizer_step"], keep="last")
    if refresh:
        cached_endpoints = pd.DataFrame()
        cached_histories = pd.DataFrame()
    have = set(cached_endpoints.run_id) if not cached_endpoints.empty else set()
    needed = design[~design.run_id.isin(have)]
    if not needed.empty:
        print(
            f"Fetching {len(needed):,} exact W&B runs with {workers} workers..."
        )
        records: list[dict[str, Any]] = []
        histories: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_one, entity, row._asdict()): row.run_id
                for row in needed.itertuples(index=False)
            }
            for index, future in enumerate(as_completed(futures), 1):
                record, history = future.result()
                records.append(record)
                if not history.empty:
                    histories.append(history)
                if index % 100 == 0 or index == len(futures):
                    print(f"  fetched {index:,}/{len(futures):,}")
                    # Checkpoint caches make long API jobs safely resumable.
                    new_endpoints = pd.DataFrame(records)
                    merged = pd.concat(
                        [cached_endpoints, new_endpoints], ignore_index=True
                    )
                    merged.drop_duplicates("run_id", keep="last").to_csv(
                        endpoint_path, index=False
                    )
                    if histories:
                        new_histories = pd.concat(histories, ignore_index=True)
                        merged_h = pd.concat(
                            [cached_histories, new_histories], ignore_index=True
                        ).drop_duplicates(
                            ["run_id", "optimizer_step"], keep="last"
                        )
                        merged_h.to_csv(history_path, index=False)
        cached_endpoints = pd.read_csv(endpoint_path)
        cached_histories = pd.read_csv(history_path)
    else:
        # Materialize this experiment's complete raw cache even when all rows
        # came from another experiment's overlapping exact-run cache.
        cached_endpoints[cached_endpoints.run_id.isin(design.run_id)].to_csv(
            endpoint_path, index=False
        )
        cached_histories[cached_histories.run_id.isin(design.run_id)].to_csv(
            history_path, index=False
        )
    endpoints = design.drop(columns=[c for c in design if c != "run_id"]).merge(
        cached_endpoints, on="run_id", how="left", validate="one_to_one"
    )
    histories = cached_histories[
        cached_histories.run_id.isin(design.run_id)
    ].copy()
    return endpoints, histories


def audit_coverage(
    design: pd.DataFrame,
    endpoints: pd.DataFrame,
    histories: pd.DataFrame,
    stem: str,
) -> pd.DataFrame:
    """Write the machine-readable coverage audit and fail before science."""
    found = set(endpoints.loc[endpoints.state.notna(), "run_id"])
    history_counts = histories.groupby("run_id").size().to_dict()
    rows: list[dict[str, Any]] = []
    for expected in design.itertuples(index=False):
        match = endpoints[endpoints.run_id == expected.run_id]
        actual = match.iloc[0] if len(match) == 1 else None
        errors: list[str] = []
        if expected.run_id not in found or actual is None:
            errors.append("missing")
        else:
            checks = {
                "run_name": (actual.run_name, expected.expected_run_name),
                "group": (actual.group, expected.expected_group),
                "state": (actual.state, "finished"),
                "cell_id": (actual.config_cell_id, expected.cell_id),
            }
            if not str(expected.condition).startswith("scratch"):
                checks.update(
                    {
                        "checkpoint_set": (
                            actual.config_checkpoint_set_id,
                            expected.checkpoint_set_id,
                        ),
                        "checkpoint_id": (
                            actual.config_checkpoint_id,
                            expected.checkpoint_id,
                        ),
                        "manifest_hash": (
                            actual.config_manifest_hash,
                            expected.checkpoint_manifest_hash,
                        ),
                        "checkpoint_hash": (
                            actual.config_checkpoint_sha256,
                            expected.checkpoint_sha256,
                        ),
                    }
                )
            for label, (observed, wanted) in checks.items():
                if str(observed) != str(wanted):
                    errors.append(f"{label}_mismatch")
            if not np.isfinite(float(actual.test_f1)):
                errors.append("missing_test_f1")
            if history_counts.get(expected.run_id, 0) < 3:
                errors.append("incomplete_history")
            if (
                pd.notna(actual.analysis_error)
                and str(actual.analysis_error).strip()
            ):
                errors.append("fetch_error")
        rows.append(
            {
                "matrix": expected.matrix,
                "species": expected.species,
                "group": expected.expected_group,
                "condition": expected.condition,
                "source_step": expected.source_step,
                "target_treatment": expected.target_treatment,
                "target_seed": expected.target_seed,
                "cell_id": expected.cell_id,
                "run_id": expected.run_id,
                "expected": 1,
                "found": int(actual is not None),
                "state": "" if actual is None else actual.state,
                "history_points": history_counts.get(expected.run_id, 0),
                "audit_errors": ";".join(sorted(set(errors))),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(CSV_DIR / f"{stem}_coverage.csv", index=False)
    failures = coverage[coverage.audit_errors != ""]
    summary = (
        coverage.groupby(
            [
                "matrix",
                "species",
                "group",
                "condition",
                "source_step",
                "target_treatment",
                "target_seed",
                "state",
            ],
            dropna=False,
        )
        .agg(expected=("expected", "sum"), found=("found", "sum"))
        .reset_index()
    )
    print("\nCoverage summary")
    print(summary.to_string(index=False))
    if not failures.empty:
        print("\nCoverage failures")
        print(failures.audit_errors.value_counts().to_string())
        raise RuntimeError(
            f"Coverage/provenance audit failed for {len(failures)} cells; see {stem}_coverage.csv"
        )
    return coverage


def _history_map(histories: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for run_id, frame in histories.groupby("run_id", sort=False):
        frame = frame.sort_values("optimizer_step").drop_duplicates(
            "optimizer_step", keep="last"
        )
        frame = frame.copy()
        frame["smooth_f1"] = frame.val_f1.rolling(3, min_periods=1).median()
        frame["cum_best_f1"] = frame.smooth_f1.cummax()
        result[str(run_id)] = frame
    return result


def _normalize_progress(best_f1: float, scratch_maximum: float) -> float:
    """Express cumulative-best F1 relative to matched scratch's maximum."""
    return best_f1 / scratch_maximum


def make_pairs(
    endpoints: pd.DataFrame, histories: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scratch = endpoints[endpoints.condition.str.startswith("scratch")].copy()
    transfer = endpoints[~endpoints.condition.str.startswith("scratch")].copy()
    scratch_columns = {
        column: f"scratch_{column}"
        for column in scratch.columns
        if column not in {"cell_id"}
    }
    paired = transfer.merge(
        scratch.rename(columns=scratch_columns),
        left_on="matched_scratch_id",
        right_on="cell_id",
        validate="many_to_one",
    )
    exact = (
        (paired.species == paired.scratch_species)
        & (paired.recording == paired.scratch_recording)
        & (paired.target_seed == paired.scratch_target_seed)
        & np.isclose(paired.target_fraction, paired.scratch_target_fraction)
        & (paired.scratch_family == paired.scratch_scratch_family)
    )
    if not exact.all():
        raise RuntimeError("Transfer-to-scratch matching invariant failed")
    paired["effect"] = paired.test_f1 - paired.scratch_test_f1
    hmap = _history_map(histories)
    efficiency_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        transfer_h = hmap[row.run_id]
        scratch_h = hmap[row.scratch_run_id]
        budgets = [int(row.planned_budget), int(row.scratch_planned_budget)]
        if budgets[0] != budgets[1]:
            raise RuntimeError(
                f"Planned-budget mismatch for {row.run_id}: {budgets}"
            )
        budget = budgets[0]
        for threshold in THRESHOLDS:
            target = threshold * float(scratch_h.smooth_f1.max())

            def crossing(frame: pd.DataFrame) -> tuple[int, bool, float, float]:
                reached = frame[frame.smooth_f1 >= target]
                if reached.empty:
                    return budget, False, np.nan, np.nan
                first = reached.iloc[0]
                return (
                    int(first.optimizer_step),
                    True,
                    float(first.processed_windows),
                    float(first.cumulative_flops),
                )

            (
                transfer_time,
                transfer_attained,
                transfer_windows,
                transfer_flops,
            ) = crossing(transfer_h)
            scratch_time, scratch_attained, scratch_windows, scratch_flops = (
                crossing(scratch_h)
            )
            if not scratch_attained:
                raise RuntimeError(
                    f"Scratch failed to attain its own {threshold:.0%} target"
                )
            if not np.isfinite(transfer_windows):
                transfer_windows = budget * float(row.effective_batch_size)
                transfer_flops = transfer_windows * float(row.flops_per_window)
            efficiency_rows.append(
                {
                    "condition": row.condition,
                    "species": row.species,
                    "subject": row.subject,
                    "recording": row.recording,
                    "target_seed": row.target_seed,
                    "source_step": int(row.source_step),
                    "threshold": threshold,
                    "planned_budget": budget,
                    "target_f1": target,
                    "scratch_time": scratch_time,
                    "transfer_time": transfer_time,
                    "steps_saved": scratch_time - transfer_time,
                    "fraction_budget_saved": (scratch_time - transfer_time)
                    / budget,
                    "attained": int(transfer_attained),
                    "scratch_attained": int(scratch_attained),
                    "scratch_windows": scratch_windows,
                    "transfer_windows": transfer_windows,
                    "windows_saved": scratch_windows - transfer_windows,
                    "scratch_flops": scratch_flops,
                    "transfer_flops": transfer_flops,
                    "flops_saved": scratch_flops - transfer_flops,
                }
            )
        # Common grid expressed in fractions of this recording's planned budget.
        for fraction in np.linspace(0.0, 1.0, 41):
            step = fraction * budget
            for kind, frame in (
                ("transfer", transfer_h),
                ("scratch", scratch_h),
            ):
                available = frame[frame.optimizer_step <= step]
                best = (
                    float(available.cum_best_f1.iloc[-1])
                    if not available.empty
                    else 0.0
                )
                dynamics_rows.append(
                    {
                        "condition": row.condition,
                        "species": row.species,
                        "subject": row.subject,
                        "recording": row.recording,
                        "target_seed": row.target_seed,
                        "source_step": int(row.source_step),
                        "kind": kind,
                        "budget_fraction": fraction,
                        "optimizer_step": step,
                        "normalized_progress": _normalize_progress(
                            best, float(scratch_h.smooth_f1.max())
                        ),
                        "attained": int(
                            best
                            >= PRIMARY_THRESHOLD
                            * float(scratch_h.smooth_f1.max())
                        ),
                    }
                )
    return paired, pd.DataFrame(efficiency_rows), pd.DataFrame(dynamics_rows)


def _hierarchy(
    frame: pd.DataFrame, values: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "species", "subject", "recording", "source_step"]
    extra = [
        key for key in ("threshold", "kind", "budget_fraction") if key in frame
    ]
    recording = frame.groupby(keys + extra, as_index=False)[list(values)].mean()
    subject = recording.groupby(
        ["condition", "species", "subject", "source_step"] + extra,
        as_index=False,
    )[list(values)].mean()
    return recording, subject


def _bootstrap_summary(
    subject: pd.DataFrame, group_keys: list[str], metrics: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for keys, frame in subject.groupby(group_keys, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        frame = frame.sort_values("subject")
        for metric in metrics:
            values = frame[metric].to_numpy(float)
            n = len(values)
            indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
            boot = values[indices].mean(axis=1)
            rows.append(
                {
                    **dict(zip(group_keys, keys)),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "median": float(np.median(values)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "subjects_positive": int((values > 0).sum()),
                    "subjects": n,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return pd.DataFrame(rows)


def aggregate_and_cache(
    paired: pd.DataFrame,
    efficiency: pd.DataFrame,
    dynamics: pd.DataFrame,
    stem: str,
) -> dict[str, pd.DataFrame]:
    perf_seed = paired[
        [
            "condition",
            "species",
            "subject",
            "recording",
            "target_seed",
            "source_step",
            "effect",
            "test_f1",
            "scratch_test_f1",
            "class_f1",
            "scratch_class_f1",
            "channel_count",
        ]
    ].copy()
    perf_recording, perf_subject = _hierarchy(
        perf_seed, ["effect", "test_f1", "scratch_test_f1"]
    )
    expected_subjects = {"minipigs": 7, "monkeys": 5}
    counts = perf_subject.groupby(
        ["condition", "species", "source_step"]
    ).subject.nunique()
    for (_, species, _), count in counts.items():
        if count != expected_subjects[species]:
            raise RuntimeError(
                f"Incomplete subject trajectory for {species}: {count}"
            )
    perf_summary = _bootstrap_summary(
        perf_subject,
        ["condition", "species", "source_step"],
        ["effect", "test_f1", "scratch_test_f1"],
    )
    eff_recording, eff_subject = _hierarchy(
        efficiency,
        [
            "scratch_time",
            "transfer_time",
            "steps_saved",
            "fraction_budget_saved",
            "attained",
            "windows_saved",
            "flops_saved",
        ],
    )
    eff_summary = _bootstrap_summary(
        eff_subject,
        ["condition", "species", "source_step", "threshold"],
        [
            "scratch_time",
            "transfer_time",
            "steps_saved",
            "fraction_budget_saved",
            "attained",
            "windows_saved",
            "flops_saved",
        ],
    )
    tables = {
        "paired_performance_seed": perf_seed,
        "paired_performance_recording": perf_recording,
        "paired_performance_subject": perf_subject,
        "performance_summary": perf_summary,
        "efficiency_seed": efficiency,
        "efficiency_recording": eff_recording,
        "efficiency_subject": eff_subject,
        "efficiency_summary": eff_summary[
            eff_summary.threshold == PRIMARY_THRESHOLD
        ],
        "threshold_sensitivity": eff_summary,
        "dynamics": dynamics,
    }
    for suffix, frame in tables.items():
        frame.to_csv(CSV_DIR / f"{stem}_{suffix}.csv", index=False)
    return tables


def _save(fig: plt.Figure, stem: str, suffix: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}_{suffix}.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def _line_band(
    ax: Any, frame: pd.DataFrame, metric: str, color: Any, label: str
) -> None:
    view = frame[frame.metric == metric].sort_values("source_step")
    ax.plot(
        view.source_step,
        view["mean"],
        marker="o",
        lw=2.4,
        color=color,
        label=label,
    )
    ax.fill_between(
        view.source_step.to_numpy(float),
        view.ci_low.to_numpy(float),
        view.ci_high.to_numpy(float),
        color=color,
        alpha=0.18,
    )


def _format_step_axis(ax: Any) -> None:
    ax.set_xscale("log")
    ax.set_xticks(SOURCE_STEPS, ["100", "300", "1k", "3k", "10k"])
    ax.grid(alpha=0.2)


def _robust_symmetric_limit(
    values: Iterable[float], quantile: float = 0.975, minimum: float = 0.01
) -> float:
    """Return an outlier-resistant, zero-centered colour limit."""
    finite = np.abs(np.asarray(list(values), dtype=float))
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return minimum
    return max(minimum, float(np.quantile(finite, quantile)))


def _wrapped_condition(condition: str, width: int = 27) -> str:
    return textwrap.fill(CONDITION_LABELS[condition], width=width)


def _subject_performance(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    subject = tables["paired_performance_subject"]
    summary = tables["performance_summary"]
    fig, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(4.8 * len(conditions), 8),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, species in enumerate(SPECIES):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            view = subject[
                (subject.species == species) & (subject.condition == condition)
            ]
            for name, trajectory in view.groupby("subject"):
                ax.plot(
                    trajectory.source_step,
                    trajectory.effect * 100,
                    color=SUBJECT_COLORS[species][name],
                    alpha=0.5,
                    lw=1,
                )
            mean = summary[
                (summary.species == species)
                & (summary.condition == condition)
                & (summary.metric == "effect")
            ].copy()
            mean[["mean", "ci_low", "ci_high"]] *= 100
            _line_band(ax, mean, "effect", "#222222", "Subject mean")
            ax.axhline(0, color="0.5", lw=0.8)
            _format_step_axis(ax)
            ax.set_title(f"{species.title()} — {CONDITION_LABELS[condition]}")
            if col == 0:
                ax.set_ylabel("Transfer − scratch F1 (percentage points)")
    fig.suptitle("Subject-level transfer-effect trajectories")
    _save(fig, stem, "subject_performance_trajectories")


def _condition_contrasts(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    summary = tables["performance_summary"]
    subject = tables["paired_performance_subject"]
    baseline = "transfer_reference"
    if len(conditions) == 1:
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.5), sharex=True, constrained_layout=True
        )
        for ax, species in zip(axes, SPECIES):
            condition = conditions[0]
            _line_band(
                ax,
                summary[
                    (summary.species == species)
                    & (summary.condition == condition)
                ],
                "effect",
                CONDITION_COLORS[condition],
                CONDITION_LABELS[condition],
            )
            ax.axhline(0, color="0.5", lw=0.8)
            _format_step_axis(ax)
            ax.set_title(species.title())
            ax.set_ylabel("Transfer effect (F1)")
        fig.suptitle("Reference-condition mean transfer trajectory")
        _save(fig, stem, "condition_performance_contrasts")
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8.5), sharex=True, constrained_layout=True
    )
    for col, species in enumerate(SPECIES):
        ax = axes[0, col]
        for condition in conditions:
            _line_band(
                ax,
                summary[
                    (summary.species == species)
                    & (summary.condition == condition)
                ],
                "effect",
                CONDITION_COLORS[condition],
                CONDITION_LABELS[condition],
            )
        ax.axhline(0, color="0.5", lw=0.8)
        _format_step_axis(ax)
        ax.set_title(species.title())
        ax.set_ylabel("Transfer effect (F1)")
        lower = axes[1, col]
        if "transfer_small" in conditions:
            comparisons = [
                ("transfer_small", baseline, "Small − Reference"),
                ("transfer_large", baseline, "Large − Reference"),
            ]
        elif "transfer_bias_free_matched" in conditions:
            comparisons = [
                (
                    "transfer_bias_free_to_biased",
                    baseline,
                    "Bias-free source / ordinary target − Reference",
                ),
                (
                    "transfer_bias_free_matched",
                    "transfer_bias_free_to_biased",
                    "Bias-free target − ordinary target (bias-free source)",
                ),
            ]
        elif "transfer_shared_retained" in conditions:
            comparisons = [
                (
                    "transfer_shared_to_ordinary",
                    baseline,
                    "Shared source / ordinary target − Reference",
                ),
                (
                    "transfer_shared_retained",
                    "transfer_shared_to_ordinary",
                    "Retained shared interface − ordinary target",
                ),
            ]
        else:
            comparisons = []
        for condition, comparator, label in comparisons:
            base = subject[
                (subject.species == species) & (subject.condition == comparator)
            ][["subject", "source_step", "effect"]].rename(
                columns={"effect": "baseline"}
            )
            contrast = subject[
                (subject.species == species) & (subject.condition == condition)
            ].merge(base, on=["subject", "source_step"], validate="one_to_one")
            contrast["contrast"] = contrast.effect - contrast.baseline
            boot = _bootstrap_summary(
                contrast.rename(columns={"contrast": "value"}),
                ["source_step"],
                ["value"],
            )
            _line_band(lower, boot, "value", CONDITION_COLORS[condition], label)
        lower.axhline(0, color="0.5", lw=0.8)
        _format_step_axis(lower)
        lower.set_ylabel("Paired contrast (F1)")
        lower.set_xlabel("Source optimizer step")
    fig.suptitle("Condition means and paired intervention contrasts")
    handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[c],
            lw=2.4,
            label=CONDITION_LABELS[c],
        )
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=8,
    )
    _save(fig, stem, "condition_performance_contrasts")


def _absolute_performance(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    summary = tables["performance_summary"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), sharey=True, constrained_layout=True
    )
    for ax, species in zip(axes, SPECIES):
        for condition in conditions:
            color = CONDITION_COLORS[condition]
            _line_band(
                ax,
                summary[
                    (summary.species == species)
                    & (summary.condition == condition)
                ],
                "test_f1",
                color,
                CONDITION_LABELS[condition],
            )
            _line_band(
                ax,
                summary[
                    (summary.species == species)
                    & (summary.condition == condition)
                ],
                "scratch_test_f1",
                color,
                f"{CONDITION_LABELS[condition]} scratch",
            )
            ax.lines[-1].set_linestyle("--")
        _format_step_axis(ax)
        ax.set_title(species.title())
        ax.set_ylabel("Held-out test supported macro-F1")
    handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[c],
            lw=2.4,
            label=CONDITION_LABELS[c],
        )
        for c in conditions
    ] + [
        Line2D([0], [0], color="0.3", lw=2.4, ls="-", label="Transfer"),
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=2.4,
            ls="--",
            label="Matched scratch",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 4),
        fontsize=8,
    )
    fig.suptitle(
        "Absolute transfer and architecture-matched scratch performance"
    )
    _save(fig, stem, "absolute_performance")


def _recording_heatmap(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    data = tables["paired_performance_recording"]
    limit = _robust_symmetric_limit(data.effect)
    fig, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(5.3 * len(conditions), 11),
        squeeze=False,
        constrained_layout=True,
    )
    mappable = None
    for row, species in enumerate(SPECIES):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            view = data[
                (data.species == species) & (data.condition == condition)
            ].copy()
            recording = view.recording.str.replace(
                "_task-AcousStim", "", regex=False
            ).str.replace("_desc-raw", "", regex=False)
            for subject in view.subject.unique():
                recording = recording.str.replace(
                    f"{subject}_", "", regex=False
                )
            view["label"] = view.subject + " | " + recording
            pivot = view.pivot(
                index="label", columns="source_step", values="effect"
            )
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                cbar=False,
            )
            if mappable is None:
                mappable = ax.collections[0]
            ax.set_title(
                f"{species.title()}\n{_wrapped_condition(condition)}",
                fontsize=10,
            )
            ax.set_xlabel("Source step")
            ax.set_ylabel("Subject | recording" if col == 0 else "")
            ax.set_xticklabels(["100", "300", "1k", "3k", "10k"], rotation=0)
            ax.tick_params(axis="y", labelsize=6)
    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            shrink=0.8,
            label=f"Transfer − scratch F1 (clipped at ±{limit:.2f})",
        )
    fig.suptitle("Recording-level transfer-effect heterogeneity")
    _save(fig, stem, "recording_effect_heatmap")


def _session_sensitivity_data(recording: pd.DataFrame) -> pd.DataFrame:
    """Center each recording's transfer effect on its source-step-100 value."""
    keys = ["condition", "species", "subject", "recording"]
    baseline = recording[recording.source_step == SOURCE_STEPS[0]][
        keys + ["effect"]
    ].rename(columns={"effect": "baseline_effect"})
    centered = recording.merge(baseline, on=keys, validate="many_to_one")
    centered["effect_change"] = centered.effect - centered.baseline_effect
    centered["effect_change_pp"] = centered.effect_change * 100
    return centered


def _session_sensitivity(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    centered = _session_sensitivity_data(tables["paired_performance_recording"])
    centered.to_csv(CSV_DIR / f"{stem}_session_sensitivity.csv", index=False)
    fig, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(max(7, 5.0 * len(conditions)), 8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, species in enumerate(SPECIES):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            view = centered[
                (centered.species == species)
                & (centered.condition == condition)
            ]
            for (subject, _, _), trajectory in view.groupby(
                ["subject", "recording", "condition"]
            ):
                ax.plot(
                    trajectory.source_step,
                    trajectory.effect_change_pp,
                    color=SUBJECT_COLORS[species][subject],
                    marker="o",
                    markersize=2.5,
                    alpha=0.48,
                    lw=1,
                )
            subject = (
                view.groupby(["subject", "source_step"], as_index=False)
                .effect_change_pp.mean()
                .rename(columns={"effect_change_pp": "value"})
            )
            summary = _bootstrap_summary(subject, ["source_step"], ["value"])
            _line_band(ax, summary, "value", "#111111", "Equal-subject mean")
            ax.axhline(0, color="0.5", lw=0.8)
            _format_step_axis(ax)
            if row == 0:
                ax.set_title(_wrapped_condition(condition), fontsize=10)
            ax.text(
                0.02,
                0.97,
                f"{view.recording.nunique()} sessions",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="0.35",
            )
            if col == 0:
                ax.set_ylabel(f"{species.title()}\nΔ transfer effect (F1 pp)")
            if row == len(SPECIES) - 1:
                ax.set_xlabel("Source optimizer step")
    subject_handles = [
        Line2D(
            [0],
            [0],
            color=SUBJECT_COLORS["minipigs"][subject],
            marker="o",
            lw=1,
            label=subject,
        )
        for subject in SUBJECT_COLORS["minipigs"]
    ]
    subject_handles.append(
        Line2D([0], [0], color="#111111", lw=2.4, label="Equal-subject mean")
    )
    fig.legend(
        handles=subject_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=8,
        title="Line color groups sessions by subject",
        title_fontsize=8,
    )
    fig.suptitle(
        "Session-wise sensitivity: within-session change from source step 100",
        y=0.97,
    )
    fig.subplots_adjust(top=0.86, bottom=0.19, hspace=0.13, wspace=0.08)
    _save(fig, stem, "session_sensitivity")


def _seed_dispersion(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    seed = tables["paired_performance_seed"].copy()
    seed["centered_effect"] = seed.effect - seed.groupby(
        ["condition", "species", "recording", "source_step"]
    ).effect.transform("mean")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, species in zip(axes, SPECIES):
        view = seed[seed.species == species]
        sns.boxplot(
            data=view,
            x="source_step",
            y="centered_effect",
            hue="condition",
            hue_order=conditions,
            palette=CONDITION_COLORS,
            ax=ax,
            showfliers=False,
        )
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(species.title())
        ax.set_ylabel("Seed effect centered within recording")
        ax.legend(fontsize=7, ncol=len(conditions))
    fig.suptitle("Target-seed dispersion (optimization replicates)")
    _save(fig, stem, "seed_dispersion")


def _classwise(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    seed = tables["paired_performance_seed"]
    rows = []
    class_names = [
        "low bass",
        "mid bass",
        "low mids",
        "midrange",
        "high mids",
        "low treble",
        "mid treble",
        "high treble",
    ]
    for row in seed.itertuples(index=False):
        transfer = json.loads(row.class_f1)
        scratch = json.loads(row.scratch_class_f1)
        for index, name in enumerate(class_names):
            if (
                index < len(transfer)
                and index < len(scratch)
                and np.isfinite(transfer[index])
                and np.isfinite(scratch[index])
            ):
                rows.append(
                    {
                        "condition": row.condition,
                        "species": row.species,
                        "subject": row.subject,
                        "recording": row.recording,
                        "source_step": row.source_step,
                        "class": name,
                        "effect": transfer[index] - scratch[index],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(CSV_DIR / f"{stem}_classwise_effects.csv", index=False)
    display = frame.groupby(
        ["condition", "species", "class", "source_step"], as_index=False
    ).effect.mean()
    limit = _robust_symmetric_limit(display.effect)
    fig, axes = plt.subplots(
        2,
        len(conditions),
        figsize=(5.0 * len(conditions), 8),
        squeeze=False,
        constrained_layout=True,
    )
    mappable = None
    for r, species in enumerate(SPECIES):
        for c, condition in enumerate(conditions):
            ax = axes[r, c]
            view = display[
                (display.species == species) & (display.condition == condition)
            ]
            pivot = view.pivot(
                index="class", columns="source_step", values="effect"
            )
            sns.heatmap(
                pivot,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                ax=ax,
                cbar=False,
            )
            if mappable is None:
                mappable = ax.collections[0]
            ax.set_title(
                f"{species.title()}\n{_wrapped_condition(condition)}",
                fontsize=10,
            )
            ax.set_xlabel("Source step")
            ax.set_ylabel("Frequency band" if c == 0 else "")
            ax.set_xticklabels(["100", "300", "1k", "3k", "10k"], rotation=0)
    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            shrink=0.82,
            label=f"Transfer − scratch class F1 (clipped at ±{limit:.2f})",
        )
    fig.suptitle(
        "Classwise transfer-minus-scratch F1 (unsupported cells excluded)"
    )
    _save(fig, stem, "classwise_effects")


def _source_association(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    path = (
        CSV_DIR
        / "20260916-MS-pretraining-architecture-viability_fixed_milestones.csv"
    )
    source = pd.read_csv(path)
    condition_map = {
        "transfer_reference": "reference_backbone",
        "transfer_small": "small_backbone",
        "transfer_large": "large_backbone",
        "transfer_bias_free_to_biased": "bias_free_adapter",
        "transfer_bias_free_matched": "bias_free_adapter",
        "transfer_shared_to_ordinary": "shared_padded_adapter",
        "transfer_shared_retained": "shared_padded_adapter",
    }
    subject = tables["paired_performance_subject"].copy()
    subject["source_condition_key"] = subject.condition.map(condition_map)
    merged = subject.merge(
        source,
        left_on=["species", "subject", "source_step", "source_condition_key"],
        right_on=["species", "subject", "milestone_step", "condition"],
        suffixes=("", "_source"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for row, species in enumerate(SPECIES):
        for col, x in enumerate(("val_loss", "val_supported_f1")):
            ax = axes[row, col]
            for condition in conditions:
                view = merged[
                    (merged.species == species)
                    & (merged.condition == condition)
                ]
                for _, trajectory in view.groupby("subject"):
                    ax.plot(
                        trajectory[x],
                        trajectory.effect,
                        marker="o",
                        alpha=0.35,
                        color=CONDITION_COLORS[condition],
                    )
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_title(
                f"{species.title()} — source {'CE' if x == 'val_loss' else 'F1'}"
            )
            ax.set_xlabel(
                "Source validation CE"
                if x == "val_loss"
                else "Source validation supported F1"
            )
            ax.set_ylabel("Downstream transfer effect" if col == 0 else "")
    fig.suptitle("Descriptive source-to-downstream association (not causal)")
    _save(fig, stem, "source_downstream_association")


def _efficiency_figures(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    subject = tables["efficiency_subject"]
    summary = tables["threshold_sensitivity"]
    primary_subject = subject[subject.threshold == PRIMARY_THRESHOLD]
    primary_summary = summary[summary.threshold == PRIMARY_THRESHOLD]
    fig, axes = plt.subplots(
        4,
        len(conditions),
        figsize=(5.0 * len(conditions), 14),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for species_index, species in enumerate(SPECIES):
        for col, condition in enumerate(conditions):
            top = axes[species_index * 2, col]
            bottom = axes[species_index * 2 + 1, col]
            view = primary_subject[
                (primary_subject.species == species)
                & (primary_subject.condition == condition)
            ]
            for name, trajectory in view.groupby("subject"):
                top.plot(
                    trajectory.source_step,
                    trajectory.steps_saved,
                    color=SUBJECT_COLORS[species][name],
                    alpha=0.5,
                    lw=1,
                )
            mean = primary_summary[
                (primary_summary.species == species)
                & (primary_summary.condition == condition)
            ]
            _line_band(top, mean, "steps_saved", "#222222", "Mean")
            _line_band(
                bottom,
                mean,
                "attained",
                CONDITION_COLORS[condition],
                "Attainment",
            )
            top.axhline(0, color="0.5", lw=0.8)
            bottom.set_ylim(-0.03, 1.03)
            _format_step_axis(top)
            _format_step_axis(bottom)
            top.set_title(
                f"{species.title()}\n{CONDITION_LABELS[condition]}", fontsize=10
            )
            if col == 0:
                top.set_ylabel("Optimizer steps saved")
                bottom.set_ylabel("Attainment proportion")
            bottom.set_xlabel("Source step")
    fig.suptitle(
        "Subject-level efficiency trajectories at 90% matched-scratch quality"
    )
    _save(fig, stem, "subject_efficiency_trajectories")

    perf = tables["performance_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, species in zip(axes, SPECIES):
        for condition in conditions:
            x = perf[
                (perf.species == species)
                & (perf.condition == condition)
                & (perf.metric == "effect")
            ].sort_values("source_step")
            y = primary_summary[
                (primary_summary.species == species)
                & (primary_summary.condition == condition)
                & (primary_summary.metric == "steps_saved")
            ].sort_values("source_step")
            ax.plot(
                x["mean"],
                y["mean"],
                marker="o",
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
            for step, xv, yv in zip(x.source_step, x["mean"], y["mean"]):
                ax.annotate(f"{int(step):,}", (xv, yv), fontsize=6)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(species.title())
        ax.set_xlabel("Transfer − scratch test F1")
        ax.set_ylabel("Optimizer steps saved")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Performance–efficiency trade-off")
    _save(fig, stem, "performance_efficiency_tradeoff")


def _dynamics_figures(
    tables: dict[str, pd.DataFrame], stem: str, conditions: tuple[str, ...]
) -> None:
    dyn = tables["dynamics"]
    # Aggregate in the same hierarchy; bands are deliberately whole-subject.
    recording = dyn.groupby(
        [
            "condition",
            "species",
            "subject",
            "recording",
            "source_step",
            "kind",
            "budget_fraction",
        ],
        as_index=False,
    )[["normalized_progress", "attained"]].mean()
    subject = recording.groupby(
        [
            "condition",
            "species",
            "subject",
            "source_step",
            "kind",
            "budget_fraction",
        ],
        as_index=False,
    )[["normalized_progress", "attained"]].mean()
    for metric, suffix, ylabel in (
        (
            "normalized_progress",
            "normalized_finetuning_dynamics",
            "Cumulative-best validation F1 / matched-scratch maximum",
        ),
        (
            "attained",
            "cumulative_attainment",
            "Cumulative attainment proportion",
        ),
    ):
        fig, axes = plt.subplots(
            2,
            5,
            figsize=(18, 7),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for r, species in enumerate(SPECIES):
            for c, step in enumerate(SOURCE_STEPS):
                ax = axes[r, c]
                for condition in conditions:
                    for kind, ls in (("transfer", "-"), ("scratch", "--")):
                        view = subject[
                            (subject.species == species)
                            & (subject.source_step == step)
                            & (subject.condition == condition)
                            & (subject.kind == kind)
                        ]
                        pivot = view.pivot(
                            index="subject",
                            columns="budget_fraction",
                            values=metric,
                        )
                        mean = pivot.mean(axis=0)
                        values = pivot.to_numpy(float)
                        rng = np.random.default_rng(
                            BOOTSTRAP_SEED
                            + r * 10_000
                            + c * 1_000
                            + conditions.index(condition) * 100
                            + (0 if kind == "transfer" else 1)
                        )
                        sampled = rng.integers(
                            0,
                            len(values),
                            size=(BOOTSTRAP_REPLICATES, len(values)),
                        )
                        boot = values[sampled].mean(axis=1)
                        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
                        ax.plot(
                            mean.index,
                            mean.values,
                            color=CONDITION_COLORS[condition],
                            ls=ls,
                            lw=1.5,
                            label=f"{CONDITION_LABELS[condition]} {kind}",
                        )
                        ax.fill_between(
                            mean.index.to_numpy(float),
                            low,
                            high,
                            color=CONDITION_COLORS[condition],
                            alpha=0.08,
                        )
                if metric == "normalized_progress":
                    ax.axhline(PRIMARY_THRESHOLD, color="0.45", lw=0.9, ls=":")
                if r == 0:
                    ax.set_title(f"Source {step:,}")
                if c == 0:
                    row_label = (
                        "Relative validation F1"
                        if metric == "normalized_progress"
                        else "Attainment proportion"
                    )
                    ax.set_ylabel(f"{species.title()}\n{row_label}")
                ax.grid(alpha=0.2)
        handles = [
            Line2D(
                [0], [0], color=CONDITION_COLORS[c], label=CONDITION_LABELS[c]
            )
            for c in conditions
        ] + [
            Line2D([0], [0], color="0.3", ls="-", label="Transfer"),
            Line2D([0], [0], color="0.3", ls="--", label="Matched scratch"),
        ]
        if metric == "normalized_progress":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.45",
                    ls=":",
                    label="90% scratch target",
                )
            )
        fig.legend(
            handles=handles,
            loc="outside right center",
            ncol=1,
            fontsize=8,
        )
        fig.suptitle(ylabel)
        fig.supxlabel("Fraction of planned optimizer budget")
        _save(fig, stem, suffix)

    sens = tables["threshold_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, species in zip(axes, SPECIES):
        for condition in conditions:
            view = sens[
                (sens.species == species)
                & (sens.condition == condition)
                & (sens.metric == "steps_saved")
            ]
            for threshold, part in view.groupby("threshold"):
                ax.plot(
                    part.source_step,
                    part["mean"],
                    marker="o",
                    color=CONDITION_COLORS[condition],
                    alpha={0.8: 0.45, 0.9: 1, 0.95: 0.7}[
                        round(float(threshold), 2)
                    ],
                    ls={0.8: ":", 0.9: "-", 0.95: "--"}[
                        round(float(threshold), 2)
                    ],
                )
        ax.axhline(0, color="0.5", lw=0.8)
        _format_step_axis(ax)
        ax.set_title(species.title())
        ax.set_xlabel("Source step")
        ax.set_ylabel("Optimizer steps saved")
    handles = [
        Line2D(
            [0],
            [0],
            color=CONDITION_COLORS[c],
            lw=2.2,
            label=CONDITION_LABELS[c],
        )
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=8,
    )
    fig.suptitle("Threshold sensitivity: 80% dotted, 90% solid, 95% dashed")
    _save(fig, stem, "threshold_sensitivity")


def _specific_figures(
    experiment: str,
    tables: dict[str, pd.DataFrame],
    stem: str,
    conditions: tuple[str, ...],
) -> None:
    if experiment == "reference":
        current = tables["performance_summary"]
        old_path = (
            CSV_DIR
            / "20260915-MS-source-validation-downstream-trajectory_paired_f1_summary.csv"
        )
        old = pd.read_csv(old_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        for ax, species in zip(axes, SPECIES):
            now = current[
                (current.species == species)
                & (current.condition == "transfer_reference")
                & (current.metric == "effect")
            ].sort_values("source_step")
            then = old[
                (old.species == species) & (old.metric == "delta_scratch")
            ].sort_values("checkpoint_step")
            ax.plot(
                now.source_step,
                now["mean"],
                marker="o",
                lw=2.5,
                color=CONDITION_COLORS["transfer_reference"],
                label="Batch 128, source seed 42",
            )
            ax.fill_between(
                now.source_step,
                now.ci_low,
                now.ci_high,
                alpha=0.18,
                color=CONDITION_COLORS["transfer_reference"],
            )
            ax.plot(
                then.checkpoint_step,
                then["mean"],
                marker="o",
                color="0.45",
                label="Historical batch 16 Phase 4F",
            )
            ax.fill_between(
                then.checkpoint_step,
                then.ci_low,
                then.ci_high,
                alpha=0.15,
                color="0.5",
            )
            ax.axhline(0, color="0.5", lw=0.8)
            _format_step_axis(ax)
            ax.set_title(species.title())
            ax.set_xlabel("Source step")
            ax.set_ylabel("Transfer − own matched scratch F1")
            ax.legend(fontsize=8)
        fig.suptitle("Historical replication context (trajectories not pooled)")
        _save(fig, stem, "historical_replication")
    if experiment == "scale":
        eff = tables["threshold_sensitivity"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric in zip(axes, ("windows_saved", "flops_saved")):
            for condition in conditions:
                view = eff[
                    (eff.threshold == 0.9)
                    & (eff.metric == metric)
                    & (eff.condition == condition)
                ]
                for species, ls in (("minipigs", "-"), ("monkeys", "--")):
                    ax.plot(
                        view[view.species == species].source_step,
                        view[view.species == species]["mean"],
                        color=CONDITION_COLORS[condition],
                        ls=ls,
                        marker="o",
                    )
            ax.axhline(0, color="0.5", lw=0.8)
            _format_step_axis(ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Source step")
        fig.suptitle("Compute-adjusted savings to 90% matched-scratch quality")
        _save(fig, stem, "compute_to_target")
        viability = pd.read_csv(
            CSV_DIR
            / "20260916-MS-pretraining-architecture-viability_fixed_milestones.csv"
        )
        param_map = {
            "reference_backbone": 507456,
            "small_backbone": 51066,
            "large_backbone": 5084598,
        }
        viability = viability[viability.condition.isin(param_map)].copy()
        viability["source_flops_proxy"] = (
            viability.milestone_step * viability.condition.map(param_map)
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        perf = tables["performance_summary"]
        for ax, species in zip(axes, SPECIES):
            for condition in conditions:
                view = perf[
                    (perf.species == species)
                    & (perf.condition == condition)
                    & (perf.metric == "effect")
                ].sort_values("source_step")
                key = {
                    "transfer_reference": "reference_backbone",
                    "transfer_small": "small_backbone",
                    "transfer_large": "large_backbone",
                }[condition]
                x = np.asarray(view.source_step) * param_map[key]
                ax.plot(
                    x,
                    view["mean"],
                    marker="o",
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition],
                )
            ax.set_xscale("log")
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_title(species.title())
            ax.set_xlabel("Source step × transferable parameters (proxy)")
            ax.set_ylabel("Transfer effect")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle(
            "Transfer trajectory on a scale-specific source-compute proxy axis"
        )
        _save(fig, stem, "source_flops")
    if experiment == "shared":
        data = tables["paired_performance_recording"]
        channel = (
            tables["paired_performance_seed"]
            .groupby(
                ["condition", "species", "subject", "recording", "source_step"],
                as_index=False,
            )
            .agg(channel_count=("channel_count", "first"))
        )
        data = data.merge(
            channel,
            on=["condition", "species", "subject", "recording", "source_step"],
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, species in zip(axes, SPECIES):
            for condition, marker in (
                ("transfer_shared_to_ordinary", "o"),
                ("transfer_shared_retained", "^"),
            ):
                view = data[
                    (data.species == species) & (data.condition == condition)
                ]
                for subject, part in view.groupby("subject"):
                    ax.scatter(
                        part.channel_count,
                        part.effect,
                        s=20,
                        alpha=0.5,
                        color=SUBJECT_COLORS[species][subject],
                        marker=marker,
                    )
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_title(species.title())
            ax.set_xlabel("Filtered real-channel count")
            ax.set_ylabel("Recording transfer effect")
            ax.grid(alpha=0.2)
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color="0.35",
                        marker="o",
                        lw=0,
                        label="Shared source / ordinary target",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="0.35",
                        marker="^",
                        lw=0,
                        label="Retained shared interface",
                    ),
                ],
                fontsize=8,
            )
        fig.suptitle(
            "Shared-interface channel-count diagnostic (recordings descriptive)"
        )
        _save(fig, stem, "channel_count")


def render_all(
    experiment: str, tables: dict[str, pd.DataFrame], stem: str
) -> None:
    conditions = tuple(EXPERIMENTS[experiment]["conditions"])
    sns.set_theme(style="whitegrid", context="notebook")
    _subject_performance(tables, stem, conditions)
    _condition_contrasts(tables, stem, conditions)
    _absolute_performance(tables, stem, conditions)
    _recording_heatmap(tables, stem, conditions)
    _session_sensitivity(tables, stem, conditions)
    _seed_dispersion(tables, stem, conditions)
    _classwise(tables, stem, conditions)
    _source_association(tables, stem, conditions)
    _efficiency_figures(tables, stem, conditions)
    _dynamics_figures(tables, stem, conditions)
    _specific_figures(experiment, tables, stem, conditions)


def print_principal_tables(tables: dict[str, pd.DataFrame]) -> None:
    perf = tables["performance_summary"]
    perf = perf[perf.metric == "effect"][
        [
            "species",
            "condition",
            "source_step",
            "mean",
            "ci_low",
            "ci_high",
            "median",
            "minimum",
            "maximum",
            "subjects_positive",
            "subjects",
        ]
    ]
    eff = tables["efficiency_summary"]
    eff = eff[
        eff.metric.isin(
            [
                "scratch_time",
                "transfer_time",
                "steps_saved",
                "fraction_budget_saved",
                "attained",
            ]
        )
    ][
        [
            "species",
            "condition",
            "source_step",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
            "median",
            "minimum",
            "maximum",
            "subjects_positive",
            "subjects",
        ]
    ]
    print("\nPrimary performance table")
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPrimary 90% efficiency table")
    print(eff.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run(experiment: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "poyo-eeg")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    spec = EXPERIMENTS[experiment]
    stem = str(spec["stem"])
    design = load_compiled_design(experiment)
    if args.skip_fetch:
        endpoint_path = CSV_DIR / f"{stem}_run_endpoints.csv"
        history_path = CSV_DIR / f"{stem}_validation_histories.csv"
        if not endpoint_path.exists() or not history_path.exists():
            raise SystemExit("--skip-fetch requested but caches are missing")
        endpoints = pd.read_csv(endpoint_path)
        histories = pd.read_csv(history_path)
        endpoints = endpoints[endpoints.run_id.isin(design.run_id)].copy()
        histories = histories[histories.run_id.isin(design.run_id)].copy()
        endpoints.to_csv(endpoint_path, index=False)
        histories.to_csv(history_path, index=False)
    else:
        endpoints, histories = fetch_or_load(
            design, stem, args.entity, args.workers, args.refresh
        )
    audit_coverage(design, endpoints, histories, stem)
    paired, efficiency, dynamics = make_pairs(endpoints, histories)
    tables = aggregate_and_cache(paired, efficiency, dynamics, stem)
    print_principal_tables(tables)
    if not args.skip_figures:
        render_all(experiment, tables, stem)
        print(f"\nFigures written under {FIGURE_DIR}")


if __name__ == "__main__":
    raise SystemExit("Use one of the four experiment entry points.")

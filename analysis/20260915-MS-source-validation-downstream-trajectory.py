"""Analyze Phase 4F source-validation versus downstream transfer.

The definitive run set is the immutable Phase 4F and Phase 4E compiled cell
lists.  Source validation histories and downstream validation histories are
fetched through ``wandb.Api``.  The primary efficiency endpoint uses the
matched scratch run's threshold and retains right censoring through a
subject-weighted Kaplan--Meier restricted mean time-to-threshold (RMST).

Usage:
    uv run python analysis/20260915-MS-source-validation-downstream-trajectory.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
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

from _wandb_utils import (
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)


ROOT = Path(__file__).resolve().parents[1]
STEM = "20260915-MS-source-validation-downstream-trajectory"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
TEST_F1 = f"test/{TASK}_supported_f1"
VAL_F1 = f"val/{TASK}_supported_f1"
FIXED_STEPS = np.array([100, 300, 1_000, 3_000, 10_000], dtype=int)
SPECIES = ("minipigs", "monkeys")
WANDB_GROUPS = {
    "phase4f": {
        "minipigs": "20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MINIPIGS",
        "monkeys": "20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MONKEYS",
    },
    "source": {
        "minipigs": "PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS",
        "monkeys": "PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS",
    },
    "scratch": {
        "minipigs": "PHASE4E_VALIDATION_LOSS_SCRATCH_MINIPIGS",
        "monkeys": "PHASE4E_VALIDATION_LOSS_SCRATCH_MONKEYS",
    },
    "loss_selected": {
        "minipigs": "PHASE4E_VALIDATION_LOSS_TRANSFER_MINIPIGS",
        "monkeys": "PHASE4E_VALIDATION_LOSS_TRANSFER_MONKEYS",
    },
}
EXPECTED = {"fixed": 2_385, "scratch": 159, "loss_selected": 477}
BOOTSTRAP_SEED = 20_260_915
COLORS = {"minipigs": "#1f77b4", "monkeys": "#d95f02"}
LABELS = {"minipigs": "Minipigs", "monkeys": "Monkeys"}


def scalar(summary: Any, key: str) -> float | None:
    """Read a finite scalar from flattened or nested W&B summaries."""
    try:
        value = summary.get(key)
    except AttributeError:
        value = None
    for unwrap_key in ("max", "min"):
        try:
            result = float(unwrap_summary_value(value, unwrap_key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def test_f1_with_file_fallback(run: Any) -> float | None:
    """Recover a test summary from W&B's synced file if GraphQL omits it."""
    value = scalar(run.summary or {}, f"{TEST_F1}.max")
    if value is not None:
        return value
    # Two completed Phase 4F runs have complete synced summary files but their
    # test namespace is absent from the public summary object.  Reading the
    # synced file remains a W&B API operation and avoids hardcoded scores.
    with tempfile.TemporaryDirectory(prefix=f"{STEM}-summary-") as directory:
        path = Path(
            run.file("wandb-summary.json")
            .download(root=directory, replace=True)
            .name
        )
        payload = json.loads(path.read_text())
    return scalar(payload, TEST_F1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def expected_downstream() -> pd.DataFrame:
    """Load the exact Phase 4F runs and reused Phase 4E 100%-data controls."""
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        path = (
            ROOT
            / "launch"
            / "phase4f"
            / f"phase4f-checkpoint-trajectory-{species}.jsonl"
        )
        for cell in load_jsonl(path):
            rows.append(
                {
                    "condition": "fixed",
                    "species": species,
                    "recording_id": cell["target_recording"],
                    "subject": cell["target_subject"],
                    "target_seed": int(cell["target_finetuning_seed"]),
                    "source_seed": int(cell["source_model_seed"]),
                    "checkpoint_step": int(
                        cell["source_condition"]["milestone_step"]
                    ),
                    "cell_id": cell["cell_id"],
                    "run_id": cell["wandb_run_id"],
                    "wandb_group": cell["wandb_group"],
                }
            )
        phase4e = (
            ROOT
            / "launch"
            / "phase4e"
            / f"phase4e-transfer-learning-curves-{species}.jsonl"
        )
        for cell in load_jsonl(phase4e):
            if not math.isclose(float(cell["target_fraction"]), 1.0):
                continue
            condition = cell["condition_id"]
            if condition not in {"scratch_matched", "transfer_validation_loss"}:
                continue
            rows.append(
                {
                    "condition": "scratch"
                    if condition == "scratch_matched"
                    else "loss_selected",
                    "species": species,
                    "recording_id": cell["target_recording"],
                    "subject": cell["target_subject"],
                    "target_seed": int(cell["target_finetuning_seed"]),
                    "source_seed": (
                        np.nan
                        if condition == "scratch_matched"
                        else int(cell["source_selection_seed"])
                    ),
                    "checkpoint_step": np.nan,
                    "cell_id": cell["cell_id"],
                    "run_id": cell["wandb_run_id"],
                    "wandb_group": cell["wandb_group"],
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame.condition.value_counts().to_dict()
    if (
        counts != EXPECTED
        or frame.run_id.duplicated().any()
        or frame.cell_id.duplicated().any()
    ):
        raise RuntimeError(f"Unexpected compiled downstream design: {counts}")
    return frame.sort_values(["condition", "species", "cell_id"]).reset_index(
        drop=True
    )


def fetch_downstream_one(
    api: Any, entity: str, row: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch one final test summary and its complete validation-F1 history."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
            history = run.history(
                keys=["_step", "trainer/global_step", VAL_F1],
                samples=10_000,
                pandas=True,
            )
            if "trainer/global_step" not in history or VAL_F1 not in history:
                raise ValueError("missing validation step/F1 history")
            history = history[["trainer/global_step", VAL_F1]].dropna().copy()
            history.columns = ["optimizer_step", "val_f1"]
            history["optimizer_step"] = history.optimizer_step.astype(int)
            history = history.sort_values("optimizer_step").drop_duplicates(
                "optimizer_step", keep="last"
            )
            if len(history) < 3:
                raise ValueError("fewer than three validation evaluations")
            record = dict(row)
            record.update(
                {
                    "run_name": str(run.name or ""),
                    "state": str(run.state),
                    "test_f1": test_f1_with_file_fallback(run),
                    "history_points": len(history),
                    "analysis_error": "",
                }
            )
            history.insert(0, "run_id", row["run_id"])
            return record, history
        except Exception as exc:  # retry transient API failures
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    record = dict(row)
    record.update(
        {
            "run_name": "",
            "state": "fetch_error",
            "test_f1": np.nan,
            "history_points": 0,
            "analysis_error": str(last_error),
        }
    )
    return record, pd.DataFrame(columns=["run_id", "optimizer_step", "val_f1"])


def fetch_downstream_chunk(
    entity: str, offset: int, limit: int, workers: int
) -> None:
    expected = expected_downstream().iloc[offset : offset + limit]
    api = wandb.Api(timeout=120)
    records: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_downstream_one, api, entity, row._asdict())
            for row in expected.itertuples(index=False)
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record, history = future.result()
            records.append(record)
            if not history.empty:
                histories.append(history)
            if index % 100 == 0:
                print(f"Fetched {index}/{len(expected)}", flush=True)
    out = csv_dir(__file__)
    pd.DataFrame(records).to_csv(
        out / f"{STEM}_downstream_{offset:04d}.csv", index=False
    )
    pd.concat(histories, ignore_index=True).to_csv(
        out / f"{STEM}_histories_{offset:04d}.csv", index=False
    )
    print(f"Wrote downstream chunk {offset}:{offset + len(expected)}")


def parse_source_name(name: str) -> tuple[str, int]:
    match = re.search(r"_(sub-\d+)_s(\d+)_m\d+$", name)
    if not match:
        raise ValueError(f"Cannot parse source run name {name!r}")
    return match.group(1), int(match.group(2))


def fetch_source(entity: str) -> pd.DataFrame:
    """Fetch the 36 source histories and select exact retained milestones."""
    api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        runs = list(
            api.runs(
                f"{entity}/{PROJECT}",
                filters={"group": WANDB_GROUPS["source"][species]},
                per_page=100,
                lazy=False,
            )
        )
        expected_count = 21 if species == "minipigs" else 15
        if len(runs) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {species} source runs, found {len(runs)}"
            )
        for run in runs:
            subject, source_seed = parse_source_name(str(run.name))
            history = run.history(
                keys=["trainer/global_step", "val/loss"],
                samples=10_000,
                pandas=True,
            )
            history = (
                history[["trainer/global_step", "val/loss"]].dropna().copy()
            )
            history.columns = ["logged_step", "source_val_loss"]
            history = history.sort_values("logged_step").drop_duplicates(
                "logged_step", keep="last"
            )
            if len(history) < 90:
                raise RuntimeError(f"Incomplete source history for {run.id}")
            for step in FIXED_STEPS:
                # Validation is logged with Lightning's pre-increment global step.
                candidates = history[history.logged_step <= int(step)]
                point = candidates.iloc[-1]
                rows.append(
                    {
                        "condition": "fixed",
                        "species": species,
                        "subject": subject,
                        "source_seed": source_seed,
                        "checkpoint_step": int(step),
                        "source_val_loss": float(point.source_val_loss),
                        "source_logged_step": int(point.logged_step),
                        "source_run_id": run.id,
                        "source_run_name": run.name,
                    }
                )
            best = history.loc[history.source_val_loss.idxmin()]
            rows.append(
                {
                    "condition": "loss_selected",
                    "species": species,
                    "subject": subject,
                    "source_seed": source_seed,
                    "checkpoint_step": int(best.logged_step) + 1,
                    "source_val_loss": float(best.source_val_loss),
                    "source_logged_step": int(best.logged_step),
                    "source_run_id": run.id,
                    "source_run_name": run.name,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 216:
        raise RuntimeError(
            f"Expected 216 source checkpoint records, found {len(frame)}"
        )
    frame.to_csv(
        csv_dir(__file__) / f"{STEM}_source_checkpoints.csv", index=False
    )
    return frame


def read_complete_cache(
    expected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = csv_dir(__file__)
    summary_paths = sorted(
        out.glob(f"{STEM}_downstream_[0-9][0-9][0-9][0-9].csv")
    )
    history_paths = sorted(
        out.glob(f"{STEM}_histories_[0-9][0-9][0-9][0-9].csv")
    )
    if not summary_paths or not history_paths:
        raise RuntimeError(
            "No chunk caches found; run with --fetch-chunk first"
        )
    downstream = pd.concat(
        [pd.read_csv(path) for path in summary_paths], ignore_index=True
    )
    histories = pd.concat(
        [pd.read_csv(path) for path in history_paths], ignore_index=True
    )
    downstream = downstream.drop_duplicates("run_id", keep="last")
    histories = histories.drop_duplicates(
        ["run_id", "optimizer_step"], keep="last"
    )
    missing = sorted(set(expected.run_id) - set(downstream.run_id))
    extra = sorted(set(downstream.run_id) - set(expected.run_id))
    if missing or extra or len(downstream) != len(expected):
        raise RuntimeError(
            f"Incomplete cache: missing={len(missing)}, extra={len(extra)}"
        )
    if not downstream.state.eq("finished").all():
        raise RuntimeError(
            f"Non-finished runs: {downstream.state.value_counts().to_dict()}"
        )
    if downstream.analysis_error.fillna("").ne("").any():
        bad = downstream[downstream.analysis_error.fillna("").ne("")]
        raise RuntimeError(f"Fetch errors remain for {len(bad)} runs")
    if downstream.test_f1.isna().any():
        raise RuntimeError(
            f"Missing test F1 for {downstream.test_f1.isna().sum()} runs"
        )
    if set(histories.run_id) != set(expected.run_id):
        raise RuntimeError("At least one expected run lacks validation history")
    return downstream, histories


def smoothed_curve(history: pd.DataFrame) -> pd.DataFrame:
    result = history.sort_values("optimizer_step").copy()
    result["smoothed_f1"] = result.val_f1.rolling(3, min_periods=3).median()
    return result.dropna(subset=["smoothed_f1"])


def attach_time_endpoints(
    downstream: pd.DataFrame, histories: pd.DataFrame
) -> pd.DataFrame:
    """Apply each matched scratch threshold to every transfer curve."""
    scratch = downstream[downstream.condition.eq("scratch")]
    thresholds: dict[tuple[str, str, int], float] = {}
    for row in scratch.itertuples(index=False):
        curve = smoothed_curve(histories[histories.run_id.eq(row.run_id)])
        thresholds[(row.species, row.recording_id, int(row.target_seed))] = (
            0.9 * float(curve.smoothed_f1.max())
        )
    output: list[dict[str, Any]] = []
    for row in downstream.to_dict("records"):
        key = (row["species"], row["recording_id"], int(row["target_seed"]))
        threshold = thresholds[key]
        curve = smoothed_curve(histories[histories.run_id.eq(row["run_id"])])
        above = curve.smoothed_f1.ge(threshold)
        crossing = (
            above
            & above.shift(-1, fill_value=False)
            & above.shift(-2, fill_value=False)
        )
        event = bool(crossing.any())
        observed = float(
            curve.loc[crossing.idxmax(), "optimizer_step"]
            if event
            else curve.optimizer_step.iloc[-1]
        )
        output.append(
            {
                **row,
                "scratch_threshold": threshold,
                "time": observed,
                "event": event,
            }
        )
    return pd.DataFrame(output)


def weighted_km_rmst(
    time_values: np.ndarray, events: np.ndarray, weights: np.ndarray, tau: float
) -> float:
    """Weighted Kaplan--Meier restricted mean survival time through ``tau``."""
    order = np.argsort(time_values)
    times = time_values[order].astype(float)
    event = events[order].astype(bool)
    weight = weights[order].astype(float)
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(times[times <= tau]):
        area += survival * (float(current) - previous)
        at_risk = weight[times >= current].sum()
        failures = weight[(times == current) & event].sum()
        if at_risk > 0:
            survival *= 1.0 - failures / at_risk
        previous = float(current)
    area += survival * max(0.0, tau - previous)
    return float(area)


def subject_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("subject").run_id.transform("count").to_numpy(float)
    return 1.0 / counts


def aggregate_condition(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    species: str,
    condition: str,
    tau: float,
) -> dict[str, float]:
    data = frame[
        (frame.species == species) & (frame.condition == condition)
    ].copy()
    f1_subject = data.groupby("subject").test_f1.mean()
    reach_subject = data.groupby("subject").event.mean()
    result = {
        "test_f1": float(f1_subject.mean()),
        "reach_probability": float(reach_subject.mean()),
        "rmst": weighted_km_rmst(
            data.time.to_numpy(float),
            data.event.to_numpy(bool),
            subject_equal_weights(data),
            tau,
        ),
    }
    src = source[(source.species == species) & (source.condition == condition)]
    result["source_val_loss"] = float(
        src.groupby("subject").source_val_loss.mean().mean()
    )
    result["checkpoint_step"] = float(
        src.groupby("subject").checkpoint_step.mean().mean()
    )
    return result


def aggregate_fixed_step(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    species: str,
    step: int,
    tau: float,
) -> dict[str, float]:
    data = frame[
        (frame.species == species)
        & frame.condition.eq("fixed")
        & frame.checkpoint_step.eq(step)
    ].copy()
    src = source[
        (source.species == species)
        & source.condition.eq("fixed")
        & source.checkpoint_step.eq(step)
    ]
    return {
        "source_val_loss": float(
            src.groupby("subject").source_val_loss.mean().mean()
        ),
        "test_f1": float(data.groupby("subject").test_f1.mean().mean()),
        "reach_probability": float(data.groupby("subject").event.mean().mean()),
        "rmst": weighted_km_rmst(
            data.time.to_numpy(float),
            data.event.to_numpy(bool),
            subject_equal_weights(data),
            tau,
        ),
    }


def km_rmst_draws(
    data: pd.DataFrame,
    subjects: np.ndarray,
    bootstrap_counts: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized subject-equal weighted KM RMST for bootstrap draws."""
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    subject_counts = data.groupby("subject").run_id.count().to_dict()
    run_subject = data.subject.map(subject_index).to_numpy(int)
    run_weight = np.array(
        [1.0 / subject_counts[subject] for subject in data.subject], dtype=float
    )
    observed = data.time.to_numpy(float)
    event = data.event.to_numpy(bool)
    event_times = np.unique(observed[observed <= tau])
    if not len(event_times):
        return np.full(len(bootstrap_counts), tau, dtype=float)
    risk = np.zeros((len(event_times), len(subjects)), dtype=float)
    failures = np.zeros_like(risk)
    for subject_number in range(len(subjects)):
        selected = run_subject == subject_number
        for time_number, current in enumerate(event_times):
            risk[time_number, subject_number] = run_weight[
                selected & (observed >= current)
            ].sum()
            failures[time_number, subject_number] = run_weight[
                selected & (observed == current) & event
            ].sum()
    at_risk = bootstrap_counts @ risk.T
    failed = bootstrap_counts @ failures.T
    hazards = np.divide(
        failed, at_risk, out=np.zeros_like(failed), where=at_risk > 0
    )
    survival_after = np.cumprod(1.0 - hazards, axis=1)
    survival_before = np.concatenate(
        [np.ones((len(bootstrap_counts), 1)), survival_after[:, :-1]], axis=1
    )
    intervals = np.diff(np.concatenate([[0.0], event_times]))
    return survival_before @ intervals + survival_after[:, -1] * max(
        0.0, tau - event_times[-1]
    )


def row_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slopes for arrays shaped (bootstrap draws, checkpoints)."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.square(x_centered).sum(axis=1)
    return np.divide(
        (x_centered * y_centered).sum(axis=1),
        denominator,
        out=np.full(len(x), np.nan),
        where=denominator > 0,
    )


def bootstrap(
    frame: pd.DataFrame, source: pd.DataFrame, replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    trajectory_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    for species in SPECIES:
        fixed = frame[(frame.species == species) & frame.condition.eq("fixed")]
        # One common restriction horizon makes checkpoint RMSTs comparable.
        tau = float(fixed.groupby("checkpoint_step").time.max().min())
        subjects = np.array(sorted(fixed.subject.unique()))
        point = {
            step: aggregate_fixed_step(frame, source, species, int(step), tau)
            for step in FIXED_STEPS
        }
        draw_metrics = {
            step: {metric: np.empty(replicates) for metric in point[step]}
            for step in FIXED_STEPS
        }
        bootstrap_counts = rng.multinomial(
            len(subjects),
            np.full(len(subjects), 1.0 / len(subjects)),
            replicates,
        )
        for step in FIXED_STEPS:
            data = fixed[fixed.checkpoint_step.eq(step)]
            src = source[
                (source.species == species)
                & source.condition.eq("fixed")
                & source.checkpoint_step.eq(step)
            ]
            subject_f1 = (
                data.groupby("subject").test_f1.mean().reindex(subjects)
            )
            subject_reach = (
                data.groupby("subject").event.mean().reindex(subjects)
            )
            subject_loss = (
                src.groupby("subject").source_val_loss.mean().reindex(subjects)
            )
            draw_metrics[step]["test_f1"] = (
                bootstrap_counts @ subject_f1.to_numpy(float) / len(subjects)
            )
            draw_metrics[step]["reach_probability"] = (
                bootstrap_counts @ subject_reach.to_numpy(float) / len(subjects)
            )
            draw_metrics[step]["source_val_loss"] = (
                bootstrap_counts @ subject_loss.to_numpy(float) / len(subjects)
            )
            draw_metrics[step]["rmst"] = km_rmst_draws(
                data, subjects, bootstrap_counts, tau
            )

        loss_draws = np.column_stack(
            [draw_metrics[step]["source_val_loss"] for step in FIXED_STEPS]
        )
        improvement_draws = loss_draws[:, [0]] - loss_draws
        log_step = np.broadcast_to(
            np.log(FIXED_STEPS), (replicates, len(FIXED_STEPS))
        )
        trend_draws = {
            "source_loss_vs_log_step": row_slopes(log_step, loss_draws),
            "f1_vs_source_improvement": row_slopes(
                improvement_draws,
                np.column_stack(
                    [draw_metrics[step]["test_f1"] for step in FIXED_STEPS]
                ),
            ),
            "rmst_vs_source_improvement": row_slopes(
                improvement_draws,
                np.column_stack(
                    [draw_metrics[step]["rmst"] for step in FIXED_STEPS]
                ),
            ),
            "reach_vs_source_improvement": row_slopes(
                improvement_draws,
                np.column_stack(
                    [
                        draw_metrics[step]["reach_probability"]
                        for step in FIXED_STEPS
                    ]
                ),
            ),
        }
        point_loss = np.array(
            [point[step]["source_val_loss"] for step in FIXED_STEPS]
        )
        point_improvement = point_loss[0] - point_loss
        point_trends = {
            "source_loss_vs_log_step": np.polyfit(
                np.log(FIXED_STEPS), point_loss, 1
            )[0],
            "f1_vs_source_improvement": np.polyfit(
                point_improvement, [point[s]["test_f1"] for s in FIXED_STEPS], 1
            )[0],
            "rmst_vs_source_improvement": np.polyfit(
                point_improvement, [point[s]["rmst"] for s in FIXED_STEPS], 1
            )[0],
            "reach_vs_source_improvement": np.polyfit(
                point_improvement,
                [point[s]["reach_probability"] for s in FIXED_STEPS],
                1,
            )[0],
        }
        for step in FIXED_STEPS:
            for metric, value in point[step].items():
                low, high = np.quantile(
                    draw_metrics[step][metric], [0.025, 0.975]
                )
                trajectory_rows.append(
                    {
                        "species": species,
                        "checkpoint_step": int(step),
                        "metric": metric,
                        "mean": value,
                        "ci_low": low,
                        "ci_high": high,
                        "tau": tau,
                    }
                )
        for metric, value in point_trends.items():
            low, high = np.quantile(trend_draws[metric], [0.025, 0.975])
            predicted = {
                "source_loss_vs_log_step": high < 0,
                "f1_vs_source_improvement": high < 0,
                "rmst_vs_source_improvement": low > 0,
                "reach_vs_source_improvement": high < 0,
            }[metric]
            trend_rows.append(
                {
                    "species": species,
                    "metric": metric,
                    "slope": value,
                    "ci_low": low,
                    "ci_high": high,
                    "predicted_direction_excludes_zero": bool(predicted),
                    "subjects": len(subjects),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(trajectory_rows), pd.DataFrame(trend_rows)


def reference_summary(
    frame: pd.DataFrame, source: pd.DataFrame, trajectory: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        tau = float(trajectory[(trajectory.species == species)].tau.iloc[0])
        for condition in ("loss_selected", "scratch"):
            if condition == "scratch":
                data = frame[
                    (frame.species == species) & frame.condition.eq("scratch")
                ]
                rows.append(
                    {
                        "species": species,
                        "condition": condition,
                        "checkpoint_step": np.nan,
                        "source_val_loss": np.nan,
                        "test_f1": float(
                            data.groupby("subject").test_f1.mean().mean()
                        ),
                        "rmst": weighted_km_rmst(
                            data.time.to_numpy(float),
                            data.event.to_numpy(bool),
                            subject_equal_weights(data),
                            tau,
                        ),
                        "reach_probability": float(
                            data.groupby("subject").event.mean().mean()
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "species": species,
                        "condition": condition,
                        **aggregate_condition(
                            frame, source, species, condition, tau
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_f1_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """Build subject-level F1 effects with the prespecified hierarchy."""
    scratch = frame[frame.condition.eq("scratch")][
        ["species", "recording_id", "subject", "target_seed", "test_f1"]
    ].rename(columns={"test_f1": "scratch_f1"})
    fixed = frame[frame.condition.eq("fixed")].merge(
        scratch,
        on=["species", "recording_id", "subject", "target_seed"],
        validate="many_to_one",
    )
    fixed["delta_scratch"] = fixed.test_f1 - fixed.scratch_f1
    # Target seeds -> recording/source seed; recordings -> subject/source seed;
    # source seeds -> subject, so every excluded subject has equal final weight.
    recording = fixed.groupby(
        [
            "species",
            "subject",
            "source_seed",
            "checkpoint_step",
            "recording_id",
        ],
        as_index=False,
    ).agg(
        delta_scratch=("delta_scratch", "mean"),
        transfer_f1=("test_f1", "mean"),
    )
    source_seed = recording.groupby(
        ["species", "subject", "source_seed", "checkpoint_step"],
        as_index=False,
    ).agg(
        delta_scratch=("delta_scratch", "mean"),
        transfer_f1=("transfer_f1", "mean"),
    )
    subject = source_seed.groupby(
        ["species", "subject", "checkpoint_step"], as_index=False
    ).agg(
        delta_scratch=("delta_scratch", "mean"),
        transfer_f1=("transfer_f1", "mean"),
    )
    early = subject[subject.checkpoint_step.eq(100)][
        ["species", "subject", "transfer_f1"]
    ].rename(columns={"transfer_f1": "step100_f1"})
    subject = subject.merge(
        early, on=["species", "subject"], validate="many_to_one"
    )
    subject["delta_step100"] = subject.transfer_f1 - subject.step100_f1
    return subject


def paired_f1_summary(subject: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows: list[dict[str, Any]] = []
    for (species, step), group in subject.groupby(
        ["species", "checkpoint_step"], sort=True
    ):
        for metric in ("delta_scratch", "delta_step100"):
            values = group[metric].to_numpy(float)
            draws = rng.choice(
                values, size=(replicates, len(values)), replace=True
            ).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "species": species,
                    "checkpoint_step": int(step),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "subjects_positive": int((values > 0).sum()),
                    "subjects_total": len(values),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def plot_paired_f1(
    subject: pd.DataFrame, summary: pd.DataFrame, destination: Path
) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(14.5, 8.3), sharex=True, constrained_layout=True
    )
    columns = [
        ("delta_scratch", "Transfer minus matched scratch F1 (pp)"),
        ("delta_step100", "Checkpoint minus step-100 F1 (pp)"),
    ]
    for row_number, species in enumerate(SPECIES):
        species_subject = subject[subject.species.eq(species)]
        for column_number, (metric, ylabel) in enumerate(columns):
            axis = axes[row_number, column_number]
            for _, values in species_subject.groupby("subject"):
                values = values.sort_values("checkpoint_step")
                axis.plot(
                    values.checkpoint_step,
                    values[metric] * 100,
                    color=COLORS[species],
                    alpha=0.22,
                    linewidth=1.0,
                    marker="o",
                    markersize=2.5,
                )
            aggregate = summary[
                summary.species.eq(species) & summary.metric.eq(metric)
            ].sort_values("checkpoint_step")
            x = aggregate.checkpoint_step.to_numpy(float)
            mean = aggregate["mean"].to_numpy(float) * 100
            low = aggregate.ci_low.to_numpy(float) * 100
            high = aggregate.ci_high.to_numpy(float) * 100
            axis.fill_between(
                x, low, high, color=COLORS[species], alpha=0.18, zorder=2
            )
            axis.plot(
                x,
                mean,
                color=COLORS[species],
                linewidth=2.7,
                marker="o",
                markersize=6,
                zorder=3,
            )
            axis.axhline(0, color="#333333", linewidth=1.0, zorder=1)
            if metric == "delta_scratch":
                span = max(float(high.max() - low.min()), 1.0)
                for x_value, y_value, positive, total in zip(
                    x,
                    high,
                    aggregate.subjects_positive,
                    aggregate.subjects_total,
                    strict=True,
                ):
                    axis.text(
                        x_value,
                        y_value + 0.035 * span,
                        f"{int(positive)}/{int(total)} > 0",
                        color=COLORS[species],
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )
            axis.set_xscale("log")
            axis.set_xticks(FIXED_STEPS)
            axis.set_xticklabels(["100", "300", "1k", "3k", "10k"])
            axis.set_xlabel("Source-pretraining optimizer step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
            axis.set_title(
                f"{LABELS[species]} — {ylabel.removesuffix(' (pp)')}"
            )
    fig.suptitle(
        "Paired downstream F1 effects\n"
        "Thin lines: excluded subjects; thick line and band: subject-balanced mean and 95% bootstrap interval",
        fontsize=14,
    )
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def paired_efficiency_subjects(
    frame: pd.DataFrame, trajectory: pd.DataFrame
) -> pd.DataFrame:
    """Compute subject-level censoring-aware efficiency contrasts."""
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        tau = float(trajectory[trajectory.species.eq(species)].tau.iloc[0])
        species_frame = frame[frame.species.eq(species)]
        for subject in sorted(species_frame.subject.unique()):
            scratch = species_frame[
                species_frame.subject.eq(subject)
                & species_frame.condition.eq("scratch")
            ]
            scratch_rmst = weighted_km_rmst(
                scratch.time.to_numpy(float),
                scratch.event.to_numpy(bool),
                np.ones(len(scratch), dtype=float),
                tau,
            )
            scratch_reach = float(scratch.event.mean())
            for step in FIXED_STEPS:
                fixed = species_frame[
                    species_frame.subject.eq(subject)
                    & species_frame.condition.eq("fixed")
                    & species_frame.checkpoint_step.eq(step)
                ]
                rmst = weighted_km_rmst(
                    fixed.time.to_numpy(float),
                    fixed.event.to_numpy(bool),
                    np.ones(len(fixed), dtype=float),
                    tau,
                )
                reach = float(fixed.event.mean())
                rows.append(
                    {
                        "species": species,
                        "subject": subject,
                        "checkpoint_step": int(step),
                        "tau": tau,
                        "rmst": rmst,
                        "scratch_rmst": scratch_rmst,
                        "delta_rmst_scratch": rmst - scratch_rmst,
                        "reach_probability": reach,
                        "scratch_reach_probability": scratch_reach,
                        "delta_reach_scratch": reach - scratch_reach,
                    }
                )
    subject = pd.DataFrame(rows)
    early = subject[subject.checkpoint_step.eq(100)][
        ["species", "subject", "rmst", "reach_probability"]
    ].rename(
        columns={
            "rmst": "step100_rmst",
            "reach_probability": "step100_reach_probability",
        }
    )
    subject = subject.merge(
        early, on=["species", "subject"], validate="many_to_one"
    )
    subject["delta_rmst_step100"] = subject.rmst - subject.step100_rmst
    subject["delta_reach_step100"] = (
        subject.reach_probability - subject.step100_reach_probability
    )
    return subject


def paired_efficiency_summary(
    subject: pd.DataFrame, replicates: int
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    metrics = (
        "delta_rmst_scratch",
        "delta_rmst_step100",
        "delta_reach_scratch",
        "delta_reach_step100",
    )
    rows: list[dict[str, Any]] = []
    for (species, step), group in subject.groupby(
        ["species", "checkpoint_step"], sort=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            draws = rng.choice(
                values, size=(replicates, len(values)), replace=True
            ).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            worse = values > 0 if "rmst" in metric else values < 0
            rows.append(
                {
                    "species": species,
                    "checkpoint_step": int(step),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "subjects_worse": int(worse.sum()),
                    "subjects_total": len(values),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def plot_paired_efficiency_metric(
    subject: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    destination: Path,
) -> None:
    if metric_name == "rmst":
        metrics = ("delta_rmst_scratch", "delta_rmst_step100")
        labels = (
            "Transfer minus matched scratch RMST (steps)",
            "Checkpoint minus step-100 RMST (steps)",
        )
        scale = 1.0
        direction = "higher is worse"
    elif metric_name == "attainment":
        metrics = ("delta_reach_scratch", "delta_reach_step100")
        labels = (
            "Transfer minus matched scratch attainment (pp)",
            "Checkpoint minus step-100 attainment (pp)",
        )
        scale = 100.0
        direction = "lower is worse"
    else:
        raise ValueError(metric_name)
    fig, axes = plt.subplots(
        2, 2, figsize=(14.5, 8.3), sharex=True, constrained_layout=True
    )
    for row_number, species in enumerate(SPECIES):
        species_subject = subject[subject.species.eq(species)]
        for column_number, (metric, ylabel) in enumerate(
            zip(metrics, labels, strict=True)
        ):
            axis = axes[row_number, column_number]
            for _, values in species_subject.groupby("subject"):
                values = values.sort_values("checkpoint_step")
                axis.plot(
                    values.checkpoint_step,
                    values[metric] * scale,
                    color=COLORS[species],
                    alpha=0.22,
                    linewidth=1.0,
                    marker="o",
                    markersize=2.5,
                )
            aggregate = summary[
                summary.species.eq(species) & summary.metric.eq(metric)
            ].sort_values("checkpoint_step")
            x = aggregate.checkpoint_step.to_numpy(float)
            mean = aggregate["mean"].to_numpy(float) * scale
            low = aggregate.ci_low.to_numpy(float) * scale
            high = aggregate.ci_high.to_numpy(float) * scale
            axis.fill_between(
                x, low, high, color=COLORS[species], alpha=0.18, zorder=2
            )
            axis.plot(
                x,
                mean,
                color=COLORS[species],
                linewidth=2.7,
                marker="o",
                markersize=6,
                zorder=3,
            )
            axis.axhline(0, color="#333333", linewidth=1.0, zorder=1)
            if metric.endswith("scratch"):
                span = max(float(high.max() - low.min()), 1.0)
                for x_value, y_value, worse, total in zip(
                    x,
                    high,
                    aggregate.subjects_worse,
                    aggregate.subjects_total,
                    strict=True,
                ):
                    axis.text(
                        x_value,
                        y_value + 0.035 * span,
                        f"{int(worse)}/{int(total)} worse",
                        color=COLORS[species],
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )
            axis.set_xscale("log")
            axis.set_xticks(FIXED_STEPS)
            axis.set_xticklabels(["100", "300", "1k", "3k", "10k"])
            axis.set_xlabel("Source-pretraining optimizer step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
            axis.margins(y=0.16)
            axis.set_title(f"{LABELS[species]} — {ylabel.rsplit(' (', 1)[0]}")
    display_name = (
        "RMST optimizer efficiency"
        if metric_name == "rmst"
        else "threshold attainment"
    )
    fig.suptitle(
        f"Paired {display_name} effects ({direction})\n"
        "Thin lines: excluded subjects; thick line and band: subject-balanced mean and 95% bootstrap interval",
        fontsize=14,
    )
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    trajectory: pd.DataFrame, references: pd.DataFrame, destination: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    specs = [
        ("source_val_loss", "Source validation cross-entropy", 1.0),
        ("test_f1", "Downstream test supported macro-F1", 1.0),
        ("rmst", "Matched-quality RMST (optimizer steps)", 1.0),
        (
            "reach_probability",
            "Reached matched quality within budget (%)",
            100.0,
        ),
    ]
    for axis, (metric, ylabel, scale) in zip(axes.flat, specs, strict=True):
        for species in SPECIES:
            data = trajectory[
                (trajectory.species == species) & trajectory.metric.eq(metric)
            ].sort_values("checkpoint_step")
            x = data.checkpoint_step.to_numpy(float)
            y = data["mean"].to_numpy(float) * scale
            low = data.ci_low.to_numpy(float) * scale
            high = data.ci_high.to_numpy(float) * scale
            axis.fill_between(x, low, high, color=COLORS[species], alpha=0.15)
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                color=COLORS[species],
                label=LABELS[species],
            )
            if metric != "source_val_loss":
                scratch = references[
                    (references.species == species)
                    & references.condition.eq("scratch")
                ].iloc[0]
                axis.axhline(
                    float(scratch[metric]) * scale,
                    color=COLORS[species],
                    linestyle=":",
                    alpha=0.55,
                )
            selected = references[
                (references.species == species)
                & references.condition.eq("loss_selected")
            ].iloc[0]
            if metric != "source_val_loss" or np.isfinite(
                selected.source_val_loss
            ):
                axis.scatter(
                    float(selected.checkpoint_step),
                    float(selected[metric]) * scale,
                    marker="D",
                    s=45,
                    color=COLORS[species],
                    edgecolor="white",
                    zorder=4,
                )
        axis.set_xscale("log")
        axis.set_xticks(FIXED_STEPS)
        axis.set_xticklabels(["100", "300", "1k", "3k", "10k"])
        axis.set_xlabel("Source-pretraining optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Source performance")
    axes[0, 1].set_title("Final downstream performance")
    axes[1, 0].set_title("Optimizer efficiency (higher is worse)")
    axes[1, 1].set_title("Threshold attainment (higher is better)")
    fig.suptitle(
        "Source-validation and downstream trajectories\nDiamonds: loss-selected checkpoint; dotted: matched scratch",
        fontsize=14,
    )
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_associations(trajectory: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    specs = [
        ("test_f1", "Downstream test supported macro-F1", 1.0),
        ("rmst", "Matched-quality RMST (steps)", 1.0),
        ("reach_probability", "Reached threshold (%)", 100.0),
    ]
    for axis, (metric, ylabel, scale) in zip(axes, specs, strict=True):
        for species in SPECIES:
            loss = trajectory[
                (trajectory.species == species)
                & trajectory.metric.eq("source_val_loss")
            ].sort_values("checkpoint_step")
            data = trajectory[
                (trajectory.species == species) & trajectory.metric.eq(metric)
            ].sort_values("checkpoint_step")
            improvement = float(loss["mean"].iloc[0]) - loss["mean"].to_numpy(
                float
            )
            values = data["mean"].to_numpy(float) * scale
            axis.plot(
                improvement,
                values,
                marker="o",
                linewidth=2.2,
                color=COLORS[species],
                label=LABELS[species],
            )
            for x, y, step in zip(
                improvement, values, FIXED_STEPS, strict=True
            ):
                axis.annotate(
                    f"{step // 1000}k" if step >= 1000 else str(step),
                    (x, y),
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xlabel("Within-source-run validation CE improvement")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Downstream outcomes versus source-validation improvement", fontsize=14
    )
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_context(trajectory: pd.DataFrame, destination: Path) -> bool:
    parent = (
        csv_dir(__file__).parent
        / "csv"
        / "20260915-MS-adapter-bias-perturbation_subject_checkpoint_means.csv"
    )
    if not parent.exists():
        parent = (
            ROOT
            / "analysis"
            / "csv"
            / "20260915-MS-adapter-bias-perturbation_subject_checkpoint_means.csv"
        )
    if not parent.exists():
        return False
    bias = pd.read_csv(parent)
    bias = bias[
        (bias.checkpoint_kind == "milestone")
        & bias.checkpoint_step.isin(FIXED_STEPS)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for species in SPECIES:
        penalty = (
            bias[bias.species == species]
            .groupby("checkpoint_step")
            .delta_cross_entropy.mean()
            .reindex(FIXED_STEPS)
        )
        f1 = trajectory[
            (trajectory.species == species) & trajectory.metric.eq("test_f1")
        ].sort_values("checkpoint_step")
        axes[0].plot(
            FIXED_STEPS,
            penalty,
            marker="o",
            color=COLORS[species],
            linewidth=2.1,
            label=LABELS[species],
        )
        axes[1].plot(
            FIXED_STEPS,
            f1["mean"],
            marker="o",
            color=COLORS[species],
            linewidth=2.1,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(FIXED_STEPS)
        axis.set_xticklabels(["100", "300", "1k", "3k", "10k"])
        axis.set_xlabel("Source-pretraining optimizer step")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Bias-derangement CE penalty")
    axes[1].set_ylabel("Downstream test supported macro-F1")
    axes[0].legend(frameon=False)
    axes[0].set_title("Context: adapter-bias reliance")
    axes[1].set_title("Current experiment: downstream trajectory")
    fig.suptitle(
        "Qualitative context only — aligned trends do not establish mediation",
        fontsize=13,
    )
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def print_summary(
    trajectory: pd.DataFrame, trends: pd.DataFrame, references: pd.DataFrame
) -> None:
    print("\nPrimary trend tests (95% subject-bootstrap intervals):")
    print(trends.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    wide = trajectory.pivot(
        index=["species", "checkpoint_step"], columns="metric", values="mean"
    ).reset_index()
    wide["test_f1"] *= 100
    wide["reach_probability"] *= 100
    print("\nCheckpoint trajectory (subject-balanced point estimates):")
    print(
        wide[
            [
                "species",
                "checkpoint_step",
                "source_val_loss",
                "test_f1",
                "rmst",
                "reach_probability",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print("\nSecondary references:")
    print(references.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def analyze(entity: str, replicates: int) -> None:
    expected = expected_downstream()
    downstream, histories = read_complete_cache(expected)
    source_path = csv_dir(__file__) / f"{STEM}_source_checkpoints.csv"
    source = (
        pd.read_csv(source_path)
        if source_path.exists()
        else fetch_source(entity)
    )
    endpoints = attach_time_endpoints(downstream, histories)
    fixed_source = source[source.condition.eq("fixed")][
        [
            "species",
            "subject",
            "source_seed",
            "checkpoint_step",
            "source_val_loss",
        ]
    ]
    endpoints = endpoints.merge(
        fixed_source,
        on=["species", "subject", "source_seed", "checkpoint_step"],
        how="left",
        validate="many_to_one",
    )
    best_source = source[source.condition.eq("loss_selected")][
        [
            "species",
            "subject",
            "source_seed",
            "checkpoint_step",
            "source_val_loss",
        ]
    ].rename(
        columns={
            "checkpoint_step": "best_checkpoint_step",
            "source_val_loss": "best_source_val_loss",
        }
    )
    endpoints = endpoints.merge(
        best_source,
        on=["species", "subject", "source_seed"],
        how="left",
        validate="many_to_one",
    )
    endpoints.loc[
        endpoints.condition.eq("loss_selected"), "checkpoint_step"
    ] = endpoints.loc[
        endpoints.condition.eq("loss_selected"), "best_checkpoint_step"
    ]
    endpoints.loc[
        endpoints.condition.eq("loss_selected"), "source_val_loss"
    ] = endpoints.loc[
        endpoints.condition.eq("loss_selected"), "best_source_val_loss"
    ]
    trajectory, trends = bootstrap(endpoints, source, replicates)
    references = reference_summary(endpoints, source, trajectory)
    out = csv_dir(__file__)
    endpoints.to_csv(out / f"{STEM}_run_endpoints.csv", index=False)
    trajectory.to_csv(out / f"{STEM}_trajectory_summary.csv", index=False)
    trends.to_csv(out / f"{STEM}_trend_summary.csv", index=False)
    references.to_csv(out / f"{STEM}_reference_summary.csv", index=False)
    paired_subjects = paired_f1_subjects(endpoints)
    paired_summary = paired_f1_summary(paired_subjects, replicates)
    paired_subjects.to_csv(out / f"{STEM}_paired_f1_subjects.csv", index=False)
    paired_summary.to_csv(out / f"{STEM}_paired_f1_summary.csv", index=False)
    efficiency_subjects = paired_efficiency_subjects(endpoints, trajectory)
    efficiency_summary = paired_efficiency_summary(
        efficiency_subjects, replicates
    )
    efficiency_subjects.to_csv(
        out / f"{STEM}_paired_efficiency_subjects.csv", index=False
    )
    efficiency_summary.to_csv(
        out / f"{STEM}_paired_efficiency_summary.csv", index=False
    )
    figures = figures_dir(__file__)
    plot_trajectories(
        trajectory, references, figures / f"{STEM}_main_trajectories.png"
    )
    plot_associations(
        trajectory, figures / f"{STEM}_source_downstream_associations.png"
    )
    plot_context(trajectory, figures / f"{STEM}_bias_context.png")
    plot_paired_f1(
        paired_subjects,
        paired_summary,
        figures / f"{STEM}_paired_f1_effects.png",
    )
    plot_paired_efficiency_metric(
        efficiency_subjects,
        efficiency_summary,
        metric_name="rmst",
        destination=figures / f"{STEM}_paired_rmst_effects.png",
    )
    plot_paired_efficiency_metric(
        efficiency_subjects,
        efficiency_summary,
        metric_name="attainment",
        destination=figures / f"{STEM}_paired_attainment_effects.png",
    )
    print_summary(trajectory, trends, references)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=default_entity(), required=default_entity() is None
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--fetch-chunk", nargs=2, type=int, metavar=("OFFSET", "LIMIT")
    )
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()
    if args.fetch_chunk:
        fetch_downstream_chunk(args.entity, *args.fetch_chunk, args.workers)
        return
    if args.fetch_source:
        source = fetch_source(args.entity)
        print(f"Fetched {len(source)} source-checkpoint records")
        return
    analyze(args.entity, args.bootstrap_replicates)


if __name__ == "__main__":
    main()

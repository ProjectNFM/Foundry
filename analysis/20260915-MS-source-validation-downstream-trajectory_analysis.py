"""Audit and fetch the Phase 4F checkpoint-trajectory experiment from W&B.

The immutable Phase 4F and Phase 4E cell lists define every expected run.
Fetches are chunkable so the 3,021-run analysis can resume without hardcoded
run IDs or metrics. Validation histories are retained for the preregistered
matched-scratch time-to-quality endpoint.

Examples:
    uv run python analysis/20260915-MS-source-validation-downstream-trajectory_analysis.py --audit-design
    uv run python analysis/20260915-MS-source-validation-downstream-trajectory_analysis.py --fetch-chunk 0 250
    uv run python analysis/20260915-MS-source-validation-downstream-trajectory_analysis.py --summarize-cache
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import wandb


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "20260915-MS-source-validation-downstream-trajectory"
PROJECT = "neurosoft_supervised_pretraining"
ENTITY = os.environ.get("WANDB_ENTITY", "poyo-eeg")
TASK = "neurosoft_acoustic_stim_8band"
TEST_F1 = f"test/{TASK}_supported_f1"
VALIDATION_F1 = f"val/{TASK}_supported_f1"
SPECIES = ("minipigs", "monkeys")
CSV_DIR = ROOT / "analysis" / "csv"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        phase4f = _jsonl(
            ROOT
            / "launch/phase4f"
            / f"phase4f-checkpoint-trajectory-{species}.jsonl"
        )
        for cell in phase4f:
            rows.append(
                {
                    "phase": "phase4f",
                    "condition": "fixed_milestone",
                    "species": species,
                    "recording": cell["target_recording"],
                    "subject": cell["target_subject"],
                    "source_seed": int(cell["source_model_seed"]),
                    "target_seed": int(cell["target_finetuning_seed"]),
                    "source_step": int(
                        cell["source_condition"]["milestone_step"]
                    ),
                    "run_id": cell["wandb_run_id"],
                    "group": cell["wandb_group"],
                    "cell_id": cell["cell_id"],
                }
            )
        phase4e = _jsonl(
            ROOT
            / "launch/phase4e"
            / f"phase4e-transfer-learning-curves-{species}.jsonl"
        )
        for cell in phase4e:
            if float(cell["target_fraction"]) != 1.0:
                continue
            condition = str(cell["condition_id"])
            if condition not in {"transfer_validation_loss", "scratch_matched"}:
                continue
            source = cell.get("source_condition") or {}
            if not isinstance(source, dict):
                source = {}
            rows.append(
                {
                    "phase": "phase4e",
                    "condition": (
                        "best_loss"
                        if condition == "transfer_validation_loss"
                        else "scratch"
                    ),
                    "species": species,
                    "recording": cell["target_recording"],
                    "subject": cell["target_subject"],
                    "source_seed": cell.get("source_model_seed"),
                    "target_seed": int(cell["target_finetuning_seed"]),
                    "source_step": source.get("milestone_step"),
                    "run_id": cell["wandb_run_id"],
                    "group": cell["wandb_group"],
                    "cell_id": cell["cell_id"],
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["phase", "species", "condition", "cell_id"]
    )
    if len(frame) != 3021 or frame.cell_id.duplicated().any():
        raise RuntimeError(
            f"Expected 3,021 unique analysis cells, found {len(frame)}"
        )
    return frame.reset_index(drop=True)


def _scalar(summary: Any, key: str) -> float | None:
    value = summary.get(key) if hasattr(summary, "get") else None
    candidates = [value]
    if hasattr(value, "get"):
        candidates = [value.get("max"), value.get("min"), value]
    for candidate in candidates:
        try:
            result = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def fetch_one(api: Any, row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{row['run_id']}")
        result["run_name"] = str(run.name or "")
        result["state"] = str(run.state)
        result["test_f1"] = _scalar(run.summary or {}, f"{TEST_F1}.max")
        history = run.history(
            keys=["_step", "trainer/global_step", VALIDATION_F1],
            samples=10_000,
            pandas=True,
        )
        step_key = (
            "trainer/global_step"
            if "trainer/global_step" in history.columns
            else "_step"
        )
        values = history[[step_key, VALIDATION_F1]].dropna().copy()
        values = values.sort_values(step_key).drop_duplicates(
            step_key, keep="last"
        )
        result["validation_steps"] = json.dumps(
            values[step_key].astype(float).tolist(), separators=(",", ":")
        )
        result["validation_f1"] = json.dumps(
            values[VALIDATION_F1].astype(float).tolist(),
            separators=(",", ":"),
        )
        result["analysis_error"] = ""
    except Exception as exc:
        result.update(
            {
                "run_name": "",
                "state": "fetch_error",
                "test_f1": None,
                "validation_steps": "[]",
                "validation_f1": "[]",
                "analysis_error": str(exc),
            }
        )
    return result


def smoothed_history(row: pd.Series) -> pd.DataFrame:
    values = pd.DataFrame(
        {
            "step": json.loads(row.validation_steps),
            "f1": json.loads(row.validation_f1),
        }
    )
    values["smoothed_f1"] = values.f1.rolling(3, min_periods=3).median()
    return values.dropna(subset=["smoothed_f1"]).reset_index(drop=True)


def matched_endpoint(
    history: pd.DataFrame, threshold: float
) -> tuple[float, bool]:
    reached = history.smoothed_f1.ge(threshold)
    stable = (
        reached
        & reached.shift(-1, fill_value=False)
        & reached.shift(-2, fill_value=False)
    )
    if stable.any():
        return float(history.loc[stable.idxmax(), "step"]), False
    return float(history.step.iloc[-1]), True


def summarize_cache() -> None:
    paths = sorted(CSV_DIR.glob(f"{PREFIX}_runs_*.csv"))
    if not paths:
        raise FileNotFoundError("No chunk caches found")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    runs = runs.drop_duplicates("cell_id", keep="last")
    scratch = runs[runs.condition.eq("scratch")].copy()
    thresholds: dict[tuple[str, str, int], float] = {}
    for _, row in scratch.iterrows():
        history = smoothed_history(row)
        thresholds[(row.species, row.recording, int(row.target_seed))] = (
            0.9 * float(history.smoothed_f1.max())
        )
    endpoint_steps: list[float | None] = []
    censored: list[bool | None] = []
    for _, row in runs.iterrows():
        key = (row.species, row.recording, int(row.target_seed))
        history = smoothed_history(row)
        if key not in thresholds or history.empty:
            endpoint_steps.append(None)
            censored.append(None)
            continue
        step, is_censored = matched_endpoint(history, thresholds[key])
        endpoint_steps.append(step)
        censored.append(is_censored)
    runs["matched_quality_step"] = endpoint_steps
    runs["matched_quality_censored"] = censored
    output = CSV_DIR / f"{PREFIX}_audited_runs.csv"
    runs.to_csv(output, index=False)
    summary = (
        runs.groupby(["species", "condition"], dropna=False)
        .agg(
            runs=("run_id", "size"),
            finished=(
                "state",
                lambda values: int((values == "finished").sum()),
            ),
            mean_test_f1=("test_f1", "mean"),
            censored_fraction=("matched_quality_censored", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-design", action="store_true")
    parser.add_argument(
        "--fetch-chunk", nargs=2, type=int, metavar=("OFFSET", "LIMIT")
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--summarize-cache", action="store_true")
    args = parser.parse_args()
    design = expected_runs()
    if args.audit_design or not (args.fetch_chunk or args.summarize_cache):
        print(
            design.groupby(["phase", "species", "condition"])
            .size()
            .rename("runs")
            .to_string()
        )
    if args.fetch_chunk:
        offset, limit = args.fetch_chunk
        selected = design.iloc[offset : offset + limit].to_dict("records")
        api = wandb.Api(timeout=60)
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, api, row) for row in selected]
            for future in as_completed(futures):
                records.append(future.result())
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        output = CSV_DIR / f"{PREFIX}_runs_{offset:04d}.csv"
        pd.DataFrame(records).sort_values("cell_id").to_csv(output, index=False)
        print(f"wrote {len(records)} rows to {output}")
    if args.summarize_cache:
        summarize_cache()


if __name__ == "__main__":
    main()

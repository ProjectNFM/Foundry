"""Fetch and summarize W&B system metrics for the Phase 4E runtime audit."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import wandb

from _wandb_utils import csv_dir


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "20260914-MS-phase4e-runtime-packing-audit"
DEFAULT_MANIFEST = ROOT / "launch/phase4e/runtime-audit/manifest.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _series_stat(events: pd.DataFrame, key: str, operation: str) -> float:
    if key not in events:
        return math.nan
    values = pd.to_numeric(events[key], errors="coerce").dropna()
    if values.empty:
        return math.nan
    if operation == "mean":
        return float(values.mean())
    if operation == "max":
        return float(values.max())
    if operation == "delta":
        return float(values.max() - values.min())
    raise ValueError(operation)


def _fetch_run(
    api: Any,
    entity: str,
    project: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    row = spec["row"]
    run_id = str(row["wandb_run_id"])
    result = {
        **{key: value for key, value in spec.items() if key != "row"},
        "cell_id": row["cell_id"],
        "run_id": run_id,
        "condition_id": row["condition_id"],
        "target_recording": row["target_recording"],
        "target_seed": row["target_finetuning_seed"],
        "source_selection_seed": row["source_selection_seed"],
    }
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        return {**result, "state": "missing", "error": str(exc)}

    summary = run.summary
    events = run.history(samples=10_000, stream="events", pandas=True)
    return {
        **result,
        "state": run.state,
        "runtime_s": summary.get("_runtime"),
        "peak_cuda_allocated_gb": summary.get(
            "compute/peak_memory_allocated_gb"
        ),
        "peak_cuda_reserved_gb": summary.get("compute/peak_memory_reserved_gb"),
        "gpu_util_mean_pct": _series_stat(events, "system.gpu.0.gpu", "mean"),
        "gpu_util_max_pct": _series_stat(events, "system.gpu.0.gpu", "max"),
        "gpu_memory_bytes_max": _series_stat(
            events, "system.gpu.0.memoryAllocatedBytes", "max"
        ),
        "process_rss_mb_max": _series_stat(
            events, "system.proc.memory.rssMB", "max"
        ),
        "host_memory_pct_max": _series_stat(
            events, "system.memory_percent", "max"
        ),
        "host_cpu_mean_pct": _series_stat(events, "system.cpu", "mean"),
        "host_cpu_max_pct": _series_stat(events, "system.cpu", "max"),
        "disk_read_delta": _series_stat(events, "system.disk.dm-0.in", "delta"),
        "disk_write_delta": _series_stat(
            events, "system.disk.dm-0.out", "delta"
        ),
        "network_recv_delta": _series_stat(
            events, "system.network.recv", "delta"
        ),
        "network_sent_delta": _series_stat(
            events, "system.network.sent", "delta"
        ),
        "event_samples": len(events),
        "error": "",
    }


def _specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in manifest["entries"]:
        rows = _load_jsonl(ROOT / entry["cell_list"])
        counts: dict[float, int] = {}
        for row in rows:
            fraction = float(row["target_fraction"])
            counts[fraction] = counts.get(fraction, 0) + 1
            run_id = str(row["wandb_run_id"])
            if run_id in seen:
                raise ValueError(f"duplicate audit W&B run ID: {run_id}")
            seen.add(run_id)
            specs.append(
                {
                    "label": entry["label"],
                    "species": entry["species"],
                    "target_fraction": fraction,
                    "tasks_per_node": entry["tasks_per_node"],
                    "cpus_per_task": entry["cpus_per_task"],
                    "num_workers": entry["num_workers"],
                    "row": row,
                }
            )
        unexpected = {
            fraction: count
            for fraction, count in counts.items()
            if count != int(entry["tasks_per_node"])
        }
        if unexpected:
            raise ValueError(
                f"{entry['label']} does not contain exactly one packed "
                f"allocation per fraction: {unexpected}"
            )
    return specs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "label",
        "species",
        "target_fraction",
        "tasks_per_node",
        "cpus_per_task",
        "num_workers",
    ]
    for values, group in runs.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, values, strict=True))
        finished = group[group.state == "finished"]
        complete = len(finished) == len(group)
        wall_s = pd.to_numeric(finished.runtime_s, errors="coerce").max()
        record.update(
            {
                "expected_cells": len(group),
                "finished_cells": len(finished),
                "allocation_wall_s": wall_s,
                "cells_per_gpu_hour": (
                    3600.0 * len(finished) / wall_s
                    if complete and wall_s and math.isfinite(wall_s)
                    else math.nan
                ),
                "median_run_s": pd.to_numeric(
                    finished.runtime_s, errors="coerce"
                ).median(),
                "max_run_s": wall_s,
            }
        )
        for key in (
            "peak_cuda_allocated_gb",
            "peak_cuda_reserved_gb",
            "gpu_memory_bytes_max",
            "process_rss_mb_max",
            "host_memory_pct_max",
            "gpu_util_max_pct",
            "host_cpu_max_pct",
        ):
            record[f"max_{key}"] = pd.to_numeric(
                finished.get(key), errors="coerce"
            ).max()
        for key in ("gpu_util_mean_pct", "host_cpu_mean_pct"):
            record[f"median_{key}"] = pd.to_numeric(
                finished.get(key), errors="coerce"
            ).median()
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    specs = _specs(manifest)
    api = wandb.Api(timeout=120)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _fetch_run,
                api,
                manifest["entity"],
                manifest["project"],
                spec,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            results.append(future.result())

    runs = pd.DataFrame(results).sort_values(
        ["species", "target_fraction", "label", "cell_id"]
    )
    summary = summarize(runs)
    output = csv_dir(__file__)
    runs_path = output / f"{PREFIX}_runs.csv"
    summary_path = output / f"{PREFIX}_allocations.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Runs CSV: {runs_path}")
    print(f"Allocations CSV: {summary_path}")


if __name__ == "__main__":
    main()

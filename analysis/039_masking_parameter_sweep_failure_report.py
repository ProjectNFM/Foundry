"""Collect failure evidence for the 2026-08-11 masking-parameter sweep.

This is deliberately a failure inventory, not a root-cause investigation. It
fetches the 90 expected downstream W&B runs, identifies non-finished runs, and
records their state, last logged progress, launcher/job metadata, and the tail
of any small text logs uploaded to W&B.

Usage:
    uv run python analysis/039_masking_parameter_sweep_failure_report.py
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb


PROJECT = "foundry_finetuning"
PRETRAIN_GROUP = "MASKING_SEQLEN_LEAK_FIXED"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = OUTPUT_DIR / "039_masking_parameter_sweep_failures.csv"
REPORT_PATH = OUTPUT_DIR / "039_masking_parameter_sweep_failure_report.md"

PRETRAIN_RUNS = (
    "pretrain_M0_baseline_leak_fixed",
    "pretrain_M1_ratio70_leak_fixed",
    "pretrain_M2_ratio80_leak_fixed",
    "pretrain_M3_ratio90_leak_fixed",
    "pretrain_M4_block20_leak_fixed",
)
DOWNSTREAM_GROUPS = (
    "KEMP_FT_DATA_SCALING",
    "KEMP_LP_DATA_SCALING",
    "PHYSIONET_FT_DATA_SCALING",
    "PHYSIONET_LP_DATA_SCALING",
    "BI_P300_FT_DATA_SCALING",
    "BI_P300_LP_DATA_SCALING",
)
TEXT_LOG_SUFFIXES = (".log", ".out", ".err")
ERROR_PATTERN = re.compile(
    r"(?:Traceback \(most recent call last\):|\b(?:Exception|Error):|"
    r"CUDA out of memory|OutOfMemory|oom-kill|Killed|SIG(?:TERM|KILL|SEGV)|"
    r"slurmstepd: error)",
    re.IGNORECASE,
)


def nested_value(data: dict[str, Any], *paths: str) -> Any:
    """Return the first present dotted path from an untrusted W&B mapping."""
    for path in paths:
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def expected_run_name(run: Any) -> bool:
    return any(pretrain_name in run.name for pretrain_name in PRETRAIN_RUNS)


def infer_pretrain_name(run_name: str) -> str:
    return next(name for name in PRETRAIN_RUNS if name in run_name)


def infer_fold(run_name: str) -> str:
    match = re.search(r"fold(\d+)", run_name)
    return match.group(1) if match else "unknown"


def text_log_tail(run: Any) -> str:
    """Return a short error-bearing tail from W&B-uploaded text logs, if any."""
    candidates = [
        file
        for file in run.files()
        if file.name.lower().endswith(TEXT_LOG_SUFFIXES)
        and file.size < 2_000_000
    ]
    for file in candidates:
        local_path = file.download(replace=True).name
        try:
            text = Path(local_path).read_text(errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        error_indexes = [
            i for i, line in enumerate(lines) if ERROR_PATTERN.search(line)
        ]
        if error_indexes:
            start = max(0, error_indexes[-1] - 2)
            return "\\n".join(lines[start : error_indexes[-1] + 60])[-8_000:]
    return ""


def classify_evidence(state: str, evidence: str) -> str:
    """Describe the observed terminal signal, without inferring a root cause."""
    if state == "running":
        return "W&B still reports running"
    if "timed-out and not checkpointable" in evidence:
        return "Timed out; not checkpointable/requeued"
    if "SIGTERMException" in evidence:
        return "SIGTERM received"
    if "Caught ValueError in DataLoader worker" in evidence:
        return "DataLoader worker ValueError"
    if "Traceback" in evidence:
        return "Python traceback (terminal exception not captured)"
    return "No uploaded error text"


def collect_runs(
    api: wandb.Api,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    path = f"{entity}/{PROJECT}"
    selected: list[Any] = []
    for group in DOWNSTREAM_GROUPS:
        runs = list(api.runs(path, filters={"group": group}))
        group_runs = [run for run in runs if expected_run_name(run)]
        print(f"{group}: {len(group_runs)} matching runs")
        selected.extend(group_runs)

    records: list[dict[str, str]] = []
    for run in sorted(
        selected, key=lambda item: (item.group, item.name, item.id)
    ):
        config = dict(run.config)
        summary = dict(run.summary)
        state = str(run.state)
        failure_evidence = text_log_tail(run) if state != "finished" else ""
        records.append(
            {
                "run_name": str(run.name),
                "run_id": str(run.id),
                "url": str(run.url),
                "group": str(run.group),
                "pretrain_run": infer_pretrain_name(run.name),
                "fold": infer_fold(run.name),
                "state": state,
                "created_at": str(run.created_at),
                "last_step": str(summary.get("_step", "")),
                "last_epoch": str(summary.get("epoch", "")),
                "slurm_job_id": str(
                    nested_value(
                        config, "hydra.job.id", "slurm_job_id", "job_id"
                    )
                    or ""
                ),
                "launcher": str(
                    nested_value(
                        config, "hydra.launcher", "hydra.launcher._target_"
                    )
                    or ""
                ),
                "failure_evidence": failure_evidence.replace("\x00", ""),
                "observed_signal": classify_evidence(state, failure_evidence),
            }
        )

    failures = [record for record in records if record["state"] != "finished"]
    return records, failures


def write_outputs(
    all_runs: list[dict[str, str]], failures: list[dict[str, str]]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(all_runs[0]) if all_runs else ["run_name", "run_id", "state"]
    with CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    state_counts = Counter(record["state"] for record in all_runs)
    signal_counts = Counter(record["observed_signal"] for record in failures)
    lines = [
        "# Masking parameter sweep — downstream failure evidence",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- W&B project: `{PROJECT}`",
        f"- Expected pretraining group: `{PRETRAIN_GROUP}`",
        "- Expected downstream runs: 90 (5 checkpoints × 6 task/mode groups × 3 folds)",
        f"- Matching W&B runs found: {len(all_runs)}",
        f"- Non-finished runs: {len(failures)}",
        "",
        "## Run states",
        "",
        "| State | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {state} | {count} |"
        for state, count in sorted(state_counts.items())
    )
    lines.extend(
        [
            "",
            "## Observed terminal signals",
            "",
            "These are recorded symptoms, not root-cause conclusions.",
            "",
            "| Signal | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {signal} | {count} |"
        for signal, count in sorted(signal_counts.items())
    )
    lines.extend(
        [
            "",
            "## Failure inventory",
            "",
            "This records W&B/launcher evidence only. It does not diagnose or fix the underlying issue.",
            "",
            "| Run | ID | Group | Fold | State | Last step | Observed signal | Evidence |",
            "|---|---|---|---:|---|---:|---|---|",
        ]
    )
    for record in failures:
        evidence = (
            record["failure_evidence"].replace("\n", " <br> ")
            or "No uploaded error text"
        )
        evidence = evidence.replace("|", "\\|")
        lines.append(
            "| [{run_name}]({url}) | `{run_id}` | `{group}` | {fold} | {state} | {last_step} | {observed_signal} | {evidence} |".format(
                **record, evidence=evidence
            )
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    api = wandb.Api()
    all_runs, failures = collect_runs(api)
    write_outputs(all_runs, failures)
    print(
        f"Found {len(all_runs)} matching downstream runs; {len(failures)} non-finished."
    )
    print(f"Failure CSV: {CSV_PATH}")
    print(f"Failure report: {REPORT_PATH}")
    if failures:
        print("\nrun_id\tstate\tgroup\tpretrain_run\tfold\tlast_step")
        for record in failures:
            print(
                "{run_id}\t{state}\t{group}\t{pretrain_run}\t{fold}\t{last_step}".format(
                    **record
                )
            )


if __name__ == "__main__":
    main()

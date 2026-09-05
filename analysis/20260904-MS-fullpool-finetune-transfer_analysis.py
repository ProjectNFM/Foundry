"""Analyze the Phase 4A full-pool full-finetuning transfer gate.

Fetches Phase-4A source/transfer groups and the existing normalized Conv--BiGRU
scratch controls through ``wandb.Api()``, checks the declared 36 + 477 run
matrix, and writes raw, paired, and subject-balanced tables.  The source seed
is averaged within each target-session/finetuning-seed pair before summaries,
so source-model replicates are not misreported as biological replicates.

Usage:
    uv run python analysis/20260904-MS-fullpool-finetune-transfer_analysis.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, unwrap_summary_value


PREFIX = "20260904-MS-fullpool-finetune-transfer"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
SEEDS = {42, 43, 44}
SOURCE_GROUPS = {
    "minipigs": "PHASE4A_FULLPOOL_SOURCE_MINIPIGS",
    "monkeys": "PHASE4A_FULLPOOL_SOURCE_MONKEYS",
}
TRANSFER_GROUPS = {
    "minipigs": "PHASE4A_FULL_FINETUNE_MINIPIGS",
    "monkeys": "PHASE4A_FULL_FINETUNE_MONKEYS",
}
SCRATCH_GROUPS = {
    "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
}
EXPECTED_SOURCE = {"minipigs": 21, "monkeys": 15}
EXPECTED_TRANSFER = {"minipigs": 360, "monkeys": 117}
RUN_COLUMNS = [
    "kind",
    "species",
    "run_id",
    "run_name",
    "state",
    "recording",
    "subject",
    "target_seed",
    "source_seed",
    "source_manifest",
    "checkpoint_manifest",
    "transfer_regime",
    "test_supported_f1",
    "best_val_supported_f1",
    "optimizer_steps",
    "best_step",
    "best_windows",
    "best_flops",
    "wall_time_s",
]


def nested(config: dict[str, Any], *keys: str) -> Any:
    """Read either a nested or Hydra-flattened W&B config field."""
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def summary_scalar(summary: dict[str, Any], key: str, aggregate: str = "max") -> float | None:
    for candidate in (f"{key}.{aggregate}", key):
        value = summary.get(candidate)
        if value is not None:
            try:
                return float(unwrap_summary_value(value, aggregate))
            except (TypeError, ValueError):
                pass
    return None


def recording_id(config: dict[str, Any], name: str) -> str | None:
    ids = nested(config, "data", "dataset_kwargs", "recording_ids")
    if isinstance(ids, list) and ids:
        return str(ids[0])
    value = nested(config, "neurosoft", "recording_id")
    if value:
        return str(value)
    match = re.search(r"(sub-\d+_ses-\d+_task-AcousStim_acq-[A-Za-z]+(?:anest)?_desc-raw)", name)
    return match.group(1) if match else None


def subject_id(recording: str | None, config: dict[str, Any]) -> str | None:
    value = nested(config, "neurosoft", "subject")
    if value:
        return str(value)
    match = re.search(r"(sub-\d+)", recording or "")
    return match.group(1) if match else None


def source_seed(config: dict[str, Any], name: str) -> int | None:
    value = nested(config, "run", "source_selection_seed")
    if value is None:
        value = nested(config, "neurosoft", "source_selection_seed")
    if value is not None:
        return int(value)
    match = re.search(r"selection[-_](42|43|44)", str(nested(config, "source_manifest") or name))
    return int(match.group(1)) if match else None


def fetch_group(api: Any, entity: str, group: str, label: str, species: str) -> list[dict[str, Any]]:
    print(f"Fetching {label}: {group}", flush=True)
    rows: list[dict[str, Any]] = []
    for run in api.runs(f"{entity}/{PROJECT}", filters={"group": group}, per_page=500, lazy=False):
        config = dict(run.config or {})
        summary = dict(run.summary or {})
        name = str(run.name or "")
        record = recording_id(config, name)
        target_seed = nested(config, "run", "seed")
        rows.append(
            {
                "kind": label,
                "species": species,
                "run_id": run.id,
                "run_name": name,
                "state": run.state,
                "recording": record,
                "subject": subject_id(record, config),
                "target_seed": int(target_seed) if target_seed is not None else None,
                "source_seed": source_seed(config, name),
                "source_manifest": nested(config, "source_manifest"),
                "checkpoint_manifest": nested(config, "run", "pretrained_checkpoint_manifest"),
                "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
                "test_supported_f1": summary_scalar(summary, f"test/{TASK}_supported_f1"),
                "best_val_supported_f1": summary_scalar(summary, f"val/{TASK}_supported_f1"),
                "optimizer_steps": summary_scalar(summary, "compute/optimizer_steps", "max"),
                "best_step": summary_scalar(summary, "compute/best_step", "max"),
                "best_windows": summary_scalar(summary, "compute/best_windows", "max"),
                "best_flops": summary_scalar(summary, "compute/best_flops", "max"),
                "wall_time_s": summary_scalar(summary, "compute/wall_time_s", "max"),
            }
        )
    return rows


def completeness(source: pd.DataFrame, transfer: pd.DataFrame) -> None:
    print("\n=== Completeness ===")
    for species in SOURCE_GROUPS:
        observed_source = len(source[source.species == species])
        observed_transfer = len(transfer[transfer.species == species])
        print(
            f"{species}: source {observed_source}/{EXPECTED_SOURCE[species]}, "
            f"full finetune {observed_transfer}/{EXPECTED_TRANSFER[species]}"
        )


def paired_effects(transfer: pd.DataFrame, scratch: pd.DataFrame) -> pd.DataFrame:
    """Average source seeds, then compare against identical session/target seeds."""
    finished_transfer = transfer[
        (transfer.state == "finished")
        & (transfer.transfer_regime == "full_finetuning")
        & transfer.recording.notna()
        & transfer.target_seed.isin(SEEDS)
        & transfer.source_seed.isin(SEEDS)
    ].copy()
    finished_scratch = scratch[
        (scratch.state == "finished")
        & scratch.recording.notna()
        & scratch.target_seed.isin(SEEDS)
    ].copy()
    if finished_transfer.empty or finished_scratch.empty:
        return pd.DataFrame()

    unit = ["species", "subject", "recording", "target_seed"]
    pretrained = (
        finished_transfer.groupby(unit, as_index=False)
        .agg(
            pretrain_replicates=("source_seed", "nunique"),
            pretrained_test_supported_f1=("test_supported_f1", "mean"),
            pretrained_best_step=("best_step", "mean"),
            pretrained_best_windows=("best_windows", "mean"),
            pretrained_best_flops=("best_flops", "mean"),
            pretrained_wall_time_s=("wall_time_s", "mean"),
        )
    )
    control = finished_scratch[unit + ["test_supported_f1", "best_step", "best_windows", "best_flops", "wall_time_s"]].rename(
        columns={
            "test_supported_f1": "scratch_test_supported_f1",
            "best_step": "scratch_best_step",
            "best_windows": "scratch_best_windows",
            "best_flops": "scratch_best_flops",
            "wall_time_s": "scratch_wall_time_s",
        }
    )
    paired = pretrained.merge(control, on=unit, how="inner", validate="one_to_one")
    paired["test_f1_gain"] = paired.pretrained_test_supported_f1 - paired.scratch_test_supported_f1
    for metric in ("best_step", "best_windows", "best_flops", "wall_time_s"):
        paired[f"{metric}_saved"] = paired[f"scratch_{metric}"] - paired[f"pretrained_{metric}"]
    return paired


def subject_balanced(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    session_means = paired.groupby(["species", "subject", "recording"], as_index=False).mean(numeric_only=True)
    return session_means.groupby("species", as_index=False).mean(numeric_only=True)


def main() -> None:
    entity = default_entity()
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or configure a default W&B entity.")

    source_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    scratch_rows: list[dict[str, Any]] = []
    for species in SOURCE_GROUPS:
        source_rows += fetch_group(api, entity, SOURCE_GROUPS[species], "source", species)
        transfer_rows += fetch_group(api, entity, TRANSFER_GROUPS[species], "transfer", species)
        scratch_rows += fetch_group(api, entity, SCRATCH_GROUPS[species], "scratch", species)

    # Declaring columns keeps a not-yet-submitted experiment analyzable: the
    # script prints zero completeness rather than failing on an empty group.
    source = pd.DataFrame(source_rows, columns=RUN_COLUMNS)
    transfer = pd.DataFrame(transfer_rows, columns=RUN_COLUMNS)
    scratch = pd.DataFrame(scratch_rows, columns=RUN_COLUMNS)
    completeness(source, transfer)
    paired = paired_effects(transfer, scratch)
    balanced = subject_balanced(paired)

    root = csv_dir(__file__)
    for label, frame in {
        "source": source,
        "transfer": transfer,
        "scratch": scratch,
        "paired": paired,
        "subject_balanced": balanced,
    }.items():
        path = root / f"{PREFIX}_{label}.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")

    if paired.empty:
        print("\nNo complete paired source/transfer/scratch cells yet.")
        return
    print("\n=== Paired test supported macro-F1 effect ===")
    print(paired.groupby("species").test_f1_gain.agg(["count", "mean", "std"]).to_string())
    print("\n=== Subject-balanced effect ===")
    print(balanced[["species", "test_f1_gain", "best_step_saved", "best_flops_saved"]].to_string(index=False))


if __name__ == "__main__":
    main()

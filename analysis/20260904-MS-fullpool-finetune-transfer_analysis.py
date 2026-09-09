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

import json
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
REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "launch/checkpoint_sets/phase4a-mila-best.jsonl"
)
CELL_LISTS = {
    species: Path(__file__).resolve().parents[1]
    / f"launch/phase4a/phase4a-downstream-{species}.jsonl"
    for species in ("minipigs", "monkeys")
}
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
SEEDS = {42, 43, 44}
SOURCE_GROUPS = {
    "minipigs": "NEUROSOFT_SOURCE_PRETRAINING_MINIPIGS",
    "monkeys": "NEUROSOFT_SOURCE_PRETRAINING_MONKEYS",
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
    "cell_id",
    "checkpoint_id",
    "state",
    "recording",
    "subject",
    "target_seed",
    "source_seed",
    "source_model_seed",
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


def summary_scalar(
    summary: dict[str, Any], key: str, aggregate: str = "max"
) -> float | None:
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
    match = re.search(
        r"(sub-\d+_ses-\d+_task-AcousStim_acq-[A-Za-z]+(?:anest)?_desc-raw)",
        name,
    )
    return match.group(1) if match else None


def subject_id(recording: str | None, config: dict[str, Any]) -> str | None:
    value = nested(config, "neurosoft", "subject")
    if value:
        return str(value)
    match = re.search(r"(sub-\d+)", recording or "")
    return match.group(1) if match else None


def source_seed(config: dict[str, Any]) -> int | None:
    value = nested(config, "run", "source_selection_seed")
    return int(value) if value is not None else None


def selected_source_runs() -> dict[str, dict[str, Any]]:
    """Index selected source W&B IDs from the committed checkpoint registry."""
    selected: dict[str, dict[str, Any]] = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        manifest = json.loads(
            Path(record["manifest_path"]).read_text(encoding="utf-8")
        )
        selected[str(manifest["wandb"]["run_id"])] = record
    return selected


def compiled_cells() -> dict[str, dict[str, Any]]:
    """Index the declared downstream matrix by immutable cell identity."""
    selected: dict[str, dict[str, Any]] = {}
    for species, path in CELL_LISTS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["species"] != species:
                raise ValueError(f"{path}: cell species mismatch")
            cell_id = str(record["cell_id"])
            if cell_id in selected:
                raise ValueError(f"Duplicate compiled cell ID: {cell_id}")
            selected[cell_id] = record
    return selected


def is_declared_transfer_run(
    run_id: str,
    run_name: str,
    config: dict[str, Any],
    species: str,
    selected_cells: dict[str, dict[str, Any]],
) -> bool:
    """Require W&B transfer provenance to match one exact compiled cell."""
    cell_id = nested(config, "run", "cell_id")
    cell = selected_cells.get(str(cell_id))
    if cell is None:
        return False
    recording_ids = nested(config, "data", "dataset_kwargs", "recording_ids")
    observed = {
        "run_name": run_name,
        "species": species,
        "target_recording": (
            str(recording_ids[0])
            if isinstance(recording_ids, list) and recording_ids
            else None
        ),
        "target_fraction": nested(config, "data", "training_fraction"),
        "target_finetuning_seed": nested(config, "run", "seed"),
        "checkpoint_id": nested(config, "run", "checkpoint_id"),
        "checkpoint_manifest": nested(
            config, "run", "pretrained_checkpoint_manifest"
        ),
        "checkpoint_manifest_hash": nested(
            config, "run", "pretrained_checkpoint_manifest_hash"
        ),
        "checkpoint_sha256": nested(
            config, "run", "pretrained_checkpoint_sha256"
        ),
        "source_selection_seed": nested(config, "run", "source_selection_seed"),
        "source_model_seed": nested(config, "run", "source_model_seed"),
        "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
    }
    expected = {key: cell[key] for key in observed}
    if observed != expected:
        print(
            f"Skipping undeclared/mismatched transfer run {run_name} ({run_id})",
            flush=True,
        )
        return False
    return True


def fetch_group(
    api: Any,
    entity: str,
    group: str,
    label: str,
    species: str,
    selected_sources: dict[str, dict[str, Any]],
    selected_cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    print(f"Fetching {label}: {group}", flush=True)
    rows: list[dict[str, Any]] = []
    for run in api.runs(
        f"{entity}/{PROJECT}",
        filters={"group": group},
        per_page=500,
        lazy=False,
    ):
        selected = selected_sources.get(str(run.id))
        if label == "source" and selected is None:
            continue
        config = dict(run.config or {})
        summary = dict(run.summary or {})
        name = str(run.name or "")
        if label == "transfer" and not is_declared_transfer_run(
            str(run.id), name, config, species, selected_cells
        ):
            continue
        record = recording_id(config, name)
        target_seed = nested(config, "run", "seed")
        rows.append(
            {
                "kind": label,
                "species": species,
                "run_id": run.id,
                "run_name": name,
                "cell_id": nested(config, "run", "cell_id"),
                "checkpoint_id": nested(config, "run", "checkpoint_id"),
                "state": run.state,
                "recording": record,
                "subject": subject_id(record, config),
                "target_seed": int(target_seed)
                if target_seed is not None
                else None,
                "source_seed": (
                    selected["source_selection_seed"]
                    if selected
                    else source_seed(config)
                ),
                "source_model_seed": (
                    selected["source_model_seed"]
                    if selected
                    else nested(config, "run", "source_model_seed")
                ),
                "source_manifest": nested(config, "source_manifest"),
                "checkpoint_manifest": nested(
                    config, "run", "pretrained_checkpoint_manifest"
                ),
                "transfer_regime": nested(
                    config, "run", "pretrained_transfer_regime"
                ),
                "test_supported_f1": summary_scalar(
                    summary, f"test/{TASK}_supported_f1"
                ),
                "best_val_supported_f1": summary_scalar(
                    summary, f"val/{TASK}_supported_f1"
                ),
                "optimizer_steps": summary_scalar(
                    summary, "compute/optimizer_steps", "max"
                ),
                "best_step": summary_scalar(
                    summary, "compute/best_step", "max"
                ),
                "best_windows": summary_scalar(
                    summary, "compute/best_windows", "max"
                ),
                "best_flops": summary_scalar(
                    summary, "compute/best_flops", "max"
                ),
                "wall_time_s": summary_scalar(
                    summary, "compute/wall_time_s", "max"
                ),
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


def paired_effects(
    transfer: pd.DataFrame, scratch: pd.DataFrame
) -> pd.DataFrame:
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
    pretrained = finished_transfer.groupby(unit, as_index=False).agg(
        pretrain_replicates=("source_seed", "nunique"),
        pretrained_test_supported_f1=("test_supported_f1", "mean"),
        pretrained_best_step=("best_step", "mean"),
        pretrained_best_windows=("best_windows", "mean"),
        pretrained_best_flops=("best_flops", "mean"),
        pretrained_wall_time_s=("wall_time_s", "mean"),
    )
    pretrained = pretrained[pretrained.pretrain_replicates == len(SEEDS)]
    control = finished_scratch[
        unit
        + [
            "test_supported_f1",
            "best_step",
            "best_windows",
            "best_flops",
            "wall_time_s",
        ]
    ].rename(
        columns={
            "test_supported_f1": "scratch_test_supported_f1",
            "best_step": "scratch_best_step",
            "best_windows": "scratch_best_windows",
            "best_flops": "scratch_best_flops",
            "wall_time_s": "scratch_wall_time_s",
        }
    )
    paired = pretrained.merge(
        control, on=unit, how="inner", validate="one_to_one"
    )
    paired["test_f1_gain"] = (
        paired.pretrained_test_supported_f1 - paired.scratch_test_supported_f1
    )
    for metric in ("best_step", "best_windows", "best_flops", "wall_time_s"):
        paired[f"{metric}_saved"] = (
            paired[f"scratch_{metric}"] - paired[f"pretrained_{metric}"]
        )
    return paired


def subject_balanced(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    session_means = paired.groupby(
        ["species", "subject", "recording"], as_index=False
    ).mean(numeric_only=True)
    return session_means.groupby("species", as_index=False).mean(
        numeric_only=True
    )


def main() -> None:
    entity = default_entity()
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError(
            "Set WANDB_ENTITY or configure a default W&B entity."
        )

    source_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    scratch_rows: list[dict[str, Any]] = []
    selected_sources = selected_source_runs()
    selected_cells = compiled_cells()
    for species in SOURCE_GROUPS:
        source_rows += fetch_group(
            api,
            entity,
            SOURCE_GROUPS[species],
            "source",
            species,
            selected_sources,
            selected_cells,
        )
        transfer_rows += fetch_group(
            api,
            entity,
            TRANSFER_GROUPS[species],
            "transfer",
            species,
            selected_sources,
            selected_cells,
        )
        scratch_rows += fetch_group(
            api,
            entity,
            SCRATCH_GROUPS[species],
            "scratch",
            species,
            selected_sources,
            selected_cells,
        )

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
    print(
        paired.groupby("species")
        .test_f1_gain.agg(["count", "mean", "std"])
        .to_string()
    )
    print("\n=== Subject-balanced effect ===")
    print(
        balanced[
            ["species", "test_f1_gain", "best_step_saved", "best_flops_saved"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

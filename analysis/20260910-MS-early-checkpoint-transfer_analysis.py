"""Analyze Phase 4B early source-checkpoint transfer with the W&B API.

Run after the four declared milestone matrices and reused controls are
complete:

    uv run python analysis/20260910-MS-early-checkpoint-transfer_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import (
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
SEEDS = {42, 43, 44}
REGISTRY = ROOT / "launch/checkpoint_sets/phase4a-mila-early-checkpoints.jsonl"
CELL_LISTS = {
    species: ROOT
    / "launch/phase4a"
    / f"phase4a-early-checkpoint-full-ft-lr1p5e3-{species}.jsonl"
    for species in ("minipigs", "monkeys")
}
TRANSFER_GROUPS = {
    "minipigs": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MINIPIGS",
    "monkeys": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MONKEYS",
}
SCRATCH_GROUPS = {
    "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
}
EXPECTED = {"minipigs": 1440, "monkeys": 468}
PREFIX = "20260910-MS-early-checkpoint-transfer"


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


def scalar(summary: dict[str, Any], key: str) -> float | None:
    for candidate in (f"{key}.max", key):
        value = summary.get(candidate)
        try:
            return float(unwrap_summary_value(value, "max"))
        except (TypeError, ValueError):
            pass
    return None


def compiled_cells() -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for species, path in CELL_LISTS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["species"] != species or row["cell_id"] in cells:
                raise RuntimeError(f"Invalid compiled identity in {path}")
            cells[str(row["cell_id"])] = row
    if {
        species: sum(row["species"] == species for row in cells.values())
        for species in EXPECTED
    } != EXPECTED:
        raise RuntimeError(
            "Compiled-cell matrix count differs from the registered design"
        )
    return cells


def milestone_by_checkpoint() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        values[str(row["checkpoint_id"])] = int(
            row["condition"]["milestone_step"]
        )
    if set(values.values()) != {500, 1500, 5000, 15000}:
        raise RuntimeError(
            "Registry does not contain exactly the declared milestones"
        )
    return values


def run_rows(
    api: Any,
    entity: str,
    group: str,
    species: str,
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in api.runs(
        f"{entity}/{PROJECT}",
        filters={"group": group},
        per_page=500,
        lazy=False,
    ):
        config, summary = dict(run.config or {}), dict(run.summary or {})
        cell_id = str(nested(config, "run", "cell_id"))
        expected = cells.get(cell_id)
        if expected is None:
            continue
        observed = {
            "species": species,
            "target_recording": nested(
                config, "data", "dataset_kwargs", "recording_ids"
            ),
            "target_finetuning_seed": nested(config, "run", "seed"),
            "checkpoint_id": nested(config, "run", "checkpoint_id"),
            "checkpoint_manifest_hash": nested(
                config, "run", "pretrained_checkpoint_manifest_hash"
            ),
            "source_selection_seed": nested(
                config, "run", "source_selection_seed"
            ),
            "source_model_seed": nested(config, "run", "source_model_seed"),
            "transfer_regime": nested(
                config, "run", "pretrained_transfer_regime"
            ),
        }
        expected_values = {
            "species": expected["species"],
            "target_recording": [expected["target_recording"]],
            "target_finetuning_seed": expected["target_finetuning_seed"],
            "checkpoint_id": expected["checkpoint_id"],
            "checkpoint_manifest_hash": expected["checkpoint_manifest_hash"],
            "source_selection_seed": expected["source_selection_seed"],
            "source_model_seed": expected["source_model_seed"],
            "transfer_regime": expected["transfer_regime"],
        }
        if (
            observed != expected_values
            or str(run.id) != expected["wandb_run_id"]
        ):
            raise RuntimeError(
                f"W&B provenance mismatch: {run.name} ({run.id})"
            )
        rows.append(
            {
                "run_id": str(run.id),
                "run_name": str(run.name),
                "state": run.state,
                "cell_id": cell_id,
                "species": species,
                "subject": expected["target_subject"],
                "recording": expected["target_recording"],
                "target_seed": expected["target_finetuning_seed"],
                "source_seed": expected["source_selection_seed"],
                "checkpoint_id": expected["checkpoint_id"],
                "test_supported_f1": scalar(
                    summary, f"test/{TASK}_supported_f1"
                ),
                "best_step": scalar(summary, "compute/best_step"),
                "best_windows": scalar(summary, "compute/best_windows"),
                "best_flops": scalar(summary, "compute/best_flops"),
                "best_wall_time_s": scalar(summary, "compute/best_wall_time_s"),
            }
        )
    return rows


def scratch_rows(
    api: Any,
    entity: str,
    group: str,
    species: str,
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = {
        (row["target_subject"], row["target_recording"])
        for row in cells.values()
        if row["species"] == species
    }
    rows: list[dict[str, Any]] = []
    for run in api.runs(
        f"{entity}/{PROJECT}",
        filters={"group": group},
        per_page=500,
        lazy=False,
    ):
        config, summary = dict(run.config or {}), dict(run.summary or {})
        recordings = nested(config, "data", "dataset_kwargs", "recording_ids")
        recording = (
            str(recordings[0])
            if isinstance(recordings, list) and recordings
            else None
        )
        subject = next(
            (item[0] for item in targets if item[1] == recording), None
        )
        seed, fraction = (
            nested(config, "run", "seed"),
            nested(config, "data", "training_fraction"),
        )
        if (
            run.state != "finished"
            or (subject, recording) not in targets
            or seed not in SEEDS
            or fraction is None
            or float(fraction) != 1.0
        ):
            continue
        rows.append(
            {
                "species": species,
                "subject": subject,
                "recording": recording,
                "target_seed": int(seed),
                "run_id": str(run.id),
                "run_name": str(run.name),
                "test_supported_f1": scalar(
                    summary, f"test/{TASK}_supported_f1"
                ),
                "best_step": scalar(summary, "compute/best_step"),
                "best_windows": scalar(summary, "compute/best_windows"),
                "best_flops": scalar(summary, "compute/best_flops"),
                "best_wall_time_s": scalar(summary, "compute/best_wall_time_s"),
            }
        )
    return rows


def paired_subject_effects(
    transfer: pd.DataFrame, scratch: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "test_supported_f1",
        "best_step",
        "best_windows",
        "best_flops",
        "best_wall_time_s",
    ]
    unit = ["species", "subject", "recording", "target_seed", "milestone_step"]
    means = transfer.groupby(unit, as_index=False).agg(
        source_replicates=("source_seed", "nunique"),
        **{metric: (metric, "mean") for metric in metrics},
    )
    if not means.source_replicates.eq(3).all():
        raise RuntimeError(
            "Each milestone/session/target-seed pair requires all three source seeds"
        )
    control = scratch.rename(
        columns={metric: f"scratch_{metric}" for metric in metrics}
    )
    paired = means.merge(
        control,
        on=["species", "subject", "recording", "target_seed"],
        how="inner",
        validate="one_to_one",
    )
    paired["test_f1_gain"] = (
        paired.test_supported_f1 - paired.scratch_test_supported_f1
    )
    for metric in metrics[1:]:
        paired[f"{metric}_saved_pct"] = (
            100
            * (paired[f"scratch_{metric}"] - paired[metric])
            / paired[f"scratch_{metric}"]
        )
    sessions = paired.groupby(
        ["species", "milestone_step", "subject", "recording"], as_index=False
    ).mean(numeric_only=True)
    subjects = sessions.groupby(
        ["species", "milestone_step", "subject"], as_index=False
    ).mean(numeric_only=True)
    return paired, subjects


def plot(subjects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        data = subjects[subjects.species.eq(species)]
        summary = data.groupby("milestone_step", as_index=False).mean(
            numeric_only=True
        )
        axis.plot(
            summary.milestone_step,
            summary.test_f1_gain,
            marker="o",
            label="test macro-F1 gain",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            title=species.title(),
            xscale="symlog",
            xlabel="Source checkpoint step",
            ylabel="Transfer − scratch test macro-F1",
        )
        axis.grid(alpha=0.25)
    fig.suptitle("Paired, subject-balanced early-checkpoint transfer effect")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    api = wandb.Api()
    entity = default_entity() or api.default_entity
    if not entity:
        raise RuntimeError(
            "Set WANDB_ENTITY or configure a default W&B entity."
        )
    cells, milestones = compiled_cells(), milestone_by_checkpoint()
    transfer_rows, scratch = [], []
    for species in TRANSFER_GROUPS:
        transfer_rows += run_rows(
            api, entity, TRANSFER_GROUPS[species], species, cells
        )
        scratch += scratch_rows(
            api, entity, SCRATCH_GROUPS[species], species, cells
        )
    transfer, scratch_frame = pd.DataFrame(transfer_rows), pd.DataFrame(scratch)
    if {
        species: int(transfer.species.eq(species).sum()) for species in EXPECTED
    } != EXPECTED:
        raise RuntimeError(
            "W&B does not yet contain exactly the declared milestone matrix"
        )
    if (
        not transfer.state.eq("finished").all()
        or transfer[["test_supported_f1", "best_step"]].isna().any().any()
    ):
        raise RuntimeError(
            "All declared transfer runs must finish with test F1 and best-step accounting"
        )
    transfer["milestone_step"] = transfer.checkpoint_id.map(milestones)
    expected_scratch = {"minipigs": 120, "monkeys": 39}
    if {
        species: int(scratch_frame.species.eq(species).sum())
        for species in expected_scratch
    } != expected_scratch:
        raise RuntimeError(
            "Matched Phase-2 scratch controls are incomplete or duplicated"
        )
    paired, subjects = paired_subject_effects(transfer, scratch_frame)
    summary = subjects.groupby(
        ["species", "milestone_step"], as_index=False
    ).mean(numeric_only=True)
    summary["n_subjects"] = (
        subjects.groupby(["species", "milestone_step"]).size().to_numpy()
    )
    output_csv, output_figures = csv_dir(__file__), figures_dir(__file__)
    for name, frame in {
        "transfer": transfer,
        "scratch": scratch_frame,
        "paired": paired,
        "subject_effects": subjects,
        "subject_balanced_summary": summary,
    }.items():
        frame.to_csv(output_csv / f"{PREFIX}_{name}.csv", index=False)
    plot(subjects, output_figures / f"{PREFIX}_subject_balanced_test_f1.png")
    print(
        summary[
            [
                "species",
                "milestone_step",
                "n_subjects",
                "test_f1_gain",
                "best_step_saved_pct",
                "best_windows_saved_pct",
                "best_flops_saved_pct",
                "best_wall_time_s_saved_pct",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()

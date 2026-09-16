#!/usr/bin/env python3
"""Generate the audited 60-cell batch-128 NeuroSoft source matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"minipigs": range(1, 8), "monkeys": range(1, 6)}
GROUPS = {
    "minipigs": "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-B128-MINIPIGS",
    "monkeys": "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-B128-MONKEYS",
}
CONDITIONS: dict[str, dict[str, Any]] = {
    "reference_backbone": {
        "input_adapter_mode": "per_session",
        "input_adapter_bias": True,
        "shared_input_channels": 32,
        "temporal_channels": 128,
        "gru_hidden_size": 128,
        "transferable_parameter_count": 507456,
    },
    "small_backbone": {
        "input_adapter_mode": "per_session",
        "input_adapter_bias": True,
        "shared_input_channels": 32,
        "temporal_channels": 38,
        "gru_hidden_size": 38,
        "transferable_parameter_count": 51066,
    },
    "large_backbone": {
        "input_adapter_mode": "per_session",
        "input_adapter_bias": True,
        "shared_input_channels": 32,
        "temporal_channels": 410,
        "gru_hidden_size": 410,
        "transferable_parameter_count": 5084598,
    },
    "bias_free": {
        "input_adapter_mode": "per_session",
        "input_adapter_bias": False,
        "shared_input_channels": 32,
        "temporal_channels": 128,
        "gru_hidden_size": 128,
        "transferable_parameter_count": 507456,
    },
    "shared_padded": {
        "input_adapter_mode": "shared_padded",
        "input_adapter_bias": True,
        "shared_input_channels": 32,
        "temporal_channels": 128,
        "gru_hidden_size": 128,
        "transferable_parameter_count": 507456,
    },
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_records(repo_root: Path, species: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition_id, model in CONDITIONS.items():
        for number in TARGETS[species]:
            subject = f"sub-{number:02d}"
            relative_manifest = (
                Path("manifests/neurosoft_supervised/v1/source_volume")
                / species
                / f"target-{subject}"
                / "fraction-1.00"
                / "selection-42.json"
            )
            manifest_path = repo_root / relative_manifest
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            source_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            expected = (species, subject, 42)
            observed = (
                source_manifest.get("target_species"),
                source_manifest.get("target_subject"),
                source_manifest.get("condition", {}).get(
                    "source_selection_seed"
                ),
            )
            if observed != expected:
                raise ValueError(
                    f"{manifest_path}: source identity {observed!r} != {expected!r}"
                )
            run_name = f"arch-{condition_id}-{species}-{subject}-b128-s42-m42"
            cell_id = (
                f"architecture_viability__{condition_id}__{species}__{subject}"
            )
            model_overrides = [
                f"model.input_adapter_mode={model['input_adapter_mode']}",
                f"model.input_adapter_bias={str(model['input_adapter_bias']).lower()}",
                f"model.shared_input_channels={model['shared_input_channels']}",
                f"model.temporal_channels={model['temporal_channels']}",
                f"model.gru_hidden_size={model['gru_hidden_size']}",
            ]
            if model["input_adapter_mode"] == "shared_padded":
                adapter_parameters = model["shared_input_channels"] * 64 + (
                    64 if model["input_adapter_bias"] else 0
                )
            else:
                adapter_parameters = sum(
                    int(recording["supported_channel_count"]) * 64
                    + (64 if model["input_adapter_bias"] else 0)
                    for recording in source_manifest["recordings"]
                )
            router_parameters = 16 * model["gru_hidden_size"] + 8
            total_parameter_count = (
                model["transferable_parameter_count"]
                + adapter_parameters
                + router_parameters
            )
            overrides = [
                f"source_manifest={relative_manifest.as_posix()}",
                f"run.source_target_subject={subject}",
                "run.seed=42",
                f"run.name={run_name}",
                f"run.group={GROUPS[species]}",
                f"++run.source_condition={condition_id}",
                f"run.tags=[source-pretraining,{species},8band,architecture-viability,{condition_id}]",
                "trainer.max_steps=10000",
                "trainer.val_check_interval=100",
                "+trainer.check_val_every_n_epoch=null",
                "hyperparameters.batch_size=128",
                "trainer.callbacks.early_stopping=null",
                "trainer.callbacks.model_checkpoint.monitor=val/loss",
                "trainer.callbacks.model_checkpoint.mode=min",
                "trainer.callbacks.compute_tracking.monitor=val/loss",
                "trainer.callbacks.compute_tracking.mode=min",
                "+trainer.callbacks.compute_milestones.milestone_fractions=[0.01,0.03,0.1,0.3,1.0]",
                *model_overrides,
            ]
            records.append(
                {
                    "cell_id": cell_id,
                    "condition_id": condition_id,
                    "species": species,
                    "target_subject": subject,
                    "source_selection_seed": 42,
                    "source_model_seed": 42,
                    "source_manifest": relative_manifest.as_posix(),
                    "source_manifest_hash": source_manifest["manifest_hash"],
                    "wandb_group": GROUPS[species],
                    "run_name": run_name,
                    "fixed_milestones": [100, 300, 1000, 3000, 10000],
                    "loss_selected_checkpoint": True,
                    "model_metadata": {
                        **model,
                        "adapter_dim": 64,
                        "temporal_kernel_samples": 64,
                        "temporal_stride": 4,
                        "conv_depth": 1,
                        "gru_num_layers": 2,
                        "gru_bidirectional": True,
                        "total_parameter_count": total_parameter_count,
                    },
                    "overrides": overrides,
                }
            )
    return records


def write_outputs(
    output_dir: Path, by_species: dict[str, list[dict[str, Any]]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for species, records in by_species.items():
        payload = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            for row in records
        )
        output = output_dir / f"source-{species}.jsonl"
        output.write_bytes(payload)
        lock = {
            "schema": "neurosoft-architecture-source-cell-lock",
            "version": 1,
            "output": output.name,
            "sha256": _digest(payload),
            "records": len(records),
            "conditions": dict(
                sorted(Counter(r["condition_id"] for r in records).items())
            ),
        }
        output.with_suffix(".lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "launch/architecture_viability",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    by_species = {
        species: build_records(args.repo_root.resolve(), species)
        for species in TARGETS
    }
    expected = {"minipigs": 35, "monkeys": 25}
    actual = {species: len(rows) for species, rows in by_species.items()}
    if actual != expected or sum(actual.values()) != 60:
        raise AssertionError(f"Unexpected source cell counts: {actual}")
    for species, rows in by_species.items():
        if len({row["cell_id"] for row in rows}) != len(rows):
            raise ValueError(f"{species}: duplicate source cell IDs")
    if not args.check:
        write_outputs(args.output_dir.resolve(), by_species)
    print(json.dumps({"counts": actual, "total": 60}, sort_keys=True))


if __name__ == "__main__":
    main()

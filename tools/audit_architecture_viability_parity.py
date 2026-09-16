#!/usr/bin/env python3
"""Fail closed on unintended Phase 4E/4F configuration differences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from foundry.config_resolvers import register_resolvers


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ALLOWED_PREFIXES = (
    "model.input_adapter_",
    "model.shared_input_channels",
    "model.temporal_channels",
    "model.gru_hidden_size",
    "run.name",
    "run.group",
    "run.tags",
    "run.source_condition",
    "logger.name",
    "logger.group",
    "logger.tags",
    "trainer.logger.name",
    "trainer.logger.group",
    "trainer.logger.tags",
    "trainer.callbacks.compute_milestones.milestone_fractions",
    "trainer.check_val_every_n_epoch",
    "hyperparameters.batch_size",
    "data.batch_size",
)
DOWNSTREAM_ALLOWED_PREFIXES = (
    "model.input_adapter_",
    "model.shared_input_channels",
    "model.temporal_channels",
    "model.gru_hidden_size",
    "run.",
    "logger.name",
    "logger.group",
    "logger.tags",
    "logger.id",
    "trainer.logger.name",
    "trainer.logger.group",
    "trainer.logger.tags",
    "trainer.logger.id",
)
DOWNSTREAM_RECIPE_INVARIANTS = (
    "audit_path",
    "eligible_session_selection",
    "base_experiments",
    "target_training_fractions",
    "target_finetuning_seeds",
    "evaluate_test",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def _compose(experiment: str, overrides: list[str]) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    register_resolvers()
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment}", *overrides],
        )
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def audit_source_cells() -> dict[str, int]:
    counts: dict[str, int] = {}
    for species in ("minipigs", "monkeys"):
        reference_rows = _load_jsonl(
            ROOT / f"launch/phase4e/phase4e-source-{species}.jsonl"
        )
        references = {
            row["target_subject"]: row
            for row in reference_rows
            if row["source_selection_seed"] == 42
            and row["source_model_seed"] == 42
        }
        rows = _load_jsonl(
            ROOT / f"launch/architecture_viability/source-{species}.jsonl"
        )
        experiment = f"pretraining/neurosoft_conv_bigru_supervised_{species}"
        for row in rows:
            new = _flatten(_compose(experiment, row["overrides"]))
            ref = _flatten(
                _compose(
                    experiment, references[row["target_subject"]]["overrides"]
                )
            )
            differing = sorted(
                key
                for key in set(new) | set(ref)
                if new.get(key) != ref.get(key)
                and not key.startswith(SOURCE_ALLOWED_PREFIXES)
            )
            if differing:
                raise ValueError(
                    f"{row['cell_id']}: unexpected Phase 4E differences: {differing}"
                )
            if new.get(
                "trainer.callbacks.compute_milestones.milestone_fractions"
            ) != [0.01, 0.03, 0.1, 0.3, 1.0]:
                raise ValueError(
                    f"{row['cell_id']}: fixed milestone schedule drift"
                )
            if new.get("hyperparameters.batch_size") != 128:
                raise ValueError(f"{row['cell_id']}: batch size drift")
            if new.get("data.batch_size") != 128:
                raise ValueError(f"{row['cell_id']}: data batch size drift")
            if new.get("trainer.check_val_every_n_epoch") is not None:
                raise ValueError(
                    f"{row['cell_id']}: epoch validation must be disabled"
                )
        counts[species] = len(rows)
    return counts


def audit_downstream_recipes() -> dict[str, str]:
    reference = yaml.safe_load(
        (
            ROOT
            / "configs/downstream_recipes/phase4f_checkpoint_trajectory.yaml"
        ).read_text()
    )
    recipes = {
        "small_backbone": "architecture_small_backbone.yaml",
        "large_backbone": "architecture_large_backbone.yaml",
        "bias_free": "architecture_bias_free.yaml",
        "shared_adapter": "architecture_shared_adapter.yaml",
    }
    results: dict[str, str] = {}
    audit = json.loads((ROOT / "docs/neurosoft-phase0-audit.json").read_text())
    target_by_species = {
        species: next(
            row["recording_id"]
            for row in audit["recordings"]
            if row.get("eligible") and row["species"] == species
        )
        for species in ("minipigs", "monkeys")
    }

    def deduplicate(overrides: list[str]) -> list[str]:
        values: dict[str, str] = {}
        for override in overrides:
            key = override.split("=", 1)[0].lstrip("+")
            values[key] = override
        return list(values.values())

    for condition, filename in recipes.items():
        recipe = yaml.safe_load(
            (ROOT / "configs/downstream_recipes" / filename).read_text()
        )
        for key in DOWNSTREAM_RECIPE_INVARIANTS:
            if recipe.get(key) != reference.get(key):
                raise ValueError(
                    f"{condition}: downstream invariant {key!r} differs from Phase 4F"
                )
        reference_fixed = {
            value
            for value in reference["fixed_overrides"]
            if not value.startswith(
                (
                    "hyperparameters.learning_rate=",
                    "hyperparameters.backbone_learning_rate=",
                    "hyperparameters.backbone_components=",
                    "hyperparameters.adapter_warmup_steps=",
                )
            )
        }
        if set(recipe.get("fixed_overrides", [])) != reference_fixed:
            raise ValueError(
                f"{condition}: scheduler/warmup fixed overrides drift"
            )
        if recipe.get("learning_rates") != [0.003]:
            raise ValueError(f"{condition}: downstream learning rate drift")
        for entry in recipe.get("condition_matrix", []):
            if entry.get("source") == "pretrained":
                if entry.get("backbone_lr_multiplier") != 0.1:
                    raise ValueError(
                        f"{condition}: backbone LR multiplier drift"
                    )
                if entry.get("adapter_warmup_steps", 0) != 0:
                    raise ValueError(f"{condition}: adapter warmup drift")
            for species, recording_id in target_by_species.items():
                common = [
                    f"data.dataset_kwargs.recording_ids=[{recording_id}]",
                    "data.training_fraction=1.0",
                    "run.seed=42",
                    "run.evaluate_test=true",
                ]
                is_pretrained = entry.get("source") == "pretrained"
                regime = entry.get("transfer_regime")
                new_overrides = [
                    *recipe["fixed_overrides"],
                    "hyperparameters.learning_rate=0.003",
                    "hyperparameters.backbone_learning_rate="
                    + ("0.0003" if is_pretrained else "null"),
                    "hyperparameters.backbone_components="
                    + ("[temporal_frontend,gru]" if is_pretrained else "null"),
                    "hyperparameters.adapter_warmup_steps=0",
                    *[
                        f"model.{key}={str(value).lower() if isinstance(value, bool) else value}"
                        for key, value in entry.get(
                            "model_overrides", {}
                        ).items()
                    ],
                    *common,
                    f"run.pretrained_transfer_regime={regime or 'null'}",
                    "run.pretrained_checkpoint_manifest="
                    + ("/synthetic/manifest.json" if is_pretrained else "null"),
                ]
                reference_overrides = [
                    *reference["fixed_overrides"],
                    *common,
                    "run.pretrained_transfer_regime="
                    + (
                        "full_finetuning_reset_router"
                        if is_pretrained
                        else "null"
                    ),
                    "run.pretrained_checkpoint_manifest="
                    + ("/synthetic/manifest.json" if is_pretrained else "null"),
                ]
                if not is_pretrained:
                    reference_overrides.extend(
                        [
                            "hyperparameters.backbone_learning_rate=null",
                            "hyperparameters.backbone_components=null",
                        ]
                    )
                experiment = recipe["base_experiments"][species]
                new_cfg = _flatten(
                    _compose(experiment, deduplicate(new_overrides))
                )
                reference_cfg = _flatten(
                    _compose(experiment, deduplicate(reference_overrides))
                )
                differing = sorted(
                    key
                    for key in set(new_cfg) | set(reference_cfg)
                    if new_cfg.get(key) != reference_cfg.get(key)
                    and not key.startswith(DOWNSTREAM_ALLOWED_PREFIXES)
                )
                if differing:
                    raise ValueError(
                        f"{condition}/{entry['id']}/{species}: unexpected "
                        f"resolved Phase 4F differences: {differing}"
                    )
        results[condition] = "phase4f-parity-ok"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("FOUNDRY_DATA_ROOT", "/synthetic/processed")
    result = {
        "source_cells": audit_source_cells(),
        "downstream_recipes": audit_downstream_recipes(),
        "status": "ok",
    }
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))


if __name__ == "__main__":
    main()

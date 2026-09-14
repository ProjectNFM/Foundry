"""Compile checkpoint registries and downstream recipes into Hydra cell lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA = "foundry-downstream-recipe"
VERSION = 1
COMPILER_VERSION = "1"
CHECKPOINT_MANIFEST_SCHEMA = "neurosoft-pretraining-checkpoint"
CHECKPOINT_MANIFEST_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    """Load and self-hash-verify a checkpoint manifest without training imports."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unexpected checkpoint manifest schema")
    if manifest.get("version") != CHECKPOINT_MANIFEST_VERSION:
        raise ValueError(f"{path}: unexpected checkpoint manifest version")
    recorded = manifest.get("manifest_hash")
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    actual = _digest(payload)
    if recorded != actual:
        raise ValueError(
            f"{path}: manifest hash mismatch; expected {recorded}, got {actual}"
        )
    return manifest


def _verify_checkpoint(manifest: dict[str, Any], checkpoint_root: Path) -> None:
    checkpoint = manifest.get("checkpoint", {})
    relative = Path(str(checkpoint.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "Manifest checkpoint.path must stay below checkpoint root"
        )
    path = checkpoint_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    actual = _file_digest(path)
    if actual != checkpoint.get("sha256"):
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path}: expected "
            f"{checkpoint.get('sha256')}, got {actual}"
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        records.append(record)
    if not records:
        raise ValueError(f"Checkpoint registry is empty: {path}")
    return records


def _required(
    record: dict[str, Any], keys: Iterable[str], context: str
) -> None:
    missing = [key for key in keys if record.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _selection_seed(manifest: dict[str, Any]) -> int:
    selection_id = str(
        manifest.get("trained_on", {}).get("source_selection_id", "")
    )
    match = re.search(r"(?:selection-|_sel)(\d+)$", selection_id)
    if not match:
        raise ValueError(
            f"Cannot recover source selection seed from manifest selection ID {selection_id!r}"
        )
    return int(match.group(1))


def _source_model_seed(manifest: dict[str, Any], manifest_path: Path) -> int:
    """Read model seed from a modern manifest or legacy run config."""
    for container in (
        manifest.get("trained_on", {}),
        manifest.get("recipe", {}),
    ):
        if container.get("source_model_seed") is not None:
            return int(container["source_model_seed"])
    legacy_config = manifest_path.parent.parent / ".hydra" / "config.yaml"
    if legacy_config.is_file():
        config = yaml.safe_load(legacy_config.read_text(encoding="utf-8"))
        seed = (
            config.get("run", {}).get("seed")
            if isinstance(config, dict)
            else None
        )
        if seed is not None:
            return int(seed)
    raise ValueError(
        f"{manifest_path}: source_model_seed is absent from the manifest and "
        "no legacy .hydra/config.yaml evidence is available"
    )


def _validate_source_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    source_manifest_root: Path,
    registry_record: dict[str, Any],
) -> None:
    """Cross-check the source selection artifact recorded by a checkpoint."""
    trained_on = manifest.get("trained_on", {})
    recorded_path = trained_on.get("source_manifest_path")
    recorded_hash = trained_on.get("source_manifest_hash")
    if not recorded_path and not recorded_hash:
        # Older/synthetic manifests predate source-selection provenance.
        return
    if not recorded_path or not recorded_hash:
        raise ValueError(
            f"{manifest_path}: incomplete source manifest provenance"
        )

    source_path = Path(str(recorded_path))
    if not source_path.is_absolute():
        source_path = source_manifest_root / source_path
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source selection manifest not found: {source_path}"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    payload = dict(source)
    source_self_hash = payload.pop("manifest_hash", None)
    actual_hash = _digest(payload)
    if source_self_hash != actual_hash or recorded_hash != actual_hash:
        raise ValueError(
            f"{manifest_path}: source selection manifest hash mismatch"
        )

    excluded = trained_on.get("excluded_target", {})
    selection_seed = _selection_seed(manifest)
    condition = source.get("condition", {})
    checks = {
        "target_species": excluded.get("species"),
        "target_subject": excluded.get("subject"),
    }
    for key, expected in checks.items():
        if source.get(key) != expected:
            raise ValueError(
                f"{manifest_path}: source manifest {key} disagrees with checkpoint"
            )
    if condition.get("source_selection_seed") != selection_seed:
        raise ValueError(
            f"{manifest_path}: source manifest selection seed disagrees with checkpoint"
        )
    if source.get("selection_id") != trained_on.get("source_selection_id"):
        raise ValueError(
            f"{manifest_path}: source selection identity disagrees with checkpoint"
        )
    declared_fraction = registry_record.get("condition", {}).get(
        "source_fraction"
    )
    if declared_fraction is not None and float(
        condition.get("requested_fraction")
    ) != float(declared_fraction):
        raise ValueError(
            f"{manifest_path}: source fraction disagrees with registry condition"
        )
    if source.get("source_test_policy") != "forbidden":
        raise ValueError(
            f"{source_path}: source_test_policy must be 'forbidden'"
        )
    if source.get("target_leakage"):
        raise ValueError(
            f"{source_path}: source manifest contains target leakage"
        )

    source_recordings = [
        str(recording.get("canonical_recording_id"))
        for recording in source.get("recordings", [])
    ]
    if source_recordings != list(trained_on.get("recordings", [])):
        raise ValueError(
            f"{manifest_path}: source recording identities disagree with source manifest"
        )
    summary = source.get("summary", {})
    summary_checks = {
        "selected_train_examples": "selected_train_examples",
        "available_train_windows": "available_train_windows",
        "represented_class_intersection": "class_intersection",
        "represented_class_union": "class_union",
    }
    for source_key, checkpoint_key in summary_checks.items():
        if summary.get(source_key) != trained_on.get(checkpoint_key):
            raise ValueError(
                f"{manifest_path}: source {source_key} disagrees with checkpoint"
            )


def validate_registry(
    registry_path: Path,
    checkpoint_root: Path,
    source_manifest_root: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Validate registry declarations against authoritative manifests/checkpoints."""
    records = _load_jsonl(registry_path)
    checkpoint_ids: set[str] = set()
    manifest_paths: set[Path] = set()
    checkpoint_hashes: set[str] = set()
    scientific_identities: set[tuple[Any, ...]] = set()
    set_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    required = (
        "checkpoint_set_id",
        "checkpoint_id",
        "manifest_path",
        "manifest_hash",
        "checkpoint_sha256",
        "species",
        "excluded_target_subject",
        "source_selection_seed",
        "source_model_seed",
        "condition",
    )
    for index, record in enumerate(records, 1):
        context = f"{registry_path} record {index}"
        _required(record, required, context)
        checkpoint_id = str(record["checkpoint_id"])
        if checkpoint_id in checkpoint_ids:
            raise ValueError(f"Duplicate checkpoint_id: {checkpoint_id}")
        checkpoint_ids.add(checkpoint_id)
        set_ids.add(str(record["checkpoint_set_id"]))

        manifest_path = Path(str(record["manifest_path"]))
        if not manifest_path.is_absolute():
            raise ValueError(f"{checkpoint_id}: manifest_path must be absolute")
        manifest = _load_checkpoint_manifest(manifest_path)
        _verify_checkpoint(manifest, checkpoint_root)
        if not isinstance(record["condition"], dict):
            raise ValueError(f"{checkpoint_id}: condition must be an object")
        _validate_source_manifest(
            manifest,
            manifest_path,
            source_manifest_root or Path.cwd(),
            record,
        )
        excluded = manifest.get("trained_on", {}).get("excluded_target", {})
        checks = {
            "manifest_hash": manifest.get("manifest_hash"),
            "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
            "species": excluded.get("species"),
            "excluded_target_subject": excluded.get("subject"),
            "source_selection_seed": _selection_seed(manifest),
            "source_model_seed": _source_model_seed(manifest, manifest_path),
        }
        for key, actual in checks.items():
            if record[key] != actual:
                raise ValueError(
                    f"{checkpoint_id}: registry {key}={record[key]!r} disagrees "
                    f"with manifest value {actual!r}"
                )
        if manifest.get("checkpoint", {}).get("kind") != record[
            "condition"
        ].get("checkpoint_kind"):
            raise ValueError(f"{checkpoint_id}: checkpoint kind mismatch")
        selection_value = manifest.get("selection", {}).get("monitor_value")
        if (
            not isinstance(selection_value, (int, float))
            or not math.isfinite(selection_value)
            or selection_value <= 0
        ):
            raise ValueError(
                f"{checkpoint_id}: source validation metric must be finite and positive"
            )

        resolved_manifest_path = manifest_path.resolve()
        checkpoint_hash = str(record["checkpoint_sha256"])
        scientific_identity = (
            str(record["species"]),
            str(record["excluded_target_subject"]),
            int(record["source_selection_seed"]),
            int(record["source_model_seed"]),
            _digest(record["condition"]),
        )
        if resolved_manifest_path in manifest_paths:
            raise ValueError(f"Duplicate checkpoint manifest: {manifest_path}")
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError(f"Duplicate checkpoint content: {checkpoint_hash}")
        if scientific_identity in scientific_identities:
            raise ValueError(
                f"Duplicate scientific checkpoint identity: {scientific_identity}"
            )
        manifest_paths.add(resolved_manifest_path)
        checkpoint_hashes.add(checkpoint_hash)
        scientific_identities.add(scientific_identity)
        validated.append({**record, "_manifest": manifest})
    if len(set_ids) != 1:
        raise ValueError(
            f"Registry must contain exactly one checkpoint_set_id: {set_ids}"
        )
    return validated, next(iter(set_ids))


def _load_recipe(path: Path) -> dict[str, Any]:
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe root must be an object: {path}")
    if recipe.get("schema") != SCHEMA or recipe.get("version") != VERSION:
        raise ValueError(
            f"Unsupported downstream recipe schema/version: {path}"
        )
    _required(
        recipe,
        (
            "recipe_id",
            "base_experiments",
            "audit_path",
            "eligible_session_selection",
            "transfer_regimes",
            "target_training_fractions",
            "target_finetuning_seeds",
            "species",
        ),
        str(path),
    )
    return recipe


def _load_audit(path: Path) -> tuple[dict[str, Any], str]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    expected = audit.get("artifact_sha256")
    payload = {
        key: value for key, value in audit.items() if key != "artifact_sha256"
    }
    actual = _digest(payload)
    if expected != actual:
        raise ValueError(
            f"Audit artifact hash mismatch: expected {expected}, got {actual}"
        )
    return audit, actual


def _slug_fraction(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p")


def _slug_learning_rate(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p").replace("-", "m")


def _quote_list(value: str) -> str:
    if any(char in value for char in "[], ' \t\n"):
        return "[" + json.dumps(value, ensure_ascii=True) + "]"
    return f"[{value}]"


def _hydra_mapping(value: dict[str, Any]) -> str:
    """Render a flat metadata mapping in Hydra's override grammar."""
    parts: list[str] = []
    for key, item in sorted(value.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", str(key)):
            raise ValueError(f"Invalid condition label key: {key!r}")
        if isinstance(item, bool):
            rendered = str(item).lower()
        elif isinstance(item, (int, float)):
            rendered = str(item)
        elif isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", item):
            rendered = item
        elif isinstance(item, str):
            rendered = json.dumps(item, ensure_ascii=True)
        else:
            raise ValueError(
                f"Condition label {key!r} must be a scalar, got {type(item)}"
            )
        parts.append(f"{key}:{rendered}")
    return "{" + ",".join(parts) + "}"


def _fixed_overrides(recipe: dict[str, Any]) -> list[str]:
    """Validate recipe-pinned Hydra overrides copied into every cell."""
    overrides = recipe.get("fixed_overrides", [])
    if not isinstance(overrides, list) or not all(
        isinstance(item, str) and item and "=" in item for item in overrides
    ):
        raise ValueError(
            "fixed_overrides must be a list of Hydra key=value strings"
        )
    keys = [_override.split("=", 1)[0].lstrip("+") for _override in overrides]
    if len(keys) != len(set(keys)):
        raise ValueError("fixed_overrides contains duplicate Hydra keys")
    reserved = {
        "data.dataset_kwargs.recording_ids",
        "data.training_fraction",
        "run.seed",
        "run.pretrained_checkpoint_manifest",
        "run.pretrained_checkpoint_manifest_hash",
        "run.pretrained_checkpoint_sha256",
        "run.pretrained_transfer_regime",
        "run.evaluate_test",
        "run.group",
        "run.tags",
        "run.cell_id",
        "run.checkpoint_set_id",
        "run.checkpoint_id",
        "run.source_selection_seed",
        "run.source_model_seed",
        "run.source_condition",
        "run.condition_labels",
        "run.target_species",
        "run.target_subject",
    }
    conflicts = sorted(set(keys) & reserved)
    if conflicts:
        raise ValueError(
            f"fixed_overrides conflicts with compiler fields: {conflicts}"
        )
    return list(overrides)


def compile_cells(
    registry_path: Path,
    recipe_path: Path,
    audit_path: Path,
    checkpoint_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    recipe = _load_recipe(recipe_path)
    registry, checkpoint_set_id = validate_registry(
        registry_path, checkpoint_root, recipe_path.resolve().parents[2]
    )
    checkpoint_filter = recipe.get("checkpoint_filter", {})
    if checkpoint_filter:
        if not isinstance(checkpoint_filter, dict):
            raise ValueError("checkpoint_filter must be a mapping")
        condition_filter = checkpoint_filter.get("condition", {})
        if not isinstance(condition_filter, dict):
            raise ValueError("checkpoint_filter.condition must be a mapping")
        registry = [
            record
            for record in registry
            if all(
                record.get(key) == value
                for key, value in checkpoint_filter.items()
                if key != "condition"
            )
            and all(
                record.get("condition", {}).get(key) == value
                for key, value in condition_filter.items()
            )
        ]
        if not registry:
            raise ValueError(
                f"checkpoint_filter selected no records from {registry_path}"
            )
    fixed_overrides = _fixed_overrides(recipe)
    unavailable_fraction_policy = recipe.get(
        "audit_unavailable_fraction_policy", "error"
    )
    if unavailable_fraction_policy not in {"error", "skip"}:
        raise ValueError(
            "audit_unavailable_fraction_policy must be 'error' or 'skip'"
        )
    audit, audit_hash = _load_audit(audit_path)
    configured_species = set(recipe["species"])
    registry_species = {str(record["species"]) for record in registry}
    if not registry_species <= configured_species:
        raise ValueError(
            f"Unexpected registry species: {registry_species - configured_species}"
        )
    missing_experiments = registry_species - set(recipe["base_experiments"])
    if missing_experiments:
        raise ValueError(
            f"Missing base experiment configs for species: {missing_experiments}"
        )
    checkpoint_counts = Counter(str(record["species"]) for record in registry)
    for species, settings in recipe["species"].items():
        expected = settings.get("expected_checkpoints")
        if expected is not None and int(expected) != checkpoint_counts.get(
            species, 0
        ):
            raise ValueError(
                f"{species}: expected {expected} checkpoints, got "
                f"{checkpoint_counts.get(species, 0)}"
            )

    expected_seed_pairs = recipe.get("expected_source_seed_pairs")
    if expected_seed_pairs is not None:
        expected_pairs = {
            (int(pair[0]), int(pair[1])) for pair in expected_seed_pairs
        }
        pairs_by_target: dict[tuple[str, str], set[tuple[int, int]]] = {}
        for record in registry:
            key = (
                str(record["species"]),
                str(record["excluded_target_subject"]),
            )
            pairs_by_target.setdefault(key, set()).add(
                (
                    int(record["source_selection_seed"]),
                    int(record["source_model_seed"]),
                )
            )
        for key, actual_pairs in sorted(pairs_by_target.items()):
            if actual_pairs != expected_pairs:
                raise ValueError(
                    f"{key[0]}/{key[1]}: expected source seed pairs "
                    f"{sorted(expected_pairs)}, got {sorted(actual_pairs)}"
                )

    selection = recipe["eligible_session_selection"]
    if set(selection) != {"eligible"} or not isinstance(
        selection["eligible"], bool
    ):
        raise ValueError(
            "eligible_session_selection must contain exactly one boolean "
            "'eligible' field"
        )
    eligible: dict[str, list[dict[str, Any]]] = {}
    for recording in audit.get("recordings", []):
        if recording.get("eligible", False) != selection["eligible"]:
            continue
        eligible.setdefault(str(recording["species"]), []).append(recording)
    for recordings in eligible.values():
        recordings.sort(key=lambda item: str(item["recording_id"]))

    by_species: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    all_outputs: set[tuple[str, str]] = set()
    all_wandb_ids: set[tuple[str, str]] = set()
    per_checkpoint: Counter[str] = Counter()
    per_condition: Counter[str] = Counter()
    skipped_cells: Counter[str] = Counter()
    unavailable_target_fractions: set[tuple[str, str, float]] = set()
    random_regime = "frozen_random_control"
    regimes = [str(regime) for regime in recipe["transfer_regimes"]]
    if len(regimes) != len(set(regimes)):
        raise ValueError("transfer_regimes contains duplicates")

    condition_matrix = recipe.get("condition_matrix")
    if condition_matrix is not None:
        if not isinstance(condition_matrix, list) or not condition_matrix:
            raise ValueError("condition_matrix must be a non-empty list")
        condition_ids: set[str] = set()
        for condition in condition_matrix:
            if not isinstance(condition, dict):
                raise ValueError("condition_matrix entries must be mappings")
            condition_id = str(condition.get("id", ""))
            if not condition_id or condition_id in condition_ids:
                raise ValueError(
                    "condition_matrix entries require unique non-empty ids"
                )
            condition_ids.add(condition_id)
            source = str(condition.get("source", ""))
            if source not in {"pretrained", "scratch"}:
                raise ValueError(
                    f"{condition_id}: source must be 'pretrained' or 'scratch'"
                )
            regime = condition.get("transfer_regime")
            if source == "pretrained" and not regime:
                raise ValueError(
                    f"{condition_id}: pretrained conditions require transfer_regime"
                )
            if source == "scratch" and regime is not None:
                raise ValueError(
                    f"{condition_id}: scratch conditions must have null transfer_regime"
                )
            warmup_steps = int(condition.get("adapter_warmup_steps", 0))
            if warmup_steps < 0:
                raise ValueError(
                    f"{condition_id}: adapter_warmup_steps must be non-negative"
                )
            multiplier = condition.get("backbone_lr_multiplier")
            if multiplier is not None and float(multiplier) <= 0:
                raise ValueError(
                    f"{condition_id}: backbone_lr_multiplier must be positive"
                )
        learning_rates = [
            float(value) for value in recipe.get("learning_rates", [])
        ]
        if not learning_rates or any(value <= 0 for value in learning_rates):
            raise ValueError(
                "condition_matrix recipes require positive learning_rates"
            )
    else:
        learning_rates = []

    def emit_cell(
        target: dict[str, Any],
        regime: str,
        checkpoint: dict[str, Any] | None,
        condition: dict[str, Any] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        species = str(target["species"])
        subject = str(target["subject"])
        recording_id = str(target["recording_id"])
        fraction_values = [
            float(value) for value in recipe["target_training_fractions"]
        ]
        species_recipe = recipe["species"][species]
        groups = species_recipe.get("wandb_groups", {})
        condition_id = str(condition["id"]) if condition is not None else regime
        wandb_group = str(
            groups.get(
                condition_id,
                groups.get(regime, species_recipe.get("wandb_group")),
            )
        )
        tags_by_regime = species_recipe.get("wandb_tags_by_regime", {})
        wandb_tags = list(
            tags_by_regime.get(
                condition_id,
                tags_by_regime.get(
                    regime, species_recipe.get("wandb_tags", [])
                ),
            )
        )
        if checkpoint is None:
            checkpoint_set = None
            checkpoint_id = None
            manifest_path = None
            manifest_hash = None
            checkpoint_sha256 = None
            source_selection_seed = None
            source_model_seed = None
            source_condition = (
                str(condition.get("source_condition", "scratch"))
                if condition is not None
                else "random_frozen_backbone_control"
            )
            condition_label = source_condition
            checkpoint_identity = (
                condition_id if condition is not None else "random-control"
            )
        else:
            checkpoint_set = checkpoint_set_id
            checkpoint_id = str(checkpoint["checkpoint_id"])
            manifest_path = checkpoint["manifest_path"]
            manifest_hash = checkpoint["manifest_hash"]
            checkpoint_sha256 = checkpoint["checkpoint_sha256"]
            source_selection_seed = int(checkpoint["source_selection_seed"])
            source_model_seed = int(checkpoint["source_model_seed"])
            source_condition = checkpoint["condition"]
            condition_label = str(
                source_condition.get("label") or checkpoint["checkpoint_set_id"]
            )
            checkpoint_identity = checkpoint_id

        for fraction_value in fraction_values:
            availability = target.get("fraction_availability", {}).get(
                f"{fraction_value:.2f}", {}
            )
            if not availability.get("available", False):
                if unavailable_fraction_policy == "skip":
                    unavailable_target_fractions.add(
                        (species, recording_id, fraction_value)
                    )
                    skipped_cells[species] += len(
                        recipe["target_finetuning_seeds"]
                    )
                    continue
                raise ValueError(
                    f"{recording_id}: target fraction {fraction_value} is unavailable"
                )
            for target_seed in recipe["target_finetuning_seeds"]:
                lr_suffix = (
                    f"lr{_slug_learning_rate(learning_rate)}__"
                    if learning_rate is not None
                    else ""
                )
                cell_id = (
                    f"{recipe['recipe_id']}__{species}__{recording_id}__"
                    f"{checkpoint_identity}__{condition_id}__"
                    f"{lr_suffix}"
                    f"f{_slug_fraction(fraction_value)}__t{int(target_seed)}"
                )
                if cell_id in all_ids:
                    raise ValueError(f"Duplicate cell ID: {cell_id}")
                all_ids.add(cell_id)
                output_identity = (wandb_group, cell_id)
                if output_identity in all_outputs:
                    raise ValueError(
                        f"Duplicate output/W&B identity: {output_identity}"
                    )
                all_outputs.add(output_identity)
                wandb_run_id = hashlib.md5(
                    cell_id.encode("utf-8"), usedforsecurity=False
                ).hexdigest()[:8]
                wandb_identity = (wandb_group, wandb_run_id)
                if wandb_identity in all_wandb_ids:
                    raise ValueError(
                        f"Duplicate deterministic W&B identity: {wandb_identity}"
                    )
                all_wandb_ids.add(wandb_identity)
                labels = dict(recipe.get("condition_labels", {}))
                if condition is not None:
                    labels.update(
                        {
                            "condition": condition_id,
                            "source": str(condition["source"]),
                            "adapter_warmup_steps": int(
                                condition.get("adapter_warmup_steps", 0)
                            ),
                        }
                    )
                    if learning_rate is not None:
                        labels["base_lr"] = learning_rate
                if checkpoint is None and condition is None:
                    labels["transfer_control"] = "random_frozen_backbone"
                elif checkpoint is None and condition is not None:
                    labels["transfer_control"] = "scratch"
                cell_overrides = list(fixed_overrides)
                if condition is not None:
                    cell_overrides.extend(
                        [
                            f"hyperparameters.learning_rate={learning_rate}",
                            "hyperparameters.backbone_learning_rate="
                            + (
                                f"{float(learning_rate) * float(condition['backbone_lr_multiplier']):.8g}"
                                if condition.get("backbone_lr_multiplier")
                                is not None
                                else "null"
                            ),
                            "hyperparameters.backbone_components="
                            + (
                                "[temporal_frontend,gru]"
                                if condition.get("backbone_lr_multiplier")
                                is not None
                                else "null"
                            ),
                            "hyperparameters.adapter_warmup_steps="
                            + str(
                                int(condition.get("adapter_warmup_steps", 0))
                            ),
                        ]
                    )
                row = {
                    "cell_id": cell_id,
                    "run_name": cell_id,
                    "base_experiment": recipe["base_experiments"][species],
                    "species": species,
                    "target_subject": subject,
                    "target_recording": recording_id,
                    "checkpoint_set_id": checkpoint_set,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_manifest": manifest_path,
                    "checkpoint_manifest_hash": manifest_hash,
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_selection_seed": source_selection_seed,
                    "source_model_seed": source_model_seed,
                    "source_condition": source_condition,
                    "condition_labels": labels,
                    "transfer_regime": (
                        condition.get("transfer_regime")
                        if condition is not None
                        else regime
                    ),
                    "condition_id": condition_id,
                    "base_learning_rate": learning_rate,
                    "adapter_warmup_steps": (
                        int(condition.get("adapter_warmup_steps", 0))
                        if condition is not None
                        else 0
                    ),
                    "target_fraction": fraction_value,
                    "target_finetuning_seed": int(target_seed),
                    "evaluate_test": bool(recipe.get("evaluate_test", False)),
                    "wandb_group": wandb_group,
                    "wandb_run_id": wandb_run_id,
                    "wandb_tags": wandb_tags,
                    "fixed_overrides": fixed_overrides,
                }
                overrides = [
                    *cell_overrides,
                    f"data.dataset_kwargs.recording_ids={_quote_list(recording_id)}",
                    f"data.training_fraction={fraction_value}",
                    f"run.seed={int(target_seed)}",
                    "run.pretrained_transfer_regime="
                    + (
                        str(condition["transfer_regime"])
                        if condition is not None
                        and condition.get("transfer_regime") is not None
                        else "null"
                        if condition is not None
                        else regime
                    ),
                    f"run.evaluate_test={str(bool(recipe.get('evaluate_test', False))).lower()}",
                    f"run.group={wandb_group}",
                    f"run.tags={json.dumps(wandb_tags, separators=(',', ':'))}",
                    f"run.cell_id={cell_id}",
                    f"run.checkpoint_set_id={checkpoint_set if checkpoint_set is not None else 'null'}",
                    f"run.checkpoint_id={checkpoint_id if checkpoint_id is not None else 'null'}",
                    f"run.source_condition={condition_label}",
                    "++run.condition_labels=" + _hydra_mapping(labels),
                    f"run.target_species={species}",
                    f"run.target_subject={subject}",
                ]
                if checkpoint is not None:
                    overrides[3:3] = [
                        f"run.pretrained_checkpoint_manifest={manifest_path}",
                        f"run.pretrained_checkpoint_manifest_hash={manifest_hash}",
                        f"run.pretrained_checkpoint_sha256={checkpoint_sha256}",
                    ]
                    overrides.extend(
                        [
                            f"run.source_selection_seed={source_selection_seed}",
                            f"run.source_model_seed={source_model_seed}",
                        ]
                    )
                else:
                    # The packed launcher requires every coupled cell vector
                    # to expose the same override keys. Keep the random
                    # control source-free with explicit nulls; main.py treats
                    # them as absent and records no source identity.
                    overrides[3:3] = [
                        "run.pretrained_checkpoint_manifest=null",
                        "run.pretrained_checkpoint_manifest_hash=null",
                        "run.pretrained_checkpoint_sha256=null",
                    ]
                    overrides.extend(
                        [
                            "run.source_selection_seed=null",
                            "run.source_model_seed=null",
                        ]
                    )
                row["overrides"] = overrides
                by_species.setdefault(species, []).append(row)
                per_condition[condition_id] += 1
                if checkpoint is not None:
                    per_checkpoint[checkpoint_id] += 1

    if condition_matrix is not None:
        for checkpoint in sorted(
            registry, key=lambda item: str(item["checkpoint_id"])
        ):
            species = str(checkpoint["species"])
            subject = str(checkpoint["excluded_target_subject"])
            compatible = [
                r
                for r in eligible.get(species, [])
                if r.get("subject") == subject
            ]
            if not compatible:
                raise ValueError(
                    f"{checkpoint['checkpoint_id']}: no eligible {species}/{subject} target sessions"
                )
            for target in compatible:
                for condition in condition_matrix:
                    if condition["source"] != "pretrained":
                        continue
                    for learning_rate in learning_rates:
                        emit_cell(
                            target,
                            str(condition["transfer_regime"]),
                            checkpoint,
                            condition,
                            learning_rate,
                        )
        for species, targets in sorted(eligible.items()):
            for target in targets:
                for condition in condition_matrix:
                    if condition["source"] != "scratch":
                        continue
                    for learning_rate in learning_rates:
                        emit_cell(
                            target,
                            "scratch",
                            None,
                            condition,
                            learning_rate,
                        )
    else:
        for checkpoint in sorted(
            registry, key=lambda item: str(item["checkpoint_id"])
        ):
            species = str(checkpoint["species"])
            subject = str(checkpoint["excluded_target_subject"])
            compatible = [
                r
                for r in eligible.get(species, [])
                if r.get("subject") == subject
            ]
            if not compatible:
                raise ValueError(
                    f"{checkpoint['checkpoint_id']}: no eligible {species}/{subject} target sessions"
                )
            for target in compatible:
                for regime in regimes:
                    if regime == random_regime:
                        continue
                    if (
                        target.get("species") != species
                        or target.get("subject") != subject
                    ):
                        raise ValueError(
                            f"Invalid target compatibility for {checkpoint['checkpoint_id']}"
                        )
                    emit_cell(target, regime, checkpoint)

        if random_regime in regimes:
            for species, targets in sorted(eligible.items()):
                for target in targets:
                    emit_cell(target, random_regime, None)

    counts = {
        species: len(rows) for species, rows in sorted(by_species.items())
    }
    for species, settings in recipe["species"].items():
        expected = settings.get("expected_cells")
        actual = counts.get(species, 0)
        if expected is not None and int(expected) != actual:
            raise ValueError(
                f"{species}: expected {expected} cells, compiled {actual}"
            )
    metadata = {
        "schema": "foundry-downstream-cell-lock",
        "version": 1,
        "compiler_version": COMPILER_VERSION,
        "compiler_sha256": _file_digest(Path(__file__).resolve()),
        "checkpoint_set_id": checkpoint_set_id,
        "recipe_id": recipe["recipe_id"],
        "output_stem": recipe.get("output_stem", recipe["recipe_id"]),
        "registry_sha256": _file_digest(registry_path),
        "recipe_sha256": _file_digest(recipe_path),
        "audit_sha256": audit_hash,
        "counts": counts,
        "per_checkpoint_counts": dict(sorted(per_checkpoint.items())),
        "per_condition_counts": dict(sorted(per_condition.items())),
        "audit_unavailable_fraction_policy": unavailable_fraction_policy,
        "skipped_cell_counts": dict(sorted(skipped_cells.items())),
        "unavailable_target_fractions": [
            {
                "species": species,
                "target_recording": recording_id,
                "target_fraction": fraction,
            }
            for species, recording_id, fraction in sorted(
                unavailable_target_fractions
            )
        ],
    }
    return by_species, metadata


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_outputs(
    output_dir: Path,
    cells: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> list[Path]:
    outputs: list[Path] = []
    for species, rows in sorted(cells.items()):
        output = output_dir / f"{metadata['output_stem']}-{species}.jsonl"
        payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        lock = {
            **metadata,
            "species": species,
            "output_path": output.name,
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "output_count": len(rows),
        }
        _atomic_write(output, payload)
        _atomic_write(
            output.with_suffix(".lock.json"),
            json.dumps(lock, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Validate and print counts only"
    )
    args = parser.parse_args()
    recipe = _load_recipe(args.recipe.resolve())
    audit = (args.audit or Path(recipe["audit_path"])).resolve()
    checkpoint_root = args.checkpoint_root or os.environ.get(
        "FOUNDRY_CHECKPOINT_ROOT"
    )
    if not checkpoint_root:
        parser.error("--checkpoint-root or FOUNDRY_CHECKPOINT_ROOT is required")
    cells, metadata = compile_cells(
        args.registry.resolve(),
        args.recipe.resolve(),
        audit,
        Path(checkpoint_root).resolve(),
    )
    for checkpoint_id, count in metadata["per_checkpoint_counts"].items():
        print(f"checkpoint {checkpoint_id}: {count} cells")
    for species, count in metadata["counts"].items():
        print(f"species {species}: {count} cells")
    if not args.check:
        for output in write_outputs(args.output_dir.resolve(), cells, metadata):
            print(
                f"wrote {len(cells[output.stem.rsplit('-', 1)[-1]])} cells: {output}"
            )


if __name__ == "__main__":
    main()

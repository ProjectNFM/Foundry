"""Typed, versioned schema objects for NeuroSoft Phase 3 source manifests.

Provides source-pool and source-selection manifests used in multi-session
supervised pretraining. Manifests are hash-verifiable and JSON-serializable.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from foundry.data.fraction_manifest import _canonical_hash

VALID_FAMILIES = frozenset(
    {
        "phase3_smoke",
        "source_volume",
        "subject_diversity",
        "eight_class_anchor",
        "species_composition",
    }
)
VALID_COMPOSITIONS = frozenset(
    {
        "minipigs_only",
        "monkeys_only",
        "mixed_50_50",
        "same_species",
    }
)

SOURCE_POOL_SCHEMA = "neurosoft-source-pool"
SOURCE_SELECTION_SCHEMA = "neurosoft-source-selection"
MANIFEST_VERSION = 1

ManifestT = TypeVar("ManifestT", bound="_HashValidatedManifest")


def canonical_recording_id(species: str, recording_id: str) -> str:
    """Return a species-qualified recording identifier."""
    if not isinstance(species, str) or not species.strip():
        raise ValueError("species must be a non-empty string")
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("recording_id must be a non-empty string")
    return f"{species}:{recording_id}"


def source_interval_identity(
    canonical_recording_id: str,
    index: int,
    start: float,
    end: float,
    label: object,
) -> str:
    """Return a stable identity for one source interval."""
    if not canonical_recording_id:
        raise ValueError("canonical_recording_id must be non-empty")
    return _canonical_hash(
        {
            "recording_id": canonical_recording_id,
            "index": index,
            "start": float(start).hex(),
            "end": float(end).hex(),
            "label": str(label),
        }
    )


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_str_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        result.append(item)
    return result


def _require_int_dict(value: object, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        result[key] = _require_int(item, f"{field_name}[{key}]")
    return result


def _require_str_dict(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        result[key] = _require_str(item, f"{field_name}[{key}]")
    return result


@dataclass(frozen=True)
class SourcePool:
    """One composition pool within a source-pool manifest."""

    composition: str
    source_species: list[str]
    source_subjects: list[str]
    source_recordings: list[str]
    source_subject_count: int
    source_recording_count: int
    class_counts: dict[str, int]
    target_leakage: list[str]
    source_train_split_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "composition": self.composition,
            "source_species": list(self.source_species),
            "source_subjects": list(self.source_subjects),
            "source_recordings": list(self.source_recordings),
            "source_subject_count": self.source_subject_count,
            "source_recording_count": self.source_recording_count,
            "class_counts": dict(self.class_counts),
            "target_leakage": list(self.target_leakage),
            "source_train_split_hashes": dict(self.source_train_split_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourcePool:
        if not isinstance(data, dict):
            raise ValueError("SourcePool payload must be a dict")
        return cls(
            composition=_require_str(data.get("composition"), "composition"),
            source_species=_require_str_list(
                data.get("source_species"), "source_species"
            ),
            source_subjects=_require_str_list(
                data.get("source_subjects"), "source_subjects"
            ),
            source_recordings=_require_str_list(
                data.get("source_recordings"), "source_recordings"
            ),
            source_subject_count=_require_int(
                data.get("source_subject_count"), "source_subject_count"
            ),
            source_recording_count=_require_int(
                data.get("source_recording_count"), "source_recording_count"
            ),
            class_counts=_require_int_dict(data.get("class_counts"), "class_counts"),
            target_leakage=_require_str_list(
                data.get("target_leakage"), "target_leakage"
            ),
            source_train_split_hashes={
                key: _require_str(value, f"source_train_split_hashes[{key}]")
                for key, value in _require_str_dict(
                    data.get("source_train_split_hashes"),
                    "source_train_split_hashes",
                ).items()
            },
        )


@dataclass(frozen=True)
class SelectionCondition:
    """Selection parameters for one source-selection manifest."""

    source_composition: str
    requested_fraction: float | None
    subject_count_bin: int | None
    source_selection_seed: int
    class_coverage_policy: str
    sensitivity_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_composition": self.source_composition,
            "requested_fraction": self.requested_fraction,
            "subject_count_bin": self.subject_count_bin,
            "source_selection_seed": self.source_selection_seed,
            "class_coverage_policy": self.class_coverage_policy,
            "sensitivity_only": self.sensitivity_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionCondition:
        if not isinstance(data, dict):
            raise ValueError("SelectionCondition payload must be a dict")
        requested_fraction = data.get("requested_fraction")
        if requested_fraction is not None and not isinstance(
            requested_fraction, (int, float)
        ):
            raise ValueError("requested_fraction must be a number or null")
        subject_count_bin = data.get("subject_count_bin")
        if subject_count_bin is not None:
            subject_count_bin = _require_int(
                subject_count_bin, "subject_count_bin"
            )
        sensitivity_only = data.get("sensitivity_only", False)
        if not isinstance(sensitivity_only, bool):
            raise ValueError("sensitivity_only must be a boolean")
        return cls(
            source_composition=_require_str(
                data.get("source_composition"), "source_composition"
            ),
            requested_fraction=(
                None if requested_fraction is None else float(requested_fraction)
            ),
            subject_count_bin=subject_count_bin,
            source_selection_seed=_require_int(
                data.get("source_selection_seed"), "source_selection_seed"
            ),
            class_coverage_policy=_require_str(
                data.get("class_coverage_policy"), "class_coverage_policy"
            ),
            sensitivity_only=sensitivity_only,
        )


@dataclass(frozen=True)
class SelectionSummary:
    """Aggregate counts for one source-selection manifest."""

    source_subject_count: int
    source_recording_count: int
    selected_train_examples: int
    available_train_windows: int
    realized_train_windows_per_epoch: int
    selected_signal_seconds: float
    validation_examples: int
    available_validation_windows: int
    represented_class_union: list[str]
    represented_class_intersection: list[str]
    requested_fraction: float | None
    realized_fraction: float | None
    sampler_implementation: str
    window_seconds: float
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_subject_count": self.source_subject_count,
            "source_recording_count": self.source_recording_count,
            "selected_train_examples": self.selected_train_examples,
            "available_train_windows": self.available_train_windows,
            "realized_train_windows_per_epoch": self.realized_train_windows_per_epoch,
            "selected_signal_seconds": self.selected_signal_seconds,
            "validation_examples": self.validation_examples,
            "available_validation_windows": self.available_validation_windows,
            "represented_class_union": list(self.represented_class_union),
            "represented_class_intersection": list(
                self.represented_class_intersection
            ),
            "requested_fraction": self.requested_fraction,
            "realized_fraction": self.realized_fraction,
            "sampler_implementation": self.sampler_implementation,
            "window_seconds": self.window_seconds,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionSummary:
        if not isinstance(data, dict):
            raise ValueError("SelectionSummary payload must be a dict")
        requested_fraction = data.get("requested_fraction")
        if requested_fraction is not None and not isinstance(
            requested_fraction, (int, float)
        ):
            raise ValueError("requested_fraction must be a number or null")
        realized_fraction = data.get("realized_fraction")
        if realized_fraction is not None and not isinstance(
            realized_fraction, (int, float)
        ):
            raise ValueError("realized_fraction must be a number or null")
        selected_signal_seconds = data.get("selected_signal_seconds")
        if not isinstance(selected_signal_seconds, (int, float)):
            raise ValueError("selected_signal_seconds must be a number")
        window_seconds = data.get("window_seconds")
        if not isinstance(window_seconds, (int, float)):
            raise ValueError("window_seconds must be a number")
        if float(window_seconds) <= 0:
            raise ValueError("window_seconds must be positive")
        return cls(
            source_subject_count=_require_int(
                data.get("source_subject_count"), "source_subject_count"
            ),
            source_recording_count=_require_int(
                data.get("source_recording_count"), "source_recording_count"
            ),
            selected_train_examples=_require_int(
                data.get("selected_train_examples"), "selected_train_examples"
            ),
            available_train_windows=_require_int(
                data.get("available_train_windows"), "available_train_windows"
            ),
            realized_train_windows_per_epoch=_require_int(
                data.get("realized_train_windows_per_epoch"),
                "realized_train_windows_per_epoch",
            ),
            selected_signal_seconds=float(selected_signal_seconds),
            validation_examples=_require_int(
                data.get("validation_examples"), "validation_examples"
            ),
            available_validation_windows=_require_int(
                data.get("available_validation_windows"),
                "available_validation_windows",
            ),
            represented_class_union=_require_str_list(
                data.get("represented_class_union"), "represented_class_union"
            ),
            represented_class_intersection=_require_str_list(
                data.get("represented_class_intersection"),
                "represented_class_intersection",
            ),
            requested_fraction=(
                None if requested_fraction is None else float(requested_fraction)
            ),
            realized_fraction=(
                None if realized_fraction is None else float(realized_fraction)
            ),
            sampler_implementation=_require_str(
                data.get("sampler_implementation"), "sampler_implementation"
            ),
            window_seconds=float(window_seconds),
            batch_size=_require_int(data.get("batch_size"), "batch_size"),
        )


@dataclass(frozen=True)
class SourceRecordingSelection:
    """Per-recording selection metadata within a source-selection manifest."""

    species: str
    subject: str
    recording_id: str
    canonical_recording_id: str
    raw_channel_count: int
    supported_channel_count: int
    train_source_intervals_hash: str
    train_selected_indices: list[int]
    train_selected_interval_ids: list[str]
    train_counts_by_class: dict[str, int]
    available_train_windows: int
    valid_source_intervals_hash: str
    valid_interval_ids: list[str]
    available_validation_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "subject": self.subject,
            "recording_id": self.recording_id,
            "canonical_recording_id": self.canonical_recording_id,
            "raw_channel_count": self.raw_channel_count,
            "supported_channel_count": self.supported_channel_count,
            "train_source_intervals_hash": self.train_source_intervals_hash,
            "train_selected_indices": list(self.train_selected_indices),
            "train_selected_interval_ids": list(self.train_selected_interval_ids),
            "train_counts_by_class": dict(self.train_counts_by_class),
            "available_train_windows": self.available_train_windows,
            "valid_source_intervals_hash": self.valid_source_intervals_hash,
            "valid_interval_ids": list(self.valid_interval_ids),
            "available_validation_windows": self.available_validation_windows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRecordingSelection:
        if not isinstance(data, dict):
            raise ValueError("SourceRecordingSelection payload must be a dict")
        raw_indices = data.get("train_selected_indices")
        if not isinstance(raw_indices, list):
            raise ValueError("train_selected_indices must be a list")
        train_selected_indices = [
            _require_int(index, f"train_selected_indices[{position}]")
            for position, index in enumerate(raw_indices)
        ]
        return cls(
            species=_require_str(data.get("species"), "species"),
            subject=_require_str(data.get("subject"), "subject"),
            recording_id=_require_str(data.get("recording_id"), "recording_id"),
            canonical_recording_id=_require_str(
                data.get("canonical_recording_id"), "canonical_recording_id"
            ),
            raw_channel_count=_require_int(
                data.get("raw_channel_count"), "raw_channel_count"
            ),
            supported_channel_count=_require_int(
                data.get("supported_channel_count"), "supported_channel_count"
            ),
            train_source_intervals_hash=_require_str(
                data.get("train_source_intervals_hash"),
                "train_source_intervals_hash",
            ),
            train_selected_indices=train_selected_indices,
            train_selected_interval_ids=_require_str_list(
                data.get("train_selected_interval_ids"),
                "train_selected_interval_ids",
            ),
            train_counts_by_class=_require_int_dict(
                data.get("train_counts_by_class"), "train_counts_by_class"
            ),
            available_train_windows=_require_int(
                data.get("available_train_windows"), "available_train_windows"
            ),
            valid_source_intervals_hash=_require_str(
                data.get("valid_source_intervals_hash"),
                "valid_source_intervals_hash",
            ),
            valid_interval_ids=_require_str_list(
                data.get("valid_interval_ids"), "valid_interval_ids"
            ),
            available_validation_windows=_require_int(
                data.get("available_validation_windows"),
                "available_validation_windows",
            ),
        )


class _HashValidatedManifest:
    """Shared serialization and hash-validation helpers for manifest types."""

    schema_name: ClassVar[str]
    schema_version: ClassVar[int] = MANIFEST_VERSION

    def validate_schema_version(self) -> None:
        schema = getattr(self, "schema")
        version = getattr(self, "version")
        if schema != self.schema_name:
            raise ValueError(
                f"Unsupported schema: expected {self.schema_name!r}, got {schema!r}"
            )
        if version != self.schema_version:
            raise ValueError(
                f"Unsupported version: expected {self.schema_version}, got {version}"
            )

    def validate_hash(self) -> None:
        payload = self.to_dict()
        payload.pop("manifest_hash")
        expected = getattr(self, "manifest_hash")
        actual = type(self).compute_hash(payload)
        if actual != expected:
            raise ValueError(
                "Manifest hash mismatch.\n"
                f"  Expected: {expected}\n"
                f"  Actual  : {actual}"
            )

    @classmethod
    def _validate_loaded(cls, manifest: ManifestT) -> ManifestT:
        manifest.validate_schema_version()
        manifest.validate_hash()
        return manifest

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls: type[ManifestT], text: str) -> ManifestT:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temp = Path(temp_path)
        try:
            temp.write_text(self.to_json(), encoding="utf-8")
            os.replace(temp, destination)
        finally:
            if temp.exists():
                temp.unlink()

    @classmethod
    def load(cls: type[ManifestT], path: str | Path) -> ManifestT:
        manifest = cls.from_json(Path(path).read_text(encoding="utf-8"))
        return cls._validate_loaded(manifest)


@dataclass(frozen=True)
class SourcePoolManifest(_HashValidatedManifest):
    """Eligible source pools for one target subject."""

    schema_name: ClassVar[str] = SOURCE_POOL_SCHEMA

    phase0_audit_sha256: str
    target_species: str
    target_subject: str
    eligible_target_recordings: list[str]
    pools: dict[str, SourcePool]
    manifest_hash: str
    schema: str = SOURCE_POOL_SCHEMA
    version: int = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "phase0_audit_sha256": self.phase0_audit_sha256,
            "target_species": self.target_species,
            "target_subject": self.target_subject,
            "eligible_target_recordings": list(self.eligible_target_recordings),
            "pools": {
                name: pool.to_dict() for name, pool in sorted(self.pools.items())
            },
            "manifest_hash": self.manifest_hash,
        }

    @staticmethod
    def compute_hash(payload: dict[str, Any]) -> str:
        hash_payload = {
            "schema": payload["schema"],
            "version": payload["version"],
            "phase0_audit_sha256": payload["phase0_audit_sha256"],
            "target_species": payload["target_species"],
            "target_subject": payload["target_subject"],
            "eligible_target_recordings": payload["eligible_target_recordings"],
            "pools": {
                name: payload["pools"][name]
                for name in sorted(payload["pools"])
            },
        }
        return _canonical_hash(hash_payload)

    def validate_no_leakage(self) -> None:
        if self.target_leakage_entries():
            raise ValueError(
                "Source pool manifest contains target leakage: "
                f"{self.target_leakage_entries()}"
            )

    def target_leakage_entries(self) -> list[str]:
        leaked: list[str] = []
        for pool_name, pool in self.pools.items():
            for entry in pool.target_leakage:
                leaked.append(f"{pool_name}:{entry}")
        return leaked

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourcePoolManifest:
        if not isinstance(data, dict):
            raise ValueError("SourcePoolManifest payload must be a dict")
        raw_pools = data.get("pools")
        if not isinstance(raw_pools, dict):
            raise ValueError("pools must be a dict")
        pools = {
            str(name): SourcePool.from_dict(pool_data)
            for name, pool_data in raw_pools.items()
        }
        manifest = cls(
            phase0_audit_sha256=_require_str(
                data.get("phase0_audit_sha256"), "phase0_audit_sha256"
            ),
            target_species=_require_str(data.get("target_species"), "target_species"),
            target_subject=_require_str(data.get("target_subject"), "target_subject"),
            eligible_target_recordings=_require_str_list(
                data.get("eligible_target_recordings"), "eligible_target_recordings"
            ),
            pools=pools,
            manifest_hash=_require_str(data.get("manifest_hash"), "manifest_hash"),
            schema=data.get("schema", SOURCE_POOL_SCHEMA),
            version=_require_int(data.get("version", MANIFEST_VERSION), "version"),
        )
        manifest.validate_schema_version()
        return manifest


@dataclass(frozen=True)
class SourceSelectionManifest(_HashValidatedManifest):
    """Complete source-data input for one pretraining run."""

    schema_name: ClassVar[str] = SOURCE_SELECTION_SCHEMA

    selection_id: str
    family: str
    phase0_audit_sha256: str
    source_pool_manifest: str
    source_pool_hash: str
    target_species: str
    target_subject: str
    condition: SelectionCondition
    summary: SelectionSummary
    subjects: list[str]
    recordings: list[SourceRecordingSelection]
    target_leakage: list[str]
    manifest_hash: str
    schema: str = SOURCE_SELECTION_SCHEMA
    version: int = MANIFEST_VERSION
    source_test_policy: str = "forbidden"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "selection_id": self.selection_id,
            "family": self.family,
            "phase0_audit_sha256": self.phase0_audit_sha256,
            "source_pool_manifest": self.source_pool_manifest,
            "source_pool_hash": self.source_pool_hash,
            "target_species": self.target_species,
            "target_subject": self.target_subject,
            "condition": self.condition.to_dict(),
            "summary": self.summary.to_dict(),
            "subjects": list(self.subjects),
            "recordings": [recording.to_dict() for recording in self.recordings],
            "source_test_policy": self.source_test_policy,
            "target_leakage": list(self.target_leakage),
            "manifest_hash": self.manifest_hash,
        }

    @staticmethod
    def compute_hash(payload: dict[str, Any]) -> str:
        hash_payload = dict(payload)
        hash_payload.pop("manifest_hash", None)
        return _canonical_hash(hash_payload)

    def validate_no_leakage(self) -> None:
        if self.target_leakage:
            raise ValueError(
                "Source selection manifest contains target leakage: "
                f"{self.target_leakage}"
            )

    def validate_test_policy(self) -> None:
        if self.source_test_policy != "forbidden":
            raise ValueError(
                "source_test_policy must be 'forbidden', "
                f"got {self.source_test_policy!r}"
            )

    def validate_summary_consistency(self) -> None:
        selected_train_examples = sum(
            len(recording.train_selected_indices) for recording in self.recordings
        )
        if selected_train_examples != self.summary.selected_train_examples:
            raise ValueError(
                "selected_train_examples does not match recordings: "
                f"summary={self.summary.selected_train_examples}, "
                f"recordings={selected_train_examples}"
            )

        available_train_windows = sum(
            recording.available_train_windows for recording in self.recordings
        )
        if available_train_windows != self.summary.available_train_windows:
            raise ValueError(
                "available_train_windows does not match recordings: "
                f"summary={self.summary.available_train_windows}, "
                f"recordings={available_train_windows}"
            )

        validation_examples = sum(
            len(recording.valid_interval_ids) for recording in self.recordings
        )
        if validation_examples != self.summary.validation_examples:
            raise ValueError(
                "validation_examples does not match recordings: "
                f"summary={self.summary.validation_examples}, "
                f"recordings={validation_examples}"
            )

        available_validation_windows = sum(
            recording.available_validation_windows for recording in self.recordings
        )
        if (
            available_validation_windows
            != self.summary.available_validation_windows
        ):
            raise ValueError(
                "available_validation_windows does not match recordings: "
                f"summary={self.summary.available_validation_windows}, "
                f"recordings={available_validation_windows}"
            )

        if len(self.recordings) != self.summary.source_recording_count:
            raise ValueError(
                "source_recording_count does not match recordings: "
                f"summary={self.summary.source_recording_count}, "
                f"recordings={len(self.recordings)}"
            )

        if len(self.subjects) != self.summary.source_subject_count:
            raise ValueError(
                "source_subject_count does not match subjects: "
                f"summary={self.summary.source_subject_count}, "
                f"subjects={len(self.subjects)}"
            )

        if self.summary.batch_size <= 0:
            raise ValueError("summary.batch_size must be positive")
        expected_realized = (
            self.summary.available_train_windows // self.summary.batch_size
            * self.summary.batch_size
        )
        if self.summary.realized_train_windows_per_epoch != expected_realized:
            raise ValueError(
                "realized_train_windows_per_epoch does not match batch dropping: "
                f"summary={self.summary.realized_train_windows_per_epoch}, "
                f"expected={expected_realized}"
            )
        expected_signal_seconds = (
            self.summary.available_train_windows * self.summary.window_seconds
        )
        if self.summary.selected_signal_seconds != expected_signal_seconds:
            raise ValueError(
                "selected_signal_seconds does not match available windows: "
                f"summary={self.summary.selected_signal_seconds}, "
                f"expected={expected_signal_seconds}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSelectionManifest:
        if not isinstance(data, dict):
            raise ValueError("SourceSelectionManifest payload must be a dict")
        family = _require_str(data.get("family"), "family")
        if family not in VALID_FAMILIES:
            raise ValueError(f"Unsupported family: {family!r}")
        raw_recordings = data.get("recordings")
        if not isinstance(raw_recordings, list):
            raise ValueError("recordings must be a list")
        recordings = [
            SourceRecordingSelection.from_dict(recording_data)
            for recording_data in raw_recordings
        ]
        source_test_policy = data.get("source_test_policy", "forbidden")
        if not isinstance(source_test_policy, str):
            raise ValueError("source_test_policy must be a string")
        manifest = cls(
            selection_id=_require_str(data.get("selection_id"), "selection_id"),
            family=family,
            phase0_audit_sha256=_require_str(
                data.get("phase0_audit_sha256"), "phase0_audit_sha256"
            ),
            source_pool_manifest=_require_str(
                data.get("source_pool_manifest"), "source_pool_manifest"
            ),
            source_pool_hash=_require_str(
                data.get("source_pool_hash"), "source_pool_hash"
            ),
            target_species=_require_str(data.get("target_species"), "target_species"),
            target_subject=_require_str(data.get("target_subject"), "target_subject"),
            condition=SelectionCondition.from_dict(data.get("condition", {})),
            summary=SelectionSummary.from_dict(data.get("summary", {})),
            subjects=_require_str_list(data.get("subjects"), "subjects"),
            recordings=recordings,
            target_leakage=_require_str_list(
                data.get("target_leakage"), "target_leakage"
            ),
            manifest_hash=_require_str(data.get("manifest_hash"), "manifest_hash"),
            schema=data.get("schema", SOURCE_SELECTION_SCHEMA),
            version=_require_int(data.get("version", MANIFEST_VERSION), "version"),
            source_test_policy=source_test_policy,
        )
        manifest.validate_schema_version()
        return manifest

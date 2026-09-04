#!/usr/bin/env python
"""Generate NeuroSoft source manifests for Phases 3-6."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundry.config_resolvers import _load_neurosoft_audit
from foundry.data.fraction_manifest import _canonical_hash
from foundry.data.datasets import NeurosoftMinipigs2026, NeurosoftMonkeys2026
from foundry.data.samplers import NeurosoftFirstFixedWindowSampler
from foundry.data.source_selection import (
    SOURCE_SELECTION_IMPLEMENTATION,
    select_class_indices,
)
from foundry.data.utils import resolve_neural_signal
from foundry.data.source_manifest import (
    MANIFEST_VERSION,
    SOURCE_POOL_SCHEMA,
    SOURCE_SELECTION_SCHEMA,
    SelectionCondition,
    SelectionSummary,
    SourcePool,
    SourcePoolManifest,
    SourceRecordingSelection,
    SourceSelectionManifest,
    canonical_recording_id,
    source_interval_identity,
)
from foundry.tasks.classification_mapping import ClassificationMapping

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOW_SECONDS = 0.5
DEFAULT_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_BATCH_SIZE = 16
SAMPLER_IMPLEMENTATION = (
    "foundry.data.samplers.NeurosoftFirstFixedWindowSampler/"
    "onset-anchored-fixed-window-v1"
)
DATASET_SPECS = {
    "minipigs": (
        NeurosoftMinipigs2026,
        REPO_ROOT / "configs/data/neurosoft_minipigs/multisess_raw.yaml",
    ),
    "monkeys": (
        NeurosoftMonkeys2026,
        REPO_ROOT / "configs/data/neurosoft_monkeys/multisess_raw.yaml",
    ),
}

PHASE3_SMOKE = {
    ("minipigs", "sub-06"): [
        "minipigs:sub-02_ses-01_task-AcousStim_acq-LH_desc-raw",
        "minipigs:sub-03_ses-06_task-AcousStim_acq-LH_desc-raw",
    ],
    ("monkeys", "sub-01"): [
        "monkeys:sub-02_ses-02_task-AcousStim_acq-RH_desc-raw",
        "monkeys:sub-05_ses-01_task-AcousStim_acq-RH_desc-raw",
    ],
}

COMPOSITIONS = (
    ("minipigs_only", "minipigs_only"),
    ("monkeys_only", "monkeys_only"),
    ("mixed_50_50", "mixed"),
)


def _load_class_mapping(task_path: Path) -> ClassificationMapping:
    with task_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return ClassificationMapping.from_dict(config["class_mapping"])


def _recording_ids(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8") as handle:
        return list(yaml.safe_load(handle)["dataset_kwargs"]["recording_ids"])


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp = Path(temp_path)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _parse_canonical_id(canonical_id: str) -> tuple[str, str, str]:
    species, separator, recording_id = canonical_id.partition(":")
    if not separator:
        raise ValueError(f"Invalid canonical recording ID: {canonical_id!r}")
    subject, _, _ = recording_id.partition("_")
    if not subject.startswith("sub-"):
        raise ValueError(
            f"Recording ID must begin with a BIDS subject token: {recording_id!r}"
        )
    return species, recording_id, subject


def _fraction_label(fraction: float) -> str:
    return f"{fraction:.2f}"


def _largest_remainder(total: int, weights: list[int]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        return []
    if sum(weights) == 0:
        base, remainder = divmod(total, len(weights))
        return [
            base + (1 if index < remainder else 0)
            for index in range(len(weights))
        ]

    exact = [total * weight / sum(weights) for weight in weights]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )
    for index in order[:remainder]:
        floors[index] += 1
    return floors


def _rank_subjects(subjects: list[str], seed: int) -> list[str]:
    ordered = sorted(subjects)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    return [ordered[index] for index in permutation]


def _audit_interval_hash(recording_id: str, intervals) -> str:
    """Return the Phase 0 raw-recording split hash for live intervals."""
    labels = (
        np.asarray(intervals.behavior_labels)
        if hasattr(intervals, "behavior_labels")
        else np.repeat("", len(intervals))
    )
    return _canonical_hash(
        [
            {
                "recording_id": recording_id,
                "index": index,
                "start": float(start).hex(),
                "end": float(end).hex(),
                "label": str(label),
            }
            for index, (start, end, label) in enumerate(
                zip(
                    np.asarray(intervals.start),
                    np.asarray(intervals.end),
                    labels,
                )
            )
        ]
    )


class _LiveSplit:
    """Causal split materialized from the processed recording, never audit counts."""

    def __init__(
        self,
        canonical_id: str,
        recording_id: str,
        intervals,
        class_mapping: ClassificationMapping,
        *,
        window_seconds: float,
    ) -> None:
        if not hasattr(intervals, "start") or not hasattr(intervals, "end"):
            raise ValueError(
                f"Intervals for {canonical_id!r} lack start/end timestamps"
            )
        self.canonical_id = canonical_id
        self.recording_id = recording_id
        self.intervals = intervals
        self.window_seconds = window_seconds
        self.starts = np.asarray(intervals.start)
        self.ends = np.asarray(intervals.end)
        self.labels = (
            np.asarray(intervals.behavior_labels)
            if hasattr(intervals, "behavior_labels")
            else np.repeat("", len(intervals))
        )
        if not (
            len(self.starts)
            == len(self.ends)
            == len(self.labels)
            == len(intervals)
        ):
            raise ValueError(f"Malformed intervals for {canonical_id!r}")
        self.interval_ids = [
            source_interval_identity(canonical_id, index, start, end, label)
            for index, (start, end, label) in enumerate(
                zip(self.starts, self.ends, self.labels)
            )
        ]
        self.intervals_hash = _canonical_hash(self.interval_ids)
        self.sampleable_mask = NeurosoftFirstFixedWindowSampler.sampleable_mask(
            self.starts, self.ends, window_seconds
        )
        self.class_indices = {
            class_name: np.flatnonzero(
                (class_mapping.map_to_class_ids(self.labels) == class_id)
                & self.sampleable_mask
            )
            .astype(int)
            .tolist()
            for class_id, class_name in enumerate(class_mapping.class_names)
        }

    @property
    def total(self) -> int:
        return len(self.interval_ids)

    def permuted_class_indices(
        self, class_name: str, class_id: int, seed: int
    ) -> list[int]:
        indices = list(self.class_indices.get(class_name, []))
        return select_class_indices(
            indices,
            canonical_recording_id=self.canonical_id,
            class_id=class_id,
            seed=seed,
            count=len(indices),
        )

    def select_class_count(
        self, class_name: str, class_id: int, seed: int, count: int
    ) -> list[int]:
        return select_class_indices(
            list(self.class_indices.get(class_name, [])),
            canonical_recording_id=self.canonical_id,
            class_id=class_id,
            seed=seed,
            count=count,
        )

    def ids_for_indices(self, indices: list[int]) -> list[str]:
        return [self.interval_ids[index] for index in indices]

    def window_count(self, indices: list[int], *, seed: int = 0) -> int:
        """Count windows by invoking the exact fixed-window sampler used in training."""
        mask = np.zeros(self.total, dtype=bool)
        mask[indices] = True
        selected = self.intervals.select_by_mask(mask)
        generator = torch.Generator().manual_seed(seed)
        sampler = NeurosoftFirstFixedWindowSampler(
            sampling_intervals={self.recording_id: selected},
            window_length=self.window_seconds,
            drop_short=True,
            generator=generator,
        )
        declared = len(sampler)
        # Account from the real iterator too: this detects a future mismatch
        # between the sampler's __len__ contract and emitted window stream.
        actual = len(list(sampler))
        if declared != actual:
            raise RuntimeError(
                "Fixed-window sampler length disagrees with its emitted windows "
                f"for {self.canonical_id!r}: len={declared}, emitted={actual}"
            )
        return actual


class _AuditIndex:
    def __init__(
        self,
        audit: dict[str, Any],
        *,
        data_root: Path,
        class_mapping: ClassificationMapping,
        window_seconds: float,
        verify_sampler_invariant: bool = True,
    ) -> None:
        self.audit = audit
        self.audit_sha256 = audit["artifact_sha256"]
        self.class_mapping = class_mapping
        self.window_seconds = window_seconds
        self.verify_sampler_invariant = verify_sampler_invariant
        self.recordings_by_canonical: dict[str, dict[str, Any]] = {}
        self.recordings_by_species_subject: dict[tuple[str, str], list[str]] = (
            defaultdict(list)
        )
        for recording in audit["recordings"]:
            canonical = canonical_recording_id(
                recording["species"], recording["recording_id"]
            )
            self.recordings_by_canonical[canonical] = recording
            self.recordings_by_species_subject[
                (recording["species"], recording["subject"])
            ].append(canonical)

        for subject_recordings in self.recordings_by_species_subject.values():
            subject_recordings.sort()

        self.splits: dict[str, dict[str, _LiveSplit]] = {
            "train": {},
            "valid": {},
            "test": {},
        }
        self.channel_counts: dict[str, tuple[int, int]] = {}
        self._load_and_verify_live_data(data_root)

    def _load_and_verify_live_data(self, data_root: Path) -> None:
        """Load both training datasets and bind every Phase 0 hash to live data."""
        seen: set[str] = set()
        for species, (dataset_class, config_path) in DATASET_SPECS.items():
            recording_ids = _recording_ids(config_path)
            dataset = dataset_class(
                root=str(data_root),
                recording_ids=recording_ids,
                task_type="acoustic_stim",
                split_type="intrasession-causal",
            )
            partitions = {
                split: dataset.get_sampling_intervals(split=split)
                for split in ("train", "valid", "test")
            }
            for recording_id in recording_ids:
                canonical_id = canonical_recording_id(species, recording_id)
                audit_recording = self.recordings_by_canonical.get(canonical_id)
                if audit_recording is None:
                    raise ValueError(
                        f"Live {canonical_id!r} is absent from the Phase 0 audit"
                    )
                seen.add(canonical_id)
                for split, intervals_by_recording in partitions.items():
                    intervals = intervals_by_recording.get(recording_id)
                    if intervals is None:
                        raise ValueError(
                            f"Live {split} split lacks {canonical_id!r}"
                        )
                    actual_audit_hash = _audit_interval_hash(
                        recording_id, intervals
                    )
                    expected_audit_hash = audit_recording["split_hashes"].get(
                        split
                    )
                    if actual_audit_hash != expected_audit_hash:
                        raise ValueError(
                            "Phase 0 split hash mismatch for "
                            f"{canonical_id!r} {split}: "
                            f"audit={expected_audit_hash}, live={actual_audit_hash}"
                        )
                    self.splits[split][canonical_id] = _LiveSplit(
                        canonical_id,
                        recording_id,
                        intervals,
                        self.class_mapping,
                        window_seconds=self.window_seconds,
                    )
                recording = dataset.get_recording(recording_id)
                _field, _signal, supported, _names = resolve_neural_signal(
                    recording
                )
                self.channel_counts[canonical_id] = (
                    len(recording.channels.id),
                    int(supported.sum()),
                )

        audited = set(self.recordings_by_canonical)
        if seen != audited:
            raise ValueError(
                "Phase 0 audit and configured live recordings differ: "
                f"missing_live={sorted(audited - seen)}, extra_live={sorted(seen - audited)}"
            )
        if self.verify_sampler_invariant:
            self._verify_one_example_one_window()

    def _verify_one_example_one_window(self) -> None:
        """Enforce one window per sampleable source example."""
        failures: list[str] = []
        for canonical_id, recording in sorted(
            self.recordings_by_canonical.items()
        ):
            if not recording.get("eligible", False):
                continue
            for split in ("train", "valid"):
                live_split = self.split(canonical_id, split)
                sampleable_indices = (
                    np.flatnonzero(live_split.sampleable_mask)
                    .astype(int)
                    .tolist()
                )
                windows = live_split.window_count(sampleable_indices)
                if windows != len(sampleable_indices):
                    failures.append(
                        f"{canonical_id} {split}: "
                        f"sampleable_intervals={len(sampleable_indices)}, "
                        f"windows={windows}"
                    )
        if failures:
            raise RuntimeError(
                "Phase 3 one-example/one-window prerequisite failed for the "
                "live fixed-window sampler. Do not freeze manifests until the "
                "allocation rule is amended.\n" + "\n".join(failures)
            )

    def recording(self, canonical_id: str) -> dict[str, Any]:
        if canonical_id not in self.recordings_by_canonical:
            raise KeyError(f"Unknown recording in audit: {canonical_id!r}")
        return self.recordings_by_canonical[canonical_id]

    def same_species_pool_name(self, target_species: str) -> str:
        return (
            "minipigs_only" if target_species == "minipigs" else "monkeys_only"
        )

    def same_species_pool(self, plan: dict[str, Any]) -> dict[str, Any]:
        pool_name = self.same_species_pool_name(plan["target_species"])
        return self.live_pool(plan["source_pools"][pool_name])

    def live_pool(self, audit_pool: dict[str, Any]) -> dict[str, Any]:
        """Recalculate source availability after fixed-window eligibility."""
        pool = dict(audit_pool)
        class_counts = {name: 0 for name in self.class_mapping.class_names}
        per_subject: dict[str, dict[str, int]] = {}
        for canonical_id in pool["source_recordings"]:
            species, _recording_id, subject = _parse_canonical_id(canonical_id)
            qualified_subject = f"{species}:{subject}"
            subject_counts = per_subject.setdefault(
                qualified_subject,
                {name: 0 for name in self.class_mapping.class_names},
            )
            split = self.split(canonical_id, "train")
            for class_name, indices in split.class_indices.items():
                count = len(indices)
                class_counts[class_name] += count
                subject_counts[class_name] += count
        pool["per_class_train_examples"] = class_counts
        pool["per_subject_class_counts"] = {
            subject: per_subject[subject] for subject in sorted(per_subject)
        }
        pool["source_subjects"] = sorted(per_subject)
        pool["source_subject_count"] = len(per_subject)
        pool["source_recording_count"] = len(pool["source_recordings"])
        return pool

    def live_diversity_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Recalculate common-class caps from sampleable source intervals."""
        pool = self.same_species_pool(plan)
        subjects = list(pool["source_subjects"])
        common_cap = {
            class_name: min(
                pool["per_subject_class_counts"][subject][class_name]
                for subject in subjects
            )
            for class_name in self.class_mapping.class_names
        }
        original = plan["phase_5_same_species_diversity"]
        return {
            "comparison_feasible": bool(subjects),
            "candidate_subjects": subjects,
            "bins": [
                int(value)
                for value in original["bins"]
                if int(value) <= len(subjects)
            ],
            "common_per_class_cap": common_cap,
        }

    def subject_recordings(
        self, pool: dict[str, Any], qualified_subject: str
    ) -> list[str]:
        species, subject_token = qualified_subject.split(":", 1)
        return sorted(
            canonical_id
            for canonical_id in pool["source_recordings"]
            if _parse_canonical_id(canonical_id)[0] == species
            and _parse_canonical_id(canonical_id)[2] == subject_token
        )

    def supported_channel_count(self, recording: dict[str, Any]) -> int:
        canonical_id = canonical_recording_id(
            recording["species"], recording["recording_id"]
        )
        return self.channel_counts[canonical_id][1]

    def raw_channel_count(self, canonical_id: str) -> int:
        return self.channel_counts[canonical_id][0]

    def split(self, canonical_id: str, split: str) -> _LiveSplit:
        return self.splits[split][canonical_id]


def _build_source_pool_manifest(
    audit_index: _AuditIndex,
    plan: dict[str, Any],
) -> SourcePoolManifest:
    pools: dict[str, SourcePool] = {}
    for name, audit_pool in plan["source_pools"].items():
        pool = audit_index.live_pool(audit_pool)
        pools[name] = SourcePool(
            composition=name,
            source_species=list(pool["source_species"]),
            source_subjects=list(pool["source_subjects"]),
            source_recordings=list(pool["source_recordings"]),
            source_subject_count=int(pool["source_subject_count"]),
            source_recording_count=int(pool["source_recording_count"]),
            class_counts=dict(pool["per_class_train_examples"]),
            target_leakage=list(pool["target_leakage"]),
            source_train_split_hashes={
                canonical_id: audit_index.split(
                    canonical_id, "train"
                ).intervals_hash
                for canonical_id in pool["source_recordings"]
            },
        )
    payload = {
        "schema": SOURCE_POOL_SCHEMA,
        "version": MANIFEST_VERSION,
        "phase0_audit_sha256": audit_index.audit_sha256,
        "target_species": plan["target_species"],
        "target_subject": plan["target_subject"],
        "eligible_target_recordings": list(plan["eligible_target_recordings"]),
        "pools": {name: pool.to_dict() for name, pool in sorted(pools.items())},
    }
    manifest_hash = SourcePoolManifest.compute_hash(payload)
    return SourcePoolManifest(
        phase0_audit_sha256=audit_index.audit_sha256,
        target_species=plan["target_species"],
        target_subject=plan["target_subject"],
        eligible_target_recordings=list(plan["eligible_target_recordings"]),
        pools=pools,
        manifest_hash=manifest_hash,
    )


def _recording_selection_full(
    audit_index: _AuditIndex,
    canonical_id: str,
    class_names: tuple[str, ...],
    seed: int,
    class_selector,
) -> SourceRecordingSelection:
    recording = audit_index.recording(canonical_id)
    species, recording_id, subject = _parse_canonical_id(canonical_id)
    train_split = audit_index.split(canonical_id, "train")
    valid_split = audit_index.split(canonical_id, "valid")

    selected_indices: list[int] = []
    train_counts_by_class: dict[str, int] = {}
    for class_id, class_name in enumerate(class_names):
        available = len(train_split.class_indices.get(class_name, []))
        selected_count = class_selector(class_name, class_id, available)
        class_selected = train_split.select_class_count(
            class_name, class_id, seed, selected_count
        )
        train_counts_by_class[class_name] = len(class_selected)
        selected_indices.extend(class_selected)
    selected_indices.sort()

    valid_indices = (
        np.flatnonzero(valid_split.sampleable_mask).astype(int).tolist()
    )
    return SourceRecordingSelection(
        species=species,
        subject=subject,
        recording_id=recording_id,
        canonical_recording_id=canonical_id,
        raw_channel_count=audit_index.raw_channel_count(canonical_id),
        supported_channel_count=audit_index.supported_channel_count(recording),
        train_source_intervals_hash=train_split.intervals_hash,
        train_counts_by_class=train_counts_by_class,
        train_selected_interval_ids_hash=_canonical_hash(
            train_split.ids_for_indices(selected_indices)
        ),
        available_train_windows=train_split.window_count(selected_indices),
        valid_source_intervals_hash=valid_split.intervals_hash,
        valid_selected_interval_ids_hash=_canonical_hash(
            valid_split.ids_for_indices(valid_indices)
        ),
        available_validation_windows=valid_split.window_count(valid_indices),
    )


def _reconstruct_train_indices(
    audit_index: _AuditIndex,
    canonical_id: str,
    train_counts_by_class: dict[str, int],
    seed: int,
) -> list[int]:
    """Rebuild a compact manifest's deterministic class-wise selection."""
    train_split = audit_index.split(canonical_id, "train")
    selected: list[int] = []
    for class_id, class_name in enumerate(
        audit_index.class_mapping.class_names
    ):
        selected.extend(
            train_split.select_class_count(
                class_name,
                class_id,
                seed,
                train_counts_by_class.get(class_name, 0),
            )
        )
    return sorted(selected)


def _summary_from_recordings(
    recordings: list[SourceRecordingSelection],
    requested_fraction: float | None,
    *,
    batch_size: int,
    window_seconds: float,
) -> SelectionSummary:
    selected_train_examples = sum(
        sum(recording.train_counts_by_class.values())
        for recording in recordings
    )
    available_train_windows = sum(
        recording.available_train_windows for recording in recordings
    )
    validation_examples = sum(
        recording.available_validation_windows for recording in recordings
    )
    available_validation_windows = sum(
        recording.available_validation_windows for recording in recordings
    )
    subjects = sorted(
        {f"{recording.species}:{recording.subject}" for recording in recordings}
    )
    class_union: set[str] = set()
    class_sets: list[set[str]] = []
    for recording in recordings:
        represented = {
            class_name
            for class_name, count in recording.train_counts_by_class.items()
            if count > 0
        }
        if represented:
            class_sets.append(represented)
            class_union.update(represented)
    class_intersection = set.intersection(*class_sets) if class_sets else set()
    realized_fraction = (
        selected_train_examples / available_train_windows
        if available_train_windows
        else None
    )
    realized_train_windows = available_train_windows // batch_size * batch_size
    return SelectionSummary(
        source_subject_count=len(subjects),
        source_recording_count=len(recordings),
        selected_train_examples=selected_train_examples,
        available_train_windows=available_train_windows,
        realized_train_windows_per_epoch=realized_train_windows,
        selected_signal_seconds=available_train_windows * window_seconds,
        validation_examples=validation_examples,
        available_validation_windows=available_validation_windows,
        represented_class_union=sorted(class_union),
        represented_class_intersection=sorted(class_intersection),
        requested_fraction=requested_fraction,
        realized_fraction=realized_fraction,
        selection_implementation=SOURCE_SELECTION_IMPLEMENTATION,
        sampler_implementation=SAMPLER_IMPLEMENTATION,
        window_seconds=window_seconds,
        batch_size=batch_size,
    )


def _finalize_selection_manifest(
    *,
    selection_id: str,
    family: str,
    audit_index: _AuditIndex,
    pool_manifest_path: Path,
    pool_manifest: SourcePoolManifest,
    target_species: str,
    target_subject: str,
    condition: SelectionCondition,
    recordings: list[SourceRecordingSelection],
    target_leakage: list[str],
    batch_size: int,
) -> SourceSelectionManifest:
    summary = _summary_from_recordings(
        recordings,
        condition.requested_fraction,
        batch_size=batch_size,
        window_seconds=audit_index.window_seconds,
    )
    subjects = sorted(
        {f"{recording.species}:{recording.subject}" for recording in recordings}
    )
    payload = {
        "schema": SOURCE_SELECTION_SCHEMA,
        "version": MANIFEST_VERSION,
        "selection_id": selection_id,
        "family": family,
        "phase0_audit_sha256": audit_index.audit_sha256,
        "source_pool_manifest": pool_manifest_path.as_posix(),
        "source_pool_hash": pool_manifest.manifest_hash,
        "target_species": target_species,
        "target_subject": target_subject,
        "condition": condition.to_dict(),
        "summary": summary.to_dict(),
        "subjects": subjects,
        "recordings": [recording.to_dict() for recording in recordings],
        "source_test_policy": "forbidden",
        "target_leakage": list(target_leakage),
    }
    manifest_hash = SourceSelectionManifest.compute_hash(payload)
    return SourceSelectionManifest(
        selection_id=selection_id,
        family=family,
        phase0_audit_sha256=audit_index.audit_sha256,
        source_pool_manifest=pool_manifest_path.as_posix(),
        source_pool_hash=pool_manifest.manifest_hash,
        target_species=target_species,
        target_subject=target_subject,
        condition=condition,
        summary=summary,
        subjects=subjects,
        recordings=recordings,
        target_leakage=list(target_leakage),
        manifest_hash=manifest_hash,
    )


def _save_pool_manifest(path: Path, manifest: SourcePoolManifest) -> None:
    _atomic_write_text(path, manifest.to_json() + "\n")


def _save_selection_manifest(
    path: Path, manifest: SourceSelectionManifest
) -> None:
    _atomic_write_text(path, manifest.to_json() + "\n")


def generate_source_pools(
    audit_index: _AuditIndex, output_dir: Path
) -> dict[tuple[str, str], tuple[Path, SourcePoolManifest]]:
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]] = {}
    for target_key, plan in sorted(audit_index.audit["source_plans"].items()):
        species = plan["target_species"]
        subject = plan["target_subject"]
        manifest = _build_source_pool_manifest(audit_index, plan)
        manifest.validate_no_leakage()
        path = output_dir / "source_pools" / species / f"target-{subject}.json"
        _save_pool_manifest(path, manifest)
        pool_paths[(species, subject)] = (path, manifest)
    return pool_paths


def generate_phase3_smoke(
    audit_index: _AuditIndex,
    output_dir: Path,
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]],
    class_names: tuple[str, ...],
    batch_size: int,
) -> list[Path]:
    written: list[Path] = []
    for (target_species, target_subject), canonical_ids in PHASE3_SMOKE.items():
        pool_path, pool_manifest = pool_paths[(target_species, target_subject)]
        # ``source_pool_manifest`` is resolved from the selection manifest's
        # own directory by both the runtime loader and the data-backed
        # validator.  Smoke manifests live one directory below the family
        # root, unlike the other selection families.
        rel_pool_path = Path(
            os.path.relpath(
                pool_path, output_dir / "phase3_smoke" / target_species
            )
        )
        recordings = [
            _recording_selection_full(
                audit_index,
                canonical_id,
                class_names,
                seed=42,
                class_selector=lambda _class_name,
                _class_id,
                available: available,
            )
            for canonical_id in canonical_ids
        ]
        selection_id = f"smoke_{target_species}_target-{target_subject}"
        manifest = _finalize_selection_manifest(
            selection_id=selection_id,
            family="phase3_smoke",
            audit_index=audit_index,
            pool_manifest_path=rel_pool_path,
            pool_manifest=pool_manifest,
            target_species=target_species,
            target_subject=target_subject,
            condition=SelectionCondition(
                source_composition="same_species",
                requested_fraction=None,
                subject_count_bin=None,
                source_selection_seed=42,
                class_coverage_policy="all_available",
                sensitivity_only=False,
            ),
            recordings=recordings,
            target_leakage=[],
            batch_size=batch_size,
        )
        path = (
            output_dir
            / "phase3_smoke"
            / target_species
            / f"target-{target_subject}.json"
        )
        _save_selection_manifest(path, manifest)
        written.append(path)
    return written


def _volume_class_selector(fraction: float):
    def selector(_class_name: str, _class_id: int, available: int) -> int:
        if available <= 0:
            return 0
        if fraction == 1.0:
            return available
        return math.ceil(fraction * available)

    return selector


def generate_volume_manifests(
    audit_index: _AuditIndex,
    output_dir: Path,
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]],
    class_names: tuple[str, ...],
    fractions: tuple[float, ...],
    seeds: tuple[int, ...],
    batch_size: int,
) -> list[Path]:
    written: list[Path] = []
    nesting_cache: dict[tuple[str, str, int, str], dict[str, list[str]]] = {}

    for target_key, plan in sorted(audit_index.audit["source_plans"].items()):
        target_species = plan["target_species"]
        target_subject = plan["target_subject"]
        pool = audit_index.same_species_pool(plan)
        pool_path, pool_manifest = pool_paths[(target_species, target_subject)]

        for seed in seeds:
            for fraction in fractions:
                rel_pool_path = Path(
                    os.path.relpath(
                        pool_path,
                        output_dir
                        / "source_volume"
                        / target_species
                        / f"target-{target_subject}"
                        / f"fraction-{_fraction_label(fraction)}",
                    )
                )
                recordings = [
                    _recording_selection_full(
                        audit_index,
                        canonical_id,
                        class_names,
                        seed,
                        _volume_class_selector(fraction),
                    )
                    for canonical_id in pool["source_recordings"]
                ]
                selection_id = (
                    f"volume_{target_species}_target-{target_subject}_"
                    f"f{_fraction_label(fraction)}_sel{seed}"
                )
                manifest = _finalize_selection_manifest(
                    selection_id=selection_id,
                    family="source_volume",
                    audit_index=audit_index,
                    pool_manifest_path=rel_pool_path,
                    pool_manifest=pool_manifest,
                    target_species=target_species,
                    target_subject=target_subject,
                    condition=SelectionCondition(
                        source_composition="same_species",
                        requested_fraction=fraction,
                        subject_count_bin=None,
                        source_selection_seed=seed,
                        class_coverage_policy="all_available",
                        sensitivity_only=False,
                    ),
                    recordings=recordings,
                    target_leakage=[],
                    batch_size=batch_size,
                )
                path = (
                    output_dir
                    / "source_volume"
                    / target_species
                    / f"target-{target_subject}"
                    / f"fraction-{_fraction_label(fraction)}"
                    / f"selection-{seed}.json"
                )
                _save_selection_manifest(path, manifest)
                written.append(path)

                for recording in manifest.recordings:
                    cache_key = (
                        target_species,
                        target_subject,
                        seed,
                        recording.canonical_recording_id,
                    )
                    selected_ids = audit_index.split(
                        recording.canonical_recording_id, "train"
                    ).ids_for_indices(
                        _reconstruct_train_indices(
                            audit_index,
                            recording.canonical_recording_id,
                            recording.train_counts_by_class,
                            seed,
                        )
                    )
                    by_fraction = nesting_cache.setdefault(cache_key, {})
                    by_fraction[_fraction_label(fraction)] = selected_ids

    _validate_volume_nesting(nesting_cache, fractions)
    return written


def _validate_volume_nesting(
    nesting_cache: dict[tuple[str, str, int, str], dict[str, list[str]]],
    fractions: tuple[float, ...],
) -> None:
    labels = [_fraction_label(fraction) for fraction in fractions]
    for cache_key, by_fraction in nesting_cache.items():
        for smaller, larger in zip(labels, labels[1:]):
            smaller_set = set(by_fraction[smaller])
            larger_set = set(by_fraction[larger])
            if not smaller_set.issubset(larger_set):
                raise ValueError(
                    "Volume nesting violation for "
                    f"{cache_key}: {_fraction_label(float(smaller))} "
                    f"not subset of {_fraction_label(float(larger))}"
                )


def _allocate_counts_across_items(
    total: int, items: list[str], weights: list[int]
) -> dict[str, int]:
    if total <= 0 or not items:
        return {item: 0 for item in items}
    allocations = _largest_remainder(total, weights)
    capped = [
        min(allocation, weight)
        for allocation, weight in zip(allocations, weights)
    ]
    result = dict(zip(items, capped))
    assigned = sum(result.values())
    remaining = total - assigned
    if remaining <= 0:
        return result

    expandable = [
        index
        for index, weight in enumerate(weights)
        if result[items[index]] < weight
    ]
    while remaining > 0 and expandable:
        progress = False
        for index in expandable:
            item = items[index]
            if result[item] >= weights[index]:
                continue
            result[item] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            break
        expandable = [
            index
            for index, weight in enumerate(weights)
            if result[items[index]] < weight
        ]
    return result


def _allocate_counts_across_recordings(
    audit_index: _AuditIndex,
    pool: dict[str, Any],
    subject: str,
    class_name: str,
    total: int,
) -> dict[str, int]:
    recordings = audit_index.subject_recordings(pool, subject)
    weights = [
        len(audit_index.split(canonical_id, "train").class_indices[class_name])
        for canonical_id in recordings
    ]
    return _allocate_counts_across_items(total, recordings, weights)


def _build_diversity_recordings(
    audit_index: _AuditIndex,
    pool: dict[str, Any],
    selected_subjects: list[str],
    class_targets: dict[str, int],
    class_names: tuple[str, ...],
    seed: int,
) -> list[SourceRecordingSelection]:
    per_recording_targets: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for class_name, total in class_targets.items():
        if total <= 0:
            continue
        subject_weights = [
            pool["per_subject_class_counts"][subject].get(class_name, 0)
            for subject in selected_subjects
        ]
        subject_allocations = _allocate_counts_across_items(
            total, selected_subjects, subject_weights
        )
        for subject, subject_total in subject_allocations.items():
            if subject_total <= 0:
                continue
            recording_allocations = _allocate_counts_across_recordings(
                audit_index,
                pool,
                subject,
                class_name,
                subject_total,
            )
            for canonical_id, count in recording_allocations.items():
                per_recording_targets[canonical_id][class_name] += count

    recordings: list[SourceRecordingSelection] = []
    for canonical_id in pool["source_recordings"]:
        subject = f"{_parse_canonical_id(canonical_id)[0]}:{_parse_canonical_id(canonical_id)[2]}"
        if subject not in selected_subjects:
            continue
        class_targets_for_recording = per_recording_targets.get(
            canonical_id, {}
        )
        if not class_targets_for_recording:
            # Preserve validation-only adapters only when the recording appears
            # in the selected subject set but received zero train allocation.
            class_targets_for_recording = {}

        def selector(
            class_name: str,
            class_id: int,
            available: int,
            targets=class_targets_for_recording,
        ) -> int:
            return min(available, int(targets.get(class_name, 0)))

        recordings.append(
            _recording_selection_full(
                audit_index,
                canonical_id,
                class_names,
                seed,
                selector,
            )
        )
    return recordings


def generate_diversity_manifests(
    audit_index: _AuditIndex,
    output_dir: Path,
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]],
    class_names: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
) -> list[Path]:
    written: list[Path] = []
    for target_key, plan in sorted(audit_index.audit["source_plans"].items()):
        target_species = plan["target_species"]
        target_subject = plan["target_subject"]
        pool = audit_index.same_species_pool(plan)
        diversity = audit_index.live_diversity_plan(plan)
        if not diversity.get("comparison_feasible", False):
            continue
        pool_path, pool_manifest = pool_paths[(target_species, target_subject)]
        class_targets = {
            class_name: int(
                diversity["common_per_class_cap"].get(class_name, 0)
            )
            for class_name in class_names
        }
        candidates = list(diversity["candidate_subjects"])

        for seed in seeds:
            ranked_subjects = _rank_subjects(candidates, seed)
            for subject_bin in diversity["bins"]:
                selected_subjects = ranked_subjects[:subject_bin]
                rel_pool_path = Path(
                    os.path.relpath(
                        pool_path,
                        output_dir
                        / "subject_diversity"
                        / "common_classes"
                        / target_species
                        / f"target-{target_subject}"
                        / f"subjects-{subject_bin}",
                    )
                )
                recordings = _build_diversity_recordings(
                    audit_index,
                    pool,
                    selected_subjects,
                    class_targets,
                    class_names,
                    seed,
                )
                selection_id = (
                    f"diversity_{target_species}_target-{target_subject}_"
                    f"subjects{subject_bin}_sel{seed}"
                )
                manifest = _finalize_selection_manifest(
                    selection_id=selection_id,
                    family="subject_diversity",
                    audit_index=audit_index,
                    pool_manifest_path=rel_pool_path,
                    pool_manifest=pool_manifest,
                    target_species=target_species,
                    target_subject=target_subject,
                    condition=SelectionCondition(
                        source_composition="same_species",
                        requested_fraction=None,
                        subject_count_bin=int(subject_bin),
                        source_selection_seed=seed,
                        class_coverage_policy="common_classes",
                        sensitivity_only=False,
                    ),
                    recordings=recordings,
                    target_leakage=[],
                    batch_size=batch_size,
                )
                path = (
                    output_dir
                    / "subject_diversity"
                    / "common_classes"
                    / target_species
                    / f"target-{target_subject}"
                    / f"subjects-{subject_bin}"
                    / f"selection-{seed}.json"
                )
                _save_selection_manifest(path, manifest)
                written.append(path)
    return written


def _eight_class_anchors(
    pool: dict[str, Any], class_names: tuple[str, ...]
) -> list[str]:
    anchors: list[str] = []
    for subject, counts in pool["per_subject_class_counts"].items():
        if all(
            int(counts.get(class_name, 0)) > 0 for class_name in class_names
        ):
            anchors.append(subject)
    return sorted(anchors)


def generate_eight_class_anchor_manifests(
    audit_index: _AuditIndex,
    output_dir: Path,
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]],
    class_names: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
) -> list[Path]:
    written: list[Path] = []
    for target_key, plan in sorted(audit_index.audit["source_plans"].items()):
        target_species = plan["target_species"]
        target_subject = plan["target_subject"]
        pool = audit_index.same_species_pool(plan)
        diversity = audit_index.live_diversity_plan(plan)
        if not diversity.get("comparison_feasible", False):
            continue
        anchors = _eight_class_anchors(pool, class_names)
        if not anchors:
            continue
        pool_path, pool_manifest = pool_paths[(target_species, target_subject)]
        candidates = list(diversity["candidate_subjects"])

        for seed in seeds:
            anchor = _rank_subjects(anchors, seed)[0]
            anchor_counts = pool["per_subject_class_counts"][anchor]
            class_targets = {
                class_name: int(anchor_counts.get(class_name, 0))
                for class_name in class_names
            }
            others = [
                subject
                for subject in _rank_subjects(candidates, seed)
                if subject != anchor
            ]
            ranked_subjects = [anchor] + others

            for subject_bin in diversity["bins"]:
                selected_subjects = ranked_subjects[:subject_bin]
                if anchor not in selected_subjects:
                    raise ValueError(
                        f"Eight-class anchor {anchor!r} missing from bin "
                        f"{subject_bin} for {target_key}"
                    )
                rel_pool_path = Path(
                    os.path.relpath(
                        pool_path,
                        output_dir
                        / "subject_diversity"
                        / "eight_class_anchor"
                        / target_species
                        / f"target-{target_subject}"
                        / f"subjects-{subject_bin}",
                    )
                )
                recordings = _build_diversity_recordings(
                    audit_index,
                    pool,
                    selected_subjects,
                    class_targets,
                    class_names,
                    seed,
                )
                selection_id = (
                    f"eight_class_{target_species}_target-{target_subject}_"
                    f"subjects{subject_bin}_sel{seed}"
                )
                manifest = _finalize_selection_manifest(
                    selection_id=selection_id,
                    family="eight_class_anchor",
                    audit_index=audit_index,
                    pool_manifest_path=rel_pool_path,
                    pool_manifest=pool_manifest,
                    target_species=target_species,
                    target_subject=target_subject,
                    condition=SelectionCondition(
                        source_composition="same_species",
                        requested_fraction=None,
                        subject_count_bin=int(subject_bin),
                        source_selection_seed=seed,
                        class_coverage_policy="eight_class_anchor",
                        sensitivity_only=True,
                    ),
                    recordings=recordings,
                    target_leakage=[],
                    batch_size=batch_size,
                )
                path = (
                    output_dir
                    / "subject_diversity"
                    / "eight_class_anchor"
                    / target_species
                    / f"target-{target_subject}"
                    / f"subjects-{subject_bin}"
                    / f"selection-{seed}.json"
                )
                _save_selection_manifest(path, manifest)
                written.append(path)
    return written


def _phase6_class_targets(
    total_examples: int,
    reference_counts: dict[str, int],
    class_names: tuple[str, ...],
    pool_caps: dict[str, dict[str, int]],
) -> dict[str, int]:
    per_class_cap = {
        class_name: min(
            int(pool_caps[pool_name].get(class_name, 0))
            for pool_name in pool_caps
        )
        for class_name in class_names
    }
    feasible_total = min(total_examples, sum(per_class_cap.values()))
    weights = [
        int(reference_counts.get(class_name, 0)) for class_name in class_names
    ]
    targets = dict(
        zip(class_names, _largest_remainder(feasible_total, weights))
    )
    for class_name in class_names:
        targets[class_name] = min(
            targets[class_name], per_class_cap[class_name]
        )

    remaining = feasible_total - sum(targets.values())
    expandable = [
        class_name
        for class_name in class_names
        if targets[class_name] < per_class_cap[class_name]
    ]
    while remaining > 0 and expandable:
        progress = False
        for class_name in expandable:
            if targets[class_name] >= per_class_cap[class_name]:
                continue
            targets[class_name] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            break
        expandable = [
            class_name
            for class_name in class_names
            if targets[class_name] < per_class_cap[class_name]
        ]
    return targets


def _build_composition_recordings(
    audit_index: _AuditIndex,
    pool: dict[str, Any],
    class_targets: dict[str, int],
    class_names: tuple[str, ...],
    seed: int,
    species_filter: set[str] | None = None,
) -> list[SourceRecordingSelection]:
    filtered_targets = dict(class_targets)
    per_recording_targets: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    subjects = sorted(
        {
            subject
            for subject in pool["source_subjects"]
            if species_filter is None
            or subject.split(":", 1)[0] in species_filter
        }
    )
    for class_name, total in filtered_targets.items():
        if total <= 0:
            continue
        subject_weights = [
            pool["per_subject_class_counts"][subject].get(class_name, 0)
            for subject in subjects
        ]
        subject_allocations = _allocate_counts_across_items(
            total, subjects, subject_weights
        )
        for subject, subject_total in subject_allocations.items():
            if subject_total <= 0:
                continue
            recording_allocations = _allocate_counts_across_recordings(
                audit_index,
                pool,
                subject,
                class_name,
                subject_total,
            )
            for canonical_id, count in recording_allocations.items():
                per_recording_targets[canonical_id][class_name] += count

    recordings: list[SourceRecordingSelection] = []
    for canonical_id in pool["source_recordings"]:
        species = _parse_canonical_id(canonical_id)[0]
        if species_filter is not None and species not in species_filter:
            continue
        class_targets_for_recording = per_recording_targets.get(
            canonical_id, {}
        )

        def selector(
            class_name: str,
            class_id: int,
            available: int,
            targets=class_targets_for_recording,
        ) -> int:
            return min(available, int(targets.get(class_name, 0)))

        recordings.append(
            _recording_selection_full(
                audit_index,
                canonical_id,
                class_names,
                seed,
                selector,
            )
        )
    return recordings


def _build_composition_recordings_for_condition(
    audit_index: _AuditIndex,
    pools: dict[str, Any],
    budget: dict[str, Any],
    composition_name: str,
    pool_name: str,
    class_targets: dict[str, int],
    class_names: tuple[str, ...],
    seed: int,
) -> list[SourceRecordingSelection]:
    if composition_name == "mixed_50_50":
        common_total = sum(class_targets.values())
        mp_total = common_total // 2
        mk_total = common_total - mp_total
        mp_targets = dict(
            zip(
                class_names,
                _largest_remainder(
                    mp_total,
                    [class_targets[class_name] for class_name in class_names],
                ),
            )
        )
        mk_targets = dict(
            zip(
                class_names,
                _largest_remainder(
                    mk_total,
                    [class_targets[class_name] for class_name in class_names],
                ),
            )
        )
        recordings = _build_composition_recordings(
            audit_index,
            pools["mixed"],
            mp_targets,
            class_names,
            seed,
            species_filter={"minipigs"},
        )
        recordings.extend(
            _build_composition_recordings(
                audit_index,
                pools["mixed"],
                mk_targets,
                class_names,
                seed,
                species_filter={"monkeys"},
            )
        )
        return recordings

    return _build_composition_recordings(
        audit_index,
        pools[pool_name],
        class_targets,
        class_names,
        seed,
    )


def _composition_selected_total(
    recordings: list[SourceRecordingSelection],
) -> int:
    return sum(
        sum(recording.train_counts_by_class.values())
        for recording in recordings
    )


def generate_composition_manifests(
    audit_index: _AuditIndex,
    output_dir: Path,
    pool_paths: dict[tuple[str, str], tuple[Path, SourcePoolManifest]],
    class_names: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
) -> list[Path]:
    written: list[Path] = []
    for target_key, plan in sorted(audit_index.audit["source_plans"].items()):
        target_species = plan["target_species"]
        target_subject = plan["target_subject"]
        budget = plan["phase_6_equal_volume_budget"]
        total_examples = int(budget["total_examples_per_condition"])
        pools = {
            name: audit_index.live_pool(audit_pool)
            for name, audit_pool in plan["source_pools"].items()
        }
        pool_caps = {
            name: pools[name]["per_class_train_examples"] for name in pools
        }
        reference_counts = audit_index.same_species_pool(plan)[
            "per_class_train_examples"
        ]
        class_targets = _phase6_class_targets(
            total_examples, reference_counts, class_names, pool_caps
        )
        pool_path, pool_manifest = pool_paths[(target_species, target_subject)]

        for seed in seeds:
            resolved_targets = dict(class_targets)
            composition_recordings: dict[
                str, list[SourceRecordingSelection]
            ] = {}
            for _attempt in range(32):
                composition_recordings = {
                    composition_name: _build_composition_recordings_for_condition(
                        audit_index,
                        pools,
                        budget,
                        composition_name,
                        pool_name,
                        resolved_targets,
                        class_names,
                        seed,
                    )
                    for composition_name, pool_name in COMPOSITIONS
                }
                totals = {
                    composition_name: _composition_selected_total(recordings)
                    for composition_name, recordings in composition_recordings.items()
                }
                if len(set(totals.values())) == 1:
                    break
                resolved_targets = _phase6_class_targets(
                    min(totals.values()),
                    reference_counts,
                    class_names,
                    pool_caps,
                )
            else:
                raise ValueError(
                    f"Could not reconcile Phase 6 totals for {target_key} seed {seed}: "
                    f"{totals}"
                )

            summaries = totals
            for composition_name, pool_name in COMPOSITIONS:
                recordings = composition_recordings[composition_name]

                rel_pool_path = Path(
                    os.path.relpath(
                        pool_path,
                        output_dir
                        / "species_composition"
                        / target_species
                        / f"target-{target_subject}"
                        / composition_name,
                    )
                )
                selection_id = (
                    f"composition_{target_species}_target-{target_subject}_"
                    f"{composition_name}_sel{seed}"
                )
                manifest = _finalize_selection_manifest(
                    selection_id=selection_id,
                    family="species_composition",
                    audit_index=audit_index,
                    pool_manifest_path=rel_pool_path,
                    pool_manifest=pool_manifest,
                    target_species=target_species,
                    target_subject=target_subject,
                    condition=SelectionCondition(
                        source_composition=composition_name,
                        requested_fraction=None,
                        subject_count_bin=None,
                        source_selection_seed=seed,
                        class_coverage_policy="matched_volume",
                        sensitivity_only=False,
                    ),
                    recordings=recordings,
                    target_leakage=[],
                    batch_size=batch_size,
                )
                summaries[composition_name] = (
                    manifest.summary.selected_train_examples
                )
                path = (
                    output_dir
                    / "species_composition"
                    / target_species
                    / f"target-{target_subject}"
                    / composition_name
                    / f"selection-{seed}.json"
                )
                _save_selection_manifest(path, manifest)
                written.append(path)
    return written


def _manifest_entry(
    path: Path, root: Path, manifest_type: str
) -> dict[str, Any]:
    rel_path = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    if manifest_type == "source_pool":
        manifest = SourcePoolManifest.from_json(text)
        manifest.validate_hash()
        manifest.validate_no_leakage()
        return {
            "manifest_type": manifest_type,
            "path": rel_path,
            "manifest_hash": manifest.manifest_hash,
            "target_species": manifest.target_species,
            "target_subject": manifest.target_subject,
            "source_pool_hash": manifest.manifest_hash,
            "family": "source_pool",
            "selection_id": f"pool_{manifest.target_species}_target-{manifest.target_subject}",
            "source_recording_count": sum(
                pool.source_recording_count for pool in manifest.pools.values()
            ),
            "eligible": True,
            "failure_reason": None,
        }

    manifest = SourceSelectionManifest.from_json(text)
    manifest.validate_hash()
    manifest.validate_no_leakage()
    manifest.validate_test_policy()
    manifest.validate_summary_consistency()
    return {
        "manifest_type": manifest_type,
        "path": rel_path,
        "manifest_hash": manifest.manifest_hash,
        "selection_id": manifest.selection_id,
        "family": manifest.family,
        "target_species": manifest.target_species,
        "target_subject": manifest.target_subject,
        "source_pool_hash": manifest.source_pool_hash,
        "source_composition": manifest.condition.source_composition,
        "requested_fraction": manifest.condition.requested_fraction,
        "subject_count_bin": manifest.condition.subject_count_bin,
        "source_selection_seed": manifest.condition.source_selection_seed,
        "class_coverage_policy": manifest.condition.class_coverage_policy,
        "sensitivity_only": manifest.condition.sensitivity_only,
        "source_subject_count": manifest.summary.source_subject_count,
        "source_recording_count": manifest.summary.source_recording_count,
        "selected_train_examples": manifest.summary.selected_train_examples,
        "available_train_windows": manifest.summary.available_train_windows,
        "realized_fraction": manifest.summary.realized_fraction,
        "represented_class_count": len(
            manifest.summary.represented_class_union
        ),
        "eligible": True,
        "failure_reason": None,
    }


def generate_index(output_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*.json")):
        if path.name == "index.json":
            continue
        manifest_type = (
            "source_pool"
            if "source_pools" in path.parts
            else "source_selection"
        )
        entries.append(_manifest_entry(path, output_dir, manifest_type))
    index = {
        "schema": "neurosoft-source-manifest-index",
        "version": MANIFEST_VERSION,
        "manifest_count": len(entries),
        "entries": sorted(entries, key=lambda item: item["path"]),
    }
    _atomic_write_json(output_dir / "index.json", index)
    return index


def generate_readme(output_dir: Path, index: dict[str, Any]) -> None:
    lines = [
        "# NeuroSoft supervised source manifests",
        "",
        "Generated by `tools/generate_neurosoft_source_manifests.py`.",
        "Counts are derived from live processed data, audited causal splits, and deterministic selection rules.",
        "",
        f"- Manifest count: {index['manifest_count']}",
        "",
        "## Families",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    family_counts: dict[str, int] = defaultdict(int)
    for entry in index["entries"]:
        family_counts[entry["family"]] += 1
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")

    lines.extend(["", "## Targets", ""])
    for target_species in ("minipigs", "monkeys"):
        target_entries = [
            entry
            for entry in index["entries"]
            if entry.get("target_species") == target_species
            and entry["family"] != "source_pool"
        ]
        if not target_entries:
            continue
        lines.append(f"### {target_species}")
        lines.append("")
        lines.append(
            "| Selection ID | Family | Examples | Recordings | Seed | Path |"
        )
        lines.append("|---|---|---:|---:|---:|---|")
        for entry in sorted(
            target_entries, key=lambda item: item["selection_id"]
        ):
            seed = entry.get("source_selection_seed")
            seed_text = "" if seed is None else str(seed)
            lines.append(
                f"| {entry['selection_id']} | {entry['family']} | "
                f"{entry.get('selected_train_examples', '—')} | "
                f"{entry.get('source_recording_count', '—')} | "
                f"{seed_text} | `{entry['path']}` |"
            )
        lines.append("")

    _atomic_write_text(
        output_dir / "README.md", "\n".join(lines).rstrip() + "\n"
    )


def validate_manifests(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Manifest output directory not found: {output_dir}"
        )

    index_path = output_dir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index["entries"]:
            manifest_path = output_dir / entry["path"]
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Indexed manifest missing: {manifest_path}"
                )
            refreshed = _manifest_entry(
                manifest_path,
                output_dir,
                entry["manifest_type"],
            )
            if refreshed["manifest_hash"] != entry["manifest_hash"]:
                raise ValueError(
                    f"Index hash mismatch for {entry['path']}: "
                    f"index={entry['manifest_hash']}, actual={refreshed['manifest_hash']}"
                )
    else:
        for path in sorted(output_dir.rglob("*.json")):
            manifest_type = (
                "source_pool"
                if "source_pools" in path.parts
                else "source_selection"
            )
            _manifest_entry(path, output_dir, manifest_type)


def validate_manifests_data_backed(
    output_dir: Path,
    *,
    data_root: Path,
    audit_path: Path,
    task_path: Path,
    batch_size: int,
    window_seconds: float,
    families: tuple[str, ...] | None = None,
) -> None:
    """Independently validate committed selections against processed data.

    This intentionally reloads both data sets and rechecks every Phase 0 split
    before inspecting manifests.  Static JSON/hash validation alone cannot
    establish that positional interval selections still denote live data.
    """
    audit = _load_neurosoft_audit(str(audit_path))
    class_mapping = _load_class_mapping(task_path)
    live = _AuditIndex(
        audit,
        data_root=data_root,
        class_mapping=class_mapping,
        window_seconds=window_seconds,
        verify_sampler_invariant=False,
    )
    pools: dict[Path, SourcePoolManifest] = {}
    for path in sorted((output_dir / "source_pools").rglob("*.json")):
        pool = SourcePoolManifest.load(path)
        key = path.resolve()
        pools[key] = pool
        plan = audit["source_plans"][
            f"{pool.target_species}:{pool.target_subject}"
        ]
        if (
            pool.eligible_target_recordings
            != plan["eligible_target_recordings"]
        ):
            raise ValueError(f"Pool target recording drift: {path}")
        for name, source_pool in pool.pools.items():
            audit_pool = plan["source_pools"].get(name)
            if audit_pool is None:
                raise ValueError(
                    f"Pool {path} contains unknown composition {name!r}"
                )
            expected = live.live_pool(audit_pool)
            if source_pool.source_recordings != expected["source_recordings"]:
                raise ValueError(f"Pool source recording drift: {path} {name}")
            if source_pool.class_counts != expected["per_class_train_examples"]:
                raise ValueError(
                    f"Pool live class-count mismatch: {path} {name}"
                )
            actual_hashes = {
                canonical_id: live.split(canonical_id, "train").intervals_hash
                for canonical_id in source_pool.source_recordings
            }
            if source_pool.source_train_split_hashes != actual_hashes:
                raise ValueError(
                    f"Pool live train split hash mismatch: {path} {name}"
                )

    family_roots = {
        "phase3_smoke": (output_dir / "phase3_smoke",),
        "source_volume": (output_dir / "source_volume",),
        "subject_diversity": (
            output_dir / "subject_diversity" / "common_classes",
        ),
        "eight_class_anchor": (
            output_dir / "subject_diversity" / "eight_class_anchor",
        ),
        "species_composition": (output_dir / "species_composition",),
    }
    if families is not None:
        unknown = sorted(set(families) - set(family_roots))
        if unknown:
            raise ValueError(f"Unknown manifest families: {unknown}")
        roots = [root for family in families for root in family_roots[family]]
    else:
        roots = list(family_roots.values())
        roots = [root for family_roots in roots for root in family_roots]

    train_cache: dict[
        tuple[str, int, tuple[tuple[str, int], ...]], list[int]
    ] = {}
    train_window_cache: dict[
        tuple[str, int, tuple[tuple[str, int], ...]], int
    ] = {}
    valid_cache: dict[str, tuple[list[int], str, int]] = {}
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            manifest = SourceSelectionManifest.load(path)
            if families is not None and manifest.family not in families:
                raise ValueError(
                    f"Unexpected manifest family under {root}: {path}"
                )
            pool_path = (path.parent / manifest.source_pool_manifest).resolve()
            pool = pools.get(pool_path)
            if pool is None:
                raise ValueError(f"Selection parent pool is missing: {path}")
            if pool.manifest_hash != manifest.source_pool_hash:
                raise ValueError(f"Selection parent pool hash mismatch: {path}")
            if (
                manifest.summary.selection_implementation
                != SOURCE_SELECTION_IMPLEMENTATION
                or manifest.summary.sampler_implementation
                != SAMPLER_IMPLEMENTATION
                or manifest.summary.window_seconds != window_seconds
                or manifest.summary.batch_size != batch_size
            ):
                raise ValueError(
                    f"Selection sampler-accounting metadata mismatch: {path}"
                )

            selected_window_total = 0
            validation_window_total = 0
            selected_examples = 0
            validation_examples = 0
            for rec in manifest.recordings:
                canonical_id = canonical_recording_id(
                    rec.species, rec.recording_id
                )
                if canonical_id != rec.canonical_recording_id:
                    raise ValueError(
                        f"Bad canonical recording ID in {path}: {canonical_id}"
                    )
                train = live.split(canonical_id, "train")
                valid = live.split(canonical_id, "valid")
                if rec.train_source_intervals_hash != train.intervals_hash:
                    raise ValueError(
                        f"Live train interval hash mismatch: {path} {canonical_id}"
                    )
                if rec.valid_source_intervals_hash != valid.intervals_hash:
                    raise ValueError(
                        f"Live validation interval hash mismatch: {path} {canonical_id}"
                    )
                train_key = (
                    canonical_id,
                    manifest.condition.source_selection_seed,
                    tuple(sorted(rec.train_counts_by_class.items())),
                )
                if train_key not in train_cache:
                    train_cache[train_key] = _reconstruct_train_indices(
                        live,
                        canonical_id,
                        rec.train_counts_by_class,
                        manifest.condition.source_selection_seed,
                    )
                train_indices = train_cache[train_key]
                if rec.train_selected_interval_ids_hash != _canonical_hash(
                    train.ids_for_indices(train_indices)
                ):
                    raise ValueError(
                        f"Live selected interval ID mismatch: {path} {canonical_id}"
                    )
                if canonical_id not in valid_cache:
                    valid_indices = (
                        np.flatnonzero(valid.sampleable_mask)
                        .astype(int)
                        .tolist()
                    )
                    valid_cache[canonical_id] = (
                        valid_indices,
                        _canonical_hash(valid.ids_for_indices(valid_indices)),
                        valid.window_count(valid_indices),
                    )
                valid_indices, valid_ids_hash, valid_windows = valid_cache[
                    canonical_id
                ]
                if rec.valid_selected_interval_ids_hash != valid_ids_hash:
                    raise ValueError(
                        f"Live validation interval ID mismatch: {path} {canonical_id}"
                    )
                class_index_sets = {
                    name: set(indices)
                    for name, indices in train.class_indices.items()
                }
                class_counts = {
                    name: sum(
                        index in class_index_sets[name]
                        for index in train_indices
                    )
                    for name in class_mapping.class_names
                }
                if rec.train_counts_by_class != class_counts:
                    raise ValueError(
                        f"Live class-count mismatch: {path} {canonical_id}"
                    )
                raw_channels, supported_channels = live.channel_counts[
                    canonical_id
                ]
                if (rec.raw_channel_count, rec.supported_channel_count) != (
                    raw_channels,
                    supported_channels,
                ):
                    raise ValueError(
                        f"Live channel-count mismatch: {path} {canonical_id}"
                    )
                if train_key not in train_window_cache:
                    train_window_cache[train_key] = train.window_count(
                        train_indices
                    )
                train_windows = train_window_cache[train_key]
                if rec.available_train_windows != train_windows:
                    raise ValueError(
                        f"Live train window-count mismatch: {path} {canonical_id}"
                    )
                if rec.available_validation_windows != valid_windows:
                    raise ValueError(
                        f"Live valid window-count mismatch: {path} {canonical_id}"
                    )
                selected_examples += len(train_indices)
                validation_examples += len(valid_indices)
                selected_window_total += train_windows
                validation_window_total += valid_windows

            summary = manifest.summary
            if (
                summary.selected_train_examples != selected_examples
                or summary.available_train_windows != selected_window_total
                or summary.validation_examples != validation_examples
                or summary.available_validation_windows
                != validation_window_total
                or summary.realized_train_windows_per_epoch
                != selected_window_total // batch_size * batch_size
                or summary.selected_signal_seconds
                != selected_window_total * window_seconds
            ):
                raise ValueError(f"Live aggregate accounting mismatch: {path}")


def _parse_fractions(values: list[str]) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if not fractions:
        raise ValueError("At least one fraction is required")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must be in the interval (0, 1]")
    if any(
        smaller >= larger for smaller, larger in zip(fractions, fractions[1:])
    ):
        raise ValueError("fractions must be strictly increasing")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NeuroSoft source manifests for Phases 3-6."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Processed NeuroSoft data root used to load and verify live recordings.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "docs/neurosoft-phase0-audit.json",
        help="Path to the Phase 0 audit JSON.",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=REPO_ROOT / "configs/tasks/neurosoft_acoustic_stim_8band.yaml",
        help="Task config providing the class order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated manifests.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        default=[str(value) for value in DEFAULT_FRACTIONS],
        help="Source-volume fractions to generate.",
    )
    parser.add_argument(
        "--selection-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Deterministic source-selection seeds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size used for drop-last window accounting.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=WINDOW_SECONDS,
        help="Fixed training-window duration in seconds.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing manifests without regenerating them.",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        raise ValueError("window-seconds must be finite and positive")

    output_dir = args.output.resolve()
    if args.validate_only:
        validate_manifests_data_backed(
            output_dir,
            data_root=args.data_root,
            audit_path=args.audit,
            task_path=args.task,
            batch_size=args.batch_size,
            window_seconds=args.window_seconds,
        )
        print(f"Validated manifests under {output_dir}")
        return

    audit = _load_neurosoft_audit(str(args.audit))
    class_mapping = _load_class_mapping(args.task.resolve())
    audit_index = _AuditIndex(
        audit,
        data_root=args.data_root,
        class_mapping=class_mapping,
        window_seconds=args.window_seconds,
    )
    class_names = tuple(class_mapping.class_names)
    fractions = _parse_fractions(args.fractions)
    seeds = tuple(int(seed) for seed in args.selection_seeds)
    if not seeds:
        raise ValueError("At least one selection seed is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_paths = generate_source_pools(audit_index, output_dir)
    generate_phase3_smoke(
        audit_index, output_dir, pool_paths, class_names, args.batch_size
    )
    generate_volume_manifests(
        audit_index,
        output_dir,
        pool_paths,
        class_names,
        fractions,
        seeds,
        args.batch_size,
    )
    generate_diversity_manifests(
        audit_index, output_dir, pool_paths, class_names, seeds, args.batch_size
    )
    generate_eight_class_anchor_manifests(
        audit_index, output_dir, pool_paths, class_names, seeds, args.batch_size
    )
    generate_composition_manifests(
        audit_index, output_dir, pool_paths, class_names, seeds, args.batch_size
    )
    index = generate_index(output_dir)
    generate_readme(output_dir, index)
    validate_manifests_data_backed(
        output_dir,
        data_root=args.data_root,
        audit_path=args.audit,
        task_path=args.task,
        batch_size=args.batch_size,
        window_seconds=args.window_seconds,
    )

    print(f"Generated {index['manifest_count']} manifests under {output_dir}")


if __name__ == "__main__":
    main()

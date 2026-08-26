"""Produce the reproducible NeuroSoft supervised-pretraining Phase 0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "minipigs": (
        "foundry.data.datasets.NeurosoftMinipigs2026",
        REPO_ROOT / "configs/data/neurosoft_minipigs/multisess_raw.yaml",
    ),
    "monkeys": (
        "foundry.data.datasets.NeurosoftMonkeys2026",
        REPO_ROOT / "configs/data/neurosoft_monkeys/multisess_raw.yaml",
    ),
}
TASK_CONFIG_PATH = (
    REPO_ROOT / "configs/tasks/neurosoft_acoustic_stim_8band.yaml"
)

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SOURCE_VOLUME_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (42, 43, 44)
CHECKPOINT_SELECTIONS = ("1%", "3%", "10%", "30%", "100%", "best")
MIN_PRESENT_CLASSES = 6

# Planning assumptions, not measured values. They are deliberately explicit so
# the estimate can be regenerated with revised pilot timings.
FULL_SESSION_GPU_HOURS = 0.25
FULL_SOURCE_PRETRAIN_GPU_HOURS = 2.0


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class RecordingStats:
    recording_id: str
    subject: str
    session: str
    acquisition: str
    species: str
    n_channels: int | None
    duration_seconds: float | None
    domain_segments: int | None
    split_interval_counts: dict[str, int] = field(default_factory=dict)
    mapped_interval_counts: dict[str, int] = field(default_factory=dict)
    per_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    present_classes: list[str] = field(default_factory=list)
    absent_classes: list[str] = field(default_factory=list)
    split_hashes: dict[str, str] = field(default_factory=dict)
    fraction_availability: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    eligible: bool = True
    ineligibility_reasons: list[str] = field(default_factory=list)


@dataclass
class SubjectStats:
    subject: str
    species: str
    recording_count: int
    eligible_recording_count: int
    total_duration_seconds: float
    channel_counts: list[int]


@dataclass
class SpeciesStats:
    species: str
    total_recordings: int
    eligible_recordings: int
    target_subjects: int
    total_duration_seconds: float
    channel_counts: list[int]
    present_class_histogram: dict[int, int]
    per_class_min: dict[str, int]
    per_class_median: dict[str, float]
    per_class_max: dict[str, int]


def _load_class_mapping():
    from foundry.tasks.classification_mapping import ClassificationMapping

    with TASK_CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    return ClassificationMapping.from_dict(config["class_mapping"])


def _load_dataset(dataset_class_path: str, root: str, recording_ids: list[str]):
    from foundry.data.datasets import (
        NeurosoftMinipigs2026,
        NeurosoftMonkeys2026,
    )

    classes = {
        "foundry.data.datasets.NeurosoftMinipigs2026": NeurosoftMinipigs2026,
        "foundry.data.datasets.NeurosoftMonkeys2026": NeurosoftMonkeys2026,
    }
    return classes[dataset_class_path](
        root=root,
        recording_ids=recording_ids,
        task_type="acoustic_stim",
        split_type="intrasession-causal",
    )


def _recording_ids(config_path: Path) -> list[str]:
    with config_path.open() as handle:
        return yaml.safe_load(handle)["dataset_kwargs"]["recording_ids"]


def _parse_recording_id(recording_id: str) -> tuple[str, str, str]:
    parts = recording_id.split("_")
    subject = next(
        (part for part in parts if part.startswith("sub-")), "unknown"
    )
    session = next(
        (part for part in parts if part.startswith("ses-")), "unknown"
    )
    acquisition = next(
        (
            part.removeprefix("acq-")
            for part in parts
            if part.startswith("acq-")
        ),
        "unknown",
    )
    return subject, session, acquisition


def _count_per_class(intervals, class_mapping) -> dict[str, int]:
    if not hasattr(intervals, "behavior_labels") or not len(intervals):
        return {name: 0 for name in class_mapping.class_names}
    mapped, _ = class_mapping.filter_and_remap(
        np.asarray(intervals.behavior_labels)
    )
    counts = Counter(mapped.tolist())
    return {
        name: counts.get(class_id, 0)
        for class_id, name in enumerate(class_mapping.class_names)
    }


def _mapped_count(intervals, class_mapping) -> int:
    if not hasattr(intervals, "behavior_labels") or not len(intervals):
        return 0
    return int(
        class_mapping.kept_mask(np.asarray(intervals.behavior_labels)).sum()
    )


def _interval_hash(recording_id: str, intervals) -> str:
    if not len(intervals):
        return _canonical_hash([])
    labels = (
        np.asarray(intervals.behavior_labels)
        if hasattr(intervals, "behavior_labels")
        else np.repeat("", len(intervals))
    )
    payload = [
        {
            "recording_id": recording_id,
            "index": index,
            "start": float(start).hex(),
            "end": float(end).hex(),
            "label": str(label),
        }
        for index, (start, end, label) in enumerate(
            zip(np.asarray(intervals.start), np.asarray(intervals.end), labels)
        )
    ]
    return _canonical_hash(payload)


def _fraction_availability(
    class_counts: dict[str, int],
    min_class_support: int,
    min_present_classes: int,
) -> dict[str, dict[str, Any]]:
    present_classes = [
        name for name, count in class_counts.items() if count > 0
    ]
    result: dict[str, dict[str, Any]] = {}
    for fraction in FRACTIONS:
        selected = {
            name: count if fraction == 1.0 else math.ceil(fraction * count)
            for name, count in class_counts.items()
        }
        unsupported = {
            name: count
            for name, count in selected.items()
            if class_counts[name] > 0 and count < min_class_support
        }
        reasons = [
            f"{name}: {count} < {min_class_support}"
            for name, count in unsupported.items()
        ]
        if len(present_classes) < min_present_classes:
            reasons.append(
                f"present classes: {len(present_classes)} < {min_present_classes}"
            )
        result[f"{fraction:.2f}"] = {
            "available": not reasons,
            "present_class_count": len(present_classes),
            "present_classes": present_classes,
            "selected_per_class": selected,
            "failure_reason": "; ".join(reasons) or None,
        }
    return result


def audit_recordings(
    root: str, min_class_support: int, min_present_classes: int
) -> tuple[
    list[RecordingStats], dict[str, list[SubjectStats]], dict[str, SpeciesStats]
]:
    """Inventory every configured recording and apply preregistered rules."""
    class_mapping = _load_class_mapping()
    all_recordings: list[RecordingStats] = []
    subjects_by_species: dict[str, list[SubjectStats]] = {}
    species_stats: dict[str, SpeciesStats] = {}

    for species, (dataset_class_path, config_path) in SPECS.items():
        recording_ids = _recording_ids(config_path)
        dataset = _load_dataset(dataset_class_path, root, recording_ids)
        partitions = {
            split: dataset.get_sampling_intervals(split=split)
            for split in ("train", "valid", "test")
        }
        species_recordings: list[RecordingStats] = []

        for recording_id in recording_ids:
            subject, session, acquisition = _parse_recording_id(recording_id)
            try:
                recording = dataset.get_recording(recording_id)
                domain_starts = np.asarray(recording.domain.start)
                domain_ends = np.asarray(recording.domain.end)
                durations = domain_ends - domain_starts
                if np.any(durations <= 0):
                    raise ValueError(
                        "recording domain contains non-positive segments"
                    )

                split_intervals = {
                    split: partitions[split][recording_id]
                    for split in ("train", "valid", "test")
                }
                per_class_counts = {
                    split: _count_per_class(intervals, class_mapping)
                    for split, intervals in split_intervals.items()
                }
                reasons: list[str] = []
                train_counts = per_class_counts["train"]
                present_classes = [
                    class_name
                    for class_name, count in train_counts.items()
                    if count > 0
                ]
                absent_classes = [
                    class_name
                    for class_name, count in train_counts.items()
                    if count == 0
                ]
                if len(present_classes) < min_present_classes:
                    reasons.append(
                        f"represented classes {len(present_classes)} < "
                        f"{min_present_classes}"
                    )
                for class_name, count in train_counts.items():
                    if 0 < count < min_class_support:
                        reasons.append(
                            f"{class_name}: causal-train support {count} < "
                            f"{min_class_support}"
                        )
                split_class_sets = {
                    split: {
                        class_name
                        for class_name, count in counts.items()
                        if count > 0
                    }
                    for split, counts in per_class_counts.items()
                }
                if len(set(map(frozenset, split_class_sets.values()))) != 1:
                    reasons.append(
                        "represented class set differs across train/valid/test"
                    )
                if not len(split_intervals["valid"]):
                    reasons.append("empty causal-valid split")
                if not len(split_intervals["test"]):
                    reasons.append("empty causal-test split")

                stats = RecordingStats(
                    recording_id=recording_id,
                    subject=subject,
                    session=session,
                    acquisition=acquisition,
                    species=species,
                    n_channels=len(recording.channels.id[:]),
                    duration_seconds=float(durations.sum()),
                    domain_segments=len(durations),
                    split_interval_counts={
                        split: len(intervals)
                        for split, intervals in split_intervals.items()
                    },
                    mapped_interval_counts={
                        split: _mapped_count(intervals, class_mapping)
                        for split, intervals in split_intervals.items()
                    },
                    per_class_counts=per_class_counts,
                    present_classes=present_classes,
                    absent_classes=absent_classes,
                    split_hashes={
                        split: _interval_hash(recording_id, intervals)
                        for split, intervals in split_intervals.items()
                    },
                    fraction_availability=_fraction_availability(
                        train_counts,
                        min_class_support,
                        min_present_classes,
                    ),
                    eligible=not reasons,
                    ineligibility_reasons=reasons,
                )
            except Exception as error:  # preserve an explicit inventory row
                stats = RecordingStats(
                    recording_id=recording_id,
                    subject=subject,
                    session=session,
                    acquisition=acquisition,
                    species=species,
                    n_channels=None,
                    duration_seconds=None,
                    domain_segments=None,
                    eligible=False,
                    ineligibility_reasons=[
                        f"load/audit error: {type(error).__name__}: {error}"
                    ],
                )
            all_recordings.append(stats)
            species_recordings.append(stats)

        subject_groups: dict[str, list[RecordingStats]] = defaultdict(list)
        for recording in species_recordings:
            subject_groups[recording.subject].append(recording)
        subject_summaries = [
            SubjectStats(
                subject=subject,
                species=species,
                recording_count=len(recordings),
                eligible_recording_count=sum(r.eligible for r in recordings),
                total_duration_seconds=sum(
                    r.duration_seconds or 0.0 for r in recordings
                ),
                channel_counts=sorted(
                    {
                        r.n_channels
                        for r in recordings
                        if r.n_channels is not None
                    }
                ),
            )
            for subject, recordings in sorted(subject_groups.items())
        ]
        subjects_by_species[species] = subject_summaries

        eligible = [
            recording for recording in species_recordings if recording.eligible
        ]
        per_class_values = {
            class_name: [
                recording.per_class_counts["train"][class_name]
                for recording in species_recordings
                if recording.per_class_counts
            ]
            for class_name in class_mapping.class_names
        }
        species_stats[species] = SpeciesStats(
            species=species,
            total_recordings=len(species_recordings),
            eligible_recordings=len(eligible),
            target_subjects=len({recording.subject for recording in eligible}),
            total_duration_seconds=sum(
                recording.duration_seconds or 0.0
                for recording in species_recordings
            ),
            channel_counts=sorted(
                {
                    recording.n_channels
                    for recording in species_recordings
                    if recording.n_channels is not None
                }
            ),
            present_class_histogram=dict(
                sorted(
                    Counter(len(r.present_classes) for r in eligible).items()
                )
            ),
            per_class_min={
                name: min(values) if values else 0
                for name, values in per_class_values.items()
            },
            per_class_median={
                name: float(np.median(values)) if values else 0.0
                for name, values in per_class_values.items()
            },
            per_class_max={
                name: max(values) if values else 0
                for name, values in per_class_values.items()
            },
        )

    return all_recordings, subjects_by_species, species_stats


def _source_pool(
    recordings: list[RecordingStats],
    source_species: set[str],
    target_species: str,
    target_subject: str,
) -> dict[str, Any]:
    selected = [
        recording
        for recording in recordings
        if recording.eligible
        and recording.species in source_species
        and not (
            recording.species == target_species
            and recording.subject == target_subject
        )
    ]
    leakage = [
        recording.recording_id
        for recording in selected
        if recording.species == target_species
        and recording.subject == target_subject
    ]
    subject_groups: dict[str, list[RecordingStats]] = defaultdict(list)
    for recording in selected:
        subject_groups[f"{recording.species}:{recording.subject}"].append(
            recording
        )

    per_class_counts: dict[str, int] = defaultdict(int)
    per_subject_class_counts: dict[str, dict[str, int]] = {}
    for qualified_subject, subject_recordings in sorted(subject_groups.items()):
        counts: dict[str, int] = defaultdict(int)
        for recording in subject_recordings:
            for class_name, count in recording.per_class_counts[
                "train"
            ].items():
                counts[class_name] += count
                per_class_counts[class_name] += count
        per_subject_class_counts[qualified_subject] = dict(counts)

    payload = {
        "source_species": sorted(source_species),
        "recordings": [
            f"{recording.species}:{recording.recording_id}"
            for recording in selected
        ],
        "split_hashes": [
            recording.split_hashes["train"] for recording in selected
        ],
    }
    return {
        "source_species": sorted(source_species),
        "source_subjects": sorted(subject_groups),
        "source_recordings": payload["recordings"],
        "source_subject_count": len(subject_groups),
        "source_recording_count": len(selected),
        "per_class_train_examples": dict(per_class_counts),
        "total_train_examples": sum(per_class_counts.values()),
        "per_subject_class_counts": per_subject_class_counts,
        "excluded_target_subject": f"{target_species}:{target_subject}",
        "target_leakage": leakage,
        "manifest_hash": _canonical_hash(payload),
    }


def _volume_caps(pool: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fraction in SOURCE_VOLUME_FRACTIONS:
        per_class = {
            name: count if fraction == 1.0 else math.ceil(fraction * count)
            for name, count in pool["per_class_train_examples"].items()
        }
        result[f"{fraction:.2f}"] = {
            "per_class_examples": per_class,
            "total_examples": sum(per_class.values()),
        }
    return result


def _diversity_plan(pool: dict[str, Any]) -> dict[str, Any]:
    subject_counts = pool["per_subject_class_counts"]
    n_subjects = len(subject_counts)
    bins = sorted({n for n in (1, 2, 4, n_subjects) if 0 < n <= n_subjects})
    if not subject_counts:
        return {
            "bins": [],
            "comparison_feasible": False,
            "reason": "no eligible source subjects",
        }

    class_names = next(iter(subject_counts.values())).keys()
    common_per_class_cap = {
        class_name: min(
            counts.get(class_name, 0) for counts in subject_counts.values()
        )
        for class_name in class_names
    }
    common_present_classes = sorted(
        class_name
        for class_name, count in common_per_class_cap.items()
        if count > 0
    )
    return {
        "bins": bins,
        "comparison_feasible": len(bins) >= 2,
        "selection_rule": (
            "For each seed, deterministically sample a source-subject set; "
            "the seed therefore varies both initialization and subject set."
        ),
        "common_per_class_cap": common_per_class_cap,
        "common_total_example_cap": sum(common_per_class_cap.values()),
        "common_present_classes": common_present_classes,
        "common_present_class_count": len(common_present_classes),
        "candidate_subjects": sorted(subject_counts),
    }


def compute_source_plans(recordings: list[RecordingStats]) -> dict[str, Any]:
    """Build target-specific, leakage-free source pools and final caps."""
    target_pairs = sorted(
        {
            (recording.species, recording.subject)
            for recording in recordings
            if recording.eligible
        }
    )
    plans: dict[str, Any] = {}
    for target_species, target_subject in target_pairs:
        key = f"{target_species}:{target_subject}"
        pools = {
            "minipigs_only": _source_pool(
                recordings,
                {"minipigs"},
                target_species,
                target_subject,
            ),
            "monkeys_only": _source_pool(
                recordings,
                {"monkeys"},
                target_species,
                target_subject,
            ),
            "mixed": _source_pool(
                recordings,
                {"minipigs", "monkeys"},
                target_species,
                target_subject,
            ),
        }
        leakage = [
            recording_id
            for pool in pools.values()
            for recording_id in pool["target_leakage"]
        ]
        if leakage:
            raise AssertionError(f"target leakage for {key}: {leakage}")

        same_species_name = (
            "minipigs_only" if target_species == "minipigs" else "monkeys_only"
        )
        same_species_pool = pools[same_species_name]
        minipig_total = pools["minipigs_only"]["total_train_examples"]
        monkey_total = pools["monkeys_only"]["total_train_examples"]
        equal_total_budget = min(minipig_total, monkey_total)
        equal_total_budget -= equal_total_budget % 2

        plans[key] = {
            "target_species": target_species,
            "target_subject": target_subject,
            "eligible_target_recordings": [
                recording.recording_id
                for recording in recordings
                if recording.eligible
                and recording.species == target_species
                and recording.subject == target_subject
            ],
            "source_pools": pools,
            "phase_4_same_species_volume_caps": _volume_caps(same_species_pool),
            "phase_5_same_species_diversity": _diversity_plan(
                same_species_pool
            ),
            "phase_6_equal_volume_budget": {
                "total_examples_per_condition": equal_total_budget,
                "minipigs_only_examples": equal_total_budget,
                "monkeys_only_examples": equal_total_budget,
                "mixed_minipig_examples": equal_total_budget // 2,
                "mixed_monkey_examples": equal_total_budget // 2,
            },
        }
    return plans


def estimate_run_counts(
    recordings: list[RecordingStats], source_plans: dict[str, Any]
) -> dict[str, Any]:
    """Estimate staged scientific jobs and GPU-hours from explicit assumptions."""
    eligible = [recording for recording in recordings if recording.eligible]
    available_pairs = [
        (recording, fraction)
        for recording in eligible
        for fraction in FRACTIONS
        if recording.fraction_availability[f"{fraction:.2f}"]["available"]
    ]
    three_fraction_pairs = [
        pair for pair in available_pairs if pair[1] in (0.05, 0.25, 1.0)
    ]
    full_pairs = [pair for pair in available_pairs if pair[1] == 1.0]
    n_target_subjects = len(source_plans)
    n_seeds = len(SEEDS)
    n_checkpoints = len(CHECKPOINT_SELECTIONS)

    def downstream_hours(pairs, multiplier: int = 1) -> float:
        return round(
            sum(fraction for _, fraction in pairs)
            * n_seeds
            * multiplier
            * FULL_SESSION_GPU_HOURS,
            1,
        )

    phase_1_runs = len(available_pairs) * n_seeds
    phase_2_runs = phase_1_runs

    phase_4_pretrain_runs = (
        n_target_subjects * len(SOURCE_VOLUME_FRACTIONS) * n_seeds
    )
    phase_4_pretrain_hours = round(
        n_target_subjects
        * n_seeds
        * sum(SOURCE_VOLUME_FRACTIONS)
        * FULL_SOURCE_PRETRAIN_GPU_HOURS,
        1,
    )

    phase_5_bins_by_target = {
        key: plan["phase_5_same_species_diversity"]["bins"]
        for key, plan in source_plans.items()
        if plan["phase_5_same_species_diversity"]["comparison_feasible"]
    }
    phase_5_pretrain_runs = sum(
        len(bins) * n_seeds for bins in phase_5_bins_by_target.values()
    )
    phase_5_downstream_runs = sum(
        len(bins) * n_seeds
        for recording, _ in available_pairs
        if (
            bins := phase_5_bins_by_target.get(
                f"{recording.species}:{recording.subject}"
            )
        )
    )
    phase_5_downstream_hours = round(
        sum(
            fraction * len(bins) * n_seeds * FULL_SESSION_GPU_HOURS
            for recording, fraction in available_pairs
            if (
                bins := phase_5_bins_by_target.get(
                    f"{recording.species}:{recording.subject}"
                )
            )
        ),
        1,
    )

    phases = {
        "phase_1_eegnet_curves": {
            "jobs": phase_1_runs,
            "matrix": (
                f"{len(available_pairs)} available session/fraction cells x "
                f"{n_seeds} seeds"
            ),
            "estimated_gpu_hours": downstream_hours(available_pairs),
        },
        "phase_2_gru_scratch": {
            "jobs": phase_2_runs,
            "matrix": "same target matrix as Phase 1",
            "estimated_gpu_hours": downstream_hours(available_pairs),
        },
        "phase_3_pretrain_smoke": {
            "jobs": 10,
            "matrix": "10 explicitly selected end-to-end smoke jobs",
            "estimated_gpu_hours": 5.0,
        },
        "phase_4_volume_stage_a": {
            "pretrain_jobs": phase_4_pretrain_runs,
            "downstream_jobs": (
                len(full_pairs)
                * len(SOURCE_VOLUME_FRACTIONS)
                * n_checkpoints
                * n_seeds
            ),
            "matrix": "100% downstream data; all 4 scales and 6 checkpoints",
            "estimated_gpu_hours": round(
                phase_4_pretrain_hours
                + downstream_hours(
                    full_pairs,
                    len(SOURCE_VOLUME_FRACTIONS) * n_checkpoints,
                ),
                1,
            ),
        },
        "phase_4_volume_stage_b_total": {
            "pretrain_jobs": phase_4_pretrain_runs,
            "downstream_jobs": (
                len(three_fraction_pairs)
                * len(SOURCE_VOLUME_FRACTIONS)
                * n_checkpoints
                * n_seeds
            ),
            "matrix": "5%, 25%, 100%; only after the transfer gate",
            "estimated_gpu_hours": round(
                phase_4_pretrain_hours
                + downstream_hours(
                    three_fraction_pairs,
                    len(SOURCE_VOLUME_FRACTIONS) * n_checkpoints,
                ),
                1,
            ),
        },
        "phase_4_volume_full_grid_total": {
            "pretrain_jobs": phase_4_pretrain_runs,
            "downstream_jobs": (
                len(available_pairs)
                * len(SOURCE_VOLUME_FRACTIONS)
                * n_checkpoints
                * n_seeds
            ),
            "matrix": "all available fractions; informative scales only",
            "estimated_gpu_hours": round(
                phase_4_pretrain_hours
                + downstream_hours(
                    available_pairs,
                    len(SOURCE_VOLUME_FRACTIONS) * n_checkpoints,
                ),
                1,
            ),
        },
        "phase_5_diversity": {
            "pretrain_jobs": phase_5_pretrain_runs,
            "downstream_jobs": phase_5_downstream_runs,
            "matrix": (
                "target-specific 1/2/4/all same-species subject bins; "
                "best checkpoints"
            ),
            "estimated_gpu_hours": round(
                phase_5_pretrain_runs * FULL_SOURCE_PRETRAIN_GPU_HOURS
                + phase_5_downstream_hours,
                1,
            ),
        },
        "phase_6_species_composition": {
            "pretrain_jobs": n_target_subjects * 3 * n_seeds,
            "downstream_jobs": len(available_pairs) * 3 * n_seeds,
            "matrix": "3 source compositions; best checkpoints",
            "estimated_gpu_hours": round(
                n_target_subjects * 3 * n_seeds * FULL_SOURCE_PRETRAIN_GPU_HOURS
                + downstream_hours(available_pairs, 3),
                1,
            ),
        },
        "phase_7_scale": {
            "pretrain_jobs": n_target_subjects * 3 * n_seeds,
            "downstream_jobs": len(available_pairs) * 3 * 2 * n_seeds,
            "matrix": "3 scales x scratch/pretrained; all available fractions",
            "estimated_gpu_hours": round(
                n_target_subjects * 3 * n_seeds * FULL_SOURCE_PRETRAIN_GPU_HOURS
                + downstream_hours(available_pairs, 6),
                1,
            ),
        },
    }
    return {
        "assumptions": {
            "full_session_gpu_hours": FULL_SESSION_GPU_HOURS,
            "full_source_pretrain_gpu_hours": FULL_SOURCE_PRETRAIN_GPU_HOURS,
            "checkpoint_selections": list(CHECKPOINT_SELECTIONS),
            "warning": (
                "GPU-hour figures are planning estimates, not measurements; "
                "replace both timing assumptions after Phase 1/3 pilots."
            ),
        },
        "phases": phases,
    }


def build_audit(
    root: str,
    min_class_support: int,
    min_present_classes: int,
) -> dict[str, Any]:
    recordings, subjects, species = audit_recordings(
        root, min_class_support, min_present_classes
    )
    source_plans = compute_source_plans(recordings)
    run_estimates = estimate_run_counts(recordings, source_plans)
    audit = {
        "manifest_version": 1,
        "protocol": {
            "split": "intrasession-causal",
            "task": "neurosoft_acoustic_stim_8band",
            "min_class_support": min_class_support,
            "min_present_classes": min_present_classes,
            "fractions": list(FRACTIONS),
            "source_volume_fractions": list(SOURCE_VOLUME_FRACTIONS),
            "seeds": list(SEEDS),
            "phase_4_and_5_fixed_composition": "same-species",
            "config_sha256": {
                str(path.relative_to(REPO_ROOT)): _file_hash(path)
                for _, path in SPECS.values()
            }
            | {
                str(TASK_CONFIG_PATH.relative_to(REPO_ROOT)): _file_hash(
                    TASK_CONFIG_PATH
                )
            },
        },
        "recordings": [asdict(recording) for recording in recordings],
        "subjects": {
            species_name: [asdict(subject) for subject in values]
            for species_name, values in subjects.items()
        },
        "species": {
            species_name: asdict(stats)
            for species_name, stats in species.items()
        },
        "source_plans": source_plans,
        "run_estimates": run_estimates,
    }
    audit["artifact_sha256"] = _canonical_hash(audit)
    return audit


def print_report(audit: dict[str, Any]) -> None:
    print("=" * 88)
    print("  NEUROSOFT SUPERVISED-PRETRAINING PHASE 0 AUDIT")
    print("=" * 88)
    for species_name in ("minipigs", "monkeys"):
        stats = audit["species"][species_name]
        print(
            f"{species_name}: {stats['eligible_recordings']}/"
            f"{stats['total_recordings']} eligible recordings across "
            f"{stats['target_subjects']} target subjects; "
            f"{stats['total_duration_seconds'] / 3600:.2f} signal hours; "
            f"channels={stats['channel_counts']}"
        )

    print("\nRecording eligibility:")
    for recording in audit["recordings"]:
        status = "ELIGIBLE" if recording["eligible"] else "INELIGIBLE"
        fraction_5 = recording["fraction_availability"].get("0.05", {})
        fraction_status = (
            "available" if fraction_5.get("available") else "unavailable"
        )
        reason = "; ".join(recording["ineligibility_reasons"]) or "—"
        print(
            f"  {status:<10} {recording['species']:<9} "
            f"{recording['recording_id']} | 5%={fraction_status} | {reason}"
        )

    print("\nTarget-specific same-species source plans:")
    for key, plan in audit["source_plans"].items():
        species_name = plan["target_species"]
        pool_name = (
            "minipigs_only" if species_name == "minipigs" else "monkeys_only"
        )
        pool = plan["source_pools"][pool_name]
        diversity = plan["phase_5_same_species_diversity"]
        volume = plan["phase_4_same_species_volume_caps"]
        print(
            f"  {key}: {pool['source_subject_count']} source subjects, "
            f"{pool['total_train_examples']} examples; "
            f"volumes={[v['total_examples'] for v in volume.values()]}; "
            f"diversity={diversity['bins']}"
        )

    print("\nRun-count and compute plan:")
    for phase, values in audit["run_estimates"]["phases"].items():
        jobs = values.get("jobs")
        if jobs is None:
            jobs = values["pretrain_jobs"] + values["downstream_jobs"]
        print(
            f"  {phase:<34} jobs={jobs:>5}  "
            f"est_gpu_h={values['estimated_gpu_hours']:>7.1f}  "
            f"{values['matrix']}"
        )
    print(f"\nArtifact SHA-256: {audit['artifact_sha256']}")


def _markdown_report(audit: dict[str, Any]) -> str:
    eligible_recordings = [
        recording for recording in audit["recordings"] if recording["eligible"]
    ]
    available_fraction_cells = sum(
        cell["available"]
        for recording in eligible_recordings
        for cell in recording["fraction_availability"].values()
    )
    unavailable_fraction_cells = sum(
        not cell["available"]
        for recording in eligible_recordings
        for cell in recording["fraction_availability"].values()
    )
    unavailable_recordings = sum(
        any(
            not cell["available"]
            for cell in recording["fraction_availability"].values()
        )
        for recording in eligible_recordings
    )
    diversity_bins_by_species: dict[str, set[tuple[int, ...]]] = defaultdict(
        set
    )
    for plan in audit["source_plans"].values():
        diversity_bins_by_species[plan["target_species"]].add(
            tuple(plan["phase_5_same_species_diversity"]["bins"])
        )
    diversity_summary = "; ".join(
        f"{species}: {sorted(map(list, bins))}"
        for species, bins in sorted(diversity_bins_by_species.items())
    )

    lines = [
        "# NeuroSoft supervised-pretraining Phase 0 data audit",
        "",
        "This report is generated by `tools/audit_neurosoft_sessions.py` from",
        "the processed NeuroSoft artifacts and committed data/task configs.",
        "",
        f"- **Artifact SHA-256:** `{audit['artifact_sha256']}`",
        "- **Split:** `intrasession-causal`",
        "- **Task:** `neurosoft_acoustic_stim_8band`",
        f"- **Minimum class support:** {audit['protocol']['min_class_support']}",
        f"- **Minimum represented classes:** {audit['protocol']['min_present_classes']}",
        f"- **Seeds:** {audit['protocol']['seeds']}",
        "- **Phase 4/5 fixed composition:** same-species",
        "",
        "## Species summary",
        "",
        "| Species | Configured sessions | Eligible sessions | Target subjects | 6/7/8-class sessions | Signal hours | Channels |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for species_name in ("minipigs", "monkeys"):
        stats = audit["species"][species_name]
        lines.append(
            f"| {species_name} | {stats['total_recordings']} | "
            f"{stats['eligible_recordings']} | {stats['target_subjects']} | "
            f"{stats['present_class_histogram'].get(6, 0)}/"
            f"{stats['present_class_histogram'].get(7, 0)}/"
            f"{stats['present_class_histogram'].get(8, 0)} | "
            f"{stats['total_duration_seconds'] / 3600:.2f} | "
            f"{', '.join(map(str, stats['channel_counts']))} |"
        )

    lines.extend(
        [
            "",
            "## Recording inventory and eligibility",
            "",
            f"A session is eligible when at least {audit['protocol']['min_present_classes']} mapped classes are represented,",
            "each represented class has at least three causal-training examples,",
            "and the represented class set is invariant across train/valid/test.",
            "Fraction availability is a separate cell-level decision.",
            "",
            "| Species | Recording | Duration (s) | Channels | Eligible | 5% | Reason |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for recording in audit["recordings"]:
        fraction = recording["fraction_availability"].get("0.05", {})
        reason = "; ".join(recording["ineligibility_reasons"]) or "—"
        if recording["eligible"] and not fraction.get("available", False):
            reason = fraction.get("failure_reason") or reason
        duration = recording["duration_seconds"]
        lines.append(
            f"| {recording['species']} | `{recording['recording_id']}` | "
            f"{duration:.1f} | {recording['n_channels']} | "
            f"{'yes' if recording['eligible'] else 'no'} | "
            f"{'yes' if fraction.get('available') else 'no'} | {reason} |"
            if duration is not None
            else f"| {recording['species']} | `{recording['recording_id']}` | — | — | no | no | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Target-specific source caps",
            "",
            "Phase 4 and Phase 5 use same-species sources, excluding every",
            "recording from the target subject. The four values are the class-aware",
            "10%, 25%, 50%, and 100% source-volume caps. Diversity uses a common",
            "class-aware cap set by the least-supported source subject. The common",
            "class count makes source-label coverage differences explicit.",
            "",
            "| Target | Source subjects | Train examples | Volume caps | Diversity bins | Common classes | Diversity cap |",
            "|---|---:|---:|---|---|---:|---:|",
        ]
    )
    for key, plan in audit["source_plans"].items():
        pool_name = (
            "minipigs_only"
            if plan["target_species"] == "minipigs"
            else "monkeys_only"
        )
        pool = plan["source_pools"][pool_name]
        volume_caps = [
            value["total_examples"]
            for value in plan["phase_4_same_species_volume_caps"].values()
        ]
        diversity = plan["phase_5_same_species_diversity"]
        lines.append(
            f"| `{key}` | {pool['source_subject_count']} | "
            f"{pool['total_train_examples']} | {volume_caps} | "
            f"{diversity['bins']} | "
            f"{diversity.get('common_present_class_count', '—')} | "
            f"{diversity.get('common_total_example_cap', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Run-count and compute estimates",
            "",
            "Counts include only audit-supported target fraction cells. Phase 4",
            "totals are cumulative alternatives, not additive. GPU-hours use the",
            "explicit planning assumptions below and must be replaced with measured",
            "Phase 1/3 pilot timings before production expansion.",
            "",
            "| Phase | Jobs | Estimated GPU-h | Matrix |",
            "|---|---:|---:|---|",
        ]
    )
    for phase, values in audit["run_estimates"]["phases"].items():
        jobs = values.get("jobs")
        if jobs is None:
            jobs = values["pretrain_jobs"] + values["downstream_jobs"]
        lines.append(
            f"| `{phase}` | {jobs} | {values['estimated_gpu_hours']:.1f} | "
            f"{values['matrix']} |"
        )
    assumptions = audit["run_estimates"]["assumptions"]
    lines.extend(
        [
            "",
            f"- Full-session run assumption: {assumptions['full_session_gpu_hours']} GPU-h.",
            f"- Full-source pretraining assumption: {assumptions['full_source_pretrain_gpu_hours']} GPU-h.",
            f"- Retained source checkpoints: {assumptions['checkpoint_selections']}.",
            "",
            "## Audit decisions",
            "",
            f"- Phase 1 has {available_fraction_cells} supported session/fraction cells; three seeds give {available_fraction_cells * len(SEEDS)} jobs.",
            f"- {unavailable_fraction_cells} fraction cells across {unavailable_recordings} otherwise eligible recordings are explicit rather than silently rebalanced.",
            f"- Same-species Phase 5 diversity bins are target-specific ({diversity_summary}).",
            "- Source-label coverage is recorded for every diversity plan and must be matched or stratified in Phase 5 analysis.",
            "- Phase 6 budgets are target-specific and equal-volume. The mixed condition receives half of its total examples from each species; it no longer doubles the single-species budget.",
            "- Every source-pool manifest has an empty `target_leakage` list and a content hash in the JSON companion artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--min-class-support", type=int, default=3)
    parser.add_argument(
        "--min-present-classes", type=int, default=MIN_PRESENT_CLASSES
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    if args.min_class_support < 1:
        parser.error("--min-class-support must be at least 1")
    if not 1 <= args.min_present_classes <= 8:
        parser.error("--min-present-classes must be between 1 and 8")

    audit = build_audit(
        args.data_root, args.min_class_support, args.min_present_classes
    )
    print_report(audit)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"Structured output: {args.output_json}")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown_report(audit) + "\n")
        print(f"Markdown report: {args.output_markdown}")


if __name__ == "__main__":
    main()

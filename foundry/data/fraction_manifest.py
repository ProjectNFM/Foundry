"""Build deterministic, nested, class-aware training-fraction manifests.

The manifest binds a selection to both the recording ID and the exact source
intervals. Positional indices are retained for efficient selection, while
stable interval IDs and hashes make the selection auditable after launch.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def interval_identity(
    recording_id: str,
    index: int,
    start: float,
    end: float,
    label: object,
) -> str:
    """Return a stable identity for one source interval."""
    return _canonical_hash(
        {
            "recording_id": recording_id,
            "index": index,
            "start": float(start).hex(),
            "end": float(end).hex(),
            "label": str(label),
        }
    )


@dataclass(frozen=True)
class FractionManifest:
    """Selection metadata for one recording, seed, and requested fraction."""

    recording_id: str
    seed: int
    requested_fraction: float
    realized_fraction: float
    per_class_counts: dict[str, int]
    per_class_total_counts: dict[str, int]
    present_classes: list[str]
    absent_classes: list[str]
    selected_indices: list[int]
    selected_interval_ids: list[str]
    total_intervals: int
    source_intervals_hash: str
    manifest_hash: str
    available: bool = True
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class FractionManifestBuilder:
    """Build nested manifests from one recording's causal training intervals.

    Each class gets one recording-specific deterministic permutation. A
    fraction selects a prefix of every class permutation, which guarantees
    nested subsets and approximately preserves the full training distribution.
    """

    def __init__(
        self,
        recording_id: str,
        train_intervals,
        class_mapping,
        seed: int,
        min_class_support: int = 1,
        min_present_classes: int = 1,
        fractions: Sequence[float] | None = None,
    ) -> None:
        if not recording_id:
            raise ValueError("recording_id must be non-empty")
        if min_class_support < 1:
            raise ValueError("min_class_support must be at least 1")

        requested_fractions = tuple(
            FRACTIONS if fractions is None else fractions
        )
        if not requested_fractions:
            raise ValueError("fractions must be non-empty")
        if any(not 0 < fraction <= 1 for fraction in requested_fractions):
            raise ValueError("fractions must be in the interval (0, 1]")
        if any(
            smaller >= larger
            for smaller, larger in zip(
                requested_fractions, requested_fractions[1:]
            )
        ):
            raise ValueError("fractions must be unique and strictly increasing")

        class_names = tuple(class_mapping.class_names)
        if not class_names:
            raise ValueError("class_mapping must define at least one class")
        if not 1 <= min_present_classes <= len(class_names):
            raise ValueError(
                "min_present_classes must be between 1 and the number of classes"
            )

        self.recording_id = recording_id
        self.train_intervals = train_intervals
        self.class_mapping = class_mapping
        self.class_names = class_names
        self.seed = int(seed)
        self.min_class_support = min_class_support
        self.min_present_classes = min_present_classes
        self.fractions = requested_fractions

        self._class_indices: dict[int, list[int]] | None = None
        self._permutations: dict[int, np.ndarray] | None = None
        self._interval_ids: dict[int, str] | None = None
        self._source_intervals_hash: str | None = None

    def _rng(self, class_id: int) -> np.random.Generator:
        recording_digest = hashlib.sha256(
            self.recording_id.encode("utf-8")
        ).digest()
        recording_words = np.frombuffer(recording_digest[:16], dtype="<u4")
        seed_sequence = np.random.SeedSequence(
            [self.seed, class_id, *(int(word) for word in recording_words)]
        )
        return np.random.default_rng(seed_sequence)

    def _prepare(self) -> None:
        """Group mapped intervals and compute immutable IDs/permutations."""
        if self._class_indices is not None:
            return

        if (
            not hasattr(self.train_intervals, "behavior_labels")
            or len(self.train_intervals) == 0
        ):
            self._class_indices = {}
            self._permutations = {}
            self._interval_ids = {}
            self._source_intervals_hash = _canonical_hash([])
            return

        values = np.asarray(self.train_intervals.behavior_labels)
        starts = np.asarray(self.train_intervals.start)
        ends = np.asarray(self.train_intervals.end)
        if not (
            len(values) == len(starts) == len(ends) == len(self.train_intervals)
        ):
            raise ValueError(
                "interval starts, ends, and labels must have equal length"
            )

        mapped, keep = self.class_mapping.filter_and_remap(values)
        kept_global_indices = np.flatnonzero(keep)

        self._class_indices = {}
        self._permutations = {}
        self._interval_ids = {}
        for class_id in range(len(self.class_names)):
            indices = kept_global_indices[mapped == class_id].tolist()
            self._class_indices[class_id] = indices
            self._permutations[class_id] = self._rng(class_id).permutation(
                len(indices)
            )
            for index in indices:
                self._interval_ids[index] = interval_identity(
                    self.recording_id,
                    index,
                    starts[index],
                    ends[index],
                    values[index],
                )

        ordered_source_ids = [
            self._interval_ids[index] for index in sorted(self._interval_ids)
        ]
        self._source_intervals_hash = _canonical_hash(ordered_source_ids)

    def build_fraction(self, fraction: float) -> FractionManifest:
        """Build one fraction manifest using the shared class permutations."""
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in the interval (0, 1]")
        self._prepare()

        assert self._class_indices is not None
        assert self._permutations is not None
        assert self._interval_ids is not None
        assert self._source_intervals_hash is not None

        selected_indices: list[int] = []
        per_class_counts: dict[str, int] = {}
        per_class_total_counts: dict[str, int] = {}
        failure_reasons: list[str] = []

        for class_id, class_name in enumerate(self.class_names):
            indices = self._class_indices.get(class_id, [])
            class_count = len(indices)
            per_class_total_counts[class_name] = class_count
            n_select = (
                class_count
                if fraction == 1.0
                else math.ceil(fraction * class_count)
            )
            permutation = self._permutations[class_id]
            class_selected = [
                indices[position] for position in permutation[:n_select]
            ]

            per_class_counts[class_name] = len(class_selected)
            selected_indices.extend(class_selected)
            if 0 < len(class_selected) < self.min_class_support:
                failure_reasons.append(
                    f"{class_name}: {len(class_selected)} < "
                    f"{self.min_class_support}"
                )

        present_classes = [
            name for name, count in per_class_total_counts.items() if count > 0
        ]
        absent_classes = [
            name for name, count in per_class_total_counts.items() if count == 0
        ]
        if len(present_classes) < self.min_present_classes:
            failure_reasons.append(
                f"present classes: {len(present_classes)} < "
                f"{self.min_present_classes}"
            )

        selected_indices.sort()
        selected_interval_ids = [
            self._interval_ids[index] for index in selected_indices
        ]
        total_intervals = sum(per_class_total_counts.values())
        realized_fraction = (
            len(selected_indices) / total_intervals if total_intervals else 0.0
        )
        failure_reason = "; ".join(failure_reasons) or None
        manifest_payload = {
            "recording_id": self.recording_id,
            "seed": self.seed,
            "requested_fraction": fraction,
            "selected_interval_ids": selected_interval_ids,
            "source_intervals_hash": self._source_intervals_hash,
            "present_classes": present_classes,
        }

        return FractionManifest(
            recording_id=self.recording_id,
            seed=self.seed,
            requested_fraction=fraction,
            realized_fraction=realized_fraction,
            per_class_counts=per_class_counts,
            per_class_total_counts=per_class_total_counts,
            present_classes=present_classes,
            absent_classes=absent_classes,
            selected_indices=selected_indices,
            selected_interval_ids=selected_interval_ids,
            total_intervals=total_intervals,
            source_intervals_hash=self._source_intervals_hash,
            manifest_hash=_canonical_hash(manifest_payload),
            available=not failure_reasons,
            failure_reason=failure_reason,
        )

    def build_all_fractions(self) -> list[FractionManifest]:
        """Build manifests for all configured fractions."""
        return [self.build_fraction(fraction) for fraction in self.fractions]

    def validate_nesting(self) -> bool:
        """Return whether every selection is contained in the next one."""
        manifests = self.build_all_fractions()
        return all(
            set(smaller.selected_interval_ids).issubset(
                larger.selected_interval_ids
            )
            for smaller, larger in zip(manifests, manifests[1:])
        )

"""Unified classification task mapping contract.

Provides a single source of truth for raw-to-mapped label relationships,
class removal, derived class counts, and display names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassificationMapping:
    """Full-enumeration classification label mapping.

    Every raw label ID that can appear in the data must be declared in
    ``raw_to_mapped``. Mapped IDs must form a contiguous range 0..N-1.
    Raw IDs mapped to ``None`` are removed from training supervision.

    Args:
        raw_to_mapped: Complete mapping from every possible raw label ID
            to either a contiguous mapped ID (int) or None for removal.
        names: Optional mapping from mapped ID to human-readable display
            name. Missing entries fall back to ``class_<id>``.
    """

    raw_to_mapped: dict[int, int | None]
    names: dict[int, str] | None = None

    # Cached derived values (computed in __post_init__)
    _num_classes: int = field(init=False, repr=False, compare=False)
    _kept_raw_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    _removed_raw_ids: frozenset[int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._validate()
        mapped_ids = {v for v in self.raw_to_mapped.values() if v is not None}
        object.__setattr__(self, "_num_classes", len(mapped_ids))
        object.__setattr__(
            self,
            "_kept_raw_ids",
            frozenset(
                k for k, v in self.raw_to_mapped.items() if v is not None
            ),
        )
        object.__setattr__(
            self,
            "_removed_raw_ids",
            frozenset(k for k, v in self.raw_to_mapped.items() if v is None),
        )

    def _validate(self) -> None:
        mapped_ids = {v for v in self.raw_to_mapped.values() if v is not None}

        if not mapped_ids:
            raise ValueError(
                "ClassificationMapping must retain at least one mapped class "
                "(all values are None or raw_to_mapped is empty)."
            )

        for mid in mapped_ids:
            if mid < 0:
                raise ValueError(
                    f"Mapped IDs must be non-negative integers, got {mid}."
                )

        expected = set(range(len(mapped_ids)))
        if mapped_ids != expected:
            raise ValueError(
                f"Mapped IDs must be contiguous integers 0..{len(mapped_ids) - 1}, "
                f"got {sorted(mapped_ids)}."
            )

        reachable = set()
        for raw_id, mapped_id in self.raw_to_mapped.items():
            if mapped_id is not None:
                reachable.add(mapped_id)

        unreachable = expected - reachable
        if unreachable:
            raise ValueError(
                f"Every mapped ID must be reachable from at least one raw ID. "
                f"Unreachable: {sorted(unreachable)}."
            )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def kept_raw_ids(self) -> set[int]:
        return set(self._kept_raw_ids)

    @property
    def removed_raw_ids(self) -> set[int]:
        return set(self._removed_raw_ids)

    @property
    def class_names(self) -> list[str]:
        """Ordered list of display names for mapped classes 0..N-1."""
        result = []
        for i in range(self._num_classes):
            if self.names and i in self.names:
                result.append(self.names[i])
            else:
                result.append(f"class_{i}")
        return result

    def apply(self, raw_values: np.ndarray) -> np.ndarray:
        """Map raw label array to canonical mapped labels.

        Removed classes get mapped to -1. Undeclared raw IDs raise immediately.

        Args:
            raw_values: Integer array of raw label IDs.

        Returns:
            Integer array of mapped label IDs (same dtype as input).

        Raises:
            ValueError: If any value in raw_values is not declared in raw_to_mapped.
        """
        declared = set(self.raw_to_mapped.keys())
        unique_raw = set(raw_values.flat)
        undeclared = unique_raw - declared
        if undeclared:
            raise ValueError(
                f"Encountered undeclared raw label IDs: {sorted(undeclared)}. "
                f"All raw IDs must be enumerated in classification_mapping.raw_to_mapped."
            )

        result = np.empty_like(raw_values)
        for raw_id, mapped_id in self.raw_to_mapped.items():
            mask = raw_values == raw_id
            if mask.any():
                result[mask] = mapped_id if mapped_id is not None else -1

        return result

    def kept_mask(self, raw_values: np.ndarray) -> np.ndarray:
        """Boolean mask where True indicates a kept (non-removed) label."""
        kept = self._kept_raw_ids
        mask = np.zeros(len(raw_values), dtype=bool)
        for raw_id in kept:
            mask |= raw_values == raw_id
        return mask

    def filter_and_remap(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter removed labels and remap kept ones in one atomic operation.

        Returns:
            (mapped_values, keep_mask) where mapped_values has length == keep_mask.sum().
        """
        keep = self.kept_mask(values)
        mapped = self.apply(values[keep])
        return mapped, keep

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationMapping:
        """Construct from a parsed YAML/dict representation.

        Accepts raw_to_mapped with string or int keys (YAML often produces strings).
        """
        raw_to_mapped_raw = data["raw_to_mapped"]
        raw_to_mapped: dict[int, int | None] = {}
        for k, v in raw_to_mapped_raw.items():
            raw_to_mapped[int(k)] = int(v) if v is not None else None

        names_raw = data.get("names")
        names: dict[int, str] | None = None
        if names_raw is not None:
            names = {int(k): str(v) for k, v in names_raw.items()}

        return cls(raw_to_mapped=raw_to_mapped, names=names)


def filter_intervals_by_mapping(
    intervals,
    mapping: ClassificationMapping,
    value_field: str,
):
    """Filter intervals to retain only those with kept (non-removed) labels.

    If the intervals object doesn't have the value_field attribute,
    returns intervals unchanged (e.g. for regression tasks or plain domains).

    Args:
        intervals: An Interval-like object with start/end arrays and optionally
            a label array accessible via value_field.
        mapping: The classification mapping defining which labels are kept.
        value_field: Name of the attribute on intervals containing raw label IDs.

    Returns:
        Filtered intervals containing only entries with retained labels.
    """
    if not hasattr(intervals, value_field):
        return intervals

    values = getattr(intervals, value_field)
    keep_mask = mapping.kept_mask(values)
    return intervals.select_by_mask(keep_mask)

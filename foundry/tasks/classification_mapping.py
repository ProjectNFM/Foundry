"""Unified classification task mapping contract.

Provides a single source of truth for input-label-to-class relationships,
implicit filtering, derived class counts, and display names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from foundry.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationMapping:
    """User-friendly classification label mapping.

    Maps input labels (int or str) to output class names. Labels not listed
    are implicitly filtered out. Integer class IDs are auto-assigned from the
    order of first appearance of unique output names, or from an explicit
    ``order``; the tokenizer assigns these IDs to target values during
    tokenization.

    Args:
        mapping: Either a dict mapping input labels to output class names
            (all keys must be the same type), or a list of labels to keep.
            When a list is provided, each label is kept with its original
            name (identity mapping) and the list order defines class IDs.
        order: Optional explicit ordering of output class names for ID
            assignment. Must contain exactly the unique values from mapping.
            Ignored when mapping is a list (the list itself defines order).
    """

    mapping: dict[int | str, str] | list[int | str]
    order: list[str] | None = None

    _input_class_to_id: dict[int | str, int] = field(
        init=False, repr=False, compare=False
    )
    _kept_input_classes: frozenset[int | str] = field(
        init=False, repr=False, compare=False
    )
    _num_classes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.mapping, list):
            labels = self.mapping
            object.__setattr__(
                self, "mapping", {label: str(label) for label in labels}
            )
            if self.order is None:
                object.__setattr__(
                    self, "order", [str(label) for label in labels]
                )

        self._validate()

        if self.order is None:
            object.__setattr__(
                self,
                "order",
                list(dict.fromkeys(self.mapping.values())),
            )

        output_class_to_id = {name: i for i, name in enumerate(self.order)}
        input_class_to_id = {
            k: output_class_to_id[v] for k, v in self.mapping.items()
        }

        object.__setattr__(self, "_input_class_to_id", input_class_to_id)
        object.__setattr__(
            self, "_kept_input_classes", frozenset(self.mapping.keys())
        )
        object.__setattr__(self, "_num_classes", len(self.order))

    def _validate(self) -> None:
        if not self.mapping:
            raise ValueError(
                "ClassificationMapping must have at least one entry in mapping."
            )

        key_types = {type(k) for k in self.mapping.keys()}
        if len(key_types) > 1:
            raise ValueError(
                f"All mapping keys must be the same type (all int or all str), "
                f"got mixed types: {key_types}."
            )

        unique_values = set(self.mapping.values())

        if self.order is not None:
            if len(self.order) != len(set(self.order)):
                raise ValueError("order must not contain duplicate entries.")
            order_set = set(self.order)
            if order_set != unique_values:
                raise ValueError(
                    f"order must contain exactly the unique output class names "
                    f"from mapping. Expected {sorted(unique_values)}, "
                    f"got {sorted(self.order)}."
                )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> list[str]:
        """Ordered list of output class names (index = class ID)."""
        return list(self.order)

    @property
    def kept_input_classes(self) -> frozenset[int | str]:
        """Set of input labels that are in the mapping."""
        return self._kept_input_classes

    def map_to_class_ids(self, raw_values: np.ndarray) -> np.ndarray:
        """Map input label array to integer class IDs.

        Labels not in the mapping get -1 (defense-in-depth; they should
        already be filtered by the sampler).

        Args:
            raw_values: Array of input labels (int or str).

        Returns:
            Int64 array of mapped class IDs.
        """
        result = np.full(len(raw_values), -1, dtype=np.int64)
        for input_label, class_id in self._input_class_to_id.items():
            mask = raw_values == input_label
            result[mask] = class_id
        return result

    def kept_mask(self, raw_values: np.ndarray) -> np.ndarray:
        """Boolean mask where True indicates a label present in the mapping."""
        mask = np.zeros(len(raw_values), dtype=bool)
        for input_label in self._kept_input_classes:
            mask |= raw_values == input_label
        return mask

    def filter_and_remap(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter unlisted labels and remap kept ones in one operation.

        Returns:
            (mapped_values, keep_mask) where mapped_values has
            length == keep_mask.sum().
        """
        keep = self.kept_mask(values)
        mapped = self.map_to_class_ids(values[keep])
        return mapped, keep

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationMapping:
        """Construct from a parsed YAML/dict representation.

        Accepts ``mapping`` as either a dict (input-label -> class-name) or a
        list of labels to keep (identity mapping).  Handles YAML's tendency to
        produce string keys for numeric values.
        """
        mapping_raw = data["mapping"]

        if isinstance(mapping_raw, list):
            return cls(mapping=mapping_raw)

        mapping: dict[int | str, str] = {}

        for k, v in mapping_raw.items():
            key: int | str
            if isinstance(k, int):
                key = k
            elif isinstance(k, str):
                try:
                    key = int(k)
                except ValueError:
                    key = k
            else:
                key = k
            mapping[key] = str(v)

        order = data.get("order")
        if order is not None:
            order = list(order)

        return cls(mapping=mapping, order=order)


def filter_intervals_by_mapping(
    intervals,
    mapping: ClassificationMapping,
    value_field: str,
):
    """Filter intervals to retain only those with labels in the mapping.

    If the intervals object doesn't have the value_field attribute,
    returns intervals unchanged (e.g. for regression tasks or plain domains).

    Args:
        intervals: An Interval-like object with start/end arrays and optionally
            a label array accessible via value_field.
        mapping: The classification mapping defining which labels are kept.
        value_field: Name of the attribute on intervals containing label values.

    Returns:
        Filtered intervals containing only entries with retained labels.
    """
    if not hasattr(intervals, value_field):
        return intervals

    values = getattr(intervals, value_field)
    keep_mask = mapping.kept_mask(values)
    return intervals.select_by_mask(keep_mask)


def validate_task_mappings(
    task_configs: dict[str, TaskConfig],
    dataset,
    max_recordings: int = 5,
) -> None:
    """Verify classification mappings have at least some overlap with data.

    Called once during setup. Scans a few recordings to inform the user about
    labels in the data that are not in the mapping (will be filtered out).

    Raises:
        ValueError: If NO labels in the data match the mapping at all
            (likely a misconfiguration).
    """
    for name, cfg in task_configs.items():
        if cfg.class_mapping is None:
            continue
        kept = cfg.class_mapping.kept_input_classes
        value_key = cfg.target_extractor["value_key"]

        if not hasattr(dataset, "recording_ids"):
            continue

        all_unique: set = set()
        sample_ids = list(dataset.recording_ids)[:max_recordings]
        for rid in sample_ids:
            rec = dataset.get_recording(rid)
            try:
                values = rec.get_nested_attribute(value_key)
            except (AttributeError, KeyError):
                continue
            unique_raw = set(np.asarray(values).flat)
            all_unique.update(unique_raw)

        if not all_unique:
            continue

        not_in_mapping = all_unique - set(kept)
        in_mapping = all_unique & set(kept)

        if not in_mapping:
            raise ValueError(
                f"Task '{name}': none of the labels found in data "
                f"({sorted(all_unique)}) match the class_mapping. "
                f"This is likely a misconfiguration."
            )

        if not_in_mapping:
            logger.info(
                "Task '%s': labels %s found in data are not in the mapping "
                "and will be filtered out during training.",
                name,
                sorted(not_in_mapping),
            )

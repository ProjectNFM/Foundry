"""Inverse-frequency class weight computation from task configs."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import TargetExtractor

logger = logging.getLogger(__name__)


def _resolve_intervals_and_values(
    dataset, rid, split_intervals, extractor, value_field: str
):
    """Resolve label values from split intervals, falling back to the recording.

    The split intervals returned by ``get_sampling_intervals`` may not carry the
    expected value field when a dataset transform renames it (e.g.
    ``PrepareSleepStages`` maps raw ``id`` → ``values``). In that case, fetch the
    parent label attribute from the recording and filter it by the split time
    windows.

    Returns ``(intervals, values)`` where both are aligned, or ``(None, None)``
    if labels cannot be resolved.
    """
    if hasattr(split_intervals, value_field):
        return split_intervals, np.asarray(
            getattr(split_intervals, value_field)
        )

    parts = extractor.value_key.split(".")
    if len(parts) < 2:
        return None, None

    parent_path = ".".join(parts[:-1])
    try:
        rec = dataset.get_recording(rid)
        label_intervals = rec.get_nested_attribute(parent_path)
    except (AttributeError, KeyError):
        return None, None

    if hasattr(label_intervals, "select_by_interval"):
        label_intervals = label_intervals.select_by_interval(split_intervals)

    if hasattr(label_intervals, value_field):
        return label_intervals, np.asarray(
            getattr(label_intervals, value_field)
        )
    if hasattr(label_intervals, "id"):
        return label_intervals, np.asarray(getattr(label_intervals, "id"))

    return None, None


def _count_labels_for_task(
    dataset,
    split: str,
    extractor: TargetExtractor,
) -> Counter:
    value_field = extractor.value_key.split(".")[-1]
    train_intervals = dataset.get_sampling_intervals(split=split)
    counts: Counter = Counter()

    for _rid, split_intervals in train_intervals.items():
        intervals, values = _resolve_intervals_and_values(
            dataset, _rid, split_intervals, extractor, value_field
        )
        if values is None:
            continue

        if extractor.classification_mapping is not None:
            mapping = extractor.classification_mapping
            mapped, keep = mapping.filter_and_remap(values)
            if not np.any(keep):
                continue
            selected = intervals.select_by_mask(keep)
            for label in np.unique(mapped):
                label_mask = mapped == label
                durations = (
                    selected.end[label_mask] - selected.start[label_mask]
                )
                counts[int(label)] += durations.sum()
            continue

        unique_labels = np.unique(values)
        for label in unique_labels:
            selected = intervals.select_by_mask(values == label)
            counts[int(label)] += sum(selected.end - selected.start)

    return counts


def _inverse_frequency_weights(
    counts: Counter, num_classes: int, smoothing: float
) -> list[float]:
    total = sum(counts.values())
    if total == 0:
        return [1.0] * num_classes

    return [
        (total / (num_classes * max(counts.get(i, 0), 1))) ** smoothing
        for i in range(num_classes)
    ]


def compute_class_weights_for_tasks(
    task_configs: dict[str, TaskConfig],
    dataset,
    split: str = "train",
    smoothing: float = 1.0,
) -> dict[str, list[float]]:
    """Compute inverse-frequency class weights using task target extractors."""
    weights: dict[str, list[float]] = {}

    for name, cfg in task_configs.items():
        if cfg.kind not in ("binary", "multiclass"):
            continue

        extractor = cfg.extractor

        counts = _count_labels_for_task(dataset, split, extractor)
        task_weights = _inverse_frequency_weights(
            counts, cfg.output_dim, smoothing
        )
        weights[name] = task_weights

        logger.info(
            "Class weights for %s (smoothing=%.2f): %s (counts: %s)",
            name,
            smoothing,
            [f"{w:.3f}" for w in task_weights],
            dict(sorted(counts.items())),
        )

    return weights

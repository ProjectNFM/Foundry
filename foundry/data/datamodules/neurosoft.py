"""Neurosoft-specific sampling controls for cross-species experiments."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from foundry.data.datamodules.base import NeuralDataModule
from foundry.tasks.classification_mapping import ClassificationMapping

logger = logging.getLogger(__name__)


def _normalize_raw_values(
    values: np.ndarray, mapping: ClassificationMapping
) -> np.ndarray:
    """Normalize byte labels when a task mapping uses string keys."""
    first_key = next(iter(mapping.mapping))
    if not isinstance(first_key, str):
        return np.asarray(values)
    return np.asarray(
        [
            value.decode()
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ]
    )


def _interval_class_ids(
    interval,
    mapping: ClassificationMapping,
    value_field: str,
) -> np.ndarray:
    if not hasattr(interval, value_field):
        raise ValueError(
            f"Sampling controls require interval field '{value_field}'"
        )
    values = _normalize_raw_values(getattr(interval, value_field), mapping)
    class_ids = mapping.map_to_class_ids(values)
    if np.any(class_ids < 0):
        raise ValueError(
            "Sampling controls received labels excluded by the task class "
            "mapping. Apply standard class filtering first."
        )
    return class_ids


def _normalize_class_ids(
    selectors: Sequence[int | str], mapping: ClassificationMapping
) -> set[int]:
    class_name_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(mapping.class_names)
    }
    normalized = set()
    for selector in selectors:
        if isinstance(selector, str):
            if selector not in class_name_to_id:
                raise ValueError(
                    f"Unknown class '{selector}'. Available: "
                    f"{mapping.class_names}"
                )
            normalized.add(class_name_to_id[selector])
        else:
            class_id = int(selector)
            if not 0 <= class_id < mapping.num_classes:
                raise ValueError(
                    f"Class id {class_id} is outside "
                    f"[0, {mapping.num_classes - 1}]"
                )
            normalized.add(class_id)
    if not normalized:
        raise ValueError("At least one class must be selected per source")
    return normalized


def filter_source_class_intervals(
    intervals: dict,
    mapping: ClassificationMapping,
    value_field: str,
    class_ids_by_source: dict[str, Sequence[int | str]],
) -> dict:
    """Keep selected classes for configured sources and preserve all others."""
    normalized = {
        str(source_id): _normalize_class_ids(selectors, mapping)
        for source_id, selectors in class_ids_by_source.items()
    }
    selected_counts = {source_id: 0 for source_id in normalized}
    filtered = {}

    for recording_id, interval in intervals.items():
        source_id, _, _ = str(recording_id).partition("/")
        allowed = normalized.get(source_id)
        if allowed is None:
            filtered[recording_id] = interval
            continue

        class_ids = _interval_class_ids(interval, mapping, value_field)
        selected = interval.select_by_mask(np.isin(class_ids, list(allowed)))
        if len(selected.start) > 0:
            filtered[recording_id] = selected
            selected_counts[source_id] += len(selected.start)

    empty_sources = [
        source_id
        for source_id, selected_count in selected_counts.items()
        if selected_count == 0
    ]
    if empty_sources:
        raise ValueError(
            "Class filtering removed all training intervals for source(s): "
            f"{empty_sources}"
        )
    return filtered


def sample_uniform_source_class_intervals(
    intervals: dict,
    mapping: ClassificationMapping,
    value_field: str,
    total_count_by_source: dict[str, int],
    *,
    seed: int,
) -> dict:
    """Sample an exact, class-balanced training volume for each source."""
    requested_counts = {
        str(source_id): int(total_count)
        for source_id, total_count in total_count_by_source.items()
    }
    for source_id, total_count in requested_counts.items():
        if total_count <= 0:
            raise ValueError(
                "Uniform sampling counts must be positive, got "
                f"{total_count} for source '{source_id}'"
            )

    candidates = {
        source_id: [[] for _ in range(mapping.num_classes)]
        for source_id in requested_counts
    }
    for recording_id in sorted(intervals):
        interval = intervals[recording_id]
        source_id, _, _ = str(recording_id).partition("/")
        if source_id not in candidates:
            continue
        class_ids = _interval_class_ids(interval, mapping, value_field)
        for local_index, class_id in enumerate(class_ids):
            candidates[source_id][int(class_id)].append(
                (recording_id, local_index)
            )

    selected_by_recording: dict[str, list[int]] = {
        recording_id: [] for recording_id in intervals
    }
    for source_id, total_count in requested_counts.items():
        source_candidates = candidates[source_id]
        available = sum(len(items) for items in source_candidates)
        if total_count > available:
            raise ValueError(
                f"Uniform sampling requested {total_count} intervals for "
                f"source '{source_id}', but only {available} are available"
            )

        source_seed = seed + sum(
            (index + 1) * ord(char) for index, char in enumerate(source_id)
        )
        rng = np.random.default_rng(source_seed)
        for class_candidates in source_candidates:
            rng.shuffle(class_candidates)

        positions = [0] * mapping.num_classes
        selected_counts = [0] * mapping.num_classes
        selected_total = 0
        while selected_total < total_count:
            made_progress = False
            for class_id, class_candidates in enumerate(source_candidates):
                position = positions[class_id]
                if position >= len(class_candidates):
                    continue
                recording_id, local_index = class_candidates[position]
                positions[class_id] += 1
                selected_by_recording[recording_id].append(local_index)
                selected_counts[class_id] += 1
                selected_total += 1
                made_progress = True
                if selected_total == total_count:
                    break
            if not made_progress:
                raise RuntimeError("Uniform sampler exhausted unexpectedly")

        logger.info(
            "Uniform sampling kept %d intervals for source %s: %s",
            selected_total,
            source_id,
            dict(zip(mapping.class_names, selected_counts)),
        )

    sampled = {}
    for recording_id, interval in intervals.items():
        source_id, _, _ = str(recording_id).partition("/")
        if source_id not in requested_counts:
            sampled[recording_id] = interval
            continue
        local_indices = selected_by_recording[recording_id]
        if not local_indices:
            continue
        keep = np.zeros(len(interval.start), dtype=bool)
        keep[np.asarray(sorted(local_indices), dtype=int)] = True
        sampled[recording_id] = interval.select_by_mask(keep)
    return sampled


class NeurosoftMultispeciesDataModule(NeuralDataModule):
    """NeuralDataModule with training-only source/class sampling controls."""

    def __init__(
        self,
        *args,
        train_band_ids_by_source: dict[str, Sequence[int | str]] | None = None,
        train_uniform_band_total_count_by_source: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        filtered_sources = set(train_band_ids_by_source or {})
        uniform_sources = set(train_uniform_band_total_count_by_source or {})
        overlap = sorted(filtered_sources & uniform_sources)
        if overlap:
            raise ValueError(
                "A source cannot use both band filtering and uniform "
                f"sampling controls: {overlap}"
            )
        self.train_band_ids_by_source = train_band_ids_by_source or {}
        self.train_uniform_band_total_count_by_source = (
            train_uniform_band_total_count_by_source or {}
        )
        super().__init__(*args, **kwargs)

    def _sampling_task(self):
        candidates = [
            cfg
            for cfg in (self._task_configs or {}).values()
            if cfg.class_mapping is not None
            and cfg.target_extractor is not None
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Neurosoft sampling controls require exactly one configured "
                "classification task"
            )
        cfg = candidates[0]
        value_field = cfg.target_extractor["value_key"].split(".")[-1]
        return cfg.class_mapping, value_field

    def _filter_intervals(self, sampling_intervals, split=None):
        intervals = super()._filter_intervals(sampling_intervals, split=split)
        if split != "train" or not (
            self.train_band_ids_by_source
            or self.train_uniform_band_total_count_by_source
        ):
            return intervals

        mapping, value_field = self._sampling_task()
        if self.train_band_ids_by_source:
            intervals = filter_source_class_intervals(
                intervals,
                mapping,
                value_field,
                self.train_band_ids_by_source,
            )
        if self.train_uniform_band_total_count_by_source:
            intervals = sample_uniform_source_class_intervals(
                intervals,
                mapping,
                value_field,
                self.train_uniform_band_total_count_by_source,
                seed=self.seed,
            )
        return intervals


__all__ = [
    "NeurosoftMultispeciesDataModule",
    "filter_source_class_intervals",
    "sample_uniform_source_class_intervals",
]

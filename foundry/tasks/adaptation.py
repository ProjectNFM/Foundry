"""Task config adaptation for class filtering and grouping.

This module provides a generic mechanism to adapt TaskConfig objects at runtime
to support class subset selection and class grouping, without modifying base
YAML configs or reintroducing torch_brain's ModalitySpec layer.

Key design: adaptation happens once at startup via TargetExtractor.label_map,
keeping raw integer IDs in the data pipeline while remapping at tokenization time.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from temporaldata import Interval

from foundry.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskClassSchema:
    """Metadata for interpreting data.classes configuration for a task.

    Attributes:
        vocabulary: Dict mapping human-readable class names to raw integer IDs
            in the data (e.g. {"Eat": 0, "Talk": 1, ...}).
        interval_filter_field: Name of the attribute on Interval objects
            to filter by (e.g. "behavior_labels", "behavior_id").
        interval_filter_mode: How to interpret the filter field:
            - "names": field contains strings matching vocabulary keys (NeuroSoft)
            - "ids": field contains raw integer IDs (Ajile)
        grouping_presets: Optional dict of preset grouping dicts. Keys are
            preset names (e.g. "3band"), values are dicts mapping class names
            to group names (e.g. {"stim_500Hz": "low", ...}).
        group_order: Optional dict specifying sort order for groups
            (e.g. {"low": 0, "medium": 1, "high": 2}). If not provided,
            groups are sorted alphabetically.
        display_name_formatter: Optional function to format effective class
            names for display (e.g. strip prefixes for confusion matrices).
    """

    vocabulary: Dict[str, int]
    interval_filter_field: str
    interval_filter_mode: str  # "names" or "ids"
    grouping_presets: Optional[Dict[str, Dict[str, str]]] = None
    group_order: Optional[Dict[str, int]] = None
    display_name_formatter: Optional[Callable[[List[str]], List[str]]] = None


def _build_label_mapping(
    classes: Optional[List[str]],
    vocabulary: Dict[str, int],
    class_grouping: Optional[str | Dict[str, str]] = None,
    grouping_presets: Optional[Dict[str, Dict[str, str]]] = None,
    group_order: Optional[Dict[str, int]] = None,
) -> tuple[Dict[int, int], List[str]]:
    """Build label mapping and effective class names.

    Args:
        classes: Selected class names, or None for no adaptation.
        vocabulary: Map from class name to raw integer ID.
        class_grouping: None for raw subset mode, str for preset, or dict for
            custom grouping.
        grouping_presets: Available preset groupings (e.g. {"3band": {...}}).
        group_order: Optional dict specifying sort order for groups.

    Returns:
        Tuple of (label_map, effective_names) where label_map maps raw IDs to
        class indices [0, num_classes-1], and effective_names is the ordered
        list of class names.

    Raises:
        ValueError: If classes contain invalid names, preset not found, or
            grouping results in empty output.
    """
    if classes is None:
        return {}, []

    # Validate that all requested classes exist in vocabulary
    invalid_names = [c for c in classes if c not in vocabulary]
    if invalid_names:
        raise ValueError(
            f"Invalid class names: {invalid_names}. "
            f"Valid options: {list(vocabulary.keys())}"
        )

    if class_grouping is None:
        # Raw subset mode: dense remap raw_id -> 0..N-1, sorted by ID
        selected_freqs_sorted = sorted(classes, key=lambda c: vocabulary[c])
        mapping = {
            vocabulary[c]: i for i, c in enumerate(selected_freqs_sorted)
        }
        return mapping, selected_freqs_sorted

    # Resolve grouping dict
    if isinstance(class_grouping, str):
        if grouping_presets is None or class_grouping not in grouping_presets:
            raise ValueError(
                f"Preset '{class_grouping}' not found. Available: "
                f"{list(grouping_presets.keys()) if grouping_presets else []}"
            )
        grouping = grouping_presets[class_grouping]
    else:
        grouping = class_grouping

    # Grouping mode: collapse selected classes into groups
    # Build mapping: raw_id -> group_name
    selected_bands = set()
    mapping_raw_to_group = {}

    for class_name in classes:
        if class_name in grouping:
            raw_id = vocabulary[class_name]
            group_name = grouping[class_name]
            selected_bands.add(group_name)
            mapping_raw_to_group[raw_id] = group_name

    if not mapping_raw_to_group:
        raise ValueError(
            f"No valid class mappings after applying grouping. "
            f"Classes: {classes}, Grouping keys: {list(grouping.keys())}"
        )

    # Sort groups by group_order if provided, else alphabetically
    if group_order is not None:
        selected_band_order = sorted(
            selected_bands, key=lambda b: group_order.get(b, float("inf"))
        )
    else:
        selected_band_order = sorted(selected_bands)

    # Build final label mapping: raw_id -> group_index
    label_mapping = {
        raw_id: selected_band_order.index(group_name)
        for raw_id, group_name in mapping_raw_to_group.items()
    }

    return label_mapping, selected_band_order


def _adapt_task_config(
    base: TaskConfig,
    label_map: Dict[int, int],
    effective_names: List[str],
    display_formatter: Optional[Callable[[List[str]], List[str]]] = None,
) -> TaskConfig:
    """Create an adapted TaskConfig with effective class count and label mapping.

    Deep-copies the base config and updates:
    - head.output_dim to len(effective_names)
    - metrics.num_classes to len(effective_names) if metrics present
    - target_extractor.label_map to the provided mapping
    - class_names to formatted effective names

    Args:
        base: Base TaskConfig from YAML.
        label_map: Mapping from raw label IDs to class indices [0, num_classes-1].
        effective_names: Ordered list of effective class names.
        display_formatter: Optional function to format names for display.

    Returns:
        New TaskConfig with adaptations applied.
    """
    # Deep copy to avoid mutating YAML-loaded config
    cfg_dict = {
        "name": base.name,
        "head": copy.deepcopy(base.head),
        "target_extractor": copy.deepcopy(base.target_extractor),
        "loss": copy.deepcopy(base.loss),
        "metrics": copy.deepcopy(base.metrics) if base.metrics else None,
        "class_names": None,
        "metric_summary_modes": copy.deepcopy(base.metric_summary_modes),
    }

    # Update dimensions
    effective_dim = len(effective_names)
    cfg_dict["head"]["output_dim"] = effective_dim

    if cfg_dict["metrics"] is not None:
        cfg_dict["metrics"]["num_classes"] = effective_dim

    # Update label mapping in target extractor
    cfg_dict["target_extractor"]["label_map"] = label_map

    # Set class names with optional formatting
    if display_formatter is not None:
        cfg_dict["class_names"] = display_formatter(effective_names)
    else:
        cfg_dict["class_names"] = effective_names

    return TaskConfig.from_dict(cfg_dict)


def adapt_task_configs(
    task_configs: Dict[str, TaskConfig],
    schemas: Dict[str, TaskClassSchema],
    classes: Optional[List[str]],
    class_grouping: Optional[str | Dict[str, str]],
) -> Dict[str, TaskConfig]:
    """Adapt task configs for class filtering/grouping.

    For each task with a schema, builds label mapping and adapts the config.
    Tasks without a schema or continuous tasks are returned unchanged.

    Args:
        task_configs: Dict of TaskConfigs from get_tasks_for_experiment.
        schemas: Dict mapping task name to TaskClassSchema.
        classes: Selected class names, or None.
        class_grouping: Grouping mode (None, preset name, or custom dict).

    Returns:
        Dict of adapted task configs (or unchanged if no adaptation needed).
    """
    if classes is None:
        return task_configs

    adapted = {}
    for name, cfg in task_configs.items():
        schema = schemas.get(name)

        # Skip tasks without schema or continuous tasks
        if schema is None or cfg.kind == "continuous":
            adapted[name] = cfg
            continue

        # Build label mapping for this task
        label_map, effective_names = _build_label_mapping(
            classes,
            schema.vocabulary,
            class_grouping,
            schema.grouping_presets,
            schema.group_order,
        )

        # Adapt the config
        adapted_cfg = _adapt_task_config(
            cfg, label_map, effective_names, schema.display_name_formatter
        )
        adapted[name] = adapted_cfg

        logger.info(
            "Adapted task '%s': %d → %d classes (grouping=%s)",
            name,
            len(schema.vocabulary),
            len(effective_names),
            class_grouping,
        )

    return adapted


def filter_sampling_intervals(
    intervals: Interval,
    schema: TaskClassSchema,
    classes: List[str],
) -> Interval:
    """Filter sampling intervals by class membership.

    Args:
        intervals: Interval object from dataset.get_sampling_intervals.
        schema: TaskClassSchema describing how to interpret the filter field.
        classes: Selected class names to keep.

    Returns:
        New Interval with only trials matching selected classes.
    """
    filter_field = schema.interval_filter_field

    if not hasattr(intervals, filter_field):
        logger.warning(
            f"Interval object missing field '{filter_field}'; returning unchanged"
        )
        return intervals

    field_values = getattr(intervals, filter_field)

    if schema.interval_filter_mode == "names":
        # Filter by string class names (NeuroSoft)
        mask = np.isin(field_values, classes)
    else:  # ids mode
        # Convert class names to raw IDs, then filter (Ajile)
        raw_ids = np.array([schema.vocabulary[c] for c in classes])
        mask = np.isin(field_values, raw_ids)

    if not np.any(mask):
        # Return empty interval
        return intervals.select_by_mask(mask)

    return intervals.select_by_mask(mask)

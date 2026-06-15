"""Startup validation for task classification mappings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foundry.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


def validate_task_mappings(
    task_configs: dict[str, "TaskConfig"],
    dataset,
    max_recordings: int = 5,
) -> None:
    """Verify classification mappings cover actual data labels.

    Called once during setup. Scans a few recordings to catch mapping/data
    mismatches before training starts.

    Raises:
        ValueError: If raw label IDs appear in data that are not declared
            in the task's classification_mapping.
    """
    for name, cfg in task_configs.items():
        if cfg.classification_mapping is None:
            continue
        declared = set(cfg.classification_mapping.raw_to_mapped.keys())
        value_key = cfg.target_extractor["value_key"]

        if not hasattr(dataset, "recording_ids"):
            continue

        sample_ids = list(dataset.recording_ids)[:max_recordings]
        for rid in sample_ids:
            rec = dataset.get_recording(rid)
            try:
                values = rec.get_nested_attribute(value_key)
            except (AttributeError, KeyError):
                continue
            unique_raw = set(np.asarray(values).flat)
            undeclared = unique_raw - declared
            if undeclared:
                raise ValueError(
                    f"Task '{name}': raw label IDs {sorted(undeclared)} found in "
                    f"recording '{rid}' but not declared in classification_mapping. "
                    f"Add them to raw_to_mapped (mapped to an int or null)."
                )

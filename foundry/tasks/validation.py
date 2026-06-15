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
    """Verify classification mappings have at least some overlap with data.

    Called once during setup. Scans a few recordings to inform the user about
    labels in the data that are not in the mapping (will be filtered out).

    Raises:
        ValueError: If NO labels in the data match the mapping at all
            (likely a misconfiguration).
    """
    for name, cfg in task_configs.items():
        if cfg.classification_mapping is None:
            continue
        kept = cfg.classification_mapping.kept_inputs
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
                f"({sorted(all_unique)}) match the classification_mapping. "
                f"This is likely a misconfiguration."
            )

        if not_in_mapping:
            logger.info(
                "Task '%s': labels %s found in data are not in the mapping "
                "and will be filtered out during training.",
                name,
                sorted(not_in_mapping),
            )

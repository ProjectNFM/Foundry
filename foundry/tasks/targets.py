"""Target extraction from ``torch_brain.data`` samples during tokenization.

These callables live in the **data layer**: they read CPU-side ``Data`` objects
and return numpy arrays for timestamps and target values.

A :class:`TargetExtractor` is configured per task via Hydra config
and invoked during dataset tokenization. Label remapping and dtype normalization
happen here so downstream loss functions and metrics see prepared targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_brain.batching import chain, collate, track_batch
from torch_brain.data import Data

if TYPE_CHECKING:
    from foundry.tasks.classification_mapping import ClassificationMapping
    from foundry.tasks.config import TaskConfig


@dataclass(frozen=True)
class TargetExtractor:
    """Extract per-sample target timestamps and values from a ``Data`` object.

    A frozen, serializable data transform invoked during tokenization. Keys use
    dot notation to reach nested fields on :class:`torch_brain.data.Data` containers.

    Args:
        timestamp_key: Dot-separated path to the timestamp array for each
            target (e.g. ``"active_behavior_trials.timestamps"``).
        value_key: Dot-separated path to the target values array (e.g.
            ``"active_behavior_trials.behavior_id"`` or
            ``"pose_trajectories.values"``).
        class_mapping: Optional unified classification mapping.
            When set, raw labels are remapped via
            :meth:`ClassificationMapping.apply`.

    Returns:
        A dict with:

        - ``"timestamps"``: numpy array of query timestamps aligned with targets.
        - ``"values"``: numpy array of class indices or regression targets.
          ``float64`` arrays are cast to ``float32``; integer dtypes are left
          unchanged.
    """

    timestamp_key: str
    value_key: str
    class_mapping: ClassificationMapping | None = None

    def __call__(self, data: Data) -> dict:
        timestamps = data.get_nested_attribute(self.timestamp_key)
        values = data.get_nested_attribute(self.value_key)

        if self.class_mapping is not None:
            values = self.class_mapping.apply(values)

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}


def extract_multitask_targets(
    task_configs: dict[str, TaskConfig],
    data: Data,
) -> tuple[
    torch.Tensor,
    dict[str, np.ndarray],
    torch.Tensor,
    dict[str, np.ndarray],
]:
    """Extract targets for all configured tasks from a single data sample.

    Iterates tasks in sorted name order, builds a :class:`TargetExtractor` per
    task (injecting ``class_mapping`` when present), and collates the
    per-task timestamps into a single ``output_timestamps`` tensor with a
    parallel ``task_index`` tensor.

    Returns:
        Tuple of ``(output_timestamps, target_values, output_task_index,
        target_weights)`` where *output_timestamps* is a 1-D float tensor,
        *target_values* / *target_weights* are dicts keyed by task name, and
        *output_task_index* is a 1-D long tensor with 1-based task indices
        (0 reserved for padding).
    """
    sorted_names = sorted(task_configs.keys())
    name_to_idx = {n: i for i, n in enumerate(sorted_names)}

    all_timestamps: list[np.ndarray] = []
    task_indices: list[int] = []
    target_values: dict[str, np.ndarray] = {}
    target_weights: dict[str, np.ndarray] = {}

    for name in sorted_names:
        cfg = task_configs[name]

        targets = cfg.extractor(data)
        timestamps = targets["timestamps"]
        if timestamps is None or len(timestamps) == 0:
            continue

        idx = name_to_idx[name]
        all_timestamps.append(timestamps)
        task_indices.append(idx)
        target_values[name] = targets["values"]
        target_weights[name] = np.ones_like(timestamps, dtype=np.float32)

    if not all_timestamps:
        raise ValueError("No targets extracted from data for configured tasks")

    if len(all_timestamps) == 1:
        output_timestamps = torch.as_tensor(all_timestamps[0])
        output_task_index = torch.full(
            (len(output_timestamps),),
            task_indices[0] + 1,
            dtype=torch.long,
        )
    else:
        output_timestamps, batch = collate(
            [
                (chain(all_timestamps[i]), track_batch(all_timestamps[i]))
                for i in range(len(all_timestamps))
            ]
        )
        output_task_index = torch.tensor(task_indices)[batch] + 1

    return output_timestamps, target_values, output_task_index, target_weights

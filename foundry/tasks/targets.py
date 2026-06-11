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
from torch_brain.data import Data

if TYPE_CHECKING:
    from foundry.tasks.classification_mapping import ClassificationMapping


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
        label_map: Optional legacy mapping from raw label values to training
            indices. Deprecated when classification_mapping is set.
        classification_mapping: Optional unified classification mapping.
            Takes precedence over label_map when set.

    Returns:
        A dict with:

        - ``"timestamps"``: numpy array of query timestamps aligned with targets.
        - ``"values"``: numpy array of class indices or regression targets.
          ``float64`` arrays are cast to ``float32``; integer dtypes are left
          unchanged.
    """

    timestamp_key: str
    value_key: str
    label_map: dict[int, int] | None = None
    classification_mapping: ClassificationMapping | None = None

    def __call__(self, data: Data) -> dict:
        timestamps = data.get_nested_attribute(self.timestamp_key)
        values = data.get_nested_attribute(self.value_key)

        if self.classification_mapping is not None:
            values = self.classification_mapping.apply(values)
        elif self.label_map is not None:
            mapped = values.copy()
            for src, dst in self.label_map.items():
                mapped[values == src] = dst
            values = mapped

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}

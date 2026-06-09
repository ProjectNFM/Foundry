"""Target extraction from ``temporaldata`` samples during tokenization.

These callables live in the **data layer**: they read CPU-side ``Data`` objects
and return numpy arrays for timestamps and target values.

A :class:`TargetExtractor` is configured per task via Hydra config
and invoked during dataset tokenization. Label remapping and dtype normalization
happen here so downstream loss functions and metrics see prepared targets only.
"""

from dataclasses import dataclass

import numpy as np
from temporaldata import Data


@dataclass(frozen=True)
class TargetExtractor:
    """Extract per-sample target timestamps and values from a ``Data`` object.

    A frozen, serializable data transform invoked during tokenization. Keys use
    dot notation to reach nested fields on :class:`temporaldata.Data` containers.

    Args:
        timestamp_key: Dot-separated path to the timestamp array for each
            target (e.g. ``"motor_imagery_trials.timestamps"``).
        value_key: Dot-separated path to the target values array (e.g.
            ``"motor_imagery_trials.movement_ids"`` or
            ``"pose_trajectories.values"``).
        label_map: Optional mapping from raw label values to training indices.
            Applied before targets reach the loss or metrics. Use to collapse
            classes (e.g. ``{1: 0, 2: 1}`` for left/right from a 5-class task).

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

    def __call__(self, data: Data) -> dict:
        timestamps = data.get_nested_attribute(self.timestamp_key)
        values = data.get_nested_attribute(self.value_key)

        if self.label_map is not None:
            mapped = np.empty_like(values)
            for src, dst in self.label_map.items():
                mapped[values == src] = dst
            values = mapped

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}

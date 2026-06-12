"""Target extraction from ``torch_brain.data`` samples during tokenization.

These callables live in the **data layer**: they read CPU-side ``Data`` objects
and return numpy arrays for timestamps and target values.

A :class:`TargetExtractor` is configured per task via Hydra config
and invoked during dataset tokenization. Label remapping and dtype normalization
happen here so downstream loss functions and metrics see prepared targets only.
"""

from dataclasses import dataclass

import numpy as np
from torch_brain.data import Data


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
            # Initialize mapped array with -1 sentinel to catch unmapped values
            mapped = np.full_like(values, -1, dtype=np.int64)
            any_mapped = False
            for src, dst in self.label_map.items():
                mask = values == src
                if np.any(mask):
                    mapped[mask] = dst
                    any_mapped = True

            if any_mapped:
                # Check for unmapped values (sentinel -1)
                unmapped_mask = mapped == -1
                if np.any(unmapped_mask):
                    unmapped_ids = np.unique(values[unmapped_mask])
                    raise ValueError(
                        f"Found {np.sum(unmapped_mask)} samples with unmapped label IDs: "
                        f"{unmapped_ids}. Label mapping is incomplete. "
                        f"Configured mapping: {self.label_map}"
                    )
                values = mapped.astype(values.dtype)
            else:
                # No labels matched - raise error rather than silently using -1
                found_ids = np.unique(values)
                raise ValueError(
                    f"No labels found in data matching label_map keys. "
                    f"Data contains: {found_ids}, Label map: {self.label_map}"
                )

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}

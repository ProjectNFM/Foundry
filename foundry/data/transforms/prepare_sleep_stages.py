from __future__ import annotations

import types

import numpy as np
from torch_brain.data import Data

_UNKNOWN_STAGE_ID = 6


class PrepareSleepStages:
    """Materialize sleep-stage timestamps and class values from interval annotations.

    Reads the ``stages`` interval (``start``, ``end``, ``id``) from the Data
    object, computes epoch midpoints as timestamps, filters out unknown stages
    (``id == 6``), and replaces ``data.stages`` with a namespace that exposes
    ``.timestamps`` and ``.values`` for the downstream TargetExtractor.
    """

    def __call__(self, data: Data) -> Data:
        start = np.asarray(data.stages.start, dtype=np.float64)
        end = np.asarray(data.stages.end, dtype=np.float64)
        ids = np.asarray(data.stages.id, dtype=np.int64)

        keep = ids != _UNKNOWN_STAGE_ID
        timestamps = ((start + end) / 2.0)[keep]
        values = ids[keep]

        data.stages = types.SimpleNamespace(
            timestamps=timestamps,
            values=values,
            start=start[keep],
            end=end[keep],
        )
        return data

from __future__ import annotations

import types

import numpy as np
from torch_brain.data import Data


class PrepareSleepStages:
    """Materialize sleep-stage timestamps from interval annotations."""

    def __call__(self, data: Data) -> Data:
        start = np.asarray(data.stages.start, dtype=np.float64)
        end = np.asarray(data.stages.end, dtype=np.float64)
        ids = np.asarray(data.stages.id, dtype=np.int64)

        timestamps = (start + end) / 2.0

        data.stages = types.SimpleNamespace(
            timestamps=timestamps,
            values=ids,
            start=start,
            end=end,
        )
        return data

from dataclasses import dataclass

import numpy as np
from torch_brain.data import Data

from foundry.tasks.targets import TargetExtractor


@dataclass
class _Trials:
    timestamps: np.ndarray
    behavior_id: np.ndarray


def _make_trials_data(behavior_ids: np.ndarray) -> Data:
    timestamps = np.arange(len(behavior_ids), dtype=np.float64) * 0.1
    return Data(
        active_behavior_trials=_Trials(
            timestamps=timestamps,
            behavior_id=behavior_ids,
        )
    )


class TestTargetExtractor:
    def test_extracts_nested_timestamps_and_values(self):
        behavior_ids = np.array([0, 1, 2], dtype=np.int64)
        data = _make_trials_data(behavior_ids)

        extractor = TargetExtractor(
            timestamp_key="active_behavior_trials.timestamps",
            value_key="active_behavior_trials.behavior_id",
        )
        result = extractor(data)

        assert np.array_equal(result["timestamps"], np.array([0.0, 0.1, 0.2]))
        assert np.array_equal(result["values"], behavior_ids)

    def test_float64_values_converted_to_float32(self):
        values = np.array([1.5, 2.5], dtype=np.float64)
        data = Data(
            pose_trajectories=_Trials(
                timestamps=np.array([0.0, 1.0]),
                behavior_id=values,
            )
        )

        extractor = TargetExtractor(
            timestamp_key="pose_trajectories.timestamps",
            value_key="pose_trajectories.behavior_id",
        )
        result = extractor(data)

        assert result["values"].dtype == np.float32
        assert np.allclose(result["values"], values.astype(np.float32))

from dataclasses import dataclass

import numpy as np
from temporaldata import Data

from foundry.tasks.targets import TargetExtractor


@dataclass
class _Trials:
    timestamps: np.ndarray
    movement_ids: np.ndarray


def _make_trials_data(movement_ids: np.ndarray) -> Data:
    timestamps = np.arange(len(movement_ids), dtype=np.float64) * 0.1
    return Data(
        motor_imagery_trials=_Trials(
            timestamps=timestamps,
            movement_ids=movement_ids,
        )
    )


class TestTargetExtractor:
    def test_extracts_nested_timestamps_and_values(self):
        movement_ids = np.array([0, 1, 2], dtype=np.int64)
        data = _make_trials_data(movement_ids)

        extractor = TargetExtractor(
            timestamp_key="motor_imagery_trials.timestamps",
            value_key="motor_imagery_trials.movement_ids",
        )
        result = extractor(data)

        assert np.array_equal(result["timestamps"], np.array([0.0, 0.1, 0.2]))
        assert np.array_equal(result["values"], movement_ids)

    def test_label_map_remaps_known_values(self):
        movement_ids = np.array([1, 2, 1, 2], dtype=np.int64)
        data = _make_trials_data(movement_ids)

        extractor = TargetExtractor(
            timestamp_key="motor_imagery_trials.timestamps",
            value_key="motor_imagery_trials.movement_ids",
            label_map={1: 0, 2: 1},
        )
        result = extractor(data)

        assert np.array_equal(result["values"], np.array([0, 1, 0, 1]))

    def test_float64_values_converted_to_float32(self):
        values = np.array([1.5, 2.5], dtype=np.float64)
        data = Data(
            pose_trajectories=_Trials(
                timestamps=np.array([0.0, 1.0]),
                movement_ids=values,
            )
        )

        extractor = TargetExtractor(
            timestamp_key="pose_trajectories.timestamps",
            value_key="pose_trajectories.movement_ids",
        )
        result = extractor(data)

        assert result["values"].dtype == np.float32
        assert np.allclose(result["values"], values.astype(np.float32))

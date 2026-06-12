from dataclasses import dataclass
import unittest

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


class TestTargetExtractor(unittest.TestCase):
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

    def test_label_map_remaps_known_values(self):
        behavior_ids = np.array([1, 2, 1, 2], dtype=np.int64)
        data = _make_trials_data(behavior_ids)

        extractor = TargetExtractor(
            timestamp_key="active_behavior_trials.timestamps",
            value_key="active_behavior_trials.behavior_id",
            label_map={1: 0, 2: 1},
        )
        result = extractor(data)

        assert np.array_equal(result["values"], np.array([0, 1, 0, 1]))

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

    def test_label_map_handles_unmapped_values_safely(self):
        """label_map with unmapped values raises clear error."""
        # Data has IDs 0, 1, 2, 3 but mapping only covers 0, 1
        behavior_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        data = _make_trials_data(behavior_ids)

        extractor = TargetExtractor(
            timestamp_key="active_behavior_trials.timestamps",
            value_key="active_behavior_trials.behavior_id",
            label_map={0: 0, 1: 1},  # Incomplete mapping
        )

        # Should raise ValueError for unmapped IDs 2, 3
        with self.assertRaises(ValueError) as context:
            extractor(data)
        
        error_msg = str(context.exception)
        assert "unmapped" in error_msg.lower()
        assert "2" in error_msg or "3" in error_msg

    def test_label_map_complete_mapping_works(self):
        """label_map with all labels mapped works correctly."""
        behavior_ids = np.array([2, 3, 2], dtype=np.int64)
        data = _make_trials_data(behavior_ids)

        extractor = TargetExtractor(
            timestamp_key="active_behavior_trials.timestamps",
            value_key="active_behavior_trials.behavior_id",
            label_map={2: 0, 3: 1},
        )
        result = extractor(data)

        assert np.array_equal(result["values"], np.array([0, 1, 0]))

from dataclasses import dataclass

import numpy as np
import torch
from torch_brain.data import Data

from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import TargetExtractor, extract_multitask_targets


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


class TestExtractMultitaskTargetsSSL:
    """Test extract_multitask_targets with SSL tasks (no extractor)."""

    def test_skips_tasks_without_extractors(self):
        behavior_ids = np.array([0, 1, 2], dtype=np.int64)
        data = _make_trials_data(behavior_ids)

        task_configs = {
            "supervised_task": TaskConfig(
                name="supervised_task",
                head={"output_dim": 3},
                target_extractor={
                    "timestamp_key": "active_behavior_trials.timestamps",
                    "value_key": "active_behavior_trials.behavior_id",
                },
                loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            ),
            "masked_reconstruction": TaskConfig(
                name="masked_reconstruction",
                head={"output_dim": 1},
                target_extractor=None,
                loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
            ),
        }

        timestamps, values, task_index, weights = extract_multitask_targets(
            task_configs, data
        )

        assert timestamps.shape[0] == 3
        assert "supervised_task" in values
        assert "masked_reconstruction" not in values

    def test_all_ssl_tasks_returns_empty_tensors(self):
        data = Data()
        task_configs = {
            "masked_reconstruction": TaskConfig(
                name="masked_reconstruction",
                head={"output_dim": 1},
                target_extractor=None,
                loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
            ),
        }

        timestamps, values, task_index, weights = extract_multitask_targets(
            task_configs, data
        )

        assert timestamps.shape == (0,)
        assert task_index.shape == (0,)
        assert len(values) == 0
        assert len(weights) == 0
        assert timestamps.dtype == torch.float32
        assert task_index.dtype == torch.long

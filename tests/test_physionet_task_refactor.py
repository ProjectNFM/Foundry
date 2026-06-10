"""End-to-end tests for Physionet task-config migration (issue 06)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from temporaldata import Data, Interval, RegularTimeSeries

from foundry.data.datamodules.physionet import PhysionetDataModule
from foundry.data.datasets.schalk_wolpaw_physionet_2009 import (
    SchalkWolpawPhysionet2009,
)
from foundry.data.datasets.mixins import TaskMixin
from foundry.models import POYOEEGModel, EEGTokenizer, FixedChannelStrategy
from foundry.models.embeddings.temporal import PatchLinearEmbedding
from foundry.tasks.config import TaskConfig
from foundry.training import FoundryModule

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "configs" / "tasks"


class MockChannels:
    def __init__(self, channel_ids):
        self.id = np.array(channel_ids)
        self.type = np.array(["EEG"] * len(channel_ids))


class MockSession:
    def __init__(self, session_id):
        self.id = session_id


def _make_physionet_task_configs(task_type: str = "RightHandFeetImagery"):
    return PhysionetDataModule.get_tasks_for_experiment(task_type)


def _make_poyo_model(task_configs: dict[str, TaskConfig]) -> POYOEEGModel:
    embed_dim = 16
    patch_samples = 50
    tokenizer = EEGTokenizer(
        channel_strategy=FixedChannelStrategy(num_channels=4),
        temporal_embedding=PatchLinearEmbedding(
            embed_dim=embed_dim,
            num_input_channels=4,
            patch_samples=patch_samples,
        ),
        embed_dim=embed_dim,
        patch_duration=0.5,
        stride=0.5,
        channel_fusion="add",
    )
    return POYOEEGModel(
        tokenizer=tokenizer,
        task_configs=task_configs,
        embed_dim=embed_dim,
        sequence_length=2.0,
        latent_step=0.5,
        num_latents_per_step=1,
        zero_output_timestamps=True,
    )


def _make_motor_imagery_sample(movement_ids: np.ndarray) -> Data:
    num_channels = 4
    num_samples = 200
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    eeg = RegularTimeSeries(
        signal=signal,
        sampling_rate=100.0,
        domain_start=0.0,
    )

    class Trials:
        def __init__(self):
            n = len(movement_ids)
            self.timestamps = np.linspace(0.5, 1.5, n, dtype=np.float64)
            self.movement_ids = movement_ids
            self.movements = np.array(
                ["right_hand" if i % 2 == 0 else "feet" for i in range(n)]
            )

    data = Data(eeg=eeg, domain=Interval(0.0, 2.0))
    data.channels = MockChannels([f"ch{i}" for i in range(num_channels)])
    data.session = MockSession("session1")
    data.motor_imagery_trials = Trials()
    data._absolute_start = 0.0
    return data


def test_dataset_uses_task_mixin_not_modality_mixin():
    assert issubclass(SchalkWolpawPhysionet2009, TaskMixin)
    assert (
        "motor_imagery_right_feet" in SchalkWolpawPhysionet2009.AVAILABLE_TASKS
    )


def test_get_tasks_for_experiment_returns_task_configs():
    task_configs = _make_physionet_task_configs("RightHandFeetImagery")

    assert set(task_configs.keys()) == {"motor_imagery_right_feet"}
    cfg = task_configs["motor_imagery_right_feet"]
    assert isinstance(cfg, TaskConfig)
    assert cfg.name == "motor_imagery_right_feet"
    assert cfg.class_names == ["Right hand", "Feet"]


def test_poyo_model_accepts_task_configs_and_exposes_router():
    task_configs = _make_physionet_task_configs("RightHandFeetImagery")
    model = _make_poyo_model(task_configs)

    assert model.task_configs == task_configs
    assert hasattr(model, "router")
    assert model.router.num_tasks == 1
    assert "motor_imagery_right_feet" in model.router.heads


def test_tokenize_uses_target_extractor_for_labels():
    task_configs = _make_physionet_task_configs("RightHandFeetImagery")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    movement_ids = np.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=np.int64)
    data = _make_motor_imagery_sample(movement_ids)
    tokens = model.tokenize(data)

    target_values = tokens["target_values"].obj
    assert "motor_imagery_right_feet" in target_values
    assert np.array_equal(
        target_values["motor_imagery_right_feet"], [0, 1, 0, 1, 0, 1, 0, 1]
    )


def test_compute_class_weights_uses_target_extractor(data_root):
    from tests.conftest import skip_if_missing_dataset

    skip_marker = skip_if_missing_dataset(
        "schalk_wolpaw_physionet_2009", data_root
    )
    if skip_marker.args[0]:
        pytest.skip(skip_marker.kwargs["reason"])

    dm = PhysionetDataModule(
        root=str(data_root),
        task_type="RightHandFeetImagery",
        fold_number=0,
        sequence_length=2.0,
    )
    dm.setup("fit")
    weights = dm.compute_class_weights(smoothing=1.0)

    assert "motor_imagery_right_feet" in weights
    assert len(weights["motor_imagery_right_feet"]) == 2
    assert all(w > 0 for w in weights["motor_imagery_right_feet"])


def test_foundry_module_training_step_on_tokenized_batch():
    from torch_brain.data import collate

    task_configs = _make_physionet_task_configs("RightHandFeetImagery")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    movement_ids = np.array([2, 3, 2, 3, 2, 3, 2, 3], dtype=np.int64)
    data = _make_motor_imagery_sample(movement_ids)
    batch = collate([model.tokenize(data)])

    module = FoundryModule(model=model)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)

"""End-to-end tests for Ajile task-config migration (issue 07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_brain.data import (
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

from foundry.models import POYOEEGModel, EEGTokenizer, FixedChannelStrategy
from foundry.models.embeddings.temporal import PatchLinearEmbedding
from foundry.tasks.config import TaskConfig
from foundry.training import FoundryModule

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "configs" / "tasks"

_TASK_TYPE_TO_YAML = {
    "behavior": ["ajile_active_behavior"],
    "active_vs_inactive": ["ajile_inactive_active"],
    "pose_estimation": ["ajile_pose_estimation"],
}


class MockChannels:
    def __init__(self, channel_ids):
        self.id = np.array(channel_ids)
        self.type = np.array(["ECOG"] * len(channel_ids))


class MockSession:
    def __init__(self, session_id):
        self.id = session_id


def _make_ajile_task_configs(task_type: str = "behavior"):
    names = _TASK_TYPE_TO_YAML[task_type]
    configs = {}
    for name in names:
        tc = TaskConfig.from_yaml(TASKS_DIR / f"{name}.yaml")
        configs[tc.name] = tc
    return configs


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


@pytest.mark.parametrize(
    ("task_type", "expected_task", "expected_kind", "class_names"),
    [
        (
            "behavior",
            "ajile_active_behavior",
            "multiclass",
            ["Eat", "Talk", "TV", "Computer/Phone", "Other Activity"],
        ),
        (
            "active_vs_inactive",
            "ajile_inactive_active",
            "binary",
            ["Active", "Inactive"],
        ),
        (
            "pose_estimation",
            "ajile_pose_estimation",
            "continuous",
            None,
        ),
    ],
)
def test_task_configs_load_correctly(
    task_type, expected_task, expected_kind, class_names
):
    task_configs = _make_ajile_task_configs(task_type)

    assert set(task_configs.keys()) == {expected_task}
    cfg = task_configs[expected_task]
    assert isinstance(cfg, TaskConfig)
    assert cfg.name == expected_task
    assert cfg.kind == expected_kind
    assert cfg.class_names == class_names


def test_poyo_model_accepts_task_configs_and_exposes_router():
    task_configs = _make_ajile_task_configs("behavior")
    model = _make_poyo_model(task_configs)

    assert model.task_configs == task_configs
    assert hasattr(model, "router")
    assert model.router.num_tasks == 1
    assert "ajile_active_behavior" in model.router.heads


def _make_behavior_sample(behavior_ids: np.ndarray) -> Data:
    num_channels = 4
    num_samples = 200
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    ecog = RegularTimeSeries(
        signal=signal,
        sampling_rate=100.0,
        domain_start=0.0,
    )

    class Trials:
        def __init__(self):
            n = len(behavior_ids)
            self.timestamps = np.linspace(0.5, 1.5, n, dtype=np.float64)
            self.behavior_id = behavior_ids

    data = Data(ecog=ecog, domain=Interval(0.0, 2.0))
    data.channels = MockChannels([f"ch{i}" for i in range(num_channels)])
    data.session = MockSession("session1")
    data.active_behavior_trials = Trials()
    data._absolute_start = 0.0
    return data


def test_tokenize_uses_target_extractor_for_behavior_labels():
    task_configs = _make_ajile_task_configs("behavior")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
    data = _make_behavior_sample(behavior_ids)
    tokens = model.tokenize(data)

    target_values = tokens["target_values"].obj
    assert "ajile_active_behavior" in target_values
    assert np.array_equal(target_values["ajile_active_behavior"], behavior_ids)


def _make_pose_sample(num_targets: int = 4) -> Data:
    num_channels = 4
    num_samples = 200
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    ecog = RegularTimeSeries(
        signal=signal,
        sampling_rate=100.0,
        domain_start=0.0,
    )

    timestamps = np.linspace(0.5, 1.5, num_targets, dtype=np.float64)
    values = np.random.randn(num_targets, 18).astype(np.float32)

    data = Data(ecog=ecog, domain=Interval(0.0, 2.0))
    data.channels = MockChannels([f"ch{i}" for i in range(num_channels)])
    data.session = MockSession("session1")
    data.pose_trajectories = IrregularTimeSeries(
        timestamps=timestamps,
        values=values,
        domain=Interval(0.0, 2.0),
    )
    data._absolute_start = 0.0
    return data


def test_tokenize_uses_target_extractor_for_pose_values():
    task_configs = _make_ajile_task_configs("pose_estimation")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    data = _make_pose_sample(num_targets=3)
    tokens = model.tokenize(data)

    target_values = tokens["target_values"].obj
    assert "ajile_pose_estimation" in target_values
    assert target_values["ajile_pose_estimation"].shape == (3, 18)


def test_foundry_module_training_step_classification():
    from torch_brain.batching import collate

    task_configs = _make_ajile_task_configs("behavior")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
    data = _make_behavior_sample(behavior_ids)
    batch = collate([model.tokenize(data)])

    module = FoundryModule(model=model)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_foundry_module_training_step_regression():
    from torch_brain.batching import collate

    task_configs = _make_ajile_task_configs("pose_estimation")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    data = _make_pose_sample(num_targets=4)
    batch = collate([model.tokenize(data)])

    module = FoundryModule(model=model)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)

"""End-to-end tests for Neurosoft task-config migration (issue 08)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_brain.data import Data, Interval, RegularTimeSeries

from foundry.data.datasets.neurosoft import NeurosoftMinipigs2026
from foundry.models import POYOEEGModel, EEGTokenizer, FixedChannelStrategy
from foundry.models.embeddings.temporal import PatchLinearEmbedding
from foundry.tasks.config import TaskConfig
from foundry.training import FoundryModule

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "configs" / "tasks"


class MockChannels:
    def __init__(self, channel_ids):
        self.id = np.array(channel_ids)
        self.type = np.array(["ECOG"] * len(channel_ids))


class MockSession:
    def __init__(self, session_id):
        self.id = session_id


def _make_neurosoft_task_configs(task_type: str = "on_vs_off"):
    return NeurosoftMinipigs2026.get_tasks_for_experiment(task_type)


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


def test_dataset_uses_task_mixin():
    from foundry.data.datasets.mixins import TaskMixin

    assert issubclass(NeurosoftMinipigs2026, TaskMixin)
    assert "neurosoft_on_vs_off" in NeurosoftMinipigs2026.AVAILABLE_TASKS
    assert "neurosoft_acoustic_stim" in NeurosoftMinipigs2026.AVAILABLE_TASKS


def test_dataset_has_no_readout_class_names():
    assert not hasattr(NeurosoftMinipigs2026, "READOUT_CLASS_NAMES") or (
        NeurosoftMinipigs2026.READOUT_CLASS_NAMES == {}
    )


@pytest.mark.parametrize(
    ("task_type", "expected_task", "expected_kind", "class_names"),
    [
        (
            "on_vs_off",
            "neurosoft_on_vs_off",
            "binary",
            ["off", "on"],
        ),
        (
            "acoustic_stim",
            "neurosoft_acoustic_stim",
            "multiclass",
            [
                "stim_100Hz",
                "stim_200Hz",
                "stim_300Hz",
                "stim_400Hz",
                "stim_500Hz",
                "stim_650Hz",
                "stim_800Hz",
                "stim_1000Hz",
                "stim_12000Hz",
                "stim_13000Hz",
                "stim_1500Hz",
                "stim_2000Hz",
                "stim_3000Hz",
                "stim_4000Hz",
                "stim_5000Hz",
                "stim_7700Hz",
                "stim_8000Hz",
                "stim_9500Hz",
                "stim_10000Hz",
                "stim_15000Hz",
                "stim_16000Hz",
                "stim_18000Hz",
                "stim_20000Hz",
                "stim_30000Hz",
                "stim_40000Hz",
                "stim_wn",
            ],
        ),
    ],
)
def test_get_tasks_for_experiment_returns_task_configs(
    task_type, expected_task, expected_kind, class_names
):
    task_configs = _make_neurosoft_task_configs(task_type)

    assert set(task_configs.keys()) == {expected_task}
    cfg = task_configs[expected_task]
    assert isinstance(cfg, TaskConfig)
    assert cfg.name == expected_task
    assert cfg.kind == expected_kind
    assert cfg.class_names == class_names


def test_poyo_model_accepts_task_configs_and_exposes_router():
    task_configs = _make_neurosoft_task_configs("on_vs_off")
    model = _make_poyo_model(task_configs)

    assert model.task_configs == task_configs
    assert hasattr(model, "router")
    assert model.router.num_tasks == 1
    assert "neurosoft_on_vs_off" in model.router.heads


def _make_on_vs_off_sample(behavior_ids: np.ndarray) -> Data:
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
            self.behavior_ids = behavior_ids

    data = Data(ecog=ecog, domain=Interval(0.0, 2.0))
    data.channels = MockChannels([f"ch{i}" for i in range(num_channels)])
    data.session = MockSession("session1")
    data.on_vs_off_trials = Trials()
    data._absolute_start = 0.0
    return data


def _make_acoustic_stim_sample(behavior_ids: np.ndarray) -> Data:
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
            self.behavior_ids = behavior_ids

    data = Data(ecog=ecog, domain=Interval(0.0, 2.0))
    data.channels = MockChannels([f"ch{i}" for i in range(num_channels)])
    data.session = MockSession("session1")
    data.acoustic_stim_trials = Trials()
    data._absolute_start = 0.0
    return data


def test_tokenize_extracts_on_vs_off_labels():
    task_configs = _make_neurosoft_task_configs("on_vs_off")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    data = _make_on_vs_off_sample(behavior_ids)
    tokens = model.tokenize(data)

    target_values = tokens["target_values"].obj
    assert "neurosoft_on_vs_off" in target_values
    assert np.array_equal(target_values["neurosoft_on_vs_off"], behavior_ids)


def test_tokenize_extracts_acoustic_stim_labels():
    task_configs = _make_neurosoft_task_configs("acoustic_stim")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 5, 10, 25, 3, 7, 12, 20], dtype=np.int64)
    data = _make_acoustic_stim_sample(behavior_ids)
    tokens = model.tokenize(data)

    target_values = tokens["target_values"].obj
    assert "neurosoft_acoustic_stim" in target_values
    assert np.array_equal(
        target_values["neurosoft_acoustic_stim"], behavior_ids
    )


def test_foundry_module_training_step_binary():
    from torch_brain.batching import collate

    task_configs = _make_neurosoft_task_configs("on_vs_off")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    data = _make_on_vs_off_sample(behavior_ids)
    batch = collate([model.tokenize(data)])

    module = FoundryModule(model=model)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_foundry_module_training_step_multiclass():
    from torch_brain.batching import collate

    task_configs = _make_neurosoft_task_configs("acoustic_stim")
    model = _make_poyo_model(task_configs)
    model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])
    model.session_emb.initialize_vocab(["session1"])

    behavior_ids = np.array([0, 5, 10, 25, 3, 7, 12, 20], dtype=np.int64)
    data = _make_acoustic_stim_sample(behavior_ids)
    batch = collate([model.tokenize(data)])

    module = FoundryModule(model=model)
    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)

"""Test suite for baseline EEG models.

Tests Linear, MLP, GRU, TemporalConvAvgPool, ShallowConvNet, and EEGNetEncoder models.
"""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from temporaldata import Data, Interval, RegularTimeSeries
from torch_brain.batching import collate

from foundry.models import (
    TemporalConvAvgPool,
    Linear,
    MLP,
    GRU,
    ShallowConvNet,
    EEGNetEncoder,
)
from foundry.tasks.config import TaskConfig


def compute_multitask_loss(
    model, outputs, target_values, target_weights, task_index
):
    """Compute multitask loss for baseline models.

    Simplified version of the logic from foundry/training/module.py.
    """
    loss_total = torch.tensor(0.0, dtype=torch.float32)
    total_sequences = 0

    for task_name, task_output in outputs.items():
        if task_name not in target_values:
            continue

        target = target_values[task_name]
        if target.numel() == 0:
            continue

        cfg = model.task_configs[task_name]
        weights = target_weights.get(task_name, 1.0)
        loss_fn = instantiate(cfg.loss)
        task_loss = loss_fn(task_output, target, weights)

        idx = model.router.get_task_index_by_name(task_name) + 1
        num_sequences = torch.any(task_index == idx, dim=1).sum()

        loss_total = loss_total + task_loss * num_sequences
        total_sequences += num_sequences

    if total_sequences > 0:
        loss_total = loss_total / total_sequences

    return loss_total


class MockChannels:
    def __init__(self, channel_ids, types=None):
        self.id = np.array(channel_ids)
        if types is not None:
            self.type = np.array(types, dtype=str)


class MockSession:
    def __init__(self, session_id):
        self.id = session_id


@pytest.fixture(scope="module")
def task_configs():
    """Task configs for baseline model tests."""
    return {
        "test_baseline_task": TaskConfig.from_dict(
            {
                "name": "test_baseline_task",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 2,
                },
                "target_extractor": {
                    "_target_": "foundry.tasks.targets.TargetExtractor",
                    "timestamp_key": "test_baseline_task.timestamps",
                    "value_key": "test_baseline_task.values",
                },
                "loss": {
                    "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"
                },
                "metrics": {
                    "_target_": "foundry.tasks.metrics.classification_metrics",
                    "num_classes": 2,
                },
                "class_names": ["class0", "class1"],
            }
        )
    }


def create_baseline_data_sample(
    num_channels=4, num_samples=200, session_id="session1"
):
    """Create a mock baseline data sample with raw EEG signal.

    Args:
        num_channels: Number of EEG channels
        num_samples: Number of time samples
        session_id: Session identifier

    Returns:
        Data object with EEG signal and task labels
    """
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    eeg = RegularTimeSeries(
        signal=signal,
        sampling_rate=100.0,
        domain_start=0.0,
    )

    channel_ids = [f"ch{i}" for i in range(num_channels)]
    channels = MockChannels(channel_ids, types=["EEG"] * num_channels)
    session = MockSession(session_id)

    data = Data(eeg=eeg, domain=Interval(0.0, num_samples / 100.0))
    data.channels = channels
    data.session = session
    data._absolute_start = 0.0

    class TestTask:
        timestamps = np.array([1.0])
        values = np.array([0])

    data.test_baseline_task = TestTask()

    return data


@pytest.fixture
def simple_model(task_configs):
    """Create SimpleClassifier instance."""
    return TemporalConvAvgPool(
        task_configs=task_configs,
        num_channels=4,
        num_filters=32,
        kernel_size=64,
    )


@pytest.fixture
def shallow_model(task_configs):
    """Create ShallowConvNet instance."""
    return ShallowConvNet(
        task_configs=task_configs,
        num_channels=4,
        num_samples=3500,
        F1=40,
    )


@pytest.fixture
def eegnet_model(task_configs):
    """Create EEGNetEncoder instance."""
    return EEGNetEncoder(
        task_configs=task_configs,
        num_channels=4,
        num_samples=512,
    )


@pytest.fixture
def linear_model(task_configs):
    """Create Linear instance."""
    return Linear(
        task_configs=task_configs,
        num_channels=4,
        num_samples=200,
    )


@pytest.fixture
def mlp_model(task_configs):
    """Create MLP instance."""
    return MLP(
        task_configs=task_configs,
        num_channels=4,
        num_samples=200,
        hidden_dims=[64, 32],
        dropout_rate=0.3,
    )


@pytest.fixture
def gru_model(task_configs):
    """Create GRU instance."""
    return GRU(
        task_configs=task_configs,
        num_channels=4,
        num_samples=200,
        input_proj_dim=64,
        hidden_size=32,
        num_layers=2,
        bidirectional=True,
        dropout_rate=0.3,
    )


# ============================================================================
# Shared Tokenize Tests (BaselineEEGModel)
# ============================================================================


class TestBaselineTokenize:
    """Test tokenize method shared across all baseline models."""

    def test_tokenize_returns_expected_keys(self, simple_model):
        """Test that tokenize returns all expected keys."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = simple_model.tokenize(data)

        expected_keys = {
            "input_values",
            "task_index",
            "target_values",
            "target_weights",
            "session_id",
            "absolute_start",
        }
        assert set(tokens.keys()) == expected_keys

    def test_tokenize_input_values_shape(self, simple_model):
        """Test that tokenized input_values has expected shape (time, channels)."""
        num_channels = 4
        num_samples = 200
        data = create_baseline_data_sample(
            num_channels=num_channels, num_samples=num_samples
        )
        tokens = simple_model.tokenize(data)

        assert tokens["input_values"].obj.shape == (num_samples, num_channels)
        assert tokens["input_values"].obj.dtype == torch.float32

    def test_tokenize_raises_without_eeg_or_ecog_or_seeg_channels(
        self, simple_model
    ):
        """Test that tokenize raises ValueError without eeg or ecog or seeg field."""
        data = Data()
        data.channels = MockChannels(["ch0", "ch1", "ch2", "ch3"])
        data.session = MockSession("session1")

        with pytest.raises(
            ValueError,
            match="Data must have an 'eeg', 'ecog', or 'seeg' channel type",
        ):
            simple_model.tokenize(data)

    def test_tokenize_extracts_configured_task_targets(self, simple_model):
        """Test that tokenize extracts targets from configured task configs."""
        data = create_baseline_data_sample()
        tokens = simple_model.tokenize(data)

        assert "test_baseline_task" in tokens["target_values"].obj

    def test_tokenize_filters_eeg_channels(self, simple_model):
        """Test that tokenize filters for EEG channels only."""
        num_channels = 4
        signal = np.random.randn(200, num_channels).astype(np.float32)

        eeg = RegularTimeSeries(
            signal=signal,
            sampling_rate=100.0,
            domain_start=0.0,
        )

        data = Data(eeg=eeg, domain=Interval(0.0, 2.0))
        data.channels = MockChannels(
            ["ch0", "ch1", "ch2", "ch3"],
            types=["EEG", "EEG", "EOG", "EEG"],
        )
        data.session = MockSession("session1")
        data._absolute_start = 0.0

        class TestTask:
            timestamps = np.array([1.0])
            values = np.array([0])

        data.test_baseline_task = TestTask()

        tokens = simple_model.tokenize(data)

        assert tokens["input_values"].obj.shape == (200, 3)


# ============================================================================
# SimpleClassifier Tests
# ============================================================================


class TestTemporalConvAvgPool:
    """Test TemporalConvAvgPool model."""

    def test_init(self, task_configs):
        """Test TemporalConvAvgPool initialization."""
        model = TemporalConvAvgPool(
            task_configs=task_configs,
            num_channels=4,
            num_filters=32,
            kernel_size=64,
        )

        assert model.num_channels == 4
        assert hasattr(model, "router")
        assert hasattr(model, "conv")
        assert hasattr(model, "bn")
        assert hasattr(model, "act")
        assert hasattr(model, "pool")

    def test_forward_backward_pass(self, simple_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = simple_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = simple_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            simple_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in simple_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, simple_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=200)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=200)

        tokens1 = simple_model.tokenize(data1)
        tokens2 = simple_model.tokenize(data2)
        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = simple_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            simple_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in simple_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# ShallowConvNet Tests
# ============================================================================


class TestShallowConvNet:
    """Test ShallowConvNet model."""

    def test_init(self, task_configs):
        """Test ShallowConvNet initialization."""
        model = ShallowConvNet(
            task_configs=task_configs,
            num_channels=4,
            num_samples=175,
            F1=40,
        )

        assert model.num_channels == 4
        assert hasattr(model, "router")
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "flatten")
        assert hasattr(model, "act")

    def test_init_computes_correct_output_dim(self, task_configs):
        """Test that output dim is correctly computed as F1 * (num_samples // 35)."""
        model = ShallowConvNet(
            task_configs=task_configs,
            num_channels=4,
            num_samples=175,
            F1=40,
        )

        expected_out_dim = 40 * (175 // 35)
        for head in model.router.heads.values():
            assert head.projection.in_features == expected_out_dim

    def test_forward_backward_pass(self, shallow_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=3500)
        tokens = shallow_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = shallow_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            shallow_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in shallow_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, shallow_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=3500)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=3500)

        tokens1 = shallow_model.tokenize(data1)
        tokens2 = shallow_model.tokenize(data2)
        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = shallow_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            shallow_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in shallow_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# EEGNetEncoder Tests
# ============================================================================


class TestEEGNetEncoder:
    """Test EEGNetEncoder model."""

    def test_init(self, task_configs):
        """Test EEGNetEncoder initialization."""
        model = EEGNetEncoder(
            task_configs=task_configs,
            num_channels=4,
            num_samples=128,
        )

        assert model.num_channels == 4
        assert hasattr(model, "router")
        assert hasattr(model, "block1")
        assert hasattr(model, "block2")

    def test_extract_features(self, eegnet_model):
        """Test extract_features returns feature maps without classification."""
        x = torch.randn(1, 4, 128)

        features = eegnet_model.extract_features(input_values=x)

        assert features.ndim == 4
        assert features.shape[0] == 1

    def test_extract_features_3d_input(self, eegnet_model):
        """Test extract_features with 3D input (auto-unsqueeze)."""
        x_3d = torch.randn(1, 4, 128)
        x_4d = torch.randn(1, 1, 4, 128)

        features_3d = eegnet_model.extract_features(x_3d)
        features_4d = eegnet_model.extract_features(x_4d)

        assert features_3d.shape == features_4d.shape
        assert features_3d.ndim == 4

    def test_forward_backward_pass(self, eegnet_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=512)
        tokens = eegnet_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = eegnet_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            eegnet_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in eegnet_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, eegnet_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=512)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=512)

        tokens1 = eegnet_model.tokenize(data1)
        tokens2 = eegnet_model.tokenize(data2)

        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = eegnet_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            eegnet_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in eegnet_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# Linear Tests
# ============================================================================


class TestLinear:
    """Test Linear model."""

    def test_init(self, task_configs):
        """Test Linear initialization."""
        model = Linear(
            task_configs=task_configs,
            num_channels=4,
            num_samples=200,
        )

        assert model.num_channels == 4
        assert model.num_samples == 200
        assert hasattr(model, "router")
        for head in model.router.heads.values():
            assert head.projection.in_features == 4 * 200

    def test_forward_backward_pass(self, linear_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = linear_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = linear_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            linear_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in linear_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, linear_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=200)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=200)

        tokens1 = linear_model.tokenize(data1)
        tokens2 = linear_model.tokenize(data2)
        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = linear_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            linear_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in linear_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# MLP Tests
# ============================================================================


class TestMLP:
    """Test MLP model."""

    def test_init(self, task_configs):
        """Test MLP initialization."""
        model = MLP(
            task_configs=task_configs,
            num_channels=4,
            num_samples=200,
            hidden_dims=[64, 32],
        )

        assert model.num_channels == 4
        assert model.num_samples == 200
        assert hasattr(model, "router")
        assert hasattr(model, "mlp")
        first_linear = model.mlp[0]
        assert first_linear.in_features == 4 * 200

    def test_forward_backward_pass(self, mlp_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = mlp_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = mlp_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            mlp_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in mlp_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, mlp_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=200)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=200)

        tokens1 = mlp_model.tokenize(data1)
        tokens2 = mlp_model.tokenize(data2)
        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = mlp_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            mlp_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in mlp_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# GRU Tests
# ============================================================================


class TestGRU:
    """Test GRU model."""

    def test_init(self, task_configs):
        """Test GRU initialization."""
        model = GRU(
            task_configs=task_configs,
            num_channels=4,
            num_samples=200,
            input_proj_dim=64,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
        )

        assert model.num_channels == 4
        assert model.num_samples == 200
        assert hasattr(model, "router")
        assert hasattr(model, "input_norm")
        assert hasattr(model, "input_proj")
        assert hasattr(model, "gru")

        # Bidirectional GRU doubles readout input dimension.
        for head in model.router.heads.values():
            assert head.projection.in_features == 64

    def test_forward_backward_pass(self, gru_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = gru_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = gru_model(
            input_values=x,
            task_index=task_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            gru_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in gru_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients

    def test_forward_backward_batched(self, gru_model):
        """Test forward + backward with batched samples."""
        data1 = create_baseline_data_sample(num_channels=4, num_samples=200)
        data2 = create_baseline_data_sample(num_channels=4, num_samples=200)

        tokens1 = gru_model.tokenize(data1)
        tokens2 = gru_model.tokenize(data2)
        batch = collate([tokens1, tokens2])

        x = batch["input_values"]
        task_index = batch["task_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = gru_model(
            input_values=x,
            task_index=task_index,
        )

        loss = compute_multitask_loss(
            gru_model,
            outputs,
            target_values,
            target_weights,
            task_index,
        )

        assert loss.requires_grad
        loss.backward()

        has_gradients = False
        for param in gru_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        assert has_gradients


# ============================================================================
# Integration Tests
# ============================================================================


class TestBaselineIntegration:
    """Integration tests for tokenize and models."""

    def test_tokenize_then_forward_simple(self, simple_model):
        """Test tokenize output can be used with TemporalConvAvgPool."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = simple_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (200, 4)

    def test_tokenize_then_forward_shallow(self, shallow_model):
        """Test tokenize output can be used with ShallowConvNet."""
        data = create_baseline_data_sample(num_channels=4, num_samples=3500)
        tokens = shallow_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (3500, 4)

    def test_tokenize_then_forward_eegnet(self, eegnet_model):
        """Test tokenize output can be used with EEGNetEncoder."""
        data = create_baseline_data_sample(num_channels=4, num_samples=512)
        tokens = eegnet_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (512, 4)

    def test_tokenize_then_forward_linear(self, linear_model):
        """Test tokenize output can be used with Linear."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = linear_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (200, 4)

    def test_tokenize_then_forward_mlp(self, mlp_model):
        """Test tokenize output can be used with MLP."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = mlp_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (200, 4)

    def test_tokenize_then_forward_gru(self, gru_model):
        """Test tokenize output can be used with GRU."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = gru_model.tokenize(data)

        assert "input_values" in tokens
        assert "task_index" in tokens
        assert tokens["input_values"].obj.shape == (200, 4)

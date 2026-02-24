"""Test suite for baseline EEG classification models.

Tests SimpleClassifier, ShallowConvNet, and EEGNetEncoder models.
"""

import numpy as np
import pytest
import torch
from temporaldata import Data, Interval, RegularTimeSeries
from torch_brain.data import collate
from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY
from torch_brain.nn.loss import CrossEntropyLoss

from foundry.models import (
    SimpleClassifier,
    ShallowConvNet,
    EEGNetEncoder,
)


def compute_multitask_loss(
    model, outputs, target_values, target_weights, output_decoder_index
):
    """Compute multitask loss for baseline models.

    Simplified version of the logic from foundry/training/task.py.

    Args:
        model: The baseline model with readout_specs
        outputs: Dict of model outputs per task
        target_values: Dict of target values per task
        target_weights: Dict of target weights per task
        output_decoder_index: Tensor indicating which task each output belongs to

    Returns:
        Scalar loss tensor
    """
    loss_total = torch.tensor(0.0, dtype=torch.float32)
    total_sequences = 0

    for readout_id, task_output in outputs.items():
        if readout_id not in target_values:
            continue

        target = target_values[readout_id]
        if target.numel() == 0:
            continue

        spec = model.readout_specs[readout_id]
        weights = target_weights.get(readout_id, 1.0)

        task_loss = spec.loss_fn(task_output, target, weights)

        num_sequences = torch.any(output_decoder_index == spec.id, dim=1).sum()

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
def readout_specs():
    """Register and return test readout specs."""
    if "test_baseline_task" not in MODALITY_REGISTRY:
        register_modality(
            "test_baseline_task",
            dim=2,
            type=DataType.BINARY,
            timestamp_key="test_baseline_task.timestamps",
            value_key="test_baseline_task.values",
            loss_fn=CrossEntropyLoss(),
        )
    return {"test_baseline_task": MODALITY_REGISTRY["test_baseline_task"]}


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
        domain=Interval(0.0, num_samples / 100.0),
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
    data.config = {
        "multitask_readout": [
            {
                "readout_id": "test_baseline_task",
            }
        ]
    }

    return data


@pytest.fixture
def simple_model(readout_specs):
    """Create SimpleClassifier instance."""
    return SimpleClassifier(
        readout_specs=readout_specs,
        num_channels=4,
        num_filters=32,
        kernel_size=64,
    )


@pytest.fixture
def shallow_model(readout_specs):
    """Create ShallowConvNet instance."""
    return ShallowConvNet(
        readout_specs=readout_specs,
        num_channels=4,
        num_samples=3500,
        F1=40,
    )


@pytest.fixture
def eegnet_model(readout_specs):
    """Create EEGNetEncoder instance."""
    return EEGNetEncoder(
        readout_specs=readout_specs,
        num_channels=4,
        num_samples=512,
    )


# ============================================================================
# Shared Tokenize Tests (BaselineModel)
# ============================================================================


class TestBaselineTokenize:
    """Test tokenize method shared across all baseline models."""

    def test_tokenize_returns_expected_keys(self, simple_model):
        """Test that tokenize returns all expected keys."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = simple_model.tokenize(data)

        expected_keys = {
            "input_values",
            "output_decoder_index",
            "target_values",
            "target_weights",
            "session_id",
            "absolute_start",
            "eval_mask",
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

    def test_tokenize_raises_without_eeg_or_ecog_channels(self, simple_model):
        """Test that tokenize raises ValueError without eeg or ecog field."""
        data = Data()
        data.channels = MockChannels(["ch0", "ch1", "ch2", "ch3"])
        data.session = MockSession("session1")

        with pytest.raises(
            ValueError, match="Data must have an 'eeg' or 'ecog' field"
        ):
            simple_model.tokenize(data)

    def test_tokenize_respects_multitask_config(self, simple_model):
        """Test that tokenize respects multitask_readout config."""
        data = create_baseline_data_sample()

        data.config["multitask_readout"] = [
            {"readout_id": "test_baseline_task"}
        ]

        tokens = simple_model.tokenize(data)

        assert tokens["target_values"] is not None

    def test_tokenize_filters_eeg_channels(self, simple_model):
        """Test that tokenize filters for EEG channels only."""
        num_channels = 4
        signal = np.random.randn(200, num_channels).astype(np.float32)

        eeg = RegularTimeSeries(
            signal=signal,
            sampling_rate=100.0,
            domain=Interval(0.0, 2.0),
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
        data.config = {
            "multitask_readout": [{"readout_id": "test_baseline_task"}]
        }

        tokens = simple_model.tokenize(data)

        assert tokens["input_values"].obj.shape == (200, 3)


# ============================================================================
# SimpleClassifier Tests
# ============================================================================


class TestSimpleClassifier:
    """Test SimpleClassifier model."""

    def test_init(self, readout_specs):
        """Test SimpleClassifier initialization."""
        model = SimpleClassifier(
            readout_specs=readout_specs,
            num_channels=4,
            num_filters=32,
            kernel_size=64,
        )

        assert model.num_channels == 4
        assert hasattr(model, "readout")
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
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = simple_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            simple_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = simple_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        loss = compute_multitask_loss(
            simple_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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

    def test_init(self, readout_specs):
        """Test ShallowConvNet initialization."""
        model = ShallowConvNet(
            readout_specs=readout_specs,
            num_channels=4,
            num_samples=175,
            F1=40,
        )

        assert model.num_channels == 4
        assert hasattr(model, "readout")
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "flatten")
        assert hasattr(model, "act")

    def test_init_computes_correct_output_dim(self, readout_specs):
        """Test that output dim is correctly computed as F1 * (num_samples // 35)."""
        model = ShallowConvNet(
            readout_specs=readout_specs,
            num_channels=4,
            num_samples=175,
            F1=40,
        )

        expected_out_dim = 40 * (175 // 35)
        for proj in model.readout.projections.values():
            assert proj.in_features == expected_out_dim

    def test_forward_backward_pass(self, shallow_model):
        """Test tokenize -> collate -> forward -> backward end-to-end."""
        data = create_baseline_data_sample(num_channels=4, num_samples=3500)
        tokens = shallow_model.tokenize(data)
        batch = collate([tokens])

        x = batch["input_values"]
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = shallow_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            shallow_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = shallow_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        loss = compute_multitask_loss(
            shallow_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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

    def test_init(self, readout_specs):
        """Test EEGNetEncoder initialization."""
        model = EEGNetEncoder(
            readout_specs=readout_specs,
            num_channels=4,
            num_samples=128,
        )

        assert model.num_channels == 4
        assert hasattr(model, "readout")
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
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = eegnet_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        assert isinstance(outputs, dict)
        assert "test_baseline_task" in outputs

        loss = compute_multitask_loss(
            eegnet_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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
        output_decoder_index = batch["output_decoder_index"]
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        outputs = eegnet_model(
            input_values=x,
            output_decoder_index=output_decoder_index,
        )

        loss = compute_multitask_loss(
            eegnet_model,
            outputs,
            target_values,
            target_weights,
            output_decoder_index,
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
# Integration Tests
# ============================================================================


class TestBaselineIntegration:
    """Integration tests for tokenize and models."""

    def test_tokenize_then_forward_simple(self, simple_model):
        """Test tokenize output can be used with SimpleClassifier."""
        data = create_baseline_data_sample(num_channels=4, num_samples=200)
        tokens = simple_model.tokenize(data)

        assert "input_values" in tokens
        assert "output_decoder_index" in tokens
        assert tokens["input_values"].obj.shape == (200, 4)

    def test_tokenize_then_forward_shallow(self, shallow_model):
        """Test tokenize output can be used with ShallowConvNet."""
        data = create_baseline_data_sample(num_channels=4, num_samples=3500)
        tokens = shallow_model.tokenize(data)

        assert "input_values" in tokens
        assert "output_decoder_index" in tokens
        assert tokens["input_values"].obj.shape == (3500, 4)

    def test_tokenize_then_forward_eegnet(self, eegnet_model):
        """Test tokenize output can be used with EEGNetEncoder."""
        data = create_baseline_data_sample(num_channels=4, num_samples=512)
        tokens = eegnet_model.tokenize(data)

        assert "input_values" in tokens
        assert "output_decoder_index" in tokens
        assert tokens["input_values"].obj.shape == (512, 4)

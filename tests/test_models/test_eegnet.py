import numpy as np
import pytest
import torch
from temporaldata import Data, RegularTimeSeries

from foundry.models import EEGNetModel


class TestEEGNetModel:
    def test_initialization(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )
        assert model.in_chans == 64
        assert model.in_times == 160
        assert model.F2 == 16  # F1 * D = 8 * 2
        assert len(model.readout_specs) == 1
        assert "motor_imagery_right_feet" in model.readout_specs

    def test_initialization_custom_F2(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
            F1=8,
            D=2,
            F2=32,
        )
        assert model.F2 == 32

    def test_initialization_multiple_tasks(self):
        model = EEGNetModel(
            readout_specs=[
                "motor_imagery_right_feet",
                "motor_imagery_left_right",
            ],
            in_chans=64,
            in_times=160,
        )
        assert len(model.readout_specs) == 2

    def test_forward_pass_shape(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )
        batch_size = 4
        input_values = torch.randn(batch_size, 64, 160)
        output_decoder_index = torch.zeros(batch_size, 1, dtype=torch.long)

        outputs = model(
            input_values=input_values, output_decoder_index=output_decoder_index
        )

        assert isinstance(outputs, dict)
        assert "motor_imagery_right_feet" in outputs
        assert outputs["motor_imagery_right_feet"].shape == (batch_size, 2)

    def test_forward_pass_multiple_tasks(self):
        model = EEGNetModel(
            readout_specs=[
                "motor_imagery_right_feet",
                "motor_imagery_left_right",
            ],
            in_chans=64,
            in_times=160,
        )
        batch_size = 4
        input_values = torch.randn(batch_size, 64, 160)
        output_decoder_index = torch.tensor(
            [[0], [1], [0], [1]], dtype=torch.long
        )

        outputs = model(
            input_values=input_values, output_decoder_index=output_decoder_index
        )

        assert isinstance(outputs, dict)
        assert "motor_imagery_right_feet" in outputs
        assert "motor_imagery_left_right" in outputs

    def test_forward_pass_wrong_channel_count(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )
        input_values = torch.randn(4, 32, 160)  # Wrong channel count
        output_decoder_index = torch.zeros(4, 1, dtype=torch.long)

        with pytest.raises(ValueError, match="Expected 64 channels"):
            model(
                input_values=input_values,
                output_decoder_index=output_decoder_index,
            )

    def test_forward_pass_wrong_dimension(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )
        input_values = torch.randn(4, 64, 160, 10)  # Wrong dimension
        output_decoder_index = torch.zeros(4, 1, dtype=torch.long)

        with pytest.raises(ValueError, match="expects input_values of shape"):
            model(
                input_values=input_values,
                output_decoder_index=output_decoder_index,
            )

    def test_tokenize_basic(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
        )

        data = Data()
        data.eeg = RegularTimeSeries(
            signal=np.random.randn(160, 2).astype(np.float32),
            sampling_rate=160.0,
            domain=(0.0, 1.0),
        )
        data.channels = type("Channels", (), {"id": np.array(["C3", "C4"])})()
        data.session = type("Session", (), {"id": "session_1"})()
        data.motor_imagery_trials = type(
            "Trials",
            (),
            {
                "timestamps": np.array([0.5]),
                "movement_ids": np.array([2]),
            },
        )()

        tokenized = model.tokenize(data)

        assert "input_values" in tokenized
        assert "output_decoder_index" in tokenized
        assert "target_values" in tokenized
        assert "target_weights" in tokenized
        assert tokenized["input_values"].shape == (1, 2, 160)

    def test_tokenize_zscore_normalization(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
            normalize_mode="zscore",
        )

        data = Data()
        signal = np.random.randn(160, 2).astype(np.float32) * 10 + 5
        data.eeg = RegularTimeSeries(
            signal=signal,
            sampling_rate=160.0,
            domain=(0.0, 1.0),
        )
        data.channels = type("Channels", (), {"id": np.array(["C3", "C4"])})()
        data.session = type("Session", (), {"id": "session_1"})()
        data.motor_imagery_trials = type(
            "Trials",
            (),
            {
                "timestamps": np.array([0.5]),
                "movement_ids": np.array([2]),
            },
        )()

        tokenized = model.tokenize(data)
        input_values = tokenized["input_values"][0]

        mean = input_values.mean(dim=-1)
        std = input_values.std(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5)

    def test_tokenize_no_normalization(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
            normalize_mode="none",
        )

        data = Data()
        signal = np.random.randn(160, 2).astype(np.float32) * 10 + 5
        data.eeg = RegularTimeSeries(
            signal=signal,
            sampling_rate=160.0,
            domain=(0.0, 1.0),
        )
        data.channels = type("Channels", (), {"id": np.array(["C3", "C4"])})()
        data.session = type("Session", (), {"id": "session_1"})()
        data.motor_imagery_trials = type(
            "Trials",
            (),
            {
                "timestamps": np.array([0.5]),
                "movement_ids": np.array([2]),
            },
        )()

        tokenized = model.tokenize(data)
        input_values = tokenized["input_values"][0]
        expected = torch.from_numpy(signal.T)

        assert torch.allclose(input_values, expected)

    def test_tokenize_padding(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
        )
        min_T = model.min_T

        data = Data()
        signal = np.random.randn(min_T - 10, 2).astype(np.float32)
        data.eeg = RegularTimeSeries(
            signal=signal,
            sampling_rate=160.0,
            domain=(0.0, 1.0),
        )
        data.channels = type("Channels", (), {"id": np.array(["C3", "C4"])})()
        data.session = type("Session", (), {"id": "session_1"})()
        data.motor_imagery_trials = type(
            "Trials",
            (),
            {
                "timestamps": np.array([0.5]),
                "movement_ids": np.array([2]),
            },
        )()

        tokenized = model.tokenize(data)
        assert tokenized["input_values"].shape[2] == min_T

    def test_tokenize_missing_eeg(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
        )

        data = Data()

        with pytest.raises(ValueError, match="must have an 'eeg' field"):
            model.tokenize(data)

    def test_tokenize_wrong_shape(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=2,
            in_times=160,
        )

        data = Data()
        data.eeg = RegularTimeSeries(
            signal=np.random.randn(160, 2, 10).astype(np.float32),
            sampling_rate=160.0,
            domain=(0.0, 1.0),
        )

        with pytest.raises(ValueError, match="Expected EEG shape"):
            model.tokenize(data)

    def test_unpack_batch(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )

        batch = {
            "input_values": torch.randn(4, 64, 160),
            "output_decoder_index": torch.zeros(4, 1, dtype=torch.long),
            "target_values": {
                "motor_imagery_right_feet": torch.randint(0, 2, (4,))
            },
            "target_weights": {"motor_imagery_right_feet": torch.ones(4)},
            "session_id": "session_1",
            "absolute_start": 0.0,
            "eval_mask": {},
        }

        model_inputs, target_values, target_weights, output_decoder_index = (
            model.unpack_batch(batch.copy())
        )

        assert "input_values" in model_inputs
        assert "output_decoder_index" in model_inputs
        assert "motor_imagery_right_feet" in target_values
        assert "motor_imagery_right_feet" in target_weights
        assert output_decoder_index is not None

    def test_conv2d_with_constraint(self):
        from foundry.models.eegnet_model import Conv2dWithConstraint

        conv = Conv2dWithConstraint(
            in_channels=8,
            out_channels=16,
            kernel_size=(64, 1),
            groups=8,
            max_norm=1.0,
        )

        x = torch.randn(2, 8, 64, 100)
        output = conv(x)

        assert output.shape == (2, 16, 1, 100)

        norms = conv.weight.norm(p=2, dim=(1, 2, 3))
        assert torch.all(norms <= 1.0 + 1e-5)

    def test_forward_features(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )

        x = torch.randn(2, 1, 64, 160)
        features = model._forward_features(x)

        assert features.shape == (2, 16, 1, 1)

    def test_resolve_readout_specs_strings(self):
        model = EEGNetModel(
            readout_specs=["motor_imagery_right_feet"],
            in_chans=64,
            in_times=160,
        )

        assert isinstance(model.readout_specs, dict)
        assert "motor_imagery_right_feet" in model.readout_specs

    def test_resolve_readout_specs_dict(self):
        from torch_brain.registry import MODALITY_REGISTRY

        readout_specs = {
            "motor_imagery_right_feet": MODALITY_REGISTRY[
                "motor_imagery_right_feet"
            ]
        }
        model = EEGNetModel(
            readout_specs=readout_specs,
            in_chans=64,
            in_times=160,
        )

        assert model.readout_specs == readout_specs

    def test_resolve_readout_specs_invalid(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            EEGNetModel(
                readout_specs=["invalid_task"],
                in_chans=64,
                in_times=160,
            )

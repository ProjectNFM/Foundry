import numpy as np
import pytest
from temporaldata import Data, Interval, RegularTimeSeries
from torch_brain.data import collate
from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY
from torch_brain.nn.loss import CrossEntropyLoss

from foundry.models import (
    POYOEEGModel,
    LinearEmbedding,
    CNNEmbedding,
    MLPEmbedding,
)


class MockChannels:
    def __init__(self, channel_ids, types=None):
        self.id = np.array(channel_ids)
        if types is not None:
            self.types = np.array(types)


class MockSession:
    def __init__(self, session_id):
        self.id = session_id


@pytest.fixture(scope="module")
def readout_specs():
    if "test_task1" not in MODALITY_REGISTRY:
        register_modality(
            "test_task1",
            dim=2,
            type=DataType.BINARY,
            timestamp_key="test_task.timestamps",
            value_key="test_task.values",
            loss_fn=CrossEntropyLoss(),
        )
    return {"test_task1": MODALITY_REGISTRY["test_task1"]}


@pytest.fixture
def model_with_linear(readout_specs, embed_dim):
    input_embedding = LinearEmbedding(embed_dim=embed_dim)
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=2.0,
        patch_duration=0.5,
        stride=0.5,
        latent_step=0.5,
        num_latents_per_step=1,
    )
    return model


@pytest.fixture
def model_with_cnn(readout_specs, embed_dim):
    input_embedding = CNNEmbedding(
        embed_dim=embed_dim, num_filters=32, kernel_size=3
    )
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=2.0,
        patch_duration=0.5,
        stride=0.5,
        latent_step=0.5,
        num_latents_per_step=1,
    )
    return model


@pytest.fixture
def model_with_mlp(readout_specs, embed_dim):
    input_embedding = MLPEmbedding(
        embed_dim=embed_dim, hidden_dims=[128, 64], activation="gelu"
    )
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=2.0,
        patch_duration=0.5,
        stride=0.5,
        latent_step=0.5,
        num_latents_per_step=1,
    )
    return model


def extract_model_inputs(batch):
    """Extract only the inputs needed for model.forward()."""
    return {
        k: v
        for k, v in batch.items()
        if k
        in [
            "input_values",
            "input_timestamps",
            "input_channel_index",
            "input_session_index",
            "input_mask",
            "latent_index",
            "latent_timestamps",
            "output_session_index",
            "output_timestamps",
            "output_decoder_index",
        ]
    }


def create_data_sample(
    num_channels, sampling_rate, duration=2.0, session_id="session1"
):
    """
    Create a mock data sample with specified channel count and sampling rate.
    """
    num_samples = int(duration * sampling_rate)
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    eeg = RegularTimeSeries(
        signal=signal,
        sampling_rate=sampling_rate,
        domain=Interval(0.0, duration),
    )

    channel_ids = [f"ch{i}" for i in range(num_channels)]
    channels = MockChannels(channel_ids, types=["EEG"] * num_channels)
    session = MockSession(session_id)

    data = Data(eeg=eeg, domain=Interval(0.0, duration))
    data.channels = channels
    data.session = session
    data._absolute_start = 0.0

    class TestTask:
        timestamps = np.array([1.0])
        values = np.array([0])

    data.test_task = TestTask()

    data.config = {
        "multitask_readout": [
            {
                "readout_id": "test_task1",
            }
        ]
    }

    return data


class TestHeterogeneousBatching:
    def test_tokenize_different_channel_counts(self, model_with_linear):
        """Test tokenization with samples having different channel counts."""
        data1 = create_data_sample(num_channels=4, sampling_rate=100.0)
        data2 = create_data_sample(num_channels=8, sampling_rate=100.0)

        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        assert tokens1["input_values"].obj.shape[0] == 4 * 4
        assert tokens2["input_values"].obj.shape[0] == 4 * 8

        assert tokens1["input_channel_index"].obj.shape[0] == 4 * 4
        assert tokens2["input_channel_index"].obj.shape[0] == 4 * 8

    def test_tokenize_different_sampling_rates(self, model_with_linear):
        """Test tokenization with samples having different sampling rates."""
        data1 = create_data_sample(num_channels=4, sampling_rate=100.0)
        data2 = create_data_sample(num_channels=4, sampling_rate=250.0)

        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(4)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        assert tokens1["input_values"].obj.shape[1] == 50
        assert tokens2["input_values"].obj.shape[1] == 125

    def test_collate_heterogeneous_channels(self, model_with_linear):
        """Test collation of batch with different channel counts."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=100.0, session_id="s2"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])

        assert batch["input_values"].shape[0] == 2
        assert batch["input_values"].shape[1] == max(4 * 4, 4 * 8)
        assert batch["input_values"].shape[2] == 50

        assert batch["input_mask"].shape[0] == 2
        assert batch["input_mask"].shape[1] == max(4 * 4, 4 * 8)

    def test_collate_heterogeneous_sampling_rates(self, model_with_linear):
        """Test collation of batch with different sampling rates."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=4, sampling_rate=250.0, session_id="s2"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(4)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])

        assert batch["input_values"].shape[0] == 2
        assert batch["input_values"].shape[1] == 4 * 4
        assert batch["input_values"].shape[2] == max(50, 125)

        assert batch["input_mask"].shape[0] == 2
        assert batch["input_mask"].shape[1] == 4 * 4

    def test_collate_fully_heterogeneous(self, model_with_linear):
        """Test collation with both different channels and sampling rates."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=250.0, session_id="s2"
        )
        data3 = create_data_sample(
            num_channels=6, sampling_rate=128.0, session_id="s3"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2", "s3"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)
        tokens3 = model_with_linear.tokenize(data3)

        batch = collate([tokens1, tokens2, tokens3])

        assert batch["input_values"].shape[0] == 3
        assert batch["input_values"].shape[1] == max(4 * 4, 4 * 8, 4 * 6)
        assert batch["input_values"].shape[2] == max(50, 125, 64)

    def test_forward_heterogeneous_channels_linear(self, model_with_linear):
        """Test forward pass with heterogeneous channels using LinearEmbedding."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=100.0, session_id="s2"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])

        output = model_with_linear(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_heterogeneous_sampling_rates_linear(
        self, model_with_linear
    ):
        """Test forward pass with heterogeneous sampling rates using LinearEmbedding."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=4, sampling_rate=250.0, session_id="s2"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(4)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])

        output = model_with_linear(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_fully_heterogeneous_linear(self, model_with_linear):
        """Test forward pass with both different channels and sampling rates."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=250.0, session_id="s2"
        )
        data3 = create_data_sample(
            num_channels=6, sampling_rate=128.0, session_id="s3"
        )

        model_with_linear.session_emb.initialize_vocab(["s1", "s2", "s3"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)
        tokens3 = model_with_linear.tokenize(data3)

        batch = collate([tokens1, tokens2, tokens3])

        output = model_with_linear(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 3

    def test_forward_heterogeneous_cnn(self, model_with_cnn):
        """Test forward pass with heterogeneous batch using CNNEmbedding."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=250.0, session_id="s2"
        )

        model_with_cnn.session_emb.initialize_vocab(["s1", "s2"])
        model_with_cnn.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_cnn.tokenize(data1)
        tokens2 = model_with_cnn.tokenize(data2)

        batch = collate([tokens1, tokens2])

        output = model_with_cnn(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_heterogeneous_mlp(self, model_with_mlp):
        """Test forward pass with heterogeneous batch using MLPEmbedding."""
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=250.0, session_id="s2"
        )

        model_with_mlp.session_emb.initialize_vocab(["s1", "s2"])
        model_with_mlp.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(8)]
        )

        tokens1 = model_with_mlp.tokenize(data1)
        tokens2 = model_with_mlp.tokenize(data2)

        batch = collate([tokens1, tokens2])

        output = model_with_mlp(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_embedding_projection_caching(self, model_with_linear):
        """Test that different patch sizes create separate cached projections."""
        data1 = create_data_sample(num_channels=4, sampling_rate=100.0)
        data2 = create_data_sample(num_channels=4, sampling_rate=250.0)

        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(4)]
        )

        model_with_linear.tokenize(data1)
        model_with_linear.tokenize(data2)

        assert len(model_with_linear.input_embedding.projections) == 0

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch1 = collate([tokens1])
        batch2 = collate([tokens2])

        model_with_linear(**extract_model_inputs(batch1))
        assert len(model_with_linear.input_embedding.projections) == 1

        model_with_linear(**extract_model_inputs(batch2))
        assert len(model_with_linear.input_embedding.projections) == 2

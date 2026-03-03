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


SAMPLING_RATE = 100.0
PATCH_DURATION = 0.5
STRIDE = 0.5
SEQUENCE_LENGTH = 2.0
NUM_CHANNELS = 8
PATCH_SAMPLES = int(PATCH_DURATION * SAMPLING_RATE)
NUM_PATCHES = int(SEQUENCE_LENGTH / STRIDE)


class MockChannels:
    def __init__(self, channel_ids, types=None):
        self.id = np.array(channel_ids)
        if types is not None:
            self.type = np.array(types)

    def __len__(self):
        return len(self.id)


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
    input_embedding = LinearEmbedding(
        embed_dim=embed_dim,
        num_channels=NUM_CHANNELS,
        patch_samples=PATCH_SAMPLES,
    )
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=SEQUENCE_LENGTH,
        patch_duration=PATCH_DURATION,
        stride=STRIDE,
        latent_step=0.5,
        num_latents_per_step=1,
    )
    return model


@pytest.fixture
def model_with_cnn(readout_specs, embed_dim):
    input_embedding = CNNEmbedding(
        embed_dim=embed_dim,
        num_channels=NUM_CHANNELS,
        patch_samples=PATCH_SAMPLES,
        num_filters=32,
        kernel_size=3,
    )
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=SEQUENCE_LENGTH,
        patch_duration=PATCH_DURATION,
        stride=STRIDE,
        latent_step=0.5,
        num_latents_per_step=1,
    )
    return model


@pytest.fixture
def model_with_mlp(readout_specs, embed_dim):
    input_embedding = MLPEmbedding(
        embed_dim=embed_dim,
        num_channels=NUM_CHANNELS,
        patch_samples=PATCH_SAMPLES,
        hidden_dims=[128, 64],
        activation="gelu",
    )
    model = POYOEEGModel(
        input_embedding=input_embedding,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=SEQUENCE_LENGTH,
        patch_duration=PATCH_DURATION,
        stride=STRIDE,
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


def create_data_sample(num_channels, session_id="session1"):
    """Create a mock data sample with specified channel count."""
    num_samples = int(SEQUENCE_LENGTH * SAMPLING_RATE)
    signal = np.random.randn(num_samples, num_channels).astype(np.float32)

    eeg = RegularTimeSeries(
        signal=signal,
        sampling_rate=SAMPLING_RATE,
        domain=Interval(0.0, SEQUENCE_LENGTH),
    )

    channel_ids = [f"ch{i}" for i in range(num_channels)]
    channels = MockChannels(channel_ids, types=["EEG"] * num_channels)
    session = MockSession(session_id)

    data = Data(eeg=eeg, domain=Interval(0.0, SEQUENCE_LENGTH))
    data.channels = channels
    data.session = session
    data._absolute_start = 0.0

    class TestTask:
        timestamps = np.array([1.0])
        values = np.array([0])

    data.test_task = TestTask()

    data.config = {"multitask_readout": [{"readout_id": "test_task1"}]}

    return data


class TestTokenizerOutput:
    def test_tokenize_shape(self, model_with_linear):
        """Tokenizer outputs signal in (P, C_padded, S) shape."""
        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = create_data_sample(num_channels=4)
        tokens = model_with_linear.tokenize(data)

        assert tokens["input_values"].shape == (
            NUM_PATCHES,
            NUM_CHANNELS,
            PATCH_SAMPLES,
        )
        assert tokens["input_timestamps"].shape == (NUM_PATCHES,)
        assert tokens["input_channel_index"].shape == (NUM_CHANNELS,)
        assert tokens["input_mask"].shape == (NUM_CHANNELS,)

    def test_channel_padding(self, model_with_linear):
        """Channels are zero-padded and mask tracks valid channels."""
        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = create_data_sample(num_channels=4)
        tokens = model_with_linear.tokenize(data)

        assert tokens["input_mask"][:4].all()
        assert not tokens["input_mask"][4:].any()

        assert (tokens["input_values"][:, 4:, :] == 0).all()

    def test_full_channels_no_padding(self, model_with_linear):
        """When data has max channels, no padding is needed."""
        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = create_data_sample(num_channels=NUM_CHANNELS)
        tokens = model_with_linear.tokenize(data)

        assert tokens["input_mask"].all()

    def test_too_many_channels_raises(self, model_with_linear):
        """Tokenizer raises if data has more channels than model expects."""
        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS + 4)]
        )

        data = create_data_sample(num_channels=NUM_CHANNELS + 4)

        with pytest.raises(ValueError, match="channels but model expects"):
            model_with_linear.tokenize(data)


class TestHeterogeneousBatching:
    def test_tokenize_different_channel_counts(self, model_with_linear):
        """Samples with different channel counts are padded to the same shape."""
        model_with_linear.session_emb.initialize_vocab(["session1"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=4)
        data2 = create_data_sample(num_channels=NUM_CHANNELS)

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        assert tokens1["input_values"].shape == tokens2["input_values"].shape
        assert (
            tokens1["input_channel_index"].shape
            == tokens2["input_channel_index"].shape
        )

    def test_collate_heterogeneous_channels(self, model_with_linear):
        """Collation produces correct batch shapes with padded channels."""
        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=4, session_id="s1")
        data2 = create_data_sample(num_channels=NUM_CHANNELS, session_id="s2")

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])

        assert batch["input_values"].shape == (
            2,
            NUM_PATCHES,
            NUM_CHANNELS,
            PATCH_SAMPLES,
        )
        assert batch["input_timestamps"].shape == (2, NUM_PATCHES)
        assert batch["input_channel_index"].shape == (2, NUM_CHANNELS)
        assert batch["input_mask"].shape == (2, NUM_CHANNELS)

    def test_forward_heterogeneous_channels_linear(self, model_with_linear):
        """Forward pass works with batched samples of different channel counts."""
        model_with_linear.session_emb.initialize_vocab(["s1", "s2"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=4, session_id="s1")
        data2 = create_data_sample(num_channels=NUM_CHANNELS, session_id="s2")

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)

        batch = collate([tokens1, tokens2])
        output = model_with_linear(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_heterogeneous_cnn(self, model_with_cnn):
        """Forward pass works with CNNEmbedding and heterogeneous channels."""
        model_with_cnn.session_emb.initialize_vocab(["s1", "s2"])
        model_with_cnn.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=4, session_id="s1")
        data2 = create_data_sample(num_channels=NUM_CHANNELS, session_id="s2")

        tokens1 = model_with_cnn.tokenize(data1)
        tokens2 = model_with_cnn.tokenize(data2)

        batch = collate([tokens1, tokens2])
        output = model_with_cnn(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_heterogeneous_mlp(self, model_with_mlp):
        """Forward pass works with MLPEmbedding and heterogeneous channels."""
        model_with_mlp.session_emb.initialize_vocab(["s1", "s2"])
        model_with_mlp.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=4, session_id="s1")
        data2 = create_data_sample(num_channels=NUM_CHANNELS, session_id="s2")

        tokens1 = model_with_mlp.tokenize(data1)
        tokens2 = model_with_mlp.tokenize(data2)

        batch = collate([tokens1, tokens2])
        output = model_with_mlp(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 2

    def test_forward_three_samples(self, model_with_linear):
        """Forward pass works with 3 samples of varying channel counts."""
        model_with_linear.session_emb.initialize_vocab(["s1", "s2", "s3"])
        model_with_linear.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = create_data_sample(num_channels=2, session_id="s1")
        data2 = create_data_sample(num_channels=6, session_id="s2")
        data3 = create_data_sample(num_channels=NUM_CHANNELS, session_id="s3")

        tokens1 = model_with_linear.tokenize(data1)
        tokens2 = model_with_linear.tokenize(data2)
        tokens3 = model_with_linear.tokenize(data3)

        batch = collate([tokens1, tokens2, tokens3])
        output = model_with_linear(**extract_model_inputs(batch))

        assert "test_task1" in output
        assert output["test_task1"].shape[0] == 3

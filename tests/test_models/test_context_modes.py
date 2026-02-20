import numpy as np
import pytest
from temporaldata import Data, Interval, RegularTimeSeries
from torch_brain.data import collate
from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY
from torch_brain.nn.loss import CrossEntropyLoss

from foundry.models import POYOEEGModel, LinearEmbedding
from foundry.data.transforms import Patching


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
    if "test_context_task" not in MODALITY_REGISTRY:
        register_modality(
            "test_context_task",
            dim=2,
            type=DataType.BINARY,
            timestamp_key="test_task.timestamps",
            value_key="test_task.values",
            loss_fn=CrossEntropyLoss(),
        )
    return {"test_context_task": MODALITY_REGISTRY["test_context_task"]}


def create_data_sample(
    num_channels=4, sampling_rate=100.0, duration=2.0, session_id="session1"
):
    """Create a mock data sample."""
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
                "readout_id": "test_context_task",
            }
        ]
    }

    return data


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


class TestContextModes:
    def test_add_mode(self, readout_specs, embed_dim):
        """Test that add mode works correctly."""
        input_embedding = LinearEmbedding(embed_dim=embed_dim)
        model = POYOEEGModel(
            input_embedding=input_embedding,
            readout_specs=readout_specs,
            embed_dim=embed_dim,
            sequence_length=2.0,
            latent_step=0.5,
            num_latents_per_step=1,
            context_mode="add",
        )

        assert model.context_mode == "add"
        assert not hasattr(model, "context_projection")

        patching = Patching(patch_duration=0.5, stride=0.5)
        data = create_data_sample(num_channels=4, sampling_rate=100.0)
        patched = patching(data)

        model.session_emb.initialize_vocab(["session1"])
        model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])

        tokens = model.tokenize(patched)
        batch = collate([tokens])

        output = model(**extract_model_inputs(batch))

        assert "test_context_task" in output
        assert output["test_context_task"].shape[0] == 1

    def test_concat_mode(self, readout_specs, embed_dim):
        """Test that concat mode works correctly."""
        input_embedding = LinearEmbedding(embed_dim=embed_dim)
        model = POYOEEGModel(
            input_embedding=input_embedding,
            readout_specs=readout_specs,
            embed_dim=embed_dim,
            sequence_length=2.0,
            latent_step=0.5,
            num_latents_per_step=1,
            context_mode="concat",
        )

        assert model.context_mode == "concat"
        assert hasattr(model, "context_projection")
        assert model.context_projection.in_features == 3 * embed_dim
        assert model.context_projection.out_features == embed_dim

        patching = Patching(patch_duration=0.5, stride=0.5)
        data = create_data_sample(num_channels=4, sampling_rate=100.0)
        patched = patching(data)

        model.session_emb.initialize_vocab(["session1"])
        model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])

        tokens = model.tokenize(patched)
        batch = collate([tokens])

        output = model(**extract_model_inputs(batch))

        assert "test_context_task" in output
        assert output["test_context_task"].shape[0] == 1

    def test_invalid_context_mode(self, readout_specs, embed_dim):
        """Test that invalid context mode raises error."""
        input_embedding = LinearEmbedding(embed_dim=embed_dim)

        with pytest.raises(
            ValueError, match="context_mode must be 'add' or 'concat'"
        ):
            POYOEEGModel(
                input_embedding=input_embedding,
                readout_specs=readout_specs,
                embed_dim=embed_dim,
                sequence_length=2.0,
                context_mode="invalid",
            )

    def test_both_modes_produce_output(self, readout_specs, embed_dim):
        """Test that both modes produce outputs of the same shape."""
        patching = Patching(patch_duration=0.5, stride=0.5)
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s2"
        )
        patched1 = patching(data1)
        patched2 = patching(data2)

        model_add = POYOEEGModel(
            input_embedding=LinearEmbedding(embed_dim=embed_dim),
            readout_specs=readout_specs,
            embed_dim=embed_dim,
            sequence_length=2.0,
            context_mode="add",
        )

        model_concat = POYOEEGModel(
            input_embedding=LinearEmbedding(embed_dim=embed_dim),
            readout_specs=readout_specs,
            embed_dim=embed_dim,
            sequence_length=2.0,
            context_mode="concat",
        )

        for model in [model_add, model_concat]:
            model.session_emb.initialize_vocab(["s1", "s2"])
            model.channel_emb.initialize_vocab([f"ch{i}" for i in range(4)])

            tokens1 = model.tokenize(patched1)
            tokens2 = model.tokenize(patched2)
            batch = collate([tokens1, tokens2])

            output = model(**extract_model_inputs(batch))

            assert "test_context_task" in output
            assert output["test_context_task"].shape[0] == 2

    def test_concat_mode_with_heterogeneous_data(
        self, readout_specs, embed_dim
    ):
        """Test concat mode works with heterogeneous channel counts."""
        input_embedding = LinearEmbedding(embed_dim=embed_dim)
        model = POYOEEGModel(
            input_embedding=input_embedding,
            readout_specs=readout_specs,
            embed_dim=embed_dim,
            sequence_length=2.0,
            context_mode="concat",
        )

        patching = Patching(patch_duration=0.5, stride=0.5)
        data1 = create_data_sample(
            num_channels=4, sampling_rate=100.0, session_id="s1"
        )
        data2 = create_data_sample(
            num_channels=8, sampling_rate=100.0, session_id="s2"
        )

        patched1 = patching(data1)
        patched2 = patching(data2)

        model.session_emb.initialize_vocab(["s1", "s2"])
        model.channel_emb.initialize_vocab([f"ch{i}" for i in range(8)])

        tokens1 = model.tokenize(patched1)
        tokens2 = model.tokenize(patched2)
        batch = collate([tokens1, tokens2])

        output = model(**extract_model_inputs(batch))

        assert "test_context_task" in output
        assert output["test_context_task"].shape[0] == 2

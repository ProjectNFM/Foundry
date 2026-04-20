"""Tests for masked reconstruction pretraining integration.

Covers:
- EEGTokenizer with masking enabled (pretokenize + forward)
- POYOEEGModel in pretrain-only mode
- POYOEEGModel in joint (supervised + pretrain) mode
- PretrainModule training step
"""

import math

import numpy as np
import pytest
import torch
from temporaldata import Data, Interval, RegularTimeSeries
from torch_brain.data import collate
from torch_brain.nn.loss import CrossEntropyLoss
from torch_brain.registry import DataType, MODALITY_REGISTRY, register_modality

from foundry.models import (
    EEGTokenizer,
    FixedChannelStrategy,
    PerChannelStrategy,
    PatchLinearEmbedding,
    POYOEEGModel,
    RandomPatchMasking,
    ContiguousSpanMasking,
    ReconstructionHead,
)
from foundry.models.embeddings.temporal import (
    PerTimepointLinearEmbedding,
)
from foundry.models.poyo_eeg import RECON_DECODER_ID
from foundry.training import PretrainModule


SAMPLING_RATE = 250.0
PATCH_DURATION = 0.1
SEQUENCE_LENGTH = 1.0
NUM_CHANNELS = 8
PATCH_SAMPLES = int(PATCH_DURATION * SAMPLING_RATE)
NUM_SAMPLES = int(SEQUENCE_LENGTH * SAMPLING_RATE)
NUM_PATCHES = (NUM_SAMPLES - PATCH_SAMPLES) // PATCH_SAMPLES + 1
EMBED_DIM = 64
INIT_FREQS = torch.logspace(math.log10(2), math.log10(50), 8).tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    key = "pretrain_test_task"
    if key not in MODALITY_REGISTRY:
        register_modality(
            key,
            dim=2,
            type=DataType.BINARY,
            timestamp_key="test_task.timestamps",
            value_key="test_task.values",
            loss_fn=CrossEntropyLoss(),
        )
    return {key: MODALITY_REGISTRY[key]}


def _make_data_sample(num_channels=4, session_id="sess1"):
    signal = np.random.randn(NUM_SAMPLES, num_channels).astype(np.float32)
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
        timestamps = np.array([0.5])
        values = np.array([0])

    data.test_task = TestTask()
    data.config = {"multitask_readout": [{"readout_id": "pretrain_test_task"}]}
    return data


def _extract_model_inputs(batch):
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
            "input_sampling_rate",
            "latent_index",
            "latent_timestamps",
            "output_session_index",
            "output_timestamps",
            "output_decoder_index",
            "masking_mask",
        ]
    }


# ---------------------------------------------------------------------------
# Tokenizer masking tests
# ---------------------------------------------------------------------------


class TestTokenizerMaskingFixedPatch:
    """Fixed channels + patch tokenizer with masking."""

    def _make_tokenizer(self):
        return EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )

    def test_pretokenize_returns_masking_fields(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )

        assert "masking_mask" in result
        assert "reconstruction_targets" in result
        assert "masked_timestamps" in result
        assert result["masking_mask"].dtype == torch.bool
        assert result["masking_mask"].shape == (NUM_PATCHES,)

    def test_reconstruction_targets_shape(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        mask = result["masking_mask"]
        targets = result["reconstruction_targets"]

        assert targets.shape == (NUM_PATCHES, NUM_CHANNELS * PATCH_SAMPLES)
        n_masked = mask.sum().item()
        assert (targets[~mask] == 0).all()
        assert targets[mask].shape == (n_masked, NUM_CHANNELS * PATCH_SAMPLES)

    def test_masked_timestamps_shape(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        n_masked = result["masking_mask"].sum().item()
        assert result["masked_timestamps"].shape == (n_masked,)

    def test_forward_with_masking_mask(self):
        tokenizer = self._make_tokenizer()
        B = 2
        x = torch.randn(B, NUM_CHANNELS, NUM_SAMPLES)
        fs = torch.full((B,), SAMPLING_RATE)
        mask = torch.zeros(B, NUM_PATCHES, dtype=torch.bool)
        mask[:, :3] = True

        out = tokenizer(x, input_sampling_rate=fs, masking_mask=mask)
        assert out.shape == (B, NUM_PATCHES, EMBED_DIM)

    def test_forward_without_masking_unchanged(self):
        tokenizer = self._make_tokenizer()
        B = 2
        x = torch.randn(B, NUM_CHANNELS, NUM_SAMPLES)
        fs = torch.full((B,), SAMPLING_RATE)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (B, NUM_PATCHES, EMBED_DIM)

    def test_no_masking_fields_without_strategy(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
        )
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        assert "masking_mask" not in result
        assert "reconstruction_targets" not in result


class TestTokenizerMaskingPerChannelPatch:
    """Per-channel + patch tokenizer with masking."""

    def _make_tokenizer(self):
        return EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=1,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )

    def test_pretokenize_mask_shape(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        total_tokens = NUM_CHANNELS * NUM_PATCHES
        assert result["masking_mask"].shape == (total_tokens,)

    def test_reconstruction_targets_per_channel(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        total_tokens = NUM_CHANNELS * NUM_PATCHES
        assert result["reconstruction_targets"].shape == (
            total_tokens,
            PATCH_SAMPLES,
        )
        mask = result["masking_mask"]
        assert result["reconstruction_targets"][mask].shape[1] == PATCH_SAMPLES


class TestTokenizerMaskingFixedPerTimepoint:
    """Fixed channels + per-timepoint with masking."""

    def _make_tokenizer(self):
        return EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PerTimepointLinearEmbedding(
                embed_dim=EMBED_DIM,
                input_dim=NUM_CHANNELS,
            ),
            embed_dim=EMBED_DIM,
            masking=RandomPatchMasking(mask_ratio=0.2),
        )

    def test_pretokenize_mask_shape(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        assert result["masking_mask"].shape == (NUM_SAMPLES,)

    def test_reconstruction_targets_shape(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        assert result["reconstruction_targets"].shape == (
            NUM_SAMPLES,
            NUM_CHANNELS,
        )
        mask = result["masking_mask"]
        assert result["reconstruction_targets"][mask].shape[1] == NUM_CHANNELS


class TestTokenizerMaskingContiguousSpan:
    """Verify ContiguousSpanMasking integrates with the tokenizer."""

    def test_pretokenize_with_span_masking(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=ContiguousSpanMasking(mask_ratio=0.3, mean_span_length=3),
        )
        signal = np.random.randn(NUM_SAMPLES, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(
            signal, tokens, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        assert result["masking_mask"].any()
        assert result["reconstruction_targets"].ndim == 2


# ---------------------------------------------------------------------------
# POYOEEGModel pretrain-only mode
# ---------------------------------------------------------------------------


class TestPOYOEEGModelPretrainOnly:
    """Model configured with reconstruction_head only (no readout_specs)."""

    def _make_model(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )
        return POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            reconstruction_head=ReconstructionHead(
                embed_dim=EMBED_DIM,
                output_dim=NUM_CHANNELS * PATCH_SAMPLES,
            ),
            latent_step=0.5,
            num_latents_per_step=1,
        )

    def test_model_creation(self):
        model = self._make_model()
        assert model.readout_specs is None
        assert model.reconstruction_head is not None

    def test_requires_at_least_one_head(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
        )
        with pytest.raises(ValueError, match="At least one"):
            POYOEEGModel(
                tokenizer=tokenizer,
                embed_dim=EMBED_DIM,
                sequence_length=SEQUENCE_LENGTH,
            )

    def test_tokenize_produces_recon_queries(self):
        model = self._make_model()
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)

        assert "masking_mask" in tokens
        assert "reconstruction_targets" in tokens
        decoder_idx = tokens["output_decoder_index"]
        assert (np.asarray(decoder_idx) == RECON_DECODER_ID).any()

    def test_forward_returns_reconstruction(self):
        model = self._make_model()
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)
        batch = collate([tokens])
        model_inputs = _extract_model_inputs(batch)

        output = model(**model_inputs)
        assert "reconstruction" in output
        assert output["reconstruction"].ndim == 2
        assert output["reconstruction"].shape[1] == NUM_CHANNELS * PATCH_SAMPLES

    def test_forward_batch_of_two(self):
        model = self._make_model()
        model.session_emb.initialize_vocab(["s1", "s2"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data1 = _make_data_sample(session_id="s1")
        data2 = _make_data_sample(session_id="s2")
        t1 = model.tokenize(data1)
        t2 = model.tokenize(data2)
        batch = collate([t1, t2])
        model_inputs = _extract_model_inputs(batch)

        output = model(**model_inputs)
        assert "reconstruction" in output


# ---------------------------------------------------------------------------
# POYOEEGModel joint mode
# ---------------------------------------------------------------------------


class TestPOYOEEGModelJointMode:
    """Model configured with both readout_specs and reconstruction_head."""

    def _make_model(self, readout_specs):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )
        return POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            readout_specs=readout_specs,
            reconstruction_head=ReconstructionHead(
                embed_dim=EMBED_DIM,
                output_dim=NUM_CHANNELS * PATCH_SAMPLES,
            ),
            latent_step=0.5,
            num_latents_per_step=1,
        )

    def test_model_has_both_heads(self, readout_specs):
        model = self._make_model(readout_specs)
        assert model.readout_specs is not None
        assert model.reconstruction_head is not None

    def test_forward_returns_both_outputs(self, readout_specs):
        model = self._make_model(readout_specs)
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)
        batch = collate([tokens])
        model_inputs = _extract_model_inputs(batch)

        output = model(**model_inputs)
        assert "reconstruction" in output
        assert "pretrain_test_task" in output


# ---------------------------------------------------------------------------
# POYOEEGModel supervised-only backward compat
# ---------------------------------------------------------------------------


class TestPOYOEEGModelSupervisedCompat:
    """Existing supervised behavior is unchanged when no masking is configured."""

    def _make_model(self, readout_specs):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
        )
        return POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            readout_specs=readout_specs,
            latent_step=0.5,
            num_latents_per_step=1,
        )

    def test_no_masking_fields_in_tokenize(self, readout_specs):
        model = self._make_model(readout_specs)
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)
        assert "masking_mask" not in tokens
        assert "reconstruction_targets" not in tokens

    def test_forward_returns_task_outputs_only(self, readout_specs):
        model = self._make_model(readout_specs)
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)
        batch = collate([tokens])

        model_inputs = {
            k: v
            for k, v in batch.items()
            if k
            in [
                "input_values",
                "input_timestamps",
                "input_channel_index",
                "input_session_index",
                "input_mask",
                "input_sampling_rate",
                "latent_index",
                "latent_timestamps",
                "output_session_index",
                "output_timestamps",
                "output_decoder_index",
            ]
        }
        output = model(**model_inputs)
        assert "pretrain_test_task" in output
        assert "reconstruction" not in output


# ---------------------------------------------------------------------------
# PretrainModule
# ---------------------------------------------------------------------------


class TestPretrainModule:
    def _make_module(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )
        model = POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            reconstruction_head=ReconstructionHead(
                embed_dim=EMBED_DIM,
                output_dim=NUM_CHANNELS * PATCH_SAMPLES,
            ),
            latent_step=0.5,
            num_latents_per_step=1,
        )
        return PretrainModule(
            model=model, learning_rate=1e-4, weight_decay=0.01
        )

    def test_instantiation(self):
        module = self._make_module()
        assert isinstance(module, PretrainModule)

    def test_training_step_returns_loss(self):
        module = self._make_module()
        model = module.model
        model.session_emb.initialize_vocab(["sess1"])
        model.channel_emb.initialize_vocab(
            [f"ch{i}" for i in range(NUM_CHANNELS)]
        )

        data = _make_data_sample()
        tokens = model.tokenize(data)
        batch = collate([tokens])

        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_invalid_loss_type(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )
        model = POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            reconstruction_head=ReconstructionHead(
                embed_dim=EMBED_DIM,
                output_dim=NUM_CHANNELS * PATCH_SAMPLES,
            ),
            latent_step=0.5,
            num_latents_per_step=1,
        )
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            PretrainModule(model=model, loss_type="l2")

    def test_smooth_l1_loss(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=NUM_CHANNELS),
            temporal_embedding=PatchLinearEmbedding(
                embed_dim=EMBED_DIM,
                num_input_channels=NUM_CHANNELS,
                patch_samples=PATCH_SAMPLES,
            ),
            embed_dim=EMBED_DIM,
            patch_duration=PATCH_DURATION,
            masking=RandomPatchMasking(mask_ratio=0.3),
        )
        model = POYOEEGModel(
            tokenizer=tokenizer,
            embed_dim=EMBED_DIM,
            sequence_length=SEQUENCE_LENGTH,
            reconstruction_head=ReconstructionHead(
                embed_dim=EMBED_DIM,
                output_dim=NUM_CHANNELS * PATCH_SAMPLES,
            ),
            latent_step=0.5,
            num_latents_per_step=1,
        )
        module = PretrainModule(model=model, loss_type="smooth_l1")
        assert isinstance(module.loss_fn, torch.nn.SmoothL1Loss)

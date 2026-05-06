import math

import numpy as np
import pytest
import torch

from foundry.models.embeddings.channel import (
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
    LinearSpatialProjector,
    PerceiverSpatialProjector,
    SessionSpatialProjector,
)
from foundry.models.embeddings.temporal import (
    PatchCNNEmbedding as CNNEmbedding,
    CWTEmbedding,
    PerTimepointIdentityEmbedding,
    PatchLinearEmbedding as LinearEmbedding,
    PerTimepointLinearEmbedding,
)
from foundry.models.tokenizer import EEGTokenizer

INIT_FREQS = torch.logspace(math.log10(2), math.log10(50), 8).tolist()


class TestMode1FixedChannelPatched:
    """Mode 1: Fixed channels, fixed fs, patched."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=8),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim, num_input_channels=8, patch_samples=25
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
            channel_fusion="add",
        )

    def test_pretokenize_keys(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(250, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        expected_keys = {
            "input_values",
            "input_channel_index",
            "input_mask",
            "input_sampling_rate",
            "input_timestamps",
            "input_seq_len",
        }
        assert set(result.keys()) == expected_keys

    def test_pretokenize_shapes(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(250, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        assert result["input_values"].shape == (8, 250)
        assert result["input_mask"].shape == (8,)
        # num_patches = (250-25)//25+1 = 10
        assert result["input_timestamps"].shape == (10,)

    def test_forward_shape(self, embed_dim, batch_size):
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(batch_size, 8, 250)
        fs = torch.full((batch_size,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (batch_size, 10, embed_dim)

    def test_forward_gradient_flow(self):
        tokenizer = self._make_tokenizer()
        x = torch.randn(1, 8, 250, requires_grad=True)
        fs = torch.full((1,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        out.sum().backward()
        assert x.grad is not None


class TestMode2aPerChannelPatched:
    """Mode 2a: Variable channels, fixed fs, per-channel patched."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=16),
            temporal_embedding=CNNEmbedding(
                embed_dim=embed_dim,
                num_input_channels=1,
                patch_samples=25,
                num_filters=16,
                kernel_size=3,
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
            channel_fusion="add",
        )

    def test_pretokenize_shapes(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(250, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        assert result["input_values"].shape == (16, 250)
        assert result["input_mask"].shape == (16,)
        # 16 channels * 10 patches
        assert result["input_timestamps"].shape == (16 * 10,)

    def test_forward_shape(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        B = batch_size
        C = 16
        T = 250
        x = torch.randn(B, C, T)
        mask = torch.ones(B, C, dtype=torch.bool)
        mask[:, 4:] = False
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, embed_dim)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        # 16 channels * 10 patches = 160
        assert out.shape == (B, C * 10, embed_dim)


class TestMode2aConcatPerChannelPatched:
    """Mode 2a-concat: Per-channel patched with channel_fusion='concat'."""

    EMBED_DIM = 64
    CHANNEL_EMB_DIM = 16
    TOKEN_EMBED_DIM = EMBED_DIM - CHANNEL_EMB_DIM

    def _make_tokenizer(self):
        return EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=16),
            temporal_embedding=CNNEmbedding(
                embed_dim=self.TOKEN_EMBED_DIM,
                num_input_channels=1,
                patch_samples=25,
                num_filters=16,
                kernel_size=3,
            ),
            embed_dim=self.EMBED_DIM,
            patch_duration=0.1,
            channel_fusion="concat",
            channel_emb_dim=self.CHANNEL_EMB_DIM,
        )

    def test_token_embed_dim_property(self):
        tokenizer = self._make_tokenizer()
        assert tokenizer.token_embed_dim == self.TOKEN_EMBED_DIM
        assert tokenizer.channel_emb_dim == self.CHANNEL_EMB_DIM

    def test_forward_shape(self):
        tokenizer = self._make_tokenizer()
        B, C, T = 2, 16, 250
        x = torch.randn(B, C, T)
        mask = torch.ones(B, C, dtype=torch.bool)
        mask[:, 4:] = False
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, self.CHANNEL_EMB_DIM)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        assert out.shape == (B, C * 10, self.EMBED_DIM)

    def test_padded_channels_zeroed(self):
        tokenizer = self._make_tokenizer()
        B, C, T = 1, 16, 250
        x = torch.randn(B, C, T)
        mask = torch.zeros(B, C, dtype=torch.bool)
        mask[:, :3] = True
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, self.CHANNEL_EMB_DIM)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        out_reshaped = out.reshape(B, C, 10, self.EMBED_DIM)
        assert (out_reshaped[:, 3:, :, :] == 0).all()

    def test_forward_gradient_flow(self):
        tokenizer = self._make_tokenizer()
        B, C, T = 1, 16, 250
        x = torch.randn(B, C, T, requires_grad=True)
        mask = torch.ones(B, C, dtype=torch.bool)
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, self.CHANNEL_EMB_DIM)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        out.sum().backward()
        assert x.grad is not None

    def test_concat_requires_channel_emb_dim(self):
        with pytest.raises(ValueError, match="channel_emb_dim is required"):
            EEGTokenizer(
                channel_strategy=PerChannelStrategy(max_channels=16),
                temporal_embedding=CNNEmbedding(
                    embed_dim=48,
                    num_input_channels=1,
                    patch_samples=25,
                    num_filters=16,
                    kernel_size=3,
                ),
                embed_dim=64,
                patch_duration=0.1,
                channel_fusion="concat",
                channel_emb_dim=None,
            )

    def test_channel_emb_dim_must_be_less_than_embed_dim(self):
        with pytest.raises(ValueError, match="must be less than"):
            EEGTokenizer(
                channel_strategy=PerChannelStrategy(max_channels=16),
                temporal_embedding=CNNEmbedding(
                    embed_dim=1,
                    num_input_channels=1,
                    patch_samples=25,
                    num_filters=16,
                    kernel_size=3,
                ),
                embed_dim=64,
                patch_duration=0.1,
                channel_fusion="concat",
                channel_emb_dim=64,
            )


class TestMode2bSpatialProjectionPatched:
    """Mode 2b: Variable channels, fixed fs, spatial projection + patched."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=8,
                projector=LinearSpatialProjector(
                    num_channels=64, num_sources=8
                ),
            ),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim, num_input_channels=8, patch_samples=25
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
            channel_fusion="add",
        )

    def test_forward_shape(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(batch_size, 64, 250)
        fs = torch.full((batch_size,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (batch_size, 10, embed_dim)


class TestMode3aSpatialProjectionCWT:
    """Mode 3a: Variable channels, variable fs, CWT."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=8,
                projector=LinearSpatialProjector(
                    num_channels=64, num_sources=8
                ),
            ),
            temporal_embedding=CWTEmbedding(
                embed_dim=embed_dim,
                num_sources=8,
                init_freqs=INIT_FREQS,
                target_token_rate=32.0,
            ),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_pretokenize_timestamps_match_target_tokens(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(250, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)
        assert result["input_timestamps"].shape == (32,)

    def test_forward_shape(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(batch_size, 64, 250)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 250, dtype=torch.long)

        out = tokenizer(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, 32, embed_dim)


class TestMode3bSpatialProjectionPerTimepoint:
    """Mode 3b: Variable channels, variable fs, per-timepoint."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=8,
                projector=LinearSpatialProjector(
                    num_channels=64, num_sources=8
                ),
            ),
            temporal_embedding=PerTimepointLinearEmbedding(
                embed_dim=embed_dim, input_dim=8
            ),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_pretokenize_timestamps_per_sample(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(200, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)
        assert result["input_timestamps"].shape == (200,)

    def test_forward_shape(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(batch_size, 64, 200)
        fs = torch.full((batch_size,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (batch_size, 200, embed_dim)


class TestMode3cSpatialProjectionIdentityTemporal:
    """Mode 3c: Variable channels, variable fs, identity temporal embedding."""

    def _make_tokenizer(self, embed_dim=8, num_sources=8):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=num_sources,
                projector=LinearSpatialProjector(
                    num_channels=64, num_sources=num_sources
                ),
            ),
            temporal_embedding=PerTimepointIdentityEmbedding(
                embed_dim=embed_dim
            ),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_forward_shape(self, batch_size):
        tokenizer = self._make_tokenizer(embed_dim=8, num_sources=8)
        x = torch.randn(batch_size, 64, 200)
        fs = torch.full((batch_size,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (batch_size, 200, 8)

    def test_forward_raises_on_dim_mismatch(self):
        tokenizer = self._make_tokenizer(embed_dim=16, num_sources=8)
        x = torch.randn(2, 64, 200)
        fs = torch.full((2,), 250.0)

        with pytest.raises(ValueError, match="match embed_dim"):
            tokenizer(x, input_sampling_rate=fs)


class TestMode4PerChannelPerTimepoint:
    """Mode 4: Variable channels, any fs, per-channel per-timepoint."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=16),
            temporal_embedding=PerTimepointLinearEmbedding(
                embed_dim=embed_dim, input_dim=1
            ),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_pretokenize_shapes(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(100, 4).astype(np.float32)
        tokens = np.arange(4)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        assert result["input_values"].shape == (16, 100)
        assert result["input_timestamps"].shape == (16 * 100,)

    def test_forward_shape(self):
        B, C, T, D = 2, 16, 100, 64
        tokenizer = self._make_tokenizer(embed_dim=D)
        x = torch.randn(B, C, T)
        mask = torch.ones(B, C, dtype=torch.bool)
        mask[:, 4:] = False
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, D)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        assert out.shape == (B, C * T, D)

    def test_padded_channels_zeroed(self):
        B, C, T, D = 1, 8, 50, 32
        tokenizer = self._make_tokenizer(embed_dim=D)
        x = torch.randn(B, C, T)
        mask = torch.zeros(B, C, dtype=torch.bool)
        mask[:, :3] = True
        ch_idx = torch.arange(C).unsqueeze(0).expand(B, C)
        fs = torch.full((B,), 250.0)

        emb = torch.nn.Embedding(C, D)
        out = tokenizer(
            x,
            input_mask=mask,
            input_channel_index=ch_idx,
            input_sampling_rate=fs,
            channel_emb_fn=emb,
        )
        # Tokens from channels 3-7 should be zeroed out
        out_reshaped = out.reshape(B, C, T, D)
        assert (out_reshaped[:, 3:, :, :] == 0).all()


class TestSpatialProjectionWithSessionConfig:
    """SpatialProjectionStrategy with per-session spatial projectors."""

    SESSION_CONFIGS = {"sessA": 20, "sessB": 32}
    NUM_CHANNELS = 32
    NUM_SOURCES = 8

    def _make_patched_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=self.NUM_CHANNELS,
                num_sources=self.NUM_SOURCES,
                projector=SessionSpatialProjector(
                    session_configs=self.SESSION_CONFIGS,
                    num_sources=self.NUM_SOURCES,
                ),
            ),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim,
                num_input_channels=self.NUM_SOURCES,
                patch_samples=25,
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
            channel_fusion="add",
        )

    def _make_cwt_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=self.NUM_CHANNELS,
                num_sources=self.NUM_SOURCES,
                projector=SessionSpatialProjector(
                    session_configs=self.SESSION_CONFIGS,
                    num_sources=self.NUM_SOURCES,
                ),
            ),
            temporal_embedding=CWTEmbedding(
                embed_dim=embed_dim,
                num_sources=self.NUM_SOURCES,
                init_freqs=INIT_FREQS,
                target_token_rate=20.0,
            ),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_pretokenize_shapes(self):
        tokenizer = self._make_patched_tokenizer()
        signal = np.random.randn(250, 15).astype(np.float32)
        tokens = np.arange(15)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        assert result["input_values"].shape == (self.NUM_CHANNELS, 250)
        assert result["input_mask"][:15].all()
        assert not result["input_mask"][15:].any()
        assert result["input_seq_len"].item() == 250
        # (250-25)//25+1 = 10
        assert result["input_timestamps"].shape == (10,)

    def test_forward_patched_heterogeneous_sessions(self):
        """Two batch items from different sessions with different true channel counts."""
        embed_dim = 64
        tokenizer = self._make_patched_tokenizer(embed_dim=embed_dim)

        B = 2
        C = self.NUM_CHANNELS
        T = 250
        x = torch.randn(B, C, T)
        fs = torch.full((B,), 250.0)

        out = tokenizer(
            x,
            input_sampling_rate=fs,
            input_session_ids=["sessA", "sessB"],
            input_channel_counts=[20, 32],
            input_seq_len=torch.tensor([T, T]),
        )
        # (250-25)//25+1 = 10 patches
        assert out.shape == (B, 10, embed_dim)

    def test_forward_cwt_heterogeneous_sessions(self):
        """CWT temporal embedding with per-session spatial projection."""
        embed_dim = 64
        tokenizer = self._make_cwt_tokenizer(embed_dim=embed_dim)

        B = 2
        C = self.NUM_CHANNELS
        T = 200
        x = torch.randn(B, C, T)
        fs = torch.full((B,), 250.0)
        seq_lens = torch.tensor([T, T])

        out = tokenizer(
            x,
            input_sampling_rate=fs,
            input_seq_len=seq_lens,
            input_session_ids=["sessA", "sessB"],
            input_channel_counts=[20, 32],
        )
        assert out.shape == (B, 16, embed_dim)

    def test_forward_gradient_flow(self):
        embed_dim = 64
        tokenizer = self._make_patched_tokenizer(embed_dim=embed_dim)

        B = 1
        C = self.NUM_CHANNELS
        T = 250
        x = torch.randn(B, C, T, requires_grad=True)
        fs = torch.full((B,), 250.0)

        out = tokenizer(
            x,
            input_sampling_rate=fs,
            input_session_ids=["sessA"],
            input_channel_counts=[20],
            input_seq_len=torch.tensor([T]),
        )
        out.sum().backward()
        assert x.grad is not None

    def test_forward_variable_seq_lens(self):
        """Batch items with different true time lengths (zero-padded to max)."""
        embed_dim = 64
        tokenizer = self._make_cwt_tokenizer(embed_dim=embed_dim)

        B = 2
        C = self.NUM_CHANNELS
        Max_T = 300
        x = torch.randn(B, C, Max_T)
        fs = torch.tensor([250.0, 500.0])
        seq_lens = torch.tensor([200, 300])

        out = tokenizer(
            x,
            input_sampling_rate=fs,
            input_seq_len=seq_lens,
            input_session_ids=["sessA", "sessB"],
            input_channel_counts=[20, 32],
        )
        assert out.shape == (B, 16, embed_dim)


class TestPerceiverSpatialPatched:
    """Perceiver cross-attention spatial projection + patched temporal embedding."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=8,
                projector=PerceiverSpatialProjector(
                    num_sources=8, d_attn=32, num_heads=4
                ),
            ),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim, num_input_channels=8, patch_samples=25
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
            channel_fusion="add",
        )

    def test_pretokenize_shapes(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(250, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)

        assert result["input_values"].shape == (64, 250)
        assert result["input_mask"][:30].all()
        assert not result["input_mask"][30:].any()
        assert result["input_timestamps"].shape == (10,)

    def test_forward_shape(self):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        B = 2
        x = torch.randn(B, 64, 250)
        mask = torch.ones(B, 64, dtype=torch.bool)
        mask[:, 30:] = False
        fs = torch.full((B,), 250.0)

        out = tokenizer(x, input_mask=mask, input_sampling_rate=fs)
        assert out.shape == (B, 10, embed_dim)

    def test_forward_gradient_flow(self):
        tokenizer = self._make_tokenizer()
        x = torch.randn(1, 64, 250, requires_grad=True)
        mask = torch.ones(1, 64, dtype=torch.bool)
        fs = torch.full((1,), 250.0)

        out = tokenizer(x, input_mask=mask, input_sampling_rate=fs)
        out.sum().backward()
        assert x.grad is not None

    def test_forward_without_mask(self):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(2, 64, 250)
        fs = torch.full((2,), 250.0)

        out = tokenizer(x, input_sampling_rate=fs)
        assert out.shape == (2, 10, embed_dim)


class TestTokenizerProperties:
    def test_uses_per_channel_true(self):
        tokenizer = EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=8),
            temporal_embedding=PerTimepointLinearEmbedding(
                embed_dim=32, input_dim=1
            ),
            embed_dim=32,
            channel_fusion="add",
        )
        assert tokenizer.uses_per_channel is True

    def test_uses_per_channel_false(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=8),
            temporal_embedding=LinearEmbedding(
                embed_dim=32, num_input_channels=8, patch_samples=25
            ),
            embed_dim=32,
            patch_duration=0.1,
            channel_fusion="add",
        )
        assert tokenizer.uses_per_channel is False

    def test_stride_defaults_to_patch_duration(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=8),
            temporal_embedding=LinearEmbedding(
                embed_dim=32, num_input_channels=8, patch_samples=25
            ),
            embed_dim=32,
            patch_duration=0.1,
            channel_fusion="add",
        )
        assert tokenizer.stride == 0.1

    def test_explicit_stride(self):
        tokenizer = EEGTokenizer(
            channel_strategy=FixedChannelStrategy(num_channels=8),
            temporal_embedding=LinearEmbedding(
                embed_dim=32, num_input_channels=8, patch_samples=25
            ),
            embed_dim=32,
            patch_duration=0.2,
            stride=0.1,
            channel_fusion="add",
        )
        assert tokenizer.stride == 0.1


class TestCWTCompressor:
    """CWTEmbedding with the learned strided-CNN compressor."""

    GRID_HZ = 100.0
    STRIDES = [5, 2]
    TOTAL_STRIDE = 10
    NUM_SOURCES = 8

    def _make_embedding(self, embed_dim=64, **overrides):
        kwargs = dict(
            embed_dim=embed_dim,
            num_sources=self.NUM_SOURCES,
            init_freqs=INIT_FREQS,
            grid_resample_hz=self.GRID_HZ,
            compressor_strides=self.STRIDES,
            compressor_num_filters=32,
            compressor_kernel_size=9,
        )
        kwargs.update(overrides)
        return CWTEmbedding(**kwargs)

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64,
                num_sources=self.NUM_SOURCES,
                projector=LinearSpatialProjector(
                    num_channels=64, num_sources=self.NUM_SOURCES
                ),
            ),
            temporal_embedding=self._make_embedding(embed_dim=embed_dim),
            embed_dim=embed_dim,
            channel_fusion="add",
        )

    def test_target_token_rate_derived_from_hz_and_strides(self):
        emb = self._make_embedding()
        assert emb.target_token_rate == self.GRID_HZ / self.TOTAL_STRIDE

    def test_output_hz_property(self):
        emb = self._make_embedding()
        assert emb.output_hz == self.GRID_HZ / self.TOTAL_STRIDE

    def test_seconds_per_token_property(self):
        emb = self._make_embedding()
        expected = 1.0 / (self.GRID_HZ / self.TOTAL_STRIDE)
        assert emb.seconds_per_token == pytest.approx(expected)

    def test_compute_num_tokens_exact(self):
        emb = self._make_embedding()
        grid_tokens = round(self.GRID_HZ * 1.0)  # 100
        after_s0 = (grid_tokens - 1) // 5 + 1  # 20
        after_s1 = (after_s0 - 1) // 2 + 1  # 10
        assert emb.compute_num_tokens(1.0) == after_s1

    def test_compute_num_tokens_fractional_duration(self):
        emb = self._make_embedding()
        duration = 0.8
        grid_tokens = round(self.GRID_HZ * duration)  # 80
        after_s0 = (grid_tokens - 1) // 5 + 1  # 16
        after_s1 = (after_s0 - 1) // 2 + 1  # 8
        assert emb.compute_num_tokens(duration) == after_s1

    def test_forward_shape_matches_compute_num_tokens(self, batch_size):
        embed_dim = 64
        emb = self._make_embedding(embed_dim=embed_dim)

        fs_val = 250.0
        T = 250
        x = torch.randn(batch_size, self.NUM_SOURCES, T)
        fs = torch.full((batch_size,), fs_val)
        seq_lens = torch.full((batch_size,), T, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        duration = T / fs_val
        expected_tokens = emb.compute_num_tokens(duration)
        assert out.shape == (batch_size, expected_tokens, embed_dim)

    def test_forward_shape_2s_duration(self, batch_size):
        embed_dim = 64
        emb = self._make_embedding(embed_dim=embed_dim)

        fs_val = 250.0
        duration = 2.0
        T = round(fs_val * duration)
        x = torch.randn(batch_size, self.NUM_SOURCES, T)
        fs = torch.full((batch_size,), fs_val)
        seq_lens = torch.full((batch_size,), T, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        expected_tokens = emb.compute_num_tokens(duration)
        assert out.shape == (batch_size, expected_tokens, embed_dim)

    def test_output_tokens_scale_with_duration(self):
        emb = self._make_embedding()
        tokens_1s = emb.compute_num_tokens(1.0)
        tokens_2s = emb.compute_num_tokens(2.0)
        assert tokens_2s == pytest.approx(tokens_1s * 2, abs=1)

    def test_pretokenize_timestamps_match_forward(self):
        """Pretokenize and forward must agree on token count."""
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)

        fs_val = 250.0
        duration = 1.0
        T = round(fs_val * duration)
        signal = np.random.randn(T, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, fs_val, duration)
        pretok_count = result["input_timestamps"].shape[0]

        x = torch.randn(1, 64, T)
        fs = torch.full((1,), fs_val)
        seq_lens = torch.full((1,), T, dtype=torch.long)
        out = tokenizer(x, input_sampling_rate=fs, input_seq_len=seq_lens)

        assert pretok_count == out.shape[1]

    def test_pretokenize_timestamps_span_duration(self):
        tokenizer = self._make_tokenizer()
        duration = 1.0
        T = 250
        signal = np.random.randn(T, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, 250.0, duration)
        ts = result["input_timestamps"]
        assert ts[0].item() == pytest.approx(0.0)
        assert ts[-1].item() == pytest.approx(duration)

    def test_gradient_flow(self):
        emb = self._make_embedding()
        x = torch.randn(1, self.NUM_SOURCES, 250, requires_grad=True)
        fs = torch.full((1,), 250.0)
        seq_lens = torch.full((1,), 250, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        out.sum().backward()
        assert x.grad is not None

    def test_full_tokenizer_forward(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)

        x = torch.randn(batch_size, 64, 250)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 250, dtype=torch.long)

        out = tokenizer(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        expected_tokens = tokenizer.temporal_embedding.compute_num_tokens(1.0)
        assert out.shape == (batch_size, expected_tokens, embed_dim)

    def test_variable_seq_lens(self):
        embed_dim = 64
        emb = self._make_embedding(embed_dim=embed_dim)

        B = 2
        Max_T = 300
        x = torch.randn(B, self.NUM_SOURCES, Max_T)
        fs = torch.tensor([250.0, 500.0])
        seq_lens = torch.tensor([200, 300])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        max_duration = max(200 / 250.0, 300 / 500.0)
        expected_tokens = emb.compute_num_tokens(max_duration)
        assert out.shape == (B, expected_tokens, embed_dim)

    def test_backward_compat_no_compressor(self, batch_size):
        """CWTEmbedding without compressor args works identically to before."""
        embed_dim = 64
        emb = CWTEmbedding(
            embed_dim=embed_dim,
            num_sources=self.NUM_SOURCES,
            init_freqs=INIT_FREQS,
            target_token_rate=32.0,
        )
        assert emb.compressor is None
        assert emb.grid_resample_hz is None
        assert emb.target_token_rate == 32.0

        x = torch.randn(batch_size, self.NUM_SOURCES, 250)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 250, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, 32, embed_dim)

    def test_compressor_strides_without_grid_hz_raises(self):
        with pytest.raises(ValueError, match="grid_resample_hz"):
            CWTEmbedding(
                embed_dim=64,
                num_sources=self.NUM_SOURCES,
                init_freqs=INIT_FREQS,
                compressor_strides=[2, 2],
            )

    def test_grid_hz_without_strides_raises(self):
        with pytest.raises(ValueError, match="compressor_strides"):
            CWTEmbedding(
                embed_dim=64,
                num_sources=self.NUM_SOURCES,
                init_freqs=INIT_FREQS,
                grid_resample_hz=100.0,
            )

    def test_three_layer_strides(self):
        emb = self._make_embedding(compressor_strides=[2, 3, 2])
        assert emb.target_token_rate == pytest.approx(self.GRID_HZ / 12)
        duration = 1.2
        grid = round(self.GRID_HZ * duration)  # 120
        l1 = (grid - 1) // 2 + 1  # 60
        l2 = (l1 - 1) // 3 + 1  # 20
        l3 = (l2 - 1) // 2 + 1  # 10
        assert emb.compute_num_tokens(duration) == l3

    def test_per_layer_kernel_sizes(self, batch_size):
        """Per-layer kernel sizes: larger first layer, smaller second."""
        embed_dim = 64
        emb = self._make_embedding(
            embed_dim=embed_dim, compressor_kernel_size=[9, 4]
        )
        assert emb._compressor_kernel_sizes == [9, 4]

        fs_val = 250.0
        T = 250
        x = torch.randn(batch_size, self.NUM_SOURCES, T)
        fs = torch.full((batch_size,), fs_val)
        seq_lens = torch.full((batch_size,), T, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        duration = T / fs_val
        expected_tokens = emb.compute_num_tokens(duration)
        assert out.shape == (batch_size, expected_tokens, embed_dim)

    def test_per_layer_kernel_sizes_token_count(self):
        """Verify exact token math with mixed odd/even kernel sizes."""
        emb = self._make_embedding(compressor_kernel_size=[9, 4])
        grid = round(self.GRID_HZ * 1.0)  # 100
        # ks=9, padding=4, stride=5: (100+8-9)//5+1 = 20
        l1 = (grid + 2 * 4 - 9) // 5 + 1
        assert l1 == 20
        # ks=4, padding=2, stride=2: (20+4-4)//2+1 = 11
        l2 = (l1 + 2 * 2 - 4) // 2 + 1
        assert l2 == 11
        assert emb.compute_num_tokens(1.0) == l2

    def test_kernel_size_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            self._make_embedding(compressor_kernel_size=[9, 4, 3])

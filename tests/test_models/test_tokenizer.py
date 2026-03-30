import math

import numpy as np
import torch

from foundry.models.embeddings.channel_strategies import (
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
)
from foundry.models.embeddings.spatial import PerceiverSpatialProjector
from foundry.models.embeddings.cnn import CNNEmbedding
from foundry.models.embeddings.cwt import CWTEmbedding
from foundry.models.embeddings.linear import LinearEmbedding
from foundry.models.embeddings.per_timepoint import PerTimepointEmbedding
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


class TestMode2bSpatialProjectionPatched:
    """Mode 2b: Variable channels, fixed fs, spatial projection + patched."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64, num_sources=8
            ),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim, num_input_channels=8, patch_samples=25
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
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
                num_channels=64, num_sources=8
            ),
            temporal_embedding=CWTEmbedding(
                embed_dim=embed_dim,
                num_sources=8,
                init_freqs=INIT_FREQS,
                target_time_tokens=32,
            ),
            embed_dim=embed_dim,
        )

    def test_pretokenize_timestamps_match_target_tokens(self):
        tokenizer = self._make_tokenizer()
        signal = np.random.randn(200, 30).astype(np.float32)
        tokens = np.arange(30)

        result = tokenizer.pretokenize(signal, tokens, 250.0, 1.0)
        assert result["input_timestamps"].shape == (32,)

    def test_forward_shape(self, batch_size):
        embed_dim = 64
        tokenizer = self._make_tokenizer(embed_dim=embed_dim)
        x = torch.randn(batch_size, 64, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = tokenizer(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, 32, embed_dim)


class TestMode3bSpatialProjectionPerTimepoint:
    """Mode 3b: Variable channels, variable fs, per-timepoint."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=64, num_sources=8
            ),
            temporal_embedding=PerTimepointEmbedding(
                embed_dim=embed_dim, input_dim=8
            ),
            embed_dim=embed_dim,
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


class TestMode4PerChannelPerTimepoint:
    """Mode 4: Variable channels, any fs, per-channel per-timepoint."""

    def _make_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=PerChannelStrategy(max_channels=16),
            temporal_embedding=PerTimepointEmbedding(
                embed_dim=embed_dim, input_dim=1
            ),
            embed_dim=embed_dim,
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
                session_configs=self.SESSION_CONFIGS,
            ),
            temporal_embedding=LinearEmbedding(
                embed_dim=embed_dim,
                num_input_channels=self.NUM_SOURCES,
                patch_samples=25,
            ),
            embed_dim=embed_dim,
            patch_duration=0.1,
        )

    def _make_cwt_tokenizer(self, embed_dim=64):
        return EEGTokenizer(
            channel_strategy=SpatialProjectionStrategy(
                num_channels=self.NUM_CHANNELS,
                num_sources=self.NUM_SOURCES,
                session_configs=self.SESSION_CONFIGS,
            ),
            temporal_embedding=CWTEmbedding(
                embed_dim=embed_dim,
                num_sources=self.NUM_SOURCES,
                init_freqs=INIT_FREQS,
                target_time_tokens=16,
            ),
            embed_dim=embed_dim,
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
            temporal_embedding=PerTimepointEmbedding(embed_dim=32, input_dim=1),
            embed_dim=32,
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
        )
        assert tokenizer.stride == 0.1

import math

import numpy as np
import pytest
import torch

from foundry.models import CWTEmbedding, ContinuousCWTLayer, EmbeddingBase


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

INIT_FREQS = torch.logspace(math.log10(2), math.log10(50), 8).tolist()
NUM_FREQS = len(INIT_FREQS)


def _make_cwt_embedding(
    embed_dim=64,
    num_channels=8,
    num_sources=4,
    target_time_tokens=32,
    **kwargs,
):
    return CWTEmbedding(
        embed_dim=embed_dim,
        num_channels=num_channels,
        num_sources=num_sources,
        init_freqs=INIT_FREQS,
        target_time_tokens=target_time_tokens,
        **kwargs,
    )


# ------------------------------------------------------------------ #
# ContinuousCWTLayer
# ------------------------------------------------------------------ #


class TestContinuousCWTLayer:
    def test_initialization(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=32)
        assert layer.target_time_tokens == 32
        assert layer.freqs.shape == (NUM_FREQS,)
        assert layer.n_cycles.shape == (NUM_FREQS,)

    def test_output_shape(self, batch_size):
        C, Max_T = 4, 200
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=32)
        x = torch.randn(batch_size, C, Max_T)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), Max_T, dtype=torch.long)

        out = layer(x, fs, seq_lens)
        assert out.shape == (batch_size, C, 2, NUM_FREQS, 32)

    def test_magnitude_nonnegative(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        x = torch.randn(1, 2, 100)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = layer(x, fs, seq_lens)
        mag = out[:, :, 0, :, :]
        assert (mag >= 0).all()

    def test_phase_in_range(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        x = torch.randn(1, 2, 100)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = layer(x, fs, seq_lens)
        phase = out[:, :, 1, :, :]
        assert phase.min() >= -1.0 - 1e-6
        assert phase.max() <= 1.0 + 1e-6

    def test_variable_sampling_rates(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        x = torch.randn(2, 3, 200)
        fs = torch.tensor([250.0, 500.0])
        seq_lens = torch.tensor([200, 200])

        out = layer(x, fs, seq_lens)
        assert out.shape == (2, 3, 2, NUM_FREQS, 16)

    def test_variable_seq_lens(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        x = torch.randn(2, 3, 200)
        fs = torch.tensor([250.0, 250.0])
        seq_lens = torch.tensor([100, 200])

        out = layer(x, fs, seq_lens)
        assert out.shape == (2, 3, 2, NUM_FREQS, 16)

    def test_gradients_flow_through_freqs(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        x = torch.randn(1, 2, 100)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = layer(x, fs, seq_lens)
        out.sum().backward()
        assert layer.freqs.grad is not None
        assert layer.n_cycles.grad is not None

    def test_learnable_parameters(self):
        layer = ContinuousCWTLayer(
            init_freqs=INIT_FREQS, target_time_tokens=16, n_cycles=5.0
        )
        assert layer.freqs.requires_grad
        assert layer.n_cycles.requires_grad
        assert (layer.n_cycles == 5.0).all()


# ------------------------------------------------------------------ #
# CWTEmbedding — initialisation
# ------------------------------------------------------------------ #


class TestCWTEmbeddingInit:
    def test_class_hierarchy(self):
        emb = _make_cwt_embedding()
        assert isinstance(emb, EmbeddingBase)

    def test_requires_patching_is_false(self):
        emb = _make_cwt_embedding()
        assert emb.requires_patching is False

    def test_attributes(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim)
        assert emb.embed_dim == embed_dim
        assert emb.num_channels == 8
        assert emb.num_sources == 4
        assert emb.target_time_tokens == 32

    def test_default_spatial_is_linear(self):
        emb = _make_cwt_embedding()
        assert isinstance(emb.spatial, torch.nn.Linear)
        assert not emb._use_session_spatial

    def test_spatial_with_hidden_dim(self):
        emb = _make_cwt_embedding(shared_spatial_hidden_dim=16)
        assert isinstance(emb.spatial, torch.nn.Sequential)
        assert not emb._use_session_spatial

    def test_spatial_with_session_configs(self):
        emb = _make_cwt_embedding(session_configs={"A": 8, "B": 16})
        assert emb._use_session_spatial

    def test_feature_proj_dimensions(self, embed_dim):
        num_sources = 4
        emb = _make_cwt_embedding(embed_dim=embed_dim, num_sources=num_sources)
        expected_in = num_sources * 2 * NUM_FREQS
        assert emb.feature_proj.in_features == expected_in
        assert emb.feature_proj.out_features == embed_dim


# ------------------------------------------------------------------ #
# CWTEmbedding — pretokenize
# ------------------------------------------------------------------ #


class TestCWTEmbeddingPretokenize:
    def test_returns_expected_keys(self):
        emb = _make_cwt_embedding(num_channels=8, target_time_tokens=32)
        signal = np.random.randn(100, 5).astype(np.float32)
        tokens = np.arange(5)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        expected_keys = {
            "input_values",
            "input_channel_index",
            "input_mask",
            "input_sampling_rate",
            "input_seq_len",
            "input_timestamps",
        }
        assert set(result.keys()) == expected_keys

    def test_pads_channels(self):
        emb = _make_cwt_embedding(num_channels=8)
        signal = np.random.randn(100, 3).astype(np.float32)
        tokens = np.array([10, 20, 30])

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        assert result["input_values"].shape == (8, 100)
        assert result["input_channel_index"].shape == (8,)
        assert result["input_mask"].shape == (8,)
        assert result["input_mask"][:3].all()
        assert not result["input_mask"][3:].any()
        assert (result["input_values"][3:, :] == 0).all()

    def test_truncates_channels(self):
        emb = _make_cwt_embedding(num_channels=4)
        signal = np.random.randn(100, 8).astype(np.float32)
        tokens = np.arange(8)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        assert result["input_values"].shape == (4, 100)
        assert result["input_channel_index"].shape == (4,)
        assert result["input_mask"].all()

    def test_exact_channels(self):
        emb = _make_cwt_embedding(num_channels=6)
        signal = np.random.randn(80, 6).astype(np.float32)
        tokens = np.arange(6)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        assert result["input_values"].shape == (6, 80)
        assert result["input_mask"].all()
        np.testing.assert_array_equal(result["input_values"].numpy(), signal.T)

    def test_preserves_channel_tokens(self):
        emb = _make_cwt_embedding(num_channels=8)
        signal = np.random.randn(100, 5).astype(np.float32)
        tokens = np.array([100, 200, 300, 400, 500])

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        np.testing.assert_array_equal(
            result["input_channel_index"][:5].numpy(), tokens
        )
        assert (result["input_channel_index"][5:] == 0).all()

    def test_sampling_rate_stored(self):
        emb = _make_cwt_embedding()
        signal = np.random.randn(100, 4).astype(np.float32)
        tokens = np.arange(4)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=500.0, sequence_length=1.0
        )

        assert result["input_sampling_rate"].item() == pytest.approx(500.0)

    def test_seq_len_stored(self):
        emb = _make_cwt_embedding()
        signal = np.random.randn(123, 4).astype(np.float32)
        tokens = np.arange(4)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=1.0
        )

        assert result["input_seq_len"].item() == 123

    def test_timestamps_shape_and_range(self):
        emb = _make_cwt_embedding(target_time_tokens=32)
        signal = np.random.randn(100, 4).astype(np.float32)
        tokens = np.arange(4)

        result = emb.pretokenize(
            signal, tokens, sampling_rate=250.0, sequence_length=2.0
        )

        ts = result["input_timestamps"]
        assert ts.shape == (32,)
        assert ts[0].item() == pytest.approx(0.0)
        assert ts[-1].item() == pytest.approx(2.0)


# ------------------------------------------------------------------ #
# CWTEmbedding — forward
# ------------------------------------------------------------------ #


class TestCWTEmbeddingForward:
    def test_output_shape(self, embed_dim, batch_size):
        target_time_tokens = 32
        emb = _make_cwt_embedding(
            embed_dim=embed_dim, target_time_tokens=target_time_tokens
        )

        x = torch.randn(batch_size, 8, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, target_time_tokens, embed_dim)

    def test_raises_without_sampling_rate(self, batch_size):
        emb = _make_cwt_embedding()
        x = torch.randn(batch_size, 8, 200)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        with pytest.raises(ValueError, match="input_sampling_rate"):
            emb(x, input_seq_len=seq_lens)

    def test_raises_without_seq_len(self, batch_size):
        emb = _make_cwt_embedding()
        x = torch.randn(batch_size, 8, 200)
        fs = torch.full((batch_size,), 250.0)

        with pytest.raises(ValueError, match="input_seq_len"):
            emb(x, input_sampling_rate=fs)

    def test_forward_single_batch(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        x = torch.randn(1, 8, 100)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (1, 16, embed_dim)

    def test_forward_with_hidden_spatial(self, embed_dim, batch_size):
        emb = _make_cwt_embedding(
            embed_dim=embed_dim,
            shared_spatial_hidden_dim=16,
            target_time_tokens=16,
        )
        x = torch.randn(batch_size, 8, 100)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 100, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, 16, embed_dim)

    def test_forward_variable_seq_lens(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        B = 2
        x = torch.randn(B, 8, 200)
        fs = torch.tensor([250.0, 500.0])
        seq_lens = torch.tensor([100, 200])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (B, 16, embed_dim)

    def test_gradients_flow(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        x = torch.randn(1, 8, 100, requires_grad=True)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        out.sum().backward()
        assert x.grad is not None

        assert emb.cwt.freqs.grad is not None
        assert emb.feature_proj.weight.grad is not None

    def test_accepts_extra_kwargs(self, batch_size):
        emb = _make_cwt_embedding()
        x = torch.randn(batch_size, 8, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = emb(
            x,
            input_sampling_rate=fs,
            input_seq_len=seq_lens,
            input_channel_index=torch.zeros(batch_size, 8),
            input_mask=torch.ones(batch_size, 8, dtype=torch.bool),
        )
        assert out.shape == (batch_size, 32, 64)

    def test_device_placement_cpu(self, batch_size):
        emb = _make_cwt_embedding()
        emb = emb.to("cpu")
        x = torch.randn(batch_size, 8, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.device.type == "cpu"

    def test_device_placement_cuda(self, batch_size):
        if not torch.cuda.is_available():
            return
        emb = _make_cwt_embedding()
        emb = emb.to("cuda")
        x = torch.randn(batch_size, 8, 200, device="cuda")
        fs = torch.full((batch_size,), 250.0, device="cuda")
        seq_lens = torch.full(
            (batch_size,), 200, dtype=torch.long, device="cuda"
        )

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.device.type == "cuda"


# ------------------------------------------------------------------ #
# End-to-end: pretokenize -> stack -> forward
# ------------------------------------------------------------------ #


class TestCWTEmbeddingEndToEnd:
    def test_pretokenize_then_forward(self, embed_dim):
        emb = _make_cwt_embedding(
            embed_dim=embed_dim,
            num_channels=8,
            num_sources=4,
            target_time_tokens=16,
        )

        signal_a = np.random.randn(100, 5).astype(np.float32)
        signal_b = np.random.randn(120, 8).astype(np.float32)
        tokens_a = np.arange(5)
        tokens_b = np.arange(8)

        tok_a = emb.pretokenize(
            signal_a, tokens_a, sampling_rate=250.0, sequence_length=1.0
        )
        tok_b = emb.pretokenize(
            signal_b, tokens_b, sampling_rate=250.0, sequence_length=1.0
        )

        max_T = max(
            tok_a["input_values"].shape[1], tok_b["input_values"].shape[1]
        )

        def _pad_time(t, max_T):
            C, T = t.shape
            if T < max_T:
                return torch.nn.functional.pad(t, (0, max_T - T))
            return t

        batched_values = torch.stack(
            [
                _pad_time(tok_a["input_values"], max_T),
                _pad_time(tok_b["input_values"], max_T),
            ]
        )
        batched_fs = torch.stack(
            [tok_a["input_sampling_rate"], tok_b["input_sampling_rate"]]
        )
        batched_seq_len = torch.stack(
            [tok_a["input_seq_len"], tok_b["input_seq_len"]]
        )

        out = emb(
            batched_values,
            input_sampling_rate=batched_fs,
            input_seq_len=batched_seq_len,
        )
        assert out.shape == (2, 16, embed_dim)

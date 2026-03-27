import numpy as np
import pytest
import torch

from foundry.models.embeddings.channel_strategies import (
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
)


class TestFixedChannelStrategy:
    def test_prepare_pretokenize_pads(self):
        strategy = FixedChannelStrategy(num_channels=8)
        signal = np.random.randn(100, 3).astype(np.float32)
        tokens = np.array([10, 20, 30])

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (8, 100)
        assert result["input_channel_index"].shape == (8,)
        assert result["input_mask"].shape == (8,)
        assert result["input_mask"][:3].all()
        assert not result["input_mask"][3:].any()
        assert (result["input_values"][3:, :] == 0).all()

    def test_prepare_pretokenize_truncates(self):
        strategy = FixedChannelStrategy(num_channels=4)
        signal = np.random.randn(100, 8).astype(np.float32)
        tokens = np.arange(8)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (4, 100)
        assert result["input_channel_index"].shape == (4,)
        assert result["input_mask"].all()

    def test_prepare_pretokenize_exact(self):
        strategy = FixedChannelStrategy(num_channels=8)
        signal = np.random.randn(100, 8).astype(np.float32)
        tokens = np.arange(8)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (8, 100)
        assert result["input_mask"].all()
        np.testing.assert_array_equal(result["input_values"].numpy(), signal.T)

    def test_preserves_channel_tokens(self):
        strategy = FixedChannelStrategy(num_channels=8)
        signal = np.random.randn(100, 5).astype(np.float32)
        tokens = np.array([100, 200, 300, 400, 500])

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        np.testing.assert_array_equal(
            result["input_channel_index"][:5].numpy(), tokens
        )
        assert (result["input_channel_index"][5:] == 0).all()

    def test_stores_sampling_rate(self):
        strategy = FixedChannelStrategy(num_channels=8)
        signal = np.random.randn(100, 4).astype(np.float32)
        tokens = np.arange(4)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=500.0
        )
        assert result["input_sampling_rate"].item() == pytest.approx(500.0)

    def test_forward_passthrough(self, batch_size):
        strategy = FixedChannelStrategy(num_channels=8)
        x = torch.randn(batch_size, 8, 200)
        out = strategy(x)
        torch.testing.assert_close(out, x)


class TestPerChannelStrategy:
    def test_prepare_pretokenize_pads(self):
        strategy = PerChannelStrategy(max_channels=128)
        signal = np.random.randn(100, 20).astype(np.float32)
        tokens = np.arange(20)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (128, 100)
        assert result["input_mask"][:20].all()
        assert not result["input_mask"][20:].any()
        assert result["input_seq_len"].item() == 100

    def test_prepare_pretokenize_truncates(self):
        strategy = PerChannelStrategy(max_channels=16)
        signal = np.random.randn(100, 32).astype(np.float32)
        tokens = np.arange(32)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (16, 100)
        assert result["input_mask"].all()

    def test_forward_reshape(self, batch_size):
        strategy = PerChannelStrategy(max_channels=128)
        x = torch.randn(batch_size, 128, 200)
        out = strategy(x)
        assert out.shape == (batch_size * 128, 1, 200)

    def test_forward_preserves_data(self):
        strategy = PerChannelStrategy(max_channels=4)
        x = torch.randn(1, 4, 100)
        out = strategy(x)
        assert out.shape == (4, 1, 100)
        for c in range(4):
            torch.testing.assert_close(out[c, 0], x[0, c])


class TestSpatialProjectionStrategy:
    def test_prepare_pretokenize(self):
        strategy = SpatialProjectionStrategy(num_channels=64, num_sources=8)
        signal = np.random.randn(100, 30).astype(np.float32)
        tokens = np.arange(30)

        result = strategy.prepare_pretokenize(
            signal, tokens, sampling_rate=250.0
        )

        assert result["input_values"].shape == (64, 100)
        assert result["input_mask"][:30].all()
        assert not result["input_mask"][30:].any()
        assert result["input_seq_len"].item() == 100

    def test_forward_linear_projection(self, batch_size):
        strategy = SpatialProjectionStrategy(num_channels=64, num_sources=8)
        x = torch.randn(batch_size, 64, 200)
        out = strategy(x)
        assert out.shape == (batch_size, 8, 200)

    def test_forward_gradient_flow(self):
        strategy = SpatialProjectionStrategy(num_channels=16, num_sources=4)
        x = torch.randn(1, 16, 50, requires_grad=True)
        out = strategy(x)
        out.sum().backward()
        assert x.grad is not None

    def test_forward_with_session_projector(self):
        strategy = SpatialProjectionStrategy(
            num_channels=16,
            num_sources=4,
            session_configs={"sessA": 8, "sessB": 16},
        )
        B = 2
        x = torch.randn(B, 16, 50)
        out = strategy(
            x,
            input_session_ids=["sessA", "sessB"],
            input_channel_counts=[8, 16],
            input_seq_len=torch.tensor([50, 50]),
        )
        assert out.shape == (B, 4, 50)

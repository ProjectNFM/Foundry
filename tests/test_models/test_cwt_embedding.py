import math

import torch

from foundry.models import CWTEmbedding, ContinuousCWTLayer


INIT_FREQS = torch.logspace(math.log10(2), math.log10(50), 8).tolist()
NUM_FREQS = len(INIT_FREQS)


def _make_cwt_embedding(
    embed_dim=64,
    num_sources=4,
    target_time_tokens=32,
    **kwargs,
):
    return CWTEmbedding(
        embed_dim=embed_dim,
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
        assert layer.freqs_unconstrained.grad is not None
        assert layer.n_cycles_unconstrained.grad is not None

    def test_learnable_parameters(self):
        layer = ContinuousCWTLayer(
            init_freqs=INIT_FREQS, target_time_tokens=16, n_cycles=5.0
        )
        assert layer.freqs.requires_grad
        assert layer.n_cycles.requires_grad
        assert (layer.n_cycles == 5.0).all()

    def test_bfloat16_input(self):
        layer = ContinuousCWTLayer(init_freqs=INIT_FREQS, target_time_tokens=16)
        layer = layer.to(torch.bfloat16)
        x = torch.randn(1, 2, 100, dtype=torch.bfloat16)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = layer(x, fs, seq_lens)
        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 2, 2, NUM_FREQS, 16)
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------ #
# CWTEmbedding — initialisation
# ------------------------------------------------------------------ #


class TestCWTEmbeddingInit:
    def test_is_nn_module(self):
        emb = _make_cwt_embedding()
        assert isinstance(emb, torch.nn.Module)

    def test_attributes(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim)
        assert emb.embed_dim == embed_dim
        assert emb.num_sources == 4
        assert emb.target_time_tokens == 32

    def test_feature_proj_dimensions(self, embed_dim):
        num_sources = 4
        emb = _make_cwt_embedding(embed_dim=embed_dim, num_sources=num_sources)
        expected_in = num_sources * 2 * NUM_FREQS
        assert emb.feature_proj.in_features == expected_in
        assert emb.feature_proj.out_features == embed_dim


# ------------------------------------------------------------------ #
# CWTEmbedding — forward
# ------------------------------------------------------------------ #


class TestCWTEmbeddingForward:
    def test_output_shape(self, embed_dim, batch_size):
        target_time_tokens = 32
        emb = _make_cwt_embedding(
            embed_dim=embed_dim, target_time_tokens=target_time_tokens
        )

        x = torch.randn(batch_size, 4, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (batch_size, target_time_tokens, embed_dim)

    def test_forward_single_batch(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        x = torch.randn(1, 4, 100)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (1, 16, embed_dim)

    def test_forward_variable_seq_lens(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        B = 2
        x = torch.randn(B, 4, 200)
        fs = torch.tensor([250.0, 500.0])
        seq_lens = torch.tensor([100, 200])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.shape == (B, 16, embed_dim)

    def test_gradients_flow(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        x = torch.randn(1, 4, 100, requires_grad=True)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        out.sum().backward()
        assert x.grad is not None

        assert emb.cwt.freqs_unconstrained.grad is not None
        assert emb.feature_proj.weight.grad is not None

    def test_device_placement_cpu(self, batch_size):
        emb = _make_cwt_embedding()
        emb = emb.to("cpu")
        x = torch.randn(batch_size, 4, 200)
        fs = torch.full((batch_size,), 250.0)
        seq_lens = torch.full((batch_size,), 200, dtype=torch.long)

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.device.type == "cpu"

    def test_bfloat16_input(self, embed_dim):
        emb = _make_cwt_embedding(embed_dim=embed_dim, target_time_tokens=16)
        emb = emb.to(torch.bfloat16)
        x = torch.randn(1, 4, 100, dtype=torch.bfloat16)
        fs = torch.tensor([250.0])
        seq_lens = torch.tensor([100])

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 16, embed_dim)
        assert torch.isfinite(out).all()

    def test_device_placement_cuda(self, batch_size):
        if not torch.cuda.is_available():
            return
        emb = _make_cwt_embedding()
        emb = emb.to("cuda")
        x = torch.randn(batch_size, 4, 200, device="cuda")
        fs = torch.full((batch_size,), 250.0, device="cuda")
        seq_lens = torch.full(
            (batch_size,), 200, dtype=torch.long, device="cuda"
        )

        out = emb(x, input_sampling_rate=fs, input_seq_len=seq_lens)
        assert out.device.type == "cuda"

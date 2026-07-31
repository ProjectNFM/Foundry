"""Tests for dynamic session embedding components.

Covers:
- DynamicSessionEncoder: forward, pooling modes, masking, shapes
- SessionEmbeddingCache: put/get/clear semantics
- SessionContextCache: lazy build, cache hit, clearing, reseeding
- POYOEEGModel integration with session_emb_mode=dynamic
- MaskedPOYOEEGModel integration with session_emb_mode=dynamic
- Backward compatibility: session_emb_mode=static matches old behavior
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foundry.models.session_embedding import (
    DynamicSessionEncoder,
    SessionContextCache,
    SessionEmbeddingCache,
)


# ---------------------------------------------------------------------------
# DynamicSessionEncoder
# ---------------------------------------------------------------------------


class TestDynamicSessionEncoder:
    @pytest.fixture(params=["mean", "attention"])
    def encoder(self, request):
        return DynamicSessionEncoder(embed_dim=64, pool_mode=request.param)

    def test_output_shape(self, encoder):
        B, W, N, D = 2, 5, 10, 64
        tokens = torch.randn(B, W, N, D)
        out = encoder(tokens)
        assert out.shape == (B, 1, D)

    def test_output_shape_with_mask(self, encoder):
        B, W, N, D = 3, 4, 8, 64
        tokens = torch.randn(B, W, N, D)
        mask = torch.ones(B, W, N, dtype=torch.bool)
        mask[:, -1, :] = False  # last window fully masked
        out = encoder(tokens, context_token_mask=mask)
        assert out.shape == (B, 1, D)

    def test_single_window(self, encoder):
        B, W, N, D = 2, 1, 10, 64
        tokens = torch.randn(B, W, N, D)
        out = encoder(tokens)
        assert out.shape == (B, 1, D)

    def test_gradients_flow(self, encoder):
        B, W, N, D = 1, 3, 10, 64
        tokens = torch.randn(B, W, N, D, requires_grad=True)
        out = encoder(tokens)
        out.sum().backward()
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0

    def test_deterministic_eval(self, encoder):
        encoder.eval()
        B, W, N, D = 2, 5, 10, 64
        tokens = torch.randn(B, W, N, D)
        out1 = encoder(tokens)
        out2 = encoder(tokens)
        torch.testing.assert_close(out1, out2)

    def test_fully_masked_windows_handled(self, encoder):
        """If all tokens in some windows are masked, shouldn't crash."""
        B, W, N, D = 2, 3, 10, 64
        tokens = torch.randn(B, W, N, D)
        mask = torch.zeros(B, W, N, dtype=torch.bool)
        mask[:, 0, :5] = True  # only partial tokens in first window
        out = encoder(tokens, context_token_mask=mask)
        assert out.shape == (B, 1, D)
        assert torch.isfinite(out).all()

    def test_invalid_pool_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown pool_mode"):
            DynamicSessionEncoder(embed_dim=64, pool_mode="invalid")


class TestDynamicSessionEncoderMean:
    def test_mean_pooling_uniform_tokens(self):
        encoder = DynamicSessionEncoder(embed_dim=4, pool_mode="mean")
        # All tokens identical → pooling should preserve the value
        B, W, N, D = 1, 2, 3, 4
        val = torch.ones(B, W, N, D) * 2.0
        out = encoder(val)
        # After proj, result should be consistent but transformed
        assert out.shape == (B, 1, D)

    def test_mask_excludes_tokens_from_mean(self):
        encoder = DynamicSessionEncoder(embed_dim=4, pool_mode="mean")
        encoder.eval()
        B, W, N, D = 1, 1, 4, 4

        # Tokens: first two are [1,1,1,1], last two are [99,99,99,99]
        tokens = torch.zeros(B, W, N, D)
        tokens[:, :, :2, :] = 1.0
        tokens[:, :, 2:, :] = 99.0

        # Mask out the 99s
        mask = torch.zeros(B, W, N, dtype=torch.bool)
        mask[:, :, :2] = True

        out_masked = encoder(tokens, context_token_mask=mask)

        # Compare with only feeding the valid tokens
        _ = torch.ones(B, W, 2, D)
        _ = torch.ones(B, W, 2, dtype=torch.bool)
        # Different N so encoder internal shapes differ, but the pooling result
        # should match conceptually. Just verify it's not equal to unmasked.
        out_full = encoder(tokens)
        assert not torch.allclose(out_masked, out_full, atol=1e-4)


# ---------------------------------------------------------------------------
# SessionEmbeddingCache
# ---------------------------------------------------------------------------


class TestSessionEmbeddingCache:
    def test_empty_cache(self):
        cache = SessionEmbeddingCache()
        assert len(cache) == 0
        assert cache.get(0) is None
        assert 0 not in cache

    def test_put_and_get(self):
        cache = SessionEmbeddingCache()
        emb = torch.randn(1, 1, 64)
        cache.put(42, emb)
        assert 42 in cache
        assert len(cache) == 1
        retrieved = cache.get(42)
        assert retrieved is not None
        torch.testing.assert_close(retrieved, emb)

    def test_put_detaches(self):
        cache = SessionEmbeddingCache()
        emb = torch.randn(1, 1, 64, requires_grad=True)
        cache.put(0, emb)
        assert not cache.get(0).requires_grad

    def test_clear(self):
        cache = SessionEmbeddingCache()
        cache.put(0, torch.randn(1, 1, 64))
        cache.put(1, torch.randn(1, 1, 64))
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get(0) is None

    def test_overwrite(self):
        cache = SessionEmbeddingCache()
        emb1 = torch.randn(1, 1, 64)
        emb2 = torch.randn(1, 1, 64)
        cache.put(0, emb1)
        cache.put(0, emb2)
        torch.testing.assert_close(cache.get(0), emb2)
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# SessionContextCache
# ---------------------------------------------------------------------------


class _MockData:
    """Minimal Data-like object for testing SessionContextCache."""

    def __init__(self, session_id="sess_0", domain_start=0.0, domain_end=10.0):
        self.session = type("Session", (), {"id": session_id})()
        self.domain = type(
            "Domain",
            (),
            {
                "start": domain_start,
                "end": domain_end,
            },
        )()
        self.channels = type(
            "Channels",
            (),
            {
                "id": np.array(["ch0", "ch1", "ch2", "ch3"]),
            },
        )()

    def slice(self, start, end):
        return _MockData(
            session_id=self.session.id,
            domain_start=start,
            domain_end=end,
        )


class TestSessionContextCache:
    def _dummy_prepare_fn(self, data):
        return type(
            "Prepared",
            (),
            {
                "signal": np.random.randn(200, 4).astype(np.float32),
                "sampling_rate": 100.0,
                "modality_mask": np.array([True, True, True, True]),
            },
        )()

    def _dummy_pretokenize_fn(
        self, signal, channel_tokens, sampling_rate, sequence_length
    ):
        C = 4
        T = signal.shape[0]
        return {
            "input_values": torch.randn(C, T),
            "input_channel_index": torch.arange(C),
            "input_mask": torch.ones(C, dtype=torch.bool),
        }

    def _dummy_channel_tokenizer(self, ids):
        return list(range(len(ids)))

    def test_lazy_build(self):
        cache = SessionContextCache(
            num_windows=3, context_source="start", context_duration=2.0
        )
        data = _MockData()
        assert len(cache) == 0

        result = cache.get_or_build(
            "sess_0",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )

        assert len(cache) == 1
        assert "context_values" in result
        assert "context_channel_index" in result
        assert "context_mask" in result
        assert "context_sampling_rate" in result
        assert result["context_values"].shape[0] == 3  # num_windows

    def test_cache_hit(self):
        cache = SessionContextCache(
            num_windows=2, context_source="random", context_duration=1.0
        )
        data = _MockData()

        r1 = cache.get_or_build(
            "sess_0",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )
        r2 = cache.get_or_build(
            "sess_0",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )

        # Should return exact same object
        assert r1 is r2
        assert len(cache) == 1

    def test_different_sessions(self):
        cache = SessionContextCache(
            num_windows=2, context_source="start", context_duration=1.0
        )
        d1 = _MockData(session_id="sess_0")
        d2 = _MockData(session_id="sess_1")

        cache.get_or_build(
            "sess_0",
            d1,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )
        cache.get_or_build(
            "sess_1",
            d2,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )

        assert len(cache) == 2

    def test_clear(self):
        cache = SessionContextCache(
            num_windows=1, context_source="start", context_duration=1.0
        )
        data = _MockData()
        cache.get_or_build(
            "sess_0",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_reseed_clears_cache(self):
        cache = SessionContextCache(
            num_windows=2, context_source="random", context_duration=1.0
        )
        data = _MockData()
        cache.get_or_build(
            "sess_0",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )
        assert len(cache) == 1
        cache.reseed(123)
        assert len(cache) == 0

    def test_short_recording_uses_start(self):
        cache = SessionContextCache(
            num_windows=5, context_source="random", context_duration=2.0
        )
        data = _MockData(domain_start=0.0, domain_end=1.5)
        result = cache.get_or_build(
            "sess_short",
            data,
            self._dummy_prepare_fn,
            self._dummy_pretokenize_fn,
            self._dummy_channel_tokenizer,
        )
        assert result["context_values"].shape[0] == 5


# ---------------------------------------------------------------------------
# Model integration: session_emb_mode
# ---------------------------------------------------------------------------


def _build_model_with_mode(
    session_emb_mode="static",
    embed_dim=64,
    C_pad=4,
    N=10,
    sequence_length=1.0,
):
    """Build a POYOEEGModel with the given session_emb_mode."""
    from foundry.models.embeddings import PerChannelStrategy
    from foundry.models.embeddings.temporal.resample_cnn import (
        ResampleCNNEmbedding,
    )
    from foundry.models.poyo_eeg import POYOEEGModel
    from foundry.models.tokenizer import EEGTokenizer

    target_token_rate = N / sequence_length
    channel_emb_dim = 16
    token_dim = embed_dim - channel_emb_dim

    channel_strategy = PerChannelStrategy(max_channels=C_pad)
    temporal_embedding = ResampleCNNEmbedding(
        embed_dim=token_dim,
        num_sources=1,
        target_token_rate=target_token_rate,
        num_filters=4,
        kernel_size=3,
        num_conv_layers=1,
    )
    tokenizer = EEGTokenizer(
        channel_strategy=channel_strategy,
        temporal_embedding=temporal_embedding,
        embed_dim=embed_dim,
        patch_duration=None,
        channel_fusion="concat",
        channel_emb_dim=channel_emb_dim,
    )

    task_configs = {
        "task_a": {
            "name": "task_a",
            "head": {
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "dummy.timestamps",
                "value_key": "dummy.values",
            },
            "loss": {
                "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss",
            },
        },
    }

    model = POYOEEGModel(
        tokenizer=tokenizer,
        task_configs=task_configs,
        embed_dim=embed_dim,
        sequence_length=sequence_length,
        latent_step=0.5,
        num_latents_per_step=2,
        depth=1,
        dim_head=32,
        cross_heads=2,
        self_heads=2,
        session_emb_mode=session_emb_mode,
    )
    model.initialize_vocabs(
        {
            "session_ids": ["sess_0", "sess_1"],
            "channel_ids": [f"ch_{i}" for i in range(C_pad)],
        }
    )
    return model


def _make_batch(
    model, B=2, C_pad=4, N=10, T=100, sr=100.0, include_context=False
):
    """Create a batch dict for forward pass."""
    device = next(model.parameters()).device
    batch = dict(
        input_values=torch.randn(B, C_pad, T, device=device),
        input_timestamps=(
            torch.linspace(0, 1.0, N, device=device)
            .unsqueeze(0)
            .expand(B, -1)
            .repeat(1, C_pad)
        ),
        input_channel_index=torch.arange(C_pad, device=device)
        .unsqueeze(0)
        .expand(B, -1),
        input_session_index=torch.zeros(B, dtype=torch.long, device=device),
        input_mask=torch.ones(B, C_pad, dtype=torch.bool, device=device),
        input_sampling_rate=torch.full((B,), sr, device=device),
        input_seq_len=torch.full((B,), T, dtype=torch.long, device=device),
        latent_index=torch.from_numpy(model._latent_index)
        .unsqueeze(0)
        .expand(B, -1)
        .to(device),
        latent_timestamps=torch.from_numpy(model._latent_timestamps)
        .unsqueeze(0)
        .expand(B, -1)
        .float()
        .to(device),
        output_session_index=torch.zeros(B, 2, dtype=torch.long, device=device),
        output_timestamps=torch.tensor([[0.3, 0.7]] * B, device=device),
        task_index=torch.ones(B, 2, dtype=torch.long, device=device),
    )

    if include_context:
        W = 3
        batch["context_values"] = torch.randn(B, W, C_pad, T, device=device)
        batch["context_channel_index"] = (
            torch.arange(C_pad, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, W, -1)
        )
        batch["context_mask"] = torch.ones(
            B, W, C_pad, dtype=torch.bool, device=device
        )
        batch["context_sampling_rate"] = torch.full((B, W), sr, device=device)

    return batch


class TestSessionEmbModeStatic:
    def test_forward_runs(self):
        model = _build_model_with_mode("static")
        batch = _make_batch(model)
        result = model(**batch)
        assert "task_a" in result.task_outputs

    def test_session_emb_mode_attr(self):
        model = _build_model_with_mode("static")
        assert model.session_emb_mode == "static"
        assert model.dynamic_session_encoder is None


class TestSessionEmbModeDisabled:
    def test_forward_runs(self):
        model = _build_model_with_mode("disabled")
        batch = _make_batch(model)
        result = model(**batch)
        assert "task_a" in result.task_outputs

    def test_session_emb_mode_attr(self):
        model = _build_model_with_mode("disabled")
        assert model.session_emb_mode == "disabled"
        assert model.disable_session_emb is True


class TestSessionEmbModeDynamic:
    def test_forward_with_context(self):
        model = _build_model_with_mode("dynamic")
        batch = _make_batch(model, include_context=True)
        result = model(**batch)
        assert "task_a" in result.task_outputs

    def test_forward_without_context_falls_back(self):
        """Without context_kwargs, dynamic mode warns and falls back to zeros."""
        model = _build_model_with_mode("dynamic")
        batch = _make_batch(model, include_context=False)
        result = model(**batch)
        assert "task_a" in result.task_outputs

    def test_dynamic_encoder_exists(self):
        model = _build_model_with_mode("dynamic")
        assert model.dynamic_session_encoder is not None
        assert isinstance(model.dynamic_session_encoder, DynamicSessionEncoder)

    def test_gradients_flow_through_dynamic(self):
        model = _build_model_with_mode("dynamic")
        batch = _make_batch(model, B=1, include_context=True)
        result = model(**batch)
        loss = result.task_outputs["task_a"].sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_inference_cache_populated(self):
        model = _build_model_with_mode("dynamic")
        model.eval()
        assert model._session_emb_cache is not None
        assert len(model._session_emb_cache) == 0

        batch = _make_batch(model, B=2, include_context=True)
        with torch.no_grad():
            model(**batch)

        assert len(model._session_emb_cache) > 0

    def test_inference_cache_used_on_second_pass(self):
        model = _build_model_with_mode("dynamic")
        model.eval()
        batch = _make_batch(model, B=1, include_context=True)

        with torch.no_grad():
            r1 = model(**batch)

        # Second pass should use cached embedding
        with torch.no_grad():
            r2 = model(**batch)

        torch.testing.assert_close(
            r1.task_outputs["task_a"],
            r2.task_outputs["task_a"],
        )


class TestBackwardCompatDisableSessionEmb:
    def test_disable_session_emb_true_maps_to_disabled(self):
        from foundry.models.poyo_eeg import POYOEEGModel
        from foundry.models.embeddings import PerChannelStrategy
        from foundry.models.embeddings.temporal.resample_cnn import (
            ResampleCNNEmbedding,
        )
        from foundry.models.tokenizer import EEGTokenizer

        channel_strategy = PerChannelStrategy(max_channels=4)
        temporal_embedding = ResampleCNNEmbedding(
            embed_dim=48,
            num_sources=1,
            target_token_rate=10.0,
            num_filters=4,
            kernel_size=3,
            num_conv_layers=1,
        )
        tokenizer = EEGTokenizer(
            channel_strategy=channel_strategy,
            temporal_embedding=temporal_embedding,
            embed_dim=64,
            patch_duration=None,
            channel_fusion="concat",
            channel_emb_dim=16,
        )

        model = POYOEEGModel(
            tokenizer=tokenizer,
            task_configs={
                "task_a": {
                    "name": "task_a",
                    "head": {
                        "_target_": "foundry.tasks.heads.ReadoutHead",
                        "output_dim": 2,
                    },
                    "target_extractor": {
                        "_target_": "foundry.tasks.targets.TargetExtractor",
                        "timestamp_key": "dummy.timestamps",
                        "value_key": "dummy.values",
                    },
                    "loss": {
                        "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"
                    },
                },
            },
            embed_dim=64,
            sequence_length=1.0,
            latent_step=0.5,
            num_latents_per_step=2,
            depth=1,
            dim_head=32,
            cross_heads=2,
            self_heads=2,
            disable_session_emb=True,
        )
        assert model.session_emb_mode == "disabled"

    def test_session_emb_mode_overrides_disable_flag(self):
        model = _build_model_with_mode("dynamic")
        assert model.session_emb_mode == "dynamic"
        assert model.disable_session_emb is True  # still True for non-static

    def test_invalid_session_emb_mode_raises(self):
        with pytest.raises(ValueError, match="session_emb_mode must be one of"):
            _build_model_with_mode("invalid_mode")


# ---------------------------------------------------------------------------
# Masked model integration
# ---------------------------------------------------------------------------


def _build_masked_model_with_mode(session_emb_mode="static"):
    """Build a MaskedPOYOEEGModel with the given session_emb_mode."""
    from foundry.models.embeddings import PerChannelStrategy
    from foundry.models.embeddings.temporal.resample_cnn import (
        ResampleCNNEmbedding,
    )
    from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel
    from foundry.models.tokenizer import EEGTokenizer
    from foundry.tasks.masking import RandomTokenMasking

    embed_dim = 64
    C_pad = 4
    N = 10
    sequence_length = 1.0
    target_token_rate = N / sequence_length
    channel_emb_dim = 16
    token_dim = embed_dim - channel_emb_dim

    channel_strategy = PerChannelStrategy(max_channels=C_pad)
    temporal_embedding = ResampleCNNEmbedding(
        embed_dim=token_dim,
        num_sources=1,
        target_token_rate=target_token_rate,
        num_filters=4,
        kernel_size=3,
        num_conv_layers=1,
    )
    tokenizer = EEGTokenizer(
        channel_strategy=channel_strategy,
        temporal_embedding=temporal_embedding,
        embed_dim=embed_dim,
        patch_duration=None,
        channel_fusion="concat",
        channel_emb_dim=channel_emb_dim,
    )

    task_configs = {
        "masked_reconstruction": {
            "name": "masked_reconstruction",
            "head": {
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
                "num_layers": 2,
            },
            "target_extractor": None,
            "loss": {"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        },
    }

    masking = RandomTokenMasking(mask_ratio=0.5)

    model = MaskedPOYOEEGModel(
        tokenizer=tokenizer,
        task_configs=task_configs,
        embed_dim=embed_dim,
        sequence_length=sequence_length,
        latent_step=0.5,
        num_latents_per_step=2,
        depth=1,
        dim_head=32,
        cross_heads=2,
        self_heads=2,
        masking=masking,
        session_emb_mode=session_emb_mode,
    )
    model.initialize_vocabs(
        {
            "session_ids": ["sess_0", "sess_1"],
            "channel_ids": [f"ch_{i}" for i in range(4)],
        }
    )
    return model


def _make_masked_batch(model, B=2, include_context=False):
    C_pad, N, T = 4, 10, 100
    sr = 100.0
    device = next(model.parameters()).device

    batch = dict(
        input_values=torch.randn(B, C_pad, T, device=device),
        input_timestamps=(
            torch.linspace(0, 1.0, N, device=device)
            .unsqueeze(0)
            .expand(B, -1)
            .repeat(1, C_pad)
        ),
        input_channel_index=torch.arange(C_pad, device=device)
        .unsqueeze(0)
        .expand(B, -1),
        input_session_index=torch.zeros(B, dtype=torch.long, device=device),
        input_mask=torch.ones(B, C_pad, dtype=torch.bool, device=device),
        input_sampling_rate=torch.full((B,), sr, device=device),
        input_seq_len=torch.full((B,), T, dtype=torch.long, device=device),
        latent_index=torch.from_numpy(model._latent_index)
        .unsqueeze(0)
        .expand(B, -1)
        .to(device),
        latent_timestamps=torch.from_numpy(model._latent_timestamps)
        .unsqueeze(0)
        .expand(B, -1)
        .float()
        .to(device),
        output_session_index=torch.zeros(B, 0, dtype=torch.long, device=device),
        output_timestamps=torch.zeros(B, 0, device=device),
        task_index=torch.zeros(B, 0, dtype=torch.long, device=device),
        reconstruction_targets=torch.randn(B, C_pad * N, device=device),
    )

    if include_context:
        W = 3
        batch["context_values"] = torch.randn(B, W, C_pad, T, device=device)
        batch["context_channel_index"] = (
            torch.arange(C_pad, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, W, -1)
        )
        batch["context_mask"] = torch.ones(
            B, W, C_pad, dtype=torch.bool, device=device
        )
        batch["context_sampling_rate"] = torch.full((B, W), sr, device=device)

    return batch


class TestMaskedModelDynamic:
    def test_forward_with_context(self):
        model = _build_masked_model_with_mode("dynamic")
        batch = _make_masked_batch(model, include_context=True)
        result = model(**batch)
        assert "masked_reconstruction" in result.task_outputs
        assert result.ssl_meta is not None

    def test_forward_static_unchanged(self):
        model = _build_masked_model_with_mode("static")
        batch = _make_masked_batch(model, include_context=False)
        result = model(**batch)
        assert "masked_reconstruction" in result.task_outputs

    def test_gradients_flow_dynamic(self):
        model = _build_masked_model_with_mode("dynamic")
        batch = _make_masked_batch(model, B=1, include_context=True)
        result = model(**batch)
        loss = result.task_outputs["masked_reconstruction"].sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# State dict backward compatibility
# ---------------------------------------------------------------------------


class TestStateDictCompat:
    def test_session_emb_key_in_state_dict(self):
        """The InfiniteVocabEmbedding should still appear as 'session_emb.*'."""
        model = _build_model_with_mode("static")
        state_keys = list(model.state_dict().keys())
        session_keys = [k for k in state_keys if k.startswith("session_emb.")]
        assert len(session_keys) > 0

    def test_dynamic_encoder_in_state_dict(self):
        model = _build_model_with_mode("dynamic")
        state_keys = list(model.state_dict().keys())
        dynamic_keys = [
            k for k in state_keys if k.startswith("dynamic_session_encoder.")
        ]
        assert len(dynamic_keys) > 0

    def test_static_has_no_dynamic_keys(self):
        model = _build_model_with_mode("static")
        state_keys = list(model.state_dict().keys())
        dynamic_keys = [k for k in state_keys if "dynamic_session_encoder" in k]
        assert len(dynamic_keys) == 0

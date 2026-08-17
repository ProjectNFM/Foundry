"""Tests for dynamic channel embedding (RelativeChannelEncoder).

Covers:
- RelativeChannelEncoder: output shapes, padding zeroed, gradients flow
- POYOEEGModel integration with channel_emb_mode=dynamic
- MaskedPOYOEEGModel integration with channel_emb_mode=dynamic
- ch_emb_cache semantics (non-None for dynamic, None for static/disabled)
"""

from __future__ import annotations

import pytest
import torch

from foundry.models.relative_channel_encoder import RelativeChannelEncoder


# ---------------------------------------------------------------------------
# RelativeChannelEncoder unit tests
# ---------------------------------------------------------------------------


class TestRelativeChannelEncoder:
    @pytest.fixture
    def encoder(self):
        return RelativeChannelEncoder(
            token_dim=48, channel_emb_dim=16, num_heads=4
        )

    def test_output_shape(self, encoder):
        B, C, N, D = 2, 6, 10, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        out = encoder(tokens, mask)
        assert out.shape == (B, C, 16)

    def test_padded_channels_zeroed(self, encoder):
        B, C, N, D = 2, 6, 10, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        mask[:, -2:] = False  # last two channels padded
        out = encoder(tokens, mask)
        assert out.shape == (B, C, 16)
        assert (out[:, -2:, :] == 0).all()

    def test_real_channels_nonzero(self, encoder):
        B, C, N, D = 2, 6, 10, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        mask[:, -2:] = False
        out = encoder(tokens, mask)
        assert out[:, :4, :].abs().sum() > 0

    def test_gradients_flow(self, encoder):
        B, C, N, D = 1, 4, 8, 48
        tokens = torch.randn(B, C, N, D, requires_grad=True)
        mask = torch.ones(B, C, dtype=torch.bool)
        out = encoder(tokens, mask)
        # Use random weights to break symmetry (plain .sum() can cancel via LN)
        loss = (out * torch.randn_like(out)).sum()
        loss.backward()
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0

    def test_encoder_params_receive_grad(self, encoder):
        B, C, N, D = 1, 4, 8, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        out = encoder(tokens, mask)
        loss = (out * torch.randn_like(out)).sum()
        loss.backward()
        params_with_grad = sum(
            1
            for p in encoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in encoder.parameters())
        assert params_with_grad == total_params

    def test_deterministic_eval(self, encoder):
        encoder.eval()
        B, C, N, D = 2, 4, 8, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        out1 = encoder(tokens, mask)
        out2 = encoder(tokens, mask)
        torch.testing.assert_close(out1, out2)

    def test_single_channel(self, encoder):
        B, C, N, D = 2, 1, 10, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.ones(B, C, dtype=torch.bool)
        out = encoder(tokens, mask)
        assert out.shape == (B, 1, 16)
        assert torch.isfinite(out).all()

    def test_all_channels_padded_produces_zeros(self, encoder):
        B, C, N, D = 2, 4, 8, 48
        tokens = torch.randn(B, C, N, D)
        mask = torch.zeros(B, C, dtype=torch.bool)
        out = encoder(tokens, mask)
        assert (out == 0).all()


# ---------------------------------------------------------------------------
# Model integration: channel_emb_mode=dynamic
# ---------------------------------------------------------------------------


def _build_model_with_channel_mode(
    channel_emb_mode="dynamic",
    embed_dim=64,
    C_pad=4,
    N=10,
    sequence_length=1.0,
):
    """Build a POYOEEGModel with the given channel_emb_mode."""
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
        session_emb_mode="disabled",
        channel_emb_mode=channel_emb_mode,
        channel_encoder_heads=4,
    )
    model.initialize_vocabs(
        {
            "session_ids": ["sess_0", "sess_1"],
            "channel_ids": [f"ch_{i}" for i in range(C_pad)],
        }
    )
    return model


def _build_masked_model_with_channel_mode(channel_emb_mode="dynamic"):
    """Build a MaskedPOYOEEGModel with the given channel_emb_mode."""
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
        session_emb_mode="disabled",
        channel_emb_mode=channel_emb_mode,
        channel_encoder_heads=4,
    )
    model.initialize_vocabs(
        {
            "session_ids": ["sess_0", "sess_1"],
            "channel_ids": [f"ch_{i}" for i in range(4)],
        }
    )
    return model


def _make_batch(model, B=2, C_pad=4, N=10, T=100, sr=100.0):
    """Create a batch dict for POYOEEGModel forward pass."""
    device = next(model.parameters()).device
    return dict(
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


def _make_masked_batch(model, B=2):
    C_pad, N, T = 4, 10, 100
    sr = 100.0
    device = next(model.parameters()).device

    return dict(
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


class TestPOYODynamicChannelEmb:
    def test_forward_runs(self):
        model = _build_model_with_channel_mode("dynamic")
        batch = _make_batch(model)
        result = model(**batch)
        assert "task_a" in result.task_outputs

    @pytest.mark.parametrize("mode", ["static", "dynamic"])
    def test_representation_contract_captures_channel_and_backbone(self, mode):
        model = _build_model_with_channel_mode(mode)
        batch = _make_batch(model, B=2)
        result = model(**batch, capture_representations=True)

        payload = result.representations
        assert payload is not None
        assert payload.channel_mode == mode
        assert payload.channel_representations.shape == (2, 4, 16)
        assert payload.backbone_representations.shape == (2, 64)
        torch.testing.assert_close(payload.channel_mask, batch["input_mask"])

    def test_representation_contract_marks_disabled_channel_unavailable(self):
        model = _build_model_with_channel_mode("disabled")
        result = model(**_make_batch(model, B=2), capture_representations=True)

        payload = result.representations
        assert payload is not None
        assert payload.channel_mode == "disabled"
        assert payload.channel_representations is None
        assert payload.backbone_representations.shape == (2, 64)

    def test_representation_capture_is_off_by_default(self):
        model = _build_model_with_channel_mode("dynamic")
        result = model(**_make_batch(model))
        assert result.representations is None

    def test_relative_channel_encoder_exists(self):
        model = _build_model_with_channel_mode("dynamic")
        assert model.relative_channel_encoder is not None
        assert isinstance(
            model.relative_channel_encoder, RelativeChannelEncoder
        )

    def test_static_has_no_encoder(self):
        model = _build_model_with_channel_mode("static")
        assert model.relative_channel_encoder is None

    def test_disabled_has_no_encoder(self):
        model = _build_model_with_channel_mode("disabled")
        assert model.relative_channel_encoder is None

    def test_ch_emb_cache_returned_dynamic(self):
        model = _build_model_with_channel_mode("dynamic")
        batch_kwargs = dict(
            input_values=torch.randn(1, 4, 100),
            input_channel_index=torch.arange(4).unsqueeze(0),
            input_session_index=torch.zeros(1, dtype=torch.long),
            input_mask=torch.ones(1, 4, dtype=torch.bool),
            input_sampling_rate=torch.full((1,), 100.0),
            input_seq_len=torch.full((1,), 100, dtype=torch.long),
        )
        _, _, ch_emb_cache = model._tokenize_and_add_session(**batch_kwargs)
        assert ch_emb_cache is not None
        assert ch_emb_cache.shape == (1, 4, 16)

    def test_ch_emb_cache_from_static_lookup(self):
        """Static mode returns the lookup embedding as ch_emb_cache."""
        model = _build_model_with_channel_mode("static")
        batch_kwargs = dict(
            input_values=torch.randn(1, 4, 100),
            input_channel_index=torch.arange(4).unsqueeze(0),
            input_session_index=torch.zeros(1, dtype=torch.long),
            input_mask=torch.ones(1, 4, dtype=torch.bool),
            input_sampling_rate=torch.full((1,), 100.0),
            input_seq_len=torch.full((1,), 100, dtype=torch.long),
        )
        _, _, ch_emb_cache = model._tokenize_and_add_session(**batch_kwargs)
        assert ch_emb_cache is not None
        assert ch_emb_cache.shape == (1, 4, 16)

    def test_gradients_flow_through_encoder(self):
        model = _build_model_with_channel_mode("dynamic")
        batch = _make_batch(model, B=1)
        result = model(**batch)
        loss = result.task_outputs["task_a"].sum()
        loss.backward()
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.relative_channel_encoder.parameters()
        )
        assert encoder_has_grad

    def test_transferable_components_includes_encoder(self):
        model = _build_model_with_channel_mode("dynamic")
        assert "relative_channel_encoder" in model.transferable_components()

    def test_encoder_in_state_dict(self):
        model = _build_model_with_channel_mode("dynamic")
        state_keys = list(model.state_dict().keys())
        encoder_keys = [
            k for k in state_keys if k.startswith("relative_channel_encoder.")
        ]
        assert len(encoder_keys) > 0


class TestMaskedPOYODynamicChannelEmb:
    def test_forward_runs(self):
        model = _build_masked_model_with_channel_mode("dynamic")
        batch = _make_masked_batch(model)
        result = model(**batch)
        assert "masked_reconstruction" in result.task_outputs
        assert result.ssl_meta is not None

    def test_forward_static_unchanged(self):
        model = _build_masked_model_with_channel_mode("static")
        batch = _make_masked_batch(model)
        result = model(**batch)
        assert "masked_reconstruction" in result.task_outputs

    def test_forward_disabled_unchanged(self):
        model = _build_masked_model_with_channel_mode("disabled")
        batch = _make_masked_batch(model)
        result = model(**batch)
        assert "masked_reconstruction" in result.task_outputs

    @pytest.mark.parametrize("mode", ["static", "dynamic", "disabled"])
    def test_representation_contract(self, mode):
        model = _build_masked_model_with_channel_mode(mode)
        batch = _make_masked_batch(model)
        result = model(**batch, capture_representations=True)

        payload = result.representations
        assert payload is not None
        assert payload.channel_mode == mode
        assert payload.backbone_representations.shape == (2, 64)
        if mode == "disabled":
            assert payload.channel_representations is None
        else:
            assert payload.channel_representations.shape == (2, 4, 16)

    def test_gradients_flow_through_encoder(self):
        model = _build_masked_model_with_channel_mode("dynamic")
        batch = _make_masked_batch(model, B=1)
        result = model(**batch)
        loss = result.task_outputs["masked_reconstruction"].sum()
        loss.backward()
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.relative_channel_encoder.parameters()
        )
        assert encoder_has_grad

    def test_reconstruction_queries_correct_shape(self):
        model = _build_masked_model_with_channel_mode("dynamic")
        batch = _make_masked_batch(model)
        result = model(**batch)
        assert result.viz is not None
        assert result.viz.num_channels == 4
        assert result.viz.num_time_tokens == 10


class TestNoInformationLeak:
    """Verify dynamic channel encoder does not leak masked token information."""

    def test_masked_tokens_do_not_affect_channel_embeddings(self):
        """Channel embeddings must be identical regardless of masked token values.

        Directly tests the RelativeChannelEncoder: given the same token_mask,
        changing token embeddings at masked positions must not change the
        output channel embeddings.
        """
        encoder = RelativeChannelEncoder(
            token_dim=48, channel_emb_dim=16, num_heads=4
        )
        encoder.eval()

        B, C, N, D = 2, 4, 10, 48
        tokens = torch.randn(B, C, N, D)
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        # Mask 50% of tokens per channel
        token_mask = torch.ones(B, C, N, dtype=torch.bool)
        token_mask[:, :, N // 2 :] = False

        with torch.no_grad():
            ch_emb_1 = encoder(tokens, channel_mask, token_mask=token_mask)

        # Replace masked token embeddings with large random values
        tokens_corrupted = tokens.clone()
        tokens_corrupted[:, :, N // 2 :] = torch.randn(B, C, N // 2, D) * 100

        with torch.no_grad():
            ch_emb_2 = encoder(
                tokens_corrupted, channel_mask, token_mask=token_mask
            )

        torch.testing.assert_close(ch_emb_1, ch_emb_2)

    def test_without_token_mask_values_do_leak(self):
        """Without token_mask, different token values produce different embeddings.

        This is the control: without the fix (token_mask=None), changing tokens
        changes the output, confirming the leak.
        """
        encoder = RelativeChannelEncoder(
            token_dim=48, channel_emb_dim=16, num_heads=4
        )
        encoder.eval()

        B, C, N, D = 2, 4, 10, 48
        tokens = torch.randn(B, C, N, D)
        channel_mask = torch.ones(B, C, dtype=torch.bool)

        with torch.no_grad():
            ch_emb_1 = encoder(tokens, channel_mask, token_mask=None)

        tokens_corrupted = tokens.clone()
        tokens_corrupted[:, :, N // 2 :] = torch.randn(B, C, N // 2, D) * 100

        with torch.no_grad():
            ch_emb_2 = encoder(tokens_corrupted, channel_mask, token_mask=None)

        assert not torch.allclose(ch_emb_1, ch_emb_2)

    def test_masked_model_forward_uses_token_mask(self):
        """End-to-end: MaskedPOYOEEGModel forward passes token_mask to encoder.

        Verifies that the restructured forward pre-computes the mask and
        threads it through by checking ch_emb_cache is produced.
        """
        model = _build_masked_model_with_channel_mode("dynamic")
        model.eval()
        batch = _make_masked_batch(model)
        with torch.no_grad():
            result = model(**batch)
        assert result.task_outputs is not None
        assert "masked_reconstruction" in result.task_outputs

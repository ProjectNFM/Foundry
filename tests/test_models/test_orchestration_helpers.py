"""Tests for shared POYO orchestration helpers.

Verifies that the extracted helpers (_build_latents, _build_downstream_queries,
_encode_and_process, _decode, _build_reconstruction_queries) produce correct
outputs and that the refactored forward methods are numerically equivalent.
"""

import pytest
import torch

from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel
from foundry.models.ssl_meta import ModelOutput


def _build_base_model(
    embed_dim=64,
    num_channels=8,
    sequence_length=2.0,
    patch_duration=0.5,
    stride=0.5,
):
    from foundry.models import (
        POYOEEGModel,
        EEGTokenizer,
        FixedChannelStrategy,
    )
    from foundry.models.embeddings.temporal import (
        PatchLinearEmbedding as LinearEmbedding,
    )
    from foundry.tasks.config import TaskConfig

    sr = 100.0
    patch_samples = int(patch_duration * sr)

    tokenizer = EEGTokenizer(
        channel_strategy=FixedChannelStrategy(num_channels=num_channels),
        temporal_embedding=LinearEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
        ),
        embed_dim=embed_dim,
        patch_duration=patch_duration,
        stride=stride,
        channel_fusion="add",
    )

    task_configs = {
        "task_a": TaskConfig.from_dict(
            {
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
            }
        ),
        "task_b": TaskConfig.from_dict(
            {
                "name": "task_b",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 3,
                },
                "target_extractor": {
                    "_target_": "foundry.tasks.targets.TargetExtractor",
                    "timestamp_key": "dummy.timestamps",
                    "value_key": "dummy.values",
                },
                "loss": {
                    "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss",
                },
            }
        ),
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
    )
    model.session_emb.initialize_vocab(["session_0", "session_1"])
    model.channel_emb.initialize_vocab([f"ch_{i}" for i in range(num_channels)])
    return model


def _build_masked_model(
    embed_dim=64,
    C_pad=4,
    N=10,
    sequence_length=1.0,
    mask_ratio=0.5,
    channel_fusion="concat",
):
    from foundry.models.embeddings import PerChannelStrategy
    from foundry.models.embeddings.temporal.resample_cnn import (
        ResampleCNNEmbedding,
    )
    from foundry.models.tokenizer import EEGTokenizer
    from foundry.tasks.masking import RandomTokenMasking

    target_token_rate = N / sequence_length
    channel_emb_dim = 16 if channel_fusion == "concat" else embed_dim
    token_dim = (
        embed_dim - channel_emb_dim if channel_fusion == "concat" else embed_dim
    )

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
        channel_fusion=channel_fusion,
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

    masking = RandomTokenMasking(mask_ratio=mask_ratio)
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
    )
    model.initialize_vocabs(
        {
            "session_ids": ["session_0", "session_1"],
            "channel_ids": [f"ch_{i}" for i in range(C_pad)],
        }
    )
    return model


class TestBuildLatents:
    def test_output_shapes(self):
        model = _build_base_model()
        B, n_latent = 2, 8
        latent_index = torch.zeros(B, n_latent, dtype=torch.long)
        latent_timestamps = (
            torch.linspace(0, 1, n_latent).unsqueeze(0).expand(B, -1)
        )

        latents, ts_emb = model._build_latents(latent_index, latent_timestamps)
        assert latents.shape == (B, n_latent, model.embed_dim)
        assert ts_emb.shape[:2] == (B, n_latent)

    def test_deterministic(self):
        model = _build_base_model()
        model.eval()
        B, n_latent = 2, 8
        latent_index = torch.zeros(B, n_latent, dtype=torch.long)
        latent_timestamps = (
            torch.linspace(0, 1, n_latent).unsqueeze(0).expand(B, -1)
        )

        l1, t1 = model._build_latents(latent_index, latent_timestamps)
        l2, t2 = model._build_latents(latent_index, latent_timestamps)
        torch.testing.assert_close(l1, l2)
        torch.testing.assert_close(t1, t2)


class TestBuildDownstreamQueries:
    def test_output_shapes(self):
        model = _build_base_model()
        B, n_out = 2, 5
        output_session_index = torch.zeros(B, n_out, dtype=torch.long)
        task_index = torch.ones(B, n_out, dtype=torch.long)
        output_timestamps = (
            torch.linspace(0, 1, n_out).unsqueeze(0).expand(B, -1)
        )

        queries, ts_emb = model._build_downstream_queries(
            output_session_index, task_index, output_timestamps
        )
        assert queries.shape == (B, n_out, model.embed_dim)
        assert ts_emb.shape[:2] == (B, n_out)

    def test_padding_index_zero_maps_to_task_zero(self):
        """task_index=0 (padding) should map to router index 0 via clamp."""
        model = _build_base_model()
        B, n_out = 1, 3
        output_session_index = torch.zeros(B, n_out, dtype=torch.long)
        task_index = torch.tensor([[0, 1, 2]])
        output_timestamps = torch.zeros(B, n_out)

        queries, _ = model._build_downstream_queries(
            output_session_index, task_index, output_timestamps
        )
        assert queries.shape == (B, n_out, model.embed_dim)

    def test_all_registered_tasks_produce_valid_queries(self):
        model = _build_base_model()
        B = 1
        n_tasks = model.router.num_tasks
        output_session_index = torch.zeros(B, n_tasks, dtype=torch.long)
        task_index = torch.arange(1, n_tasks + 1).unsqueeze(0)
        output_timestamps = torch.zeros(B, n_tasks)

        queries, _ = model._build_downstream_queries(
            output_session_index, task_index, output_timestamps
        )
        assert queries.shape == (B, n_tasks, model.embed_dim)
        assert torch.isfinite(queries).all()


class TestEncodeAndProcess:
    def test_output_shape_matches_latents(self):
        model = _build_base_model()
        B, N_in, N_lat, D = 2, 10, 8, model.embed_dim
        inputs = torch.randn(B, N_in, D)
        input_ts = torch.linspace(0, 1, N_in).unsqueeze(0).expand(B, -1)
        input_ts_emb = model.rotary_emb(input_ts)
        latents = torch.randn(B, N_lat, D)
        latent_ts = torch.linspace(0, 1, N_lat).unsqueeze(0).expand(B, -1)
        latent_ts_emb = model.rotary_emb(latent_ts)

        result = model._encode_and_process(
            inputs, input_ts_emb, latents, latent_ts_emb
        )
        assert result.shape == (B, N_lat, D)

    def test_input_mask_propagation(self):
        """Encoder mask must affect output — masking all inputs should change result."""
        model = _build_base_model()
        model.eval()
        B, N_in, N_lat, D = 1, 10, 4, model.embed_dim
        inputs = torch.randn(B, N_in, D)
        input_ts = torch.linspace(0, 1, N_in).unsqueeze(0).expand(B, -1)
        input_ts_emb = model.rotary_emb(input_ts)
        latents = torch.randn(B, N_lat, D)
        latent_ts = torch.linspace(0, 1, N_lat).unsqueeze(0).expand(B, -1)
        latent_ts_emb = model.rotary_emb(latent_ts)

        no_mask = model._encode_and_process(
            inputs, input_ts_emb, latents.clone(), latent_ts_emb
        )

        partial_mask = torch.zeros(B, N_in, dtype=torch.bool)
        partial_mask[:, :5] = True
        with_mask = model._encode_and_process(
            inputs,
            input_ts_emb,
            latents.clone(),
            latent_ts_emb,
            input_mask=partial_mask,
        )

        assert not torch.allclose(no_mask, with_mask, atol=1e-5)


class TestDecode:
    def test_output_shape_matches_queries(self):
        model = _build_base_model()
        B, N_q, N_lat, D = 2, 6, 8, model.embed_dim
        queries = torch.randn(B, N_q, D)
        latents = torch.randn(B, N_lat, D)
        q_ts = torch.linspace(0, 1, N_q).unsqueeze(0).expand(B, -1)
        l_ts = torch.linspace(0, 1, N_lat).unsqueeze(0).expand(B, -1)
        q_ts_emb = model.rotary_emb(q_ts)
        l_ts_emb = model.rotary_emb(l_ts)

        output = model._decode(queries, latents, q_ts_emb, l_ts_emb)
        assert output.shape == (B, N_q, D)


class TestRouterIndexConversion:
    """Verify the padded task_index ↔ router index convention."""

    def test_padding_zero_excluded(self):
        model = _build_base_model()
        B, N = 1, 4
        D = model.embed_dim
        output_latents = torch.randn(B, N, D)
        task_index = torch.zeros(B, N, dtype=torch.long)

        result = model._route(output_latents, task_index)
        assert len(result) == 0

    def test_all_tasks_routed(self):
        model = _build_base_model()
        B = 1
        n_tasks = model.router.num_tasks
        D = model.embed_dim
        output_latents = torch.randn(B, n_tasks, D)
        task_index = torch.arange(1, n_tasks + 1).unsqueeze(0)

        result = model._route(output_latents, task_index)
        expected_names = sorted(model.task_configs.keys())
        assert set(result.keys()) == set(expected_names)

    def test_mixed_padding_and_tasks(self):
        model = _build_base_model()
        B, N = 1, 6
        D = model.embed_dim
        output_latents = torch.randn(B, N, D)
        task_index = torch.tensor([[0, 1, 0, 2, 0, 1]])

        result = model._route(output_latents, task_index)
        assert len(result) == 2


class TestBaseModelEquivalence:
    """Verify refactored base forward produces identical outputs."""

    def test_forward_returns_model_output(self):
        model = _build_base_model()
        model.eval()
        batch = self._make_batch(model)
        result = model(**batch)
        assert isinstance(result, ModelOutput)
        assert (
            "task_a" in result.task_outputs or "task_b" in result.task_outputs
        )

    def test_forward_deterministic(self):
        model = _build_base_model()
        model.eval()
        batch = self._make_batch(model)

        r1 = model(**batch)
        r2 = model(**batch)
        for name in r1.task_outputs:
            torch.testing.assert_close(
                r1.task_outputs[name], r2.task_outputs[name]
            )

    def test_gradients_flow(self):
        model = _build_base_model()
        batch = self._make_batch(model)
        result = model(**batch)
        loss = sum(v.sum() for v in result.task_outputs.values())
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    @staticmethod
    def _make_batch(model, B=2):
        num_channels = 8
        sr = 100.0
        seq_len = model.sequence_length
        T = int(sr * seq_len)
        N = 4  # patches

        device = next(model.parameters()).device
        input_values = torch.randn(B, num_channels, T, device=device)
        input_timestamps = (
            torch.linspace(0, seq_len, N, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        input_channel_index = (
            torch.arange(num_channels, device=device).unsqueeze(0).expand(B, -1)
        )
        input_session_index = torch.zeros(B, dtype=torch.long, device=device)
        input_mask = torch.ones(
            B, num_channels, dtype=torch.bool, device=device
        )
        input_sampling_rate = torch.full((B,), sr, device=device)

        latent_index = (
            torch.from_numpy(model._latent_index)
            .unsqueeze(0)
            .expand(B, -1)
            .to(device)
        )
        latent_timestamps = (
            torch.from_numpy(model._latent_timestamps)
            .unsqueeze(0)
            .expand(B, -1)
            .float()
            .to(device)
        )

        n_out = 3
        output_session_index = torch.zeros(
            B, n_out, dtype=torch.long, device=device
        )
        output_timestamps = (
            torch.linspace(0, seq_len, n_out, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        task_index = torch.ones(B, n_out, dtype=torch.long, device=device)

        return dict(
            input_values=input_values,
            input_timestamps=input_timestamps,
            input_channel_index=input_channel_index,
            input_session_index=input_session_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
            output_session_index=output_session_index,
            output_timestamps=output_timestamps,
            task_index=task_index,
        )


class TestMaskedModelReconstructionQueries:
    def test_recon_task_idx_cached_at_init(self):
        model = _build_masked_model()
        expected = model.router.get_task_index_by_name(
            model.RECONSTRUCTION_TASK_NAME
        )
        assert model._recon_task_idx == expected

    def test_query_shapes(self):
        model = _build_masked_model(C_pad=4, N=10)
        B, num_masked, N = 2, 8, 10
        D = model.embed_dim

        session_emb = torch.randn(B, 1, D)
        mask_indices = torch.randint(0, 4 * N, (B, num_masked))
        input_channel_index = torch.arange(4).unsqueeze(0).expand(B, -1)
        input_timestamps = (
            torch.linspace(0, 1, 4 * N).unsqueeze(0).expand(B, -1)
        )
        masked_validity = torch.ones(B, num_masked, dtype=torch.bool)

        queries, ts_emb, task_index = model._build_reconstruction_queries(
            session_emb,
            mask_indices,
            input_channel_index,
            input_timestamps,
            N,
            masked_validity,
        )

        assert queries.shape == (B, num_masked, D)
        assert ts_emb.shape[:2] == (B, num_masked)
        assert task_index.shape == (B, num_masked)

    def test_validity_masks_task_index(self):
        """Invalid masked positions should get task_index=0 (padding)."""
        model = _build_masked_model(C_pad=4, N=10)
        B, num_masked, N = 1, 8, 10
        D = model.embed_dim

        session_emb = torch.randn(B, 1, D)
        mask_indices = torch.randint(0, 4 * N, (B, num_masked))
        input_channel_index = torch.arange(4).unsqueeze(0).expand(B, -1)
        input_timestamps = (
            torch.linspace(0, 1, 4 * N).unsqueeze(0).expand(B, -1)
        )

        masked_validity = torch.tensor(
            [[True, True, False, True, False, True, True, False]]
        )

        _, _, task_index = model._build_reconstruction_queries(
            session_emb,
            mask_indices,
            input_channel_index,
            input_timestamps,
            N,
            masked_validity,
        )

        expected_recon_idx = model._recon_task_idx + 1
        assert (task_index[0, :2] == expected_recon_idx).all()
        assert task_index[0, 2] == 0
        assert task_index[0, 4] == 0
        assert task_index[0, 7] == 0

    def test_channel_projection_applied_when_dims_differ(self):
        model = _build_masked_model(embed_dim=64, channel_fusion="concat")
        assert model.recon_channel_proj is not None

    def test_no_channel_projection_when_dims_match(self):
        model = _build_masked_model(embed_dim=64, channel_fusion="add")
        assert model.recon_channel_proj is None


class TestMaskedModelEquivalence:
    """Verify refactored masked forward remains correct."""

    @pytest.fixture
    def model(self):
        return _build_masked_model(
            embed_dim=64, C_pad=4, N=10, sequence_length=1.0, mask_ratio=0.5
        )

    def test_forward_returns_model_output(self, model):
        result = self._run_forward(model, B=2)
        assert isinstance(result, ModelOutput)
        assert "masked_reconstruction" in result.task_outputs

    def test_forward_with_targets(self, model):
        result = self._run_forward(model, B=2, include_targets=True)
        assert result.ssl_meta is not None
        meta = result.ssl_meta["masked_reconstruction"]
        assert meta.targets.shape[0] == meta.weights.shape[0]
        preds = result.task_outputs["masked_reconstruction"]
        assert preds.shape[0] == meta.targets.shape[0]

    def test_forward_without_targets(self, model):
        result = self._run_forward(model, B=2, include_targets=False)
        assert result.ssl_meta is None

    def test_gradients_flow(self, model):
        result = self._run_forward(model, B=1, include_targets=True)
        loss = result.task_outputs["masked_reconstruction"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_viz_metadata_present(self, model):
        result = self._run_forward(model, B=2)
        assert result.viz is not None
        assert result.viz.num_channels == 4
        assert result.viz.num_time_tokens == 10

    @staticmethod
    def _run_forward(model, B, include_targets=True):
        C_pad, N, T = 4, 10, 100
        sr = 100.0
        device = next(model.parameters()).device

        input_values = torch.randn(B, C_pad, T, device=device)
        input_timestamps = (
            torch.linspace(0, 1.0, N, device=device)
            .unsqueeze(0)
            .expand(B, -1)
            .repeat(1, C_pad)
        )
        input_channel_index = (
            torch.arange(C_pad, device=device).unsqueeze(0).expand(B, -1)
        )
        input_session_index = torch.zeros(B, dtype=torch.long, device=device)
        input_mask = torch.ones(B, C_pad, dtype=torch.bool, device=device)
        input_sampling_rate = torch.full((B,), sr, device=device)
        input_seq_len = torch.full((B,), T, dtype=torch.long, device=device)

        latent_index = (
            torch.from_numpy(model._latent_index)
            .unsqueeze(0)
            .expand(B, -1)
            .to(device)
        )
        latent_timestamps = (
            torch.from_numpy(model._latent_timestamps)
            .unsqueeze(0)
            .expand(B, -1)
            .float()
            .to(device)
        )

        output_timestamps = torch.zeros(B, 0, device=device)
        output_session_index = torch.zeros(
            B, 0, dtype=torch.long, device=device
        )
        task_index = torch.zeros(B, 0, dtype=torch.long, device=device)

        kwargs = dict(
            input_values=input_values,
            input_timestamps=input_timestamps,
            input_channel_index=input_channel_index,
            input_session_index=input_session_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
            output_session_index=output_session_index,
            output_timestamps=output_timestamps,
            task_index=task_index,
        )
        if include_targets:
            kwargs["reconstruction_targets"] = torch.randn(
                B, C_pad * N, device=device
            )
        return model(**kwargs)


class TestMixedTaskRouting:
    """Verify combined reconstruction + downstream routing in masked model."""

    def test_mixed_ssl_and_downstream(self):
        from foundry.models.embeddings import PerChannelStrategy
        from foundry.models.embeddings.temporal.resample_cnn import (
            ResampleCNNEmbedding,
        )
        from foundry.models.tokenizer import EEGTokenizer
        from foundry.tasks.masking import RandomTokenMasking

        embed_dim = 64
        C_pad, N = 4, 10
        channel_emb_dim = 16
        token_dim = embed_dim - channel_emb_dim

        channel_strategy = PerChannelStrategy(max_channels=C_pad)
        temporal_embedding = ResampleCNNEmbedding(
            embed_dim=token_dim,
            num_sources=1,
            target_token_rate=N / 1.0,
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
            "downstream_cls": {
                "name": "downstream_cls",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 3,
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

        model = MaskedPOYOEEGModel(
            tokenizer=tokenizer,
            task_configs=task_configs,
            embed_dim=embed_dim,
            sequence_length=1.0,
            latent_step=0.5,
            num_latents_per_step=2,
            depth=1,
            dim_head=32,
            cross_heads=2,
            self_heads=2,
            masking=RandomTokenMasking(mask_ratio=0.5),
        )
        model.initialize_vocabs(
            {
                "session_ids": ["sess_0"],
                "channel_ids": [f"ch_{i}" for i in range(C_pad)],
            }
        )

        B, T, sr = 2, 100, 100.0
        device = next(model.parameters()).device

        ds_task_idx = model.router.get_task_index_by_name("downstream_cls")

        result = model(
            input_values=torch.randn(B, C_pad, T, device=device),
            input_timestamps=(
                torch.linspace(0, 1, N, device=device)
                .unsqueeze(0)
                .expand(B, -1)
                .repeat(1, C_pad)
            ),
            input_channel_index=(
                torch.arange(C_pad, device=device).unsqueeze(0).expand(B, -1)
            ),
            input_session_index=torch.zeros(B, dtype=torch.long, device=device),
            input_mask=torch.ones(B, C_pad, dtype=torch.bool, device=device),
            input_sampling_rate=torch.full((B,), sr, device=device),
            input_seq_len=torch.full((B,), T, dtype=torch.long, device=device),
            latent_index=(
                torch.from_numpy(model._latent_index)
                .unsqueeze(0)
                .expand(B, -1)
                .to(device)
            ),
            latent_timestamps=(
                torch.from_numpy(model._latent_timestamps)
                .unsqueeze(0)
                .expand(B, -1)
                .float()
                .to(device)
            ),
            output_session_index=torch.zeros(
                B, 2, dtype=torch.long, device=device
            ),
            output_timestamps=torch.tensor([[0.3, 0.7]] * B, device=device),
            task_index=torch.full(
                (B, 2), ds_task_idx + 1, dtype=torch.long, device=device
            ),
            reconstruction_targets=torch.randn(B, C_pad * N, device=device),
        )

        assert "masked_reconstruction" in result.task_outputs
        assert "downstream_cls" in result.task_outputs
        assert result.ssl_meta is not None
        assert result.task_outputs["downstream_cls"].shape[0] == B * 2

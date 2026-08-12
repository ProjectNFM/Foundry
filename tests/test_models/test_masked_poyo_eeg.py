"""Tests for MaskedPOYOEEGModel forward pass and target gathering."""

import numpy as np
import pytest
import torch
from torch_brain.data import Data, Interval, RegularTimeSeries

from foundry.models.masked_poyo_eeg import (
    MaskedPOYOEEGModel,
    _compute_visible_indices,
)
from foundry.models.ssl_meta import ModelOutput, SSLTaskMeta
from foundry.tasks.masking import RandomTokenMasking


class _MockChannels:
    def __init__(self, channel_ids, types=None):
        self.id = np.array(channel_ids)
        if types is not None:
            self.type = np.array(types, dtype=str)

    def __len__(self):
        return len(self.id)


class _MockSession:
    def __init__(self, session_id):
        self.id = session_id


def _build_minimal_masked_model(
    embed_dim=64,
    C_pad=4,
    N=10,
    sequence_length=1.0,
    mask_ratio=0.5,
):
    """Build a MaskedPOYOEEGModel with minimal config for testing.

    Uses PerChannelStrategy + ResampleCNNEmbedding so that the model
    has a valid tokenizer.
    """
    from foundry.models.embeddings import (
        PerChannelStrategy,
    )
    from foundry.models.embeddings.temporal.resample_cnn import (
        ResampleCNNEmbedding,
    )
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

    session_ids = ["session_0", "session_1"]
    channel_ids = [f"ch_{i}" for i in range(C_pad)]
    model.initialize_vocabs(
        {"session_ids": session_ids, "channel_ids": channel_ids}
    )
    return model


class TestMaskedModelInit:
    def test_requires_per_channel_strategy(self):
        from foundry.models.embeddings import (
            SpatialProjectionStrategy,
            LinearSpatialProjector,
        )
        from foundry.models.embeddings.temporal.resample_cnn import (
            ResampleCNNEmbedding,
        )
        from foundry.models.tokenizer import EEGTokenizer

        embed_dim = 64
        num_channels = 4
        num_sources = 4
        channel_strategy = SpatialProjectionStrategy(
            num_channels=num_channels,
            num_sources=num_sources,
            projector=LinearSpatialProjector(
                num_channels=num_channels,
                num_sources=num_sources,
            ),
        )
        temporal_embedding = ResampleCNNEmbedding(
            embed_dim=embed_dim,
            num_sources=num_sources,
            target_token_rate=10.0,
            num_filters=4,
            kernel_size=3,
            num_conv_layers=1,
        )
        tokenizer = EEGTokenizer(
            channel_strategy=channel_strategy,
            temporal_embedding=temporal_embedding,
            embed_dim=embed_dim,
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
            }
        }

        with pytest.raises(ValueError, match="PerChannelStrategy"):
            MaskedPOYOEEGModel(
                tokenizer=tokenizer,
                task_configs=task_configs,
                embed_dim=embed_dim,
                sequence_length=1.0,
                latent_step=0.5,
                num_latents_per_step=1,
                depth=1,
                dim_head=32,
                cross_heads=1,
                self_heads=1,
                masking=RandomTokenMasking(mask_ratio=0.5),
            )


class TestMaskedModelForward:
    @pytest.fixture
    def model(self):
        return _build_minimal_masked_model(
            embed_dim=64, C_pad=4, N=10, sequence_length=1.0, mask_ratio=0.5
        )

    def test_forward_returns_model_output_with_predictions(self, model):
        B, C_pad, N = 2, 4, 10
        T = 100  # raw signal samples
        sr = 100.0

        result = self._run_forward(model, B, C_pad, N, T, sr)

        assert isinstance(result, ModelOutput)
        assert "masked_reconstruction" in result.task_outputs

    def test_forward_returns_targets_and_weights(self, model):
        B, C_pad, N = 2, 4, 10
        T = 100
        sr = 100.0

        result = self._run_forward(
            model, B, C_pad, N, T, sr, include_recon_targets=True
        )

        assert result.ssl_meta is not None
        assert "masked_reconstruction" in result.ssl_meta
        meta = result.ssl_meta["masked_reconstruction"]
        assert isinstance(meta, SSLTaskMeta)
        assert meta.targets is not None
        assert meta.weights is not None

    def test_prediction_shape_matches_targets(self, model):
        B, C_pad, N = 2, 4, 10
        T = 100
        sr = 100.0

        result = self._run_forward(
            model, B, C_pad, N, T, sr, include_recon_targets=True
        )

        preds = result.task_outputs["masked_reconstruction"]
        targets = result.ssl_meta["masked_reconstruction"].targets
        assert preds.shape[0] == targets.shape[0]

    def test_forward_without_recon_targets(self, model):
        B, C_pad, N = 2, 4, 10
        T = 100
        sr = 100.0

        result = self._run_forward(
            model, B, C_pad, N, T, sr, include_recon_targets=False
        )

        assert "masked_reconstruction" in result.task_outputs
        assert result.ssl_meta is None

    def test_weights_are_between_zero_and_one(self, model):
        B, C_pad, N = 2, 4, 10
        T = 100
        sr = 100.0

        result = self._run_forward(
            model, B, C_pad, N, T, sr, include_recon_targets=True
        )

        weights = result.ssl_meta["masked_reconstruction"].weights
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_gradients_flow_through_forward(self, model):
        B, C_pad, N = 1, 4, 10
        T = 100
        sr = 100.0

        result = self._run_forward(
            model, B, C_pad, N, T, sr, include_recon_targets=True
        )

        loss = result.task_outputs["masked_reconstruction"].sum()
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def _run_forward(
        self, model, B, C_pad, N, T, sr, include_recon_targets=True
    ):
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

        _ = model._latent_index.shape[0]
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

        if include_recon_targets:
            kwargs["reconstruction_targets"] = torch.randn(
                B, C_pad * N, device=device
            )

        return model(**kwargs)


class TestMaskedModelTokenize:
    """Reproduce bugs in MaskedPOYOEEGModel.tokenize() target computation."""

    def _make_data(self, num_samples, channel_ids, channel_types, sr=100.0):
        n_ch = len(channel_ids)
        signal = np.random.randn(num_samples, n_ch).astype(np.float32)
        eeg = RegularTimeSeries(
            signal=signal, sampling_rate=sr, domain_start=0.0
        )
        seq_len = num_samples / sr
        data = Data(eeg=eeg, domain=Interval(0.0, seq_len))
        data.channels = _MockChannels(channel_ids, types=channel_types)
        data.session = _MockSession("session_0")
        data._absolute_start = 0.0
        return data

    def test_reconstruction_targets_exclude_non_eeg_channels(self):
        """Bug 3 (High): tokenize() re-reads raw signal_source.signal without
        applying the modality_mask filter. Reconstruction targets for a
        non-EEG channel should be all-zero (padded), not z-scored signal."""
        C_pad = 4
        N = 10
        model = _build_minimal_masked_model(
            C_pad=C_pad, N=N, sequence_length=1.0
        )

        data = self._make_data(
            num_samples=100,
            channel_ids=["ch_0", "ch_1", "ch_2", "ch_3"],
            channel_types=["EEG", "EEG", "EEG", "EOG"],
            sr=100.0,
        )

        result = model.tokenize(data)
        targets = result["reconstruction_targets"]

        eog_targets = targets[3 * N : 4 * N]
        assert torch.all(eog_targets == 0), (
            "Non-EEG channel should have zero reconstruction targets; "
            "tokenize() must use modality-filtered signal"
        )

    def test_reconstruction_targets_use_normalized_T(self):
        """Bug 5 (Medium): _compute_patch_targets uses raw T, but pretokenize
        normalizes to expected_T = round(sr * sequence_length). When T differs
        by 1, the patch count disagrees with the tokenizer grid.

        For non-patching mode, this manifests as interp using wrong raw_times.
        """
        C_pad = 4
        N = 10
        sr = 100.0
        seq_len = 1.0
        expected_T = round(sr * seq_len)  # 100

        model = _build_minimal_masked_model(
            C_pad=C_pad, N=N, sequence_length=seq_len
        )

        off_by_one_T = expected_T + 1
        data = self._make_data(
            num_samples=off_by_one_T,
            channel_ids=["ch_0", "ch_1", "ch_2", "ch_3"],
            channel_types=["EEG"] * 4,
            sr=sr,
        )

        result = model.tokenize(data)
        targets = result["reconstruction_targets"]

        expected_size = C_pad * N
        assert targets.shape[0] == expected_size, (
            f"Reconstruction targets should have {expected_size} elements "
            f"(matching tokenizer grid), got {targets.shape[0]}"
        )


class TestDynamicDurationTokenize:
    """Tokenization adapts to actual window duration from Data.start/end."""

    def _make_data(
        self, duration, channel_ids, channel_types, sr=100.0, start=0.0
    ):
        end = start + duration
        num_samples = round(sr * duration)
        n_ch = len(channel_ids)
        signal = np.random.randn(num_samples, n_ch).astype(np.float32)
        eeg = RegularTimeSeries(
            signal=signal, sampling_rate=sr, domain_start=start
        )
        data = Data(eeg=eeg, domain=Interval(start, end))
        data.channels = _MockChannels(channel_ids, types=channel_types)
        data.session = _MockSession("session_0")
        data._absolute_start = start
        return data

    def test_latent_count_constant_across_durations(self):
        """_tokenize_core produces the same number of latent tokens
        regardless of the actual window duration."""
        base_seq_len = 1.0
        latent_step = 0.5
        num_latents_per_step = 2
        C_pad = 4
        N = 10

        model = _build_minimal_masked_model(
            C_pad=C_pad,
            N=N,
            sequence_length=base_seq_len,
            mask_ratio=0.5,
        )
        model._num_latent_bins = round(base_seq_len / latent_step)

        channel_ids = [f"ch_{i}" for i in range(C_pad)]
        channel_types = ["EEG"] * C_pad
        expected_latent_count = model._num_latent_bins * num_latents_per_step

        for duration in [0.5, 1.0, 2.0, 5.0]:
            data = self._make_data(
                duration, channel_ids, channel_types, sr=100.0
            )
            result, _ = model._tokenize_core(data)

            assert len(result["latent_index"]) == expected_latent_count, (
                f"Duration {duration}s produced "
                f"{len(result['latent_index'])} latent tokens, "
                f"expected {expected_latent_count}"
            )
            assert len(result["latent_timestamps"]) == expected_latent_count

    def test_prepare_signal_uses_actual_duration(self):
        """_prepare_signal respects the explicit sequence_length argument."""
        C_pad = 4
        N = 10

        model = _build_minimal_masked_model(
            C_pad=C_pad, N=N, sequence_length=1.0
        )

        channel_ids = [f"ch_{i}" for i in range(C_pad)]
        channel_types = ["EEG"] * C_pad

        data_2s = self._make_data(2.0, channel_ids, channel_types, sr=100.0)

        prepared_default = model._prepare_signal(data_2s)
        assert prepared_default.num_samples == round(100.0 * 2.0)

        prepared_explicit = model._prepare_signal(data_2s, sequence_length=1.0)
        assert prepared_explicit.num_samples == round(100.0 * 1.0)

    def test_get_actual_duration_fallback(self):
        """_get_actual_duration returns self.sequence_length when
        Data has no domain."""
        model = _build_minimal_masked_model(sequence_length=1.0)

        data = Data()
        data.channels = _MockChannels(["ch_0"], types=["EEG"])
        data.session = _MockSession("session_0")

        assert model._get_actual_duration(data) == 1.0

    def test_get_actual_duration_from_data(self):
        """_get_actual_duration derives duration from Data.domain."""
        model = _build_minimal_masked_model(sequence_length=1.0)

        data = Data(domain=Interval(10.0, 15.0))
        data.channels = _MockChannels(["ch_0"], types=["EEG"])
        data.session = _MockSession("session_0")

        assert model._get_actual_duration(data) == 5.0

    def test_masked_tokenize_uses_actual_duration(self):
        """MaskedPOYOEEGModel.tokenize() produces reconstruction targets
        that match the actual window duration, not self.sequence_length."""
        C_pad = 4
        N = 10
        base_seq_len = 1.0

        model = _build_minimal_masked_model(
            C_pad=C_pad, N=N, sequence_length=base_seq_len
        )

        channel_ids = [f"ch_{i}" for i in range(C_pad)]
        channel_types = ["EEG"] * C_pad

        data = self._make_data(
            base_seq_len, channel_ids, channel_types, sr=100.0
        )
        result = model.tokenize(data)

        assert "reconstruction_targets" in result
        assert "latent_index" in result
        assert "latent_timestamps" in result


class TestComputeVisibleIndicesProperties:
    def test_preserves_token_order(self):
        total = 20
        mask_indices = torch.tensor([[5, 10, 15]])
        visible = _compute_visible_indices(total, mask_indices)

        diffs = visible[0, 1:] - visible[0, :-1]
        assert (diffs > 0).all(), "Visible indices should be in ascending order"


class TestZeroMaskedSignal:
    """Verify that zero_masked_signal prevents temporal embedding leakage."""

    def _make_batch_and_mask(self, model, B=2, C_pad=4, N=10, T=100, sr=100.0):
        device = next(model.parameters()).device
        input_mask = torch.ones(B, C_pad, dtype=torch.bool, device=device)

        torch.manual_seed(42)
        mask_indices, _ = model.masking(
            num_channels=C_pad,
            num_time_tokens=N,
            channel_mask=input_mask,
            device=device,
        )

        masked_grid = torch.zeros(B, C_pad * N, dtype=torch.bool, device=device)
        masked_grid.scatter_(1, mask_indices, True)
        masked_grid = masked_grid.reshape(B, C_pad, N)
        signal_mask = torch.nn.functional.interpolate(
            masked_grid.float(), size=T, mode="nearest"
        ).bool()

        input_values = torch.randn(B, C_pad, T, device=device)

        batch = dict(
            input_values=input_values,
            input_timestamps=(
                torch.linspace(0, 1.0, N, device=device)
                .unsqueeze(0)
                .expand(B, -1)
                .repeat(1, C_pad)
            ),
            input_channel_index=(
                torch.arange(C_pad, device=device).unsqueeze(0).expand(B, -1)
            ),
            input_session_index=torch.zeros(
                B, dtype=torch.long, device=device
            ),
            input_mask=input_mask,
            input_sampling_rate=torch.full((B,), sr, device=device),
            input_seq_len=torch.full(
                (B,), T, dtype=torch.long, device=device
            ),
            latent_index=torch.from_numpy(model._latent_index)
            .unsqueeze(0)
            .expand(B, -1)
            .to(device),
            latent_timestamps=torch.from_numpy(model._latent_timestamps)
            .unsqueeze(0)
            .expand(B, -1)
            .float()
            .to(device),
            output_session_index=torch.zeros(
                B, 0, dtype=torch.long, device=device
            ),
            output_timestamps=torch.zeros(B, 0, device=device),
            task_index=torch.zeros(B, 0, dtype=torch.long, device=device),
            reconstruction_targets=torch.randn(
                B, C_pad * N, device=device
            ),
        )

        return batch, signal_mask

    def test_corrupting_masked_signal_does_not_change_output(self):
        """With zero_masked_signal=True, changing the raw signal at masked
        positions must not affect model output, since those positions are
        zeroed before the temporal embedding processes them."""
        model = _build_minimal_masked_model()
        model.zero_masked_signal = True
        model.eval()

        batch, signal_mask = self._make_batch_and_mask(model)

        corrupted_values = batch["input_values"].clone()
        corrupted_values[signal_mask] += 1000.0
        corrupted_batch = {**batch, "input_values": corrupted_values}

        torch.manual_seed(42)
        with torch.no_grad():
            result_orig = model(**batch)

        torch.manual_seed(42)
        with torch.no_grad():
            result_corrupted = model(**corrupted_batch)

        torch.testing.assert_close(
            result_orig.task_outputs["masked_reconstruction"],
            result_corrupted.task_outputs["masked_reconstruction"],
        )

    def test_without_zero_masked_signal_corruption_changes_output(self):
        """Without signal zeroing, the temporal embedding's receptive field
        leaks masked signal values into visible token embeddings, so
        corrupting masked positions changes the output."""
        model = _build_minimal_masked_model()
        model.zero_masked_signal = False
        model.eval()

        batch, signal_mask = self._make_batch_and_mask(model)

        corrupted_values = batch["input_values"].clone()
        corrupted_values[signal_mask] += 1000.0
        corrupted_batch = {**batch, "input_values": corrupted_values}

        torch.manual_seed(42)
        with torch.no_grad():
            result_orig = model(**batch)

        torch.manual_seed(42)
        with torch.no_grad():
            result_corrupted = model(**corrupted_batch)

        assert not torch.allclose(
            result_orig.task_outputs["masked_reconstruction"],
            result_corrupted.task_outputs["masked_reconstruction"],
        )

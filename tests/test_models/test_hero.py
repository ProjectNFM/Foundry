"""Contract tests for the hierarchical EEG representation encoder."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_brain.batching import collate
from torch_brain.data import Data, Interval, RegularTimeSeries

from foundry.models.hero import (
    AlignedGatedResidual,
    HEROModel,
    SpatialSlotMixer,
    TaskQueryCrossAttention,
    TemporalReduction,
)
from foundry.tasks.config import TaskConfig


@pytest.fixture
def model():
    """A deliberately small CPU model; geometry, rather than capacity, is tested."""
    torch.manual_seed(7)
    return HEROModel(
        task_configs={},
        num_channels=64,
        embed_dim=16,
        num_attn_heads=4,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
    ).eval()


def encode(model, signal, *, rate=128, channel_mask=None, sample_mask=None):
    with torch.no_grad():
        return model.encode(
            signal=signal,
            sampling_rate=rate,
            channel_mask=channel_mask,
            sample_mask=sample_mask,
        )


@pytest.mark.parametrize(("duration", "channels"), [(1, 2), (4, 16), (30, 64)])
def test_hierarchy_lengths_timestamps_and_channel_independent_shape(
    model, duration, channels
):
    signal = torch.randn(1, channels, 128 * duration)
    rep = encode(model, signal)

    assert rep.content.shape == (1, 128 * duration, 16)
    assert rep.coverage.mid_valid.shape == (1, 32 * duration)
    assert rep.coverage.coarse_valid.shape == (1, 8 * duration)
    assert torch.allclose(
        rep.content_timestamps[0],
        (torch.arange(128 * duration) + 0.5) / 128,
    )
    assert torch.all(
        rep.content_timestamps[:, 1:] > rep.content_timestamps[:, :-1]
    )
    assert rep.coverage.fine_rf_intervals.shape == (1, 128 * duration, 2)
    assert torch.all(
        rep.coverage.fine_rf_intervals[..., 0]
        < rep.coverage.fine_rf_intervals[..., 1]
    )


def test_permutation_and_masked_channel_values_do_not_change_content(model):
    signal = torch.randn(1, 5, 128)
    mask = torch.tensor([[True, True, True, False, False]])
    baseline = encode(model, signal, channel_mask=mask)

    perm = torch.tensor([2, 4, 0, 3, 1])
    permuted = encode(model, signal[:, perm], channel_mask=mask[:, perm])
    assert torch.allclose(
        baseline.content, permuted.content, atol=2e-5, rtol=2e-5
    )
    assert torch.equal(baseline.content_timestamps, permuted.content_timestamps)
    for field_name, baseline_value in vars(baseline.coverage).items():
        assert torch.equal(
            baseline_value, getattr(permuted.coverage, field_name)
        ), field_name

    noisy = signal.clone()
    noisy[:, ~mask[0]] = 1e6
    masked = encode(model, noisy, channel_mask=mask)
    assert torch.allclose(
        baseline.content, masked.content, atol=2e-5, rtol=2e-5
    )


def test_sample_masks_block_values_and_are_reflected_in_coverage(model):
    signal = torch.randn(1, 4, 256)
    sample_mask = torch.ones(1, 256, dtype=torch.bool)
    sample_mask[:, 100:120] = False
    baseline = encode(model, signal, sample_mask=sample_mask)

    noisy = signal.clone()
    noisy[:, :, 100:120] = 1e6
    masked = encode(model, noisy, sample_mask=sample_mask)
    assert torch.allclose(
        baseline.content, masked.content, atol=2e-5, rtol=2e-5
    )
    assert not baseline.coverage.fine_valid[:, 100:120].any()
    assert torch.equal(baseline.coverage.sample_support.bool(), sample_mask)
    assert torch.isfinite(baseline.content).all()


def test_missing_real_channel_reduces_coverage_but_keeps_output_finite(model):
    signal = torch.randn(1, 4, 128)
    all_channels = encode(model, signal)
    mask = torch.tensor([[True, True, True, False]])
    missing_channel = encode(model, signal, channel_mask=mask)

    assert torch.isfinite(missing_channel.content).all()
    assert torch.all(missing_channel.coverage.channel_count == 3)
    assert torch.all(all_channels.coverage.channel_count == 4)
    assert torch.all(missing_channel.coverage.channel_fraction == 3 / 4)


def test_resampling_uses_physical_time_and_canonical_grid(model):
    seconds = 2
    # ``torchaudio.resample`` defines its first sample at t=0.
    time_64 = torch.arange(64 * seconds) / 64
    time_128 = torch.arange(128 * seconds) / 128
    signal_64 = torch.sin(2 * torch.pi * 7 * time_64)[None, None]
    signal_128 = torch.sin(2 * torch.pi * 7 * time_128)[None, None]

    low_rate = encode(model, signal_64, rate=64)
    canonical = encode(model, signal_128)
    assert low_rate.content.shape == canonical.content.shape == (1, 256, 16)
    assert torch.allclose(
        low_rate.content_timestamps, canonical.content_timestamps
    )
    # Finite filters differ at the boundaries; compare the unaffected interior.
    assert torch.allclose(
        low_rate.content[:, 24:-24],
        canonical.content[:, 24:-24],
        atol=0.08,
        rtol=0.08,
    )


def test_internal_level_timestamps_follow_documented_rates(model):
    timestamps = {}

    def save_mid(_module, _args, output):
        timestamps["mid"] = output[1]

    def save_coarse(_module, _args, output):
        timestamps["coarse"] = output[1]

    mid_hook = model.encoder.fine_to_mid.register_forward_hook(save_mid)
    coarse_hook = model.encoder.mid_to_coarse.register_forward_hook(save_coarse)
    try:
        representation = encode(model, torch.randn(1, 2, 256))
    finally:
        mid_hook.remove()
        coarse_hook.remove()

    assert torch.allclose(
        timestamps["mid"], representation.content_timestamps[:, 2::4]
    )
    assert torch.allclose(timestamps["coarse"], timestamps["mid"][:, 2::4])
    assert torch.allclose(
        timestamps["mid"][:, 1:] - timestamps["mid"][:, :-1],
        torch.full((1, 63), 1 / 32),
    )
    assert torch.allclose(
        timestamps["coarse"][:, 1:] - timestamps["coarse"][:, :-1],
        torch.full((1, 15), 1 / 8),
    )


@pytest.mark.parametrize("source_rate", [64, 256])
@pytest.mark.parametrize("waveform", ["chirp", "combination"])
def test_analytic_signals_agree_across_sampling_rates(
    model, source_rate, waveform
):
    seconds = 4

    def continuous_signal(rate):
        time = torch.arange(rate * seconds) / rate
        if waveform == "chirp":
            # Linear chirp from 2 to 10 Hz over the four-second interval.
            phase = 2 * torch.pi * (2 * time + time.square())
            return torch.sin(phase)
        return 0.7 * torch.sin(2 * torch.pi * 3 * time + 0.2) + 0.3 * torch.cos(
            2 * torch.pi * 11 * time - 0.1
        )

    source = encode(
        model, continuous_signal(source_rate)[None, None], rate=source_rate
    )
    canonical = encode(model, continuous_signal(128)[None, None])

    assert torch.allclose(
        source.content_timestamps, canonical.content_timestamps
    )
    # Resampling and the causal convolution stack have finite-filter boundary
    # transients. The interior tolerance is fixed by the 64 Hz worst case.
    assert torch.allclose(
        source.content[:, 32:-32],
        canonical.content[:, 32:-32],
        atol=0.12,
        rtol=0.12,
    )


def test_explicit_timestamps_preserve_their_physical_domain(model):
    timestamps = 3.0 + (torch.arange(128)[None] + 0.5) / 128
    rep = model.encode(signal=torch.randn(1, 2, 128), timestamps=timestamps)
    assert torch.allclose(rep.content_timestamps, timestamps)


def test_near_canonical_timestamps_preserve_their_physical_domain(model):
    rate = 127.6
    timestamps = 3.0 + (torch.arange(128)[None] + 0.5) / rate
    rep = model.encode(signal=torch.randn(1, 2, 128), timestamps=timestamps)
    assert torch.allclose(rep.content_timestamps, timestamps)


def test_masked_time_padding_does_not_change_valid_content():
    torch.manual_seed(7)
    model = HEROModel(
        task_configs={},
        num_channels=4,
        embed_dim=16,
        num_attn_heads=4,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=1,
    ).eval()
    signal = torch.randn(1, 4, 128)
    unpadded = encode(model, signal)

    padded_signal = torch.cat([signal, torch.randn(1, 4, 128)], dim=-1)
    sample_mask = torch.cat(
        [
            torch.ones(1, 128, dtype=torch.bool),
            torch.zeros(1, 128, dtype=torch.bool),
        ],
        dim=1,
    )
    padded = encode(model, padded_signal, sample_mask=sample_mask)

    assert torch.allclose(unpadded.content, padded.content[:, :128], atol=2e-5)
    assert torch.equal(
        unpadded.coverage.fine_rf_intervals,
        padded.coverage.fine_rf_intervals[:, :128],
    )


def test_mixed_valid_lengths_match_independent_unpadded_examples(model):
    torch.manual_seed(11)
    full = torch.randn(1, 3, 256)
    short = torch.randn(1, 3, 128)
    padded_short = torch.cat([short, torch.randn(1, 3, 128) * 1e5], dim=-1)
    signal = torch.cat([full, padded_short], dim=0)
    sample_mask = torch.ones(2, 256, dtype=torch.bool)
    sample_mask[1, 128:] = False

    batched = encode(model, signal, sample_mask=sample_mask)
    full_alone = encode(model, full)
    short_alone = encode(model, short)

    assert torch.allclose(batched.content[0], full_alone.content[0], atol=2e-5)
    assert torch.allclose(
        batched.content[1, :128], short_alone.content[0], atol=2e-5
    )
    assert not batched.coverage.fine_valid[1, 128:].any()
    assert torch.equal(
        batched.coverage.fine_rf_intervals[1, :128],
        short_alone.coverage.fine_rf_intervals[0],
    )


def test_resampling_does_not_lose_isolated_invalid_source_samples(model):
    signal = torch.randn(1, 1, 256)
    sample_mask = torch.ones(1, 256, dtype=torch.bool)
    sample_mask[:, 1] = False

    _, resampled_mask = model._resample_signal(signal, sample_mask, 256, 128)

    assert not resampled_mask.all()


def test_temporal_slots_are_not_mean_pooling_and_keep_partial_tail_policy():
    torch.manual_seed(2)
    reduction = TemporalReduction(
        embed_dim=8, num_temporal_slots=2, num_heads=2
    ).eval()
    x = torch.zeros(1, 128, 8)
    x[:, 64] = 1
    timestamps = torch.arange(128)[None] / 128
    with torch.no_grad():
        reduced, _, _, _ = reduction(x, timestamps)
    assert reduced.shape[1] == 32
    assert not torch.allclose(reduced[:, 16], torch.zeros_like(reduced[:, 16]))


def test_temporal_slots_distinguish_equal_mean_local_patterns():
    torch.manual_seed(12)
    reduction = TemporalReduction(
        embed_dim=8, num_temporal_slots=2, num_heads=2
    ).eval()
    first = torch.zeros(1, 256, 8)
    second = torch.zeros_like(first)
    first[:, 128] = 1
    first[:, 129] = -1
    second[:, 130] = 1
    second[:, 131] = -1
    assert torch.equal(first.mean(dim=1), second.mean(dim=1))
    timestamps = torch.arange(256)[None] / 128

    with torch.no_grad():
        first_reduced, _, _, _ = reduction(first, timestamps)
        second_reduced, _, _, _ = reduction(second, timestamps)

    assert not torch.allclose(first_reduced, second_reduced)


def test_spatial_slots_are_sensitive_to_an_unmasked_channel():
    torch.manual_seed(3)
    mixer = SpatialSlotMixer(embed_dim=8, num_slots=2, num_heads=2)
    features = torch.randn(1, 3, 4, 8, requires_grad=True)
    output, valid = mixer(features, torch.ones(1, 3, dtype=torch.bool))
    output.square().sum().backward()
    assert valid.all()
    per_channel_gradient = features.grad.abs().sum(dim=(0, 2, 3))
    assert torch.all(per_channel_gradient > 0)


def test_spatial_mixer_returns_zero_for_no_valid_channels_after_training_biases():
    mixer = SpatialSlotMixer(embed_dim=8, num_slots=2, num_heads=2)
    mixer.layer_norm.bias.data.fill_(1)
    output, valid = mixer(
        torch.randn(1, 2, 3, 8), torch.zeros(1, 2, dtype=torch.bool)
    )
    assert not valid.any()
    assert torch.equal(output, torch.zeros_like(output))


def test_top_down_alignment_uses_receptive_field_overlap():
    align = AlignedGatedResidual(embed_dim=4).eval()
    fine = torch.randn(1, 1, 4)
    coarse = torch.randn(1, 1, 4)
    fine_intervals = torch.tensor([[[0.95, 1.05]]])
    coarse_intervals = torch.tensor([[[0.0, 1.0]]])

    overlapping, overlapping_intervals = align(
        fine,
        coarse,
        fine_rf_intervals=fine_intervals,
        coarse_rf_intervals=coarse_intervals,
        coarse_valid=torch.tensor([[True]]),
    )
    nonoverlapping, _ = align(
        fine,
        coarse,
        fine_rf_intervals=fine_intervals + 10,
        coarse_rf_intervals=coarse_intervals,
        coarse_valid=torch.tensor([[True]]),
    )

    assert not torch.equal(overlapping, fine)
    assert torch.equal(nonoverlapping, fine)
    assert torch.equal(overlapping_intervals, torch.tensor([[[0.0, 1.05]]]))


def test_top_down_zero_gate_preserves_nonzero_fine_residual():
    align = AlignedGatedResidual(embed_dim=4).eval()
    align.gate_proj.weight.data.zero_()
    align.gate_proj.bias.data.fill_(-100)
    fine = torch.randn(1, 3, 4)
    coarse = torch.randn(1, 2, 4)
    fine_intervals = torch.tensor([[[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]]])
    coarse_intervals = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])

    output, _ = align(
        fine,
        coarse,
        fine_rf_intervals=fine_intervals,
        coarse_rf_intervals=coarse_intervals,
    )

    assert fine.abs().sum() > 0
    assert torch.equal(output, fine)


def test_top_down_alignment_does_not_materialize_dense_pairwise_weights():
    align = AlignedGatedResidual(embed_dim=4).eval()
    fine = torch.randn(1, 64, 4)
    coarse = torch.randn(1, 16, 4)
    fine_starts = torch.arange(64, dtype=torch.float32) / 64
    coarse_starts = torch.arange(16, dtype=torch.float32) / 16
    fine_intervals = torch.stack([fine_starts, fine_starts + 1 / 64], dim=-1)[
        None
    ]
    coarse_intervals = torch.stack(
        [coarse_starts - 1 / 16, coarse_starts + 2 / 16], dim=-1
    )[None]

    with patch("torch.bmm", wraps=torch.bmm) as batch_matmul:
        align(
            fine,
            coarse,
            fine_rf_intervals=fine_intervals,
            coarse_rf_intervals=coarse_intervals,
        )

    dense_shape = (1, fine.shape[1], coarse.shape[1])
    assert all(
        call.args[0].shape != dense_shape
        for call in batch_matmul.call_args_list
    )


def test_reported_receptive_fields_contain_every_impulse_affected_token_at_all_levels():
    torch.manual_seed(4)
    impulse_model = HEROModel(
        task_configs={},
        num_channels=1,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=1,
    ).eval()
    baseline = torch.zeros(1, 1, 1024)
    impulse = baseline.clone()
    impulse_index = 512
    impulse[:, :, impulse_index] = 1

    def encode_with_internal_levels(signal):
        levels = {}

        def save_coarse(_module, args):
            levels["coarse"] = args[1].detach().clone()

        def save_mid(_module, args):
            levels["mid"] = args[1].detach().clone()

        coarse_hook = (
            impulse_model.encoder.coarse_to_mid_align.register_forward_pre_hook(
                save_coarse
            )
        )
        mid_hook = (
            impulse_model.encoder.mid_to_fine_align.register_forward_pre_hook(
                save_mid
            )
        )
        try:
            representation = encode(impulse_model, signal)
        finally:
            coarse_hook.remove()
            mid_hook.remove()
        levels["fine"] = representation.content
        return representation, levels

    zero_rep, zero_levels = encode_with_internal_levels(baseline)
    impulse_rep, impulse_levels = encode_with_internal_levels(impulse)
    impulse_time = (impulse_index + 0.5) / 128
    intervals_by_level = {
        "fine": impulse_rep.coverage.fine_rf_intervals,
        "mid": impulse_rep.coverage.mid_rf_intervals,
        "coarse": impulse_rep.coverage.coarse_rf_intervals,
    }

    for level in ("fine", "mid", "coarse"):
        affected = (zero_levels[level] - impulse_levels[level]).abs().amax(
            dim=-1
        ) > 1e-6
        intervals = intervals_by_level[level]
        assert affected.any(), f"impulse did not reach {level} level"
        assert torch.all(intervals[..., 0][affected] <= impulse_time)
        assert torch.all(intervals[..., 1][affected] >= impulse_time)


def test_default_attention_path_has_finite_gradients():
    torch.manual_seed(5)
    gradient_model = HEROModel(
        task_configs={},
        num_channels=2,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=2,
    )
    signal = torch.randn(1, 2, 128, requires_grad=True)

    representation = gradient_model.encode(signal=signal, sampling_rate=128)
    representation.content.square().mean().backward()

    assert signal.grad is not None
    assert torch.isfinite(signal.grad).all()
    assert gradient_model.encoder.fine_attns[0].qkv.weight.grad is not None


def test_duration_growth_and_channel_growth_only_change_expected_dimensions(
    model,
):
    short = encode(model, torch.randn(1, 2, 128))
    long = encode(model, torch.randn(1, 2, 256))
    many_channels = encode(model, torch.randn(1, 32, 128))

    assert long.content.shape[1] == 2 * short.content.shape[1]
    assert (
        long.coverage.mid_valid.shape[1]
        == 2 * short.coverage.mid_valid.shape[1]
    )
    assert many_channels.content.shape == short.content.shape


def test_batch_order_does_not_change_each_example(model):
    signal = torch.randn(2, 3, 128)
    original = encode(model, signal)
    reversed_batch = encode(model, signal.flip(0))
    assert torch.allclose(original.content, reversed_batch.content.flip(0))


def test_fixed_lowpass_suppresses_energy_above_the_canonical_band():
    reduction = TemporalReduction(
        embed_dim=1, num_temporal_slots=1, num_heads=1
    )
    indices = torch.arange(256, dtype=torch.float32)
    low = torch.sin(2 * torch.pi * 0.04 * indices)
    high = torch.sin(2 * torch.pi * 0.30 * indices)
    valid = torch.ones(1, 256, dtype=torch.bool)
    low_filtered = reduction._apply_lowpass(low[None, :, None], valid)
    high_filtered = reduction._apply_lowpass(high[None, :, None], valid)
    interior = slice(32, -32)
    low_rms = low_filtered[:, interior].square().mean().sqrt()
    high_rms = high_filtered[:, interior].square().mean().sqrt()
    assert high_rms < low_rms * 0.25


def test_task_query_attention_learns_task_specific_temporal_spans():
    decoder = TaskQueryCrossAttention(
        embed_dim=2,
        num_tasks=2,
        num_heads=1,
        query_chunk_size=1,
    ).eval()
    with torch.no_grad():
        decoder.task_queries.weight.zero_()
        decoder.q_proj.weight.zero_()
        decoder.k_proj.weight.zero_()
        decoder.v_proj.weight.copy_(torch.eye(2))
        decoder.out_proj.weight.copy_(torch.eye(2))
        decoder.out_proj.bias.zero_()
        for parameter in decoder.ffn.parameters():
            parameter.zero_()
        decoder.log_time_decay[0].fill_(5.0)
        decoder.log_time_decay[1].fill_(-20.0)

    content = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    decoded = decoder(
        content,
        torch.tensor([[1, 1, 2]]),
        content_timestamps=torch.tensor([[0.0, 10.0]]),
        output_timestamps=torch.tensor([[0.0, 10.0, 0.0]]),
    )

    # Task 1 is configured as a short-range read and follows the requested
    # timestamp. Task 2 has nearly global attention and averages the opposing
    # content even when queried at t=0.
    assert decoded[0, 0, 0] > 0.9
    assert decoded[0, 1, 0] < -0.9
    assert decoded[0, 2].abs().max() < 1e-3


def test_task_query_attention_respects_masks_and_chunking():
    torch.manual_seed(12)
    decoder = TaskQueryCrossAttention(
        embed_dim=8,
        num_tasks=2,
        num_heads=2,
        query_chunk_size=1,
    ).eval()
    content = torch.randn(2, 5, 8)
    timestamps = torch.arange(5, dtype=torch.float32).expand(2, -1)
    task_index = torch.tensor([[1, 2, 1], [2, 1, 2]])
    output_timestamps = torch.tensor([[0.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    chunked = decoder(
        content,
        task_index,
        content_timestamps=timestamps,
        output_timestamps=output_timestamps,
        content_valid=valid,
    )
    decoder.query_chunk_size = 32
    unchunked = decoder(
        content,
        task_index,
        content_timestamps=timestamps,
        output_timestamps=output_timestamps,
        content_valid=valid,
    )
    changed_padding = content.clone()
    changed_padding[0, 3:] = 1e6
    masked = decoder(
        changed_padding,
        task_index,
        content_timestamps=timestamps,
        output_timestamps=output_timestamps,
        content_valid=valid,
    )
    global_read = decoder(content, task_index, content_valid=valid)

    assert torch.allclose(chunked, unchunked, atol=1e-6)
    assert torch.allclose(unchunked, masked, atol=1e-6)
    assert global_read.shape == (2, 3, 8)
    assert torch.isfinite(global_read).all()


def test_two_level_lowpass_prevents_high_frequency_coarse_alias():
    reduction = TemporalReduction(
        embed_dim=1, num_temporal_slots=1, num_heads=1
    )
    indices = torch.arange(2048, dtype=torch.float32)
    low = torch.sin(2 * torch.pi * 2 * indices / 128)
    high = torch.sin(2 * torch.pi * 48 * indices / 128)

    def coarse_prefiltered(signal):
        valid = torch.ones(1, signal.numel(), dtype=torch.bool)
        mid = reduction._apply_lowpass(signal[None, :, None], valid)[:, 2::4]
        mid_valid = torch.ones(1, mid.shape[1], dtype=torch.bool)
        return reduction._apply_lowpass(mid, mid_valid)[:, 2::4]

    low_coarse = coarse_prefiltered(low)
    high_coarse = coarse_prefiltered(high)
    interior = slice(16, -16)
    low_rms = low_coarse[:, interior].square().mean().sqrt()
    high_rms = high_coarse[:, interior].square().mean().sqrt()
    assert high_rms < low_rms * 0.01


def test_tokenize_collate_and_forward_integration():
    task = TaskConfig.from_dict(
        {
            "name": "hero_test",
            "head": {
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "timestamp_key": "hero_test.timestamps",
                "value_key": "hero_test.values",
            },
            "loss": {"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        }
    )
    integration_model = HEROModel(
        task_configs={task.name: task},
        num_channels=4,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
    ).eval()

    eeg = RegularTimeSeries(
        signal=np.random.default_rng(3)
        .normal(size=(200, 3))
        .astype(np.float32),
        sampling_rate=100.0,
        domain_start=2.0,
    )
    data = Data(eeg=eeg, domain=Interval(2.0, 4.0))
    data.channels = type(
        "Channels",
        (),
        {
            "id": np.array(["a", "b", "c"]),
            "type": np.array(["EEG", "EEG", "EEG"]),
            "__len__": lambda self: 3,
        },
    )()
    data.session = type("Session", (), {"id": "session"})()
    data._absolute_start = 2.0
    data.hero_test = type(
        "HeroTarget",
        (),
        {"timestamps": np.array([3.0]), "values": np.array([1])},
    )()

    batch = collate([integration_model.tokenize(data)])
    assert batch["output_timestamps"].shape == batch["task_index"].shape
    with torch.no_grad():
        output = integration_model(
            input_values=batch["input_values"],
            input_timestamps=batch["input_timestamps"],
            output_timestamps=batch["output_timestamps"],
            channel_mask=batch["channel_mask"],
            sample_mask=batch["sample_mask"],
            task_index=batch["task_index"],
        )

    assert batch["input_values"].shape == (1, 4, 256)
    assert output.task_outputs["hero_test"].shape == (1, 2)

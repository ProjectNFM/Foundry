"""Contract tests for the hierarchical EEG representation encoder."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_brain.batching import collate
from torch_brain.data import Data, Interval, RegularTimeSeries

from foundry.models.hero import (
    AbsolutePositionEncoder,
    AlignedGatedResidual,
    CanonicalSignalViews,
    ChannelTypeEncoder,
    HEROModel,
    RelationalContextEncoder,
    SharedLocalChannelEncoder,
    SharedLocalContextEncoder,
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


def test_dilated_channel_encoder_has_expected_receptive_field_and_is_causal():
    torch.manual_seed(3)
    encoder = SharedLocalChannelEncoder(
        embed_dim=8,
        num_layers=4,
        kernel_size=7,
        dilations=[1, 2, 4, 8],
    ).eval()
    baseline = torch.randn(1, 1, 128)
    changed_future = baseline.clone()
    changed_future[..., 96:] += 100.0

    with torch.no_grad():
        baseline_output = encoder(baseline)
        changed_output = encoder(changed_future)

    assert encoder.left_paddings == (6, 12, 24, 48)
    assert encoder.receptive_field_samples == 91
    torch.testing.assert_close(
        baseline_output[..., :96], changed_output[..., :96]
    )


def test_channel_encoder_rejects_invalid_dilation_configuration():
    with pytest.raises(ValueError, match="one value per layer"):
        SharedLocalChannelEncoder(num_layers=3, dilations=[1, 2])
    with pytest.raises(ValueError, match="positive"):
        SharedLocalChannelEncoder(num_layers=2, dilations=[1, 0])


@pytest.fixture
def relational_context_model():
    """Small Phase-1 model with both local and relational context enabled."""
    torch.manual_seed(19)
    return HEROModel(
        task_configs={},
        num_channels=4,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
        channel_context_mode="relational",
        context_dim=8,
        context_encoder_layers=2,
        context_encoder_kernel_size=5,
        context_pool_factor=2,
        relational_context_blocks=2,
        relational_context_heads=2,
    ).eval()


def test_local_context_is_channel_independent_before_relational_attention(
    relational_context_model,
):
    signal = torch.randn(1, 3, 128)
    baseline = relational_context_model.encode_channel_context(
        signal=signal, sampling_rate=128
    )
    changed_signal = signal.clone()
    changed_signal[:, 1] = torch.randn_like(changed_signal[:, 1]) * 17 + 5
    changed = relational_context_model.encode_channel_context(
        signal=changed_signal, sampling_rate=128
    )

    assert baseline.local.shape == (1, 3, 8)
    assert baseline.context_width == 8
    assert torch.equal(baseline.local[:, 0], changed.local[:, 0])
    assert not torch.allclose(baseline.local[:, 1], changed.local[:, 1])
    assert baseline.relational is not None
    assert changed.relational is not None
    assert not torch.allclose(
        baseline.relational[:, 0], changed.relational[:, 0]
    )


def test_context_masks_block_padded_channels_and_all_masked_rows_are_finite(
    relational_context_model,
):
    signal = torch.randn(2, 4, 128)
    channel_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )
    sample_mask = torch.ones(2, 128, dtype=torch.bool)
    sample_mask[0, 40:56] = False
    baseline = relational_context_model.encode_channel_context(
        signal=signal,
        sampling_rate=128,
        channel_mask=channel_mask,
        sample_mask=sample_mask,
    )

    corrupted = signal.clone()
    corrupted[0, 3] = float("nan")
    corrupted[0, :3, 40:56] = float("inf")
    corrupted[1] = 1e20
    changed = relational_context_model.encode_channel_context(
        signal=corrupted,
        sampling_rate=128,
        channel_mask=channel_mask,
        sample_mask=sample_mask,
    )

    assert torch.equal(baseline.local, changed.local)
    assert torch.equal(baseline.relational, changed.relational)
    assert torch.equal(baseline.local_attention, changed.local_attention)
    assert torch.equal(
        baseline.relational_attention, changed.relational_attention
    )
    assert torch.equal(baseline.local[:, 3], torch.zeros(2, 8))
    assert torch.equal(baseline.relational[:, 3], torch.zeros(2, 8))
    assert torch.equal(baseline.local[1], torch.zeros(4, 8))
    assert torch.equal(baseline.relational[1], torch.zeros(4, 8))
    assert torch.isfinite(baseline.local).all()
    assert torch.isfinite(baseline.relational).all()


def test_relational_context_and_diagnostics_are_permutation_equivariant(
    relational_context_model,
):
    signal = torch.randn(1, 4, 128)
    channel_mask = torch.tensor([[True, True, False, True]])
    baseline = relational_context_model.encode_channel_context(
        signal=signal,
        sampling_rate=128,
        channel_mask=channel_mask,
    )
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = relational_context_model.encode_channel_context(
        signal=signal[:, permutation],
        sampling_rate=128,
        channel_mask=channel_mask[:, permutation],
    )

    assert baseline.local.ndim == 3
    assert baseline.relational.ndim == 3
    assert torch.allclose(permuted.local, baseline.local[:, permutation])
    assert torch.allclose(
        permuted.relational, baseline.relational[:, permutation], atol=2e-6
    )
    assert torch.allclose(
        permuted.local_attention,
        baseline.local_attention[:, permutation],
    )
    assert torch.allclose(
        baseline.local_attention[0, channel_mask[0]].sum(dim=-1),
        torch.ones(channel_mask[0].sum()),
    )
    assert torch.equal(
        baseline.local_attention[0, ~channel_mask[0]],
        torch.zeros_like(baseline.local_attention[0, ~channel_mask[0]]),
    )
    expected_attention = baseline.relational_attention.index_select(
        -2, permutation
    ).index_select(-1, permutation)
    assert torch.allclose(
        permuted.relational_attention, expected_attention, atol=2e-6
    )

    valid_attention = baseline.relational_attention[0, :, :, channel_mask[0]]
    assert torch.allclose(
        valid_attention.sum(dim=-1),
        torch.ones_like(valid_attention.sum(dim=-1)),
    )
    assert torch.equal(
        baseline.relational_attention[..., ~channel_mask[0], :],
        torch.zeros_like(
            baseline.relational_attention[..., ~channel_mask[0], :]
        ),
    )


def test_context_gradients_and_full_model_are_finite_across_raw_scales():
    torch.manual_seed(23)
    context_model = HEROModel(
        task_configs={},
        num_channels=3,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=1,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
        channel_context_mode="relational",
        context_dim=8,
        context_encoder_layers=2,
        relational_context_blocks=2,
        relational_context_heads=2,
    )
    scales = torch.tensor([1e-4, 1.0, 1e4]).view(1, 3, 1)
    signal = (torch.randn(1, 3, 128) * scales).requires_grad_()

    context = context_model.encode_channel_context(
        signal=signal, sampling_rate=128
    )
    representation = context_model.encode(signal=signal, sampling_rate=128)
    feature_weights = torch.arange(1, 9, dtype=signal.dtype)
    loss = (
        (context.local * feature_weights).mean()
        + (context.relational * feature_weights.flip(0)).mean()
        + representation.content.square().mean()
    )
    loss.backward()

    local_grad = context_model.channel_context_encoder.local_encoder.convs[
        0
    ].weight.grad
    relation_grad = (
        context_model.channel_context_encoder.relational_encoder.blocks[
            0
        ].qkv.weight.grad
    )
    assert torch.isfinite(loss)
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert local_grad is not None and torch.isfinite(local_grad).all()
    assert relation_grad is not None and torch.isfinite(relation_grad).all()
    assert local_grad.abs().sum() > 0
    assert relation_grad.abs().sum() > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in context_model.parameters()
    )


def test_context_modes_are_explicit_and_validate_dimensions():
    disabled = HEROModel(
        task_configs={}, num_channels=2, embed_dim=8, num_attn_heads=2
    )
    assert disabled.channel_context_encoder is None
    with pytest.raises(RuntimeError, match="Channel context is disabled"):
        disabled.encode_channel_context(
            signal=torch.randn(1, 2, 128), sampling_rate=128
        )

    local = HEROModel(
        task_configs={},
        num_channels=2,
        embed_dim=8,
        num_attn_heads=2,
        channel_context_mode="local",
        context_dim=8,
    )
    local_context = local.encode_channel_context(
        signal=torch.randn(1, 2, 128), sampling_rate=128
    )
    assert local_context.relational is None
    assert local_context.relational_attention is None

    with pytest.raises(ValueError, match="context_dim"):
        RelationalContextEncoder(context_dim=7, num_heads=2)
    with pytest.raises(ValueError, match="kernel_size"):
        SharedLocalContextEncoder(kernel_size=4)


def _small_routing_model(
    *,
    mode="relational_position",
    channel_type_enabled=True,
):
    torch.manual_seed(31)
    return HEROModel(
        task_configs={},
        num_channels=4,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=2,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
        temporal_mode="flat",
        channel_context_mode=mode,
        context_dim=8,
        context_encoder_layers=2,
        context_pool_factor=2,
        relational_context_blocks=2,
        relational_context_heads=2,
        channel_type_enabled=channel_type_enabled,
        position_num_fourier_bands=3,
    ).eval()


def _set_routing_gates(model, **values):
    with torch.no_grad():
        for source, value in values.items():
            model.spatial_mixer.context_gates[f"source_{source}"].fill_(value)


def test_static_metadata_encoders_are_masked_and_permutation_equivariant():
    torch.manual_seed(29)
    type_encoder = ChannelTypeEncoder(context_dim=8)
    position_encoder = AbsolutePositionEncoder(
        context_dim=8, num_fourier_bands=3
    )
    channel_type = torch.tensor([[1, 2, 3, 0]])
    position = torch.randn(1, 4, 3)
    position_valid = torch.tensor([[True, False, True, False]])
    channel_mask = torch.tensor([[True, True, True, False]])
    permutation = torch.tensor([2, 0, 3, 1])

    type_context = type_encoder(channel_type, channel_mask)
    position_context = position_encoder(position, position_valid, channel_mask)
    permuted_type = type_encoder(
        channel_type[:, permutation], channel_mask[:, permutation]
    )
    permuted_position = position_encoder(
        position[:, permutation],
        position_valid[:, permutation],
        channel_mask[:, permutation],
    )

    assert torch.equal(permuted_type, type_context[:, permutation])
    assert torch.equal(permuted_position, position_context[:, permutation])
    assert torch.equal(type_context[:, 3], torch.zeros(1, 8))
    assert torch.equal(position_context[:, 3], torch.zeros(1, 8))

    arbitrary_missing = position.clone()
    arbitrary_missing[:, 1] = 1e6
    changed_missing = position_encoder(
        arbitrary_missing, position_valid, channel_mask
    )
    assert torch.equal(position_context[:, 1], changed_missing[:, 1])
    real_zero = position.clone()
    real_zero[:, 1] = 0
    real_zero_context = position_encoder(
        real_zero,
        torch.tensor([[True, True, True, False]]),
        channel_mask,
    )
    assert not torch.equal(position_context[:, 1], real_zero_context[:, 1])


def test_tokenize_maps_types_resolves_montage_and_ignores_ambiguous_positions():
    tokenizer_model = HEROModel(
        task_configs={},
        num_channels=5,
        embed_dim=8,
        num_attn_heads=2,
        num_local_attn_blocks=0,
    )
    eeg = RegularTimeSeries(
        signal=np.random.default_rng(4)
        .normal(size=(128, 4))
        .astype(np.float32),
        sampling_rate=128.0,
    )
    data = Data(eeg=eeg, domain=Interval(0.0, 1.0))
    data.channels = type(
        "Channels",
        (),
        {
            "id": np.array(["s/C3", "s/not-a-site", "s/G1", "s/D1"]),
            "type": np.array(["EEG", "EEG", "ECoG", "SEEG"]),
            "position": np.ones((4, 3), dtype=np.float32),
            "position_valid": np.ones(4, dtype=bool),
            "position_frame": np.array(["head", "mri", "head", "head"]),
            "position_units": np.array(["m"] * 4),
            "__len__": lambda self: 4,
        },
    )()
    data.session = type("Session", (), {"id": "session"})()
    data._absolute_start = 0.0

    tokenized = tokenizer_model.tokenize(data)

    assert torch.equal(tokenized["channel_type"], torch.tensor([1, 1, 2, 3, 0]))
    assert torch.equal(
        tokenized["channel_mask"],
        torch.tensor([True, True, True, True, False]),
    )
    # Mixed coordinate frames invalidate explicit positions. C3 is then
    # resolved numerically from the standard montage; unresolved EEG and iEEG
    # channels remain valid signals with missing positions.
    assert torch.equal(
        tokenized["channel_position_valid"],
        torch.tensor([True, False, False, False, False]),
    )
    assert torch.isfinite(tokenized["channel_position"]).all()
    assert tokenized["channel_position"][0].abs().sum() > 0
    assert torch.equal(tokenized["channel_position"][1:], torch.zeros(4, 3))
    assert "channel_name" not in tokenized


def test_zero_context_gates_exactly_reproduce_signal_only_model():
    signal_only = _small_routing_model(
        mode="disabled", channel_type_enabled=False
    )
    context_model = _small_routing_model()
    context_state = context_model.state_dict()
    for name, value in signal_only.state_dict().items():
        if name in context_state and context_state[name].shape == value.shape:
            context_state[name] = value
    context_model.load_state_dict(context_state)

    signal = torch.randn(1, 4, 128)
    channel_type = torch.tensor([[1, 1, 2, 3]])
    position = torch.randn(1, 4, 3)
    position_valid = torch.tensor([[True, False, True, True]])
    with torch.no_grad():
        baseline = signal_only.encode(signal=signal, sampling_rate=128)
        contextual = context_model.encode(
            signal=signal,
            sampling_rate=128,
            channel_type=channel_type,
            channel_position=position,
            channel_position_valid=position_valid,
        )

    assert torch.equal(baseline.content, contextual.content)
    assert all(
        torch.equal(value, torch.zeros_like(value))
        for value in contextual.spatial_routing.gate_values.values()
    )


def test_context_gate_initialization_is_configurable():
    mixer = SpatialSlotMixer(
        embed_dim=8,
        num_slots=2,
        num_heads=2,
        context_dim=8,
        context_sources=("position",),
        context_gate_init=0.1,
    )

    torch.testing.assert_close(
        mixer.context_gates["source_position"], torch.full((2,), 0.1)
    )


def test_all_context_sources_preserve_joint_channel_permutation_invariance():
    routing_model = _small_routing_model()
    _set_routing_gates(
        routing_model, local=0.4, relational=-0.7, type=0.3, position=0.9
    )
    signal = torch.randn(1, 4, 128)
    channel_mask = torch.tensor([[True, True, False, True]])
    channel_type = torch.tensor([[1, 2, 0, 3]])
    position = torch.randn(1, 4, 3)
    position_valid = torch.tensor([[True, False, False, True]])
    permutation = torch.tensor([3, 1, 0, 2])

    with torch.no_grad():
        baseline = routing_model.encode(
            signal=signal,
            sampling_rate=128,
            channel_mask=channel_mask,
            channel_type=channel_type,
            channel_position=position,
            channel_position_valid=position_valid,
        )
        permuted = routing_model.encode(
            signal=signal[:, permutation],
            sampling_rate=128,
            channel_mask=channel_mask[:, permutation],
            channel_type=channel_type[:, permutation],
            channel_position=position[:, permutation],
            channel_position_valid=position_valid[:, permutation],
        )

    assert torch.allclose(baseline.content, permuted.content, atol=2e-5)
    expected_attention = baseline.spatial_routing.attention_mean.index_select(
        -1, permutation
    )
    assert torch.allclose(
        permuted.spatial_routing.attention_mean,
        expected_attention,
        atol=2e-6,
    )


def test_relational_shuffling_changes_routing_and_validates_hook():
    routing_model = _small_routing_model(
        mode="relational", channel_type_enabled=False
    )
    _set_routing_gates(routing_model, local=0.0, relational=4.0)
    signal = torch.randn(1, 4, 128)
    permutation = torch.tensor([2, 3, 0, 1])

    with torch.no_grad():
        bound = routing_model.encode(signal=signal, sampling_rate=128)
        shuffled = routing_model.encode(
            signal=signal,
            sampling_rate=128,
            relational_context_permutation=permutation,
        )

    assert not torch.allclose(bound.content, shuffled.content)
    assert not torch.allclose(
        bound.spatial_routing.attention_mean,
        shuffled.spatial_routing.attention_mean,
    )
    with pytest.raises(ValueError, match="permute every channel"):
        routing_model.encode(
            signal=signal,
            sampling_rate=128,
            relational_context_permutation=torch.tensor([0, 0, 1, 2]),
        )


def test_context_is_routing_only_and_diagnostics_are_separate_by_source():
    torch.manual_seed(37)
    mixer = SpatialSlotMixer(
        embed_dim=8,
        num_slots=2,
        num_heads=2,
        context_dim=8,
        context_sources=("local", "relational", "type", "position"),
    ).eval()
    with torch.no_grad():
        for gate in mixer.context_gates.values():
            gate.fill_(2.0)
    # Every channel has the same normalized content value. Routing weights can
    # change, but no context can become a slot value.
    one_channel = torch.randn(1, 1, 16, 8)
    content = one_channel.expand(1, 3, 16, 8).clone()
    zeros = {source: torch.zeros(1, 3, 8) for source in mixer.context_sources}
    varied = {source: torch.randn(1, 3, 8) for source in mixer.context_sources}
    channel_mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        baseline, _, baseline_diag = mixer(
            content,
            channel_mask,
            routing_context=zeros,
            return_diagnostics=True,
        )
        routed, _, routed_diag = mixer(
            content,
            channel_mask,
            routing_context=varied,
            return_diagnostics=True,
        )

    assert torch.allclose(baseline, routed, atol=2e-6)
    assert not torch.allclose(
        baseline_diag.attention_mean, routed_diag.attention_mean
    )
    assert set(routed_diag.gate_values) == {
        "local",
        "relational",
        "type",
        "position",
    }
    assert set(routed_diag.logit_rms) == {
        "content",
        "local",
        "relational",
        "type",
        "position",
    }
    assert routed_diag.attention_mean.shape == (1, 2, 2, 3)


def test_missing_positions_are_absent_from_routing_not_shared_coordinates():
    position_model = _small_routing_model(
        mode="position", channel_type_enabled=False
    )
    _set_routing_gates(position_model, position=3.0)
    signal = torch.randn(1, 4, 128)
    arbitrary = torch.randn(1, 4, 3) * 1e6
    missing = torch.zeros(1, 4, dtype=torch.bool)
    one_real = torch.tensor([[True, False, False, False]])

    with torch.no_grad():
        absent_a = position_model.encode(
            signal=signal,
            sampling_rate=128,
            channel_position=arbitrary,
            channel_position_valid=missing,
        )
        absent_b = position_model.encode(
            signal=signal,
            sampling_rate=128,
            channel_position=torch.zeros_like(arbitrary),
            channel_position_valid=missing,
        )
        anchored = position_model.encode(
            signal=signal,
            sampling_rate=128,
            channel_position=torch.zeros_like(arbitrary),
            channel_position_valid=one_real,
        )

    assert torch.equal(absent_a.content, absent_b.content)
    assert not torch.allclose(absent_a.content, anchored.content)


def test_relational_cost_is_quadratic_and_duration_independent():
    relational = _small_routing_model(
        mode="relational", channel_type_enabled=False
    )
    cost_16 = relational.estimate_relational_context_cost(16, batch_size=2)
    cost_64 = relational.estimate_relational_context_cost(64, batch_size=2)
    assert cost_64["attention_elements"] == 16 * cost_16["attention_elements"]
    assert cost_64["multiply_adds"] == 16 * cost_16["multiply_adds"]
    assert "duration" not in cost_64

    disabled = _small_routing_model(mode="disabled", channel_type_enabled=False)
    assert disabled.estimate_relational_context_cost(64) == {
        "attention_elements": 0,
        "multiply_adds": 0,
    }


def test_routing_gradients_reach_every_active_source_and_gate():
    routing_model = _small_routing_model().train()
    _set_routing_gates(
        routing_model, local=0.5, relational=0.5, type=0.5, position=0.5
    )
    signal = torch.randn(1, 4, 128, requires_grad=True)
    channel_type = torch.tensor([[1, 2, 3, 4]])
    position = torch.randn(1, 4, 3)
    position_valid = torch.ones(1, 4, dtype=torch.bool)

    representation = routing_model.encode(
        signal=signal,
        sampling_rate=128,
        channel_type=channel_type,
        channel_position=position,
        channel_position_valid=position_valid,
    )
    weights = torch.arange(1, 9, dtype=signal.dtype)
    (representation.content * weights).mean().backward()

    parameters = [
        routing_model.channel_context_encoder.local_encoder.convs[0].weight,
        routing_model.channel_context_encoder.relational_encoder.blocks[
            0
        ].qkv.weight,
        routing_model.channel_type_encoder.embedding.weight,
        routing_model.position_encoder.mlp[0].weight,
        routing_model.spatial_mixer.context_k_proj["source_relational"].weight,
        routing_model.spatial_mixer.context_gates["source_position"],
    ]
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_context_does_not_change_post_fusion_temporal_scaling():
    routing_model = _small_routing_model(
        mode="relational", channel_type_enabled=False
    )
    _set_routing_gates(routing_model, local=0.5, relational=0.5)
    relational_cost = routing_model.estimate_relational_context_cost(4)
    with torch.no_grad():
        short = routing_model.encode(
            signal=torch.randn(1, 4, 128), sampling_rate=128
        )
        long = routing_model.encode(
            signal=torch.randn(1, 4, 512), sampling_rate=128
        )

    assert short.content.shape == (1, 128, 8)
    assert long.content.shape == (1, 512, 8)
    assert len(routing_model.encoder.fine_attns) == 0
    assert routing_model.estimate_relational_context_cost(4) == relational_cost


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


def test_single_resample_views_share_grid_masks_and_source_statistics(model):
    time = torch.arange(512, dtype=torch.float32)
    signal = torch.stack(
        [10 + torch.sin(time / 9), -3 + 2 * torch.cos(time / 13), time],
        dim=0,
    ).unsqueeze(0)
    channel_mask = torch.tensor([[True, True, False]])
    sample_mask = torch.ones(1, 512, dtype=torch.bool)
    sample_mask[:, 300] = False
    signal[:, 0, 200] = float("nan")
    signal[:, 2] = float("inf")

    with patch.object(
        model, "_resample_signal", wraps=model._resample_signal
    ) as resample:
        views = model._prepare_signal_views(
            signal,
            channel_mask,
            sample_mask,
            256,
            domain_start=torch.tensor([[2.0]]),
        )

    assert resample.call_count == 1
    assert views.raw.shape == views.content.shape == (1, 3, 256)
    assert views.timestamps.shape == views.sample_mask.shape == (1, 256)
    assert torch.allclose(
        views.timestamps,
        2 + (torch.arange(256, dtype=torch.float32) + 0.5) / 128,
    )
    valid = views.sample_mask[:, None] & channel_mask[:, :, None]
    assert torch.equal(
        views.raw.masked_select(~valid),
        torch.zeros_like(views.raw.masked_select(~valid)),
    )
    assert torch.equal(
        views.content.masked_select(~valid),
        torch.zeros_like(views.content.masked_select(~valid)),
    )
    expected_content = (
        views.raw - views.source_mean.unsqueeze(-1)
    ) / views.source_std.unsqueeze(-1)
    assert torch.allclose(
        views.content.masked_select(valid),
        expected_content.masked_select(valid),
    )
    source_valid = sample_mask[0] & torch.isfinite(signal[0, :2]).all(dim=0)
    expected_source = signal[0, :2, source_valid]
    assert torch.allclose(views.source_mean[0, :2], expected_source.mean(dim=1))
    assert torch.allclose(
        views.source_std[0, :2],
        expected_source.std(dim=1, correction=0),
    )
    assert torch.equal(views.source_mean[:, 2], torch.zeros(1))
    assert torch.equal(views.source_std[:, 2], torch.ones(1))
    assert torch.isfinite(views.raw).all()
    assert torch.isfinite(views.content).all()

    changed_invalid = signal.clone()
    changed_invalid[:, 0, 200] = float("-inf")
    changed_invalid[:, :, 300] = 1e20
    changed_invalid[:, 2] = -1e20
    changed = model._prepare_signal_views(
        changed_invalid,
        channel_mask,
        sample_mask,
        256,
        domain_start=torch.tensor([[2.0]]),
    )
    assert torch.equal(views.sample_mask, changed.sample_mask)
    assert torch.allclose(views.raw, changed.raw)
    assert torch.allclose(views.content, changed.content)

    corrupted_raw = views.raw.clone()
    corrupted_content = views.content.clone()
    corrupted_raw[:, 2] = float("nan")
    corrupted_content[:, 2] = float("nan")
    invalid_time = (~views.sample_mask[0]).nonzero()[0, 0]
    corrupted_raw[:, :2, invalid_time] = float("inf")
    corrupted_content[:, :2, invalid_time] = float("-inf")
    corrupted_views = CanonicalSignalViews(
        raw=corrupted_raw,
        content=corrupted_content,
        timestamps=views.timestamps,
        sample_mask=views.sample_mask,
        source_mean=views.source_mean,
        source_std=views.source_std,
    )
    with torch.no_grad():
        baseline_rep = model._encode_prepared_views(views, channel_mask)
        corrupted_rep = model._encode_prepared_views(
            corrupted_views, channel_mask
        )
    assert torch.allclose(baseline_rep.content, corrupted_rep.content)


def test_refactored_normalization_matches_v1_interior_and_trains(model):
    time = torch.arange(512, dtype=torch.float32) / 256
    raw = (
        7
        + 0.4 * torch.sin(2 * torch.pi * 5 * time)
        + 0.2 * torch.cos(2 * torch.pi * 17 * time)
    )[None, None]
    channel_mask = torch.ones(1, 1, dtype=torch.bool)
    sample_mask = torch.ones(1, 512, dtype=torch.bool)
    views = model._prepare_signal_views(
        raw,
        channel_mask,
        sample_mask,
        256,
        domain_start=torch.zeros(1, 1),
    )

    source_mean = raw.mean(dim=-1, keepdim=True)
    source_std = raw.std(dim=-1, correction=0, keepdim=True)
    legacy_source = (raw - source_mean) / source_std
    legacy_content, legacy_mask = model._resample_signal(
        legacy_source, sample_mask, 256, 128
    )

    assert torch.equal(views.sample_mask, legacy_mask)
    assert not torch.equal(views.content, legacy_content)
    interior_error = (
        views.content[:, :, 24:-24] - legacy_content[:, :, 24:-24]
    ).abs()
    assert interior_error.mean() < 0.02
    assert interior_error.max() < 0.02

    legacy_views = CanonicalSignalViews(
        raw=views.raw,
        content=legacy_content,
        timestamps=views.timestamps,
        sample_mask=legacy_mask,
        source_mean=views.source_mean,
        source_std=views.source_std,
    )
    with torch.no_grad():
        refactored_rep = model._encode_prepared_views(
            views, channel_mask
        ).content
        legacy_rep = model._encode_prepared_views(
            legacy_views, channel_mask
        ).content
    representation_error = (refactored_rep - legacy_rep).abs()
    assert 0 < representation_error.mean() < 0.1
    assert torch.isfinite(representation_error).all()

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for prepared in (legacy_views, views):
        optimizer.zero_grad()
        loss = (
            model._encode_prepared_views(prepared, channel_mask)
            .content.square()
            .mean()
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
    assert all(
        torch.isfinite(parameter).all() for parameter in model.parameters()
    )


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


def test_no_temporal_slots_uses_mask_aware_mean_reduction():
    torch.manual_seed(13)
    reduction = TemporalReduction(
        embed_dim=8,
        num_temporal_slots=2,
        num_heads=2,
        aggregation="mean",
    ).eval()
    x = torch.randn(1, 128, 8)
    valid = torch.ones(1, 128, dtype=torch.bool)
    valid[:, 64] = False

    with torch.no_grad():
        reduced, timestamps, reduced_valid, support = reduction(
            x, torch.arange(128)[None] / 128, valid
        )

    assert reduction.aggregation == "mean"
    assert not hasattr(reduction, "slot_queries")
    assert not hasattr(reduction, "gate_proj")
    assert reduced.shape == (1, 32, 8)
    assert timestamps.shape == reduced_valid.shape == support.shape == (1, 32)
    assert torch.isfinite(reduced).all()


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


def test_top_down_add_control_uses_ungated_aligned_residual():
    align = AlignedGatedResidual(embed_dim=4, fusion="add").eval()
    fine = torch.zeros(1, 1, 4)
    coarse = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
    intervals = torch.tensor([[[0.0, 1.0]]])

    output, output_intervals = align(
        fine,
        coarse,
        fine_rf_intervals=intervals,
        coarse_rf_intervals=intervals,
    )

    expected = align.proj(align.layer_norm(coarse))
    assert align.fusion == "add"
    assert not hasattr(align, "gate_proj")
    assert torch.allclose(output, expected)
    assert torch.equal(output_intervals, intervals)


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
            B, C, T = signal.shape
            representation = impulse_model._encode_prepared_views(
                CanonicalSignalViews(
                    raw=signal,
                    content=signal,
                    timestamps=(
                        (torch.arange(T, dtype=signal.dtype) + 0.5) / 128
                    ).expand(B, -1),
                    sample_mask=torch.ones(B, T, dtype=torch.bool),
                    source_mean=torch.zeros(B, C),
                    source_std=torch.ones(B, C),
                ),
                torch.ones(B, C, dtype=torch.bool),
            )
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


def test_flat_control_reuses_prefusion_path_and_has_configurable_depth():
    flat = HEROModel(
        task_configs={},
        num_channels=4,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=1,
        num_local_attn_blocks=1,
        temporal_mode="flat",
        flat_num_local_attn_blocks=3,
    ).eval()

    representation = encode(flat, torch.randn(1, 4, 128))

    assert flat.temporal_mode == "flat"
    assert flat.spatial_mixer.num_slots == 1
    assert len(flat.encoder.fine_attns) == 3
    assert not hasattr(flat.encoder, "fine_to_mid")
    assert representation.content.shape == (1, 128, 8)
    assert representation.coverage.mid_valid.shape == (1, 0)
    assert representation.coverage.coarse_valid.shape == (1, 0)
    assert representation.coverage.mid_rf_intervals.shape == (1, 0, 2)
    assert representation.coverage.coarse_rf_intervals.shape == (1, 0, 2)


def test_hierarchical_ablation_options_compose():
    ablation = HEROModel(
        task_configs={},
        num_channels=2,
        embed_dim=8,
        num_attn_heads=2,
        num_spatial_slots=1,
        num_temporal_slots=2,
        num_local_attn_blocks=0,
        temporal_reduction="mean",
        top_down_fusion="add",
    ).eval()

    representation = encode(ablation, torch.randn(1, 2, 128))

    assert ablation.encoder.fine_to_mid.aggregation == "mean"
    assert ablation.encoder.mid_to_coarse.aggregation == "mean"
    assert ablation.encoder.coarse_to_mid_align.fusion == "add"
    assert ablation.encoder.mid_to_fine_align.fusion == "add"
    assert representation.content.shape == (1, 128, 8)
    assert torch.isfinite(representation.content).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temporal_mode": "global"}, "temporal_mode"),
        ({"flat_num_local_attn_blocks": -1}, "flat_num_local_attn_blocks"),
        ({"temporal_reduction": "max"}, "temporal_reduction"),
        ({"top_down_fusion": "replace"}, "top_down_fusion"),
        ({"channel_context_mode": "global"}, "channel_context_mode"),
        (
            {
                "channel_context_mode": "relational",
                "relational_context_blocks": 0,
            },
            "relational_context_blocks",
        ),
    ],
)
def test_invalid_experiment_ladder_options_fail_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HEROModel(
            task_configs={},
            num_channels=2,
            embed_dim=8,
            num_attn_heads=2,
            **kwargs,
        )


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

    with patch.object(
        integration_model,
        "_resolve_signal_source",
        wraps=integration_model._resolve_signal_source,
    ) as resolve_signal:
        batch = collate([integration_model.tokenize(data)])
    assert resolve_signal.call_count == 1
    assert batch["output_timestamps"].shape == batch["task_index"].shape
    with torch.no_grad():
        output = integration_model(
            input_values=batch["input_values"],
            input_content_values=batch["input_content_values"],
            input_normalization_mean=batch["input_normalization_mean"],
            input_normalization_std=batch["input_normalization_std"],
            input_timestamps=batch["input_timestamps"],
            output_timestamps=batch["output_timestamps"],
            channel_mask=batch["channel_mask"],
            channel_type=batch["channel_type"],
            channel_position=batch["channel_position"],
            channel_position_valid=batch["channel_position_valid"],
            sample_mask=batch["sample_mask"],
            task_index=batch["task_index"],
        )

    assert batch["input_values"].shape == (1, 4, 256)
    assert batch["input_content_values"].shape == (1, 4, 256)
    assert batch["input_normalization_mean"].shape == (1, 4)
    assert batch["input_normalization_std"].shape == (1, 4)
    assert batch["channel_type"].shape == (1, 4)
    assert batch["channel_position"].shape == (1, 4, 3)
    assert batch["channel_position_valid"].shape == (1, 4)
    valid = batch["sample_mask"][:, None] & batch["channel_mask"][:, :, None]
    expected_content = (
        batch["input_values"] - batch["input_normalization_mean"].unsqueeze(-1)
    ) / batch["input_normalization_std"].unsqueeze(-1)
    assert torch.allclose(
        batch["input_content_values"].masked_select(valid),
        expected_content.masked_select(valid),
    )
    assert torch.equal(
        batch["input_values"].masked_select(~valid),
        torch.zeros_like(batch["input_values"].masked_select(~valid)),
    )
    assert torch.equal(
        batch["input_content_values"].masked_select(~valid),
        torch.zeros_like(batch["input_content_values"].masked_select(~valid)),
    )
    assert output.task_outputs["hero_test"].shape == (1, 2)
    assert output.diagnostics is not None
    assert "hero/routing/content_logit_rms_head0" in output.diagnostics
    assert "hero/routing/attention_entropy" in output.diagnostics
    assert "hero/routing/channel_00_attention" in output.diagnostics

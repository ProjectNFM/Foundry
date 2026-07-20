"""Tests for validated, atomic pretrained weight transfer.

Covers:
- Uncompiled and compiled Lightning checkpoints
- Missing state_dict, empty transfer, missing expected key, unexpected key,
  shape mismatch, dtype mismatch, and prefix collision
- Atomicity: destination state unchanged after every validation failure
- Freeze applies only to successfully transferred parameters
- Tiny real MaskedPOYOEEGModel → POYOEEGModel transfer
- Dataset/session/task-specific embeddings remain freshly initialized
- Permissive mode degrades gracefully
- TransferReport structure and summary
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from foundry.training.pretrained import (
    PretrainedTransferError,
    TransferMode,
    TransferReport,
    _normalize_checkpoint_keys,
    _validate_transfer,
    load_pretrained_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    """Minimal model mirroring POYOEEGModel submodule names."""

    _TRANSFERABLE_COMPONENTS = (
        "tokenizer",
        "backbone",
        "rotary_emb",
        "latent_emb",
    )

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.tokenizer = nn.Linear(4, embed_dim)
        self.backbone = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = nn.Linear(embed_dim, embed_dim)
        self.latent_emb = nn.Embedding(4, embed_dim)
        self.session_emb = nn.Linear(embed_dim, embed_dim)
        self.channel_emb = nn.Linear(embed_dim, embed_dim)
        self.task_emb = nn.Embedding(2, embed_dim)
        self.router = nn.Linear(embed_dim, 5)

    def transferable_components(self) -> tuple[str, ...]:
        return self._TRANSFERABLE_COMPONENTS


class _NoTransferModel(nn.Module):
    """Model without transferable_components method."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


def _save_lightning_ckpt(
    model: nn.Module,
    path: Path,
    compiled: bool = False,
    extra_state: dict | None = None,
) -> None:
    """Create a Lightning-style checkpoint file."""
    prefix = "model._orig_mod." if compiled else "model."
    state_dict = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
    ckpt = {"state_dict": state_dict}
    if extra_state:
        ckpt.update(extra_state)
    torch.save(ckpt, path)


def _snapshot_state(model: nn.Module) -> dict[str, object]:
    """Deep-copy all parameters, buffers, and vocab entries for comparison."""
    import copy as _copy_mod

    result = {}
    for k, v in model.state_dict().items():
        result[k] = (
            v.clone() if isinstance(v, torch.Tensor) else _copy_mod.deepcopy(v)
        )
    return result


def _assert_states_equal(
    before: dict[str, object],
    after: dict[str, object],
    msg: str = "",
) -> None:
    """Assert that two snapshots are identical (handles tensors and non-tensors)."""
    assert before.keys() == after.keys(), f"Key sets differ {msg}"
    for key in before:
        b, a = before[key], after[key]
        if isinstance(b, torch.Tensor) and isinstance(a, torch.Tensor):
            assert torch.equal(b, a), f"State changed for {key} {msg}"
        else:
            assert b == a, f"State changed for {key} {msg}"


# ---------------------------------------------------------------------------
# TransferReport
# ---------------------------------------------------------------------------


class TestTransferReport:
    def test_summary_format(self):
        report = TransferReport(
            loaded=["a", "b"],
            skipped_excluded=["c"],
            missing_in_checkpoint=["d"],
        )
        s = report.summary()
        assert "loaded:" in s
        assert "2" in s

    def test_has_errors_on_missing(self):
        report = TransferReport(missing_in_checkpoint=["x"])
        assert report.has_errors

    def test_has_errors_on_shape(self):
        report = TransferReport(shape_mismatched=["x: (2,) vs (3,)"])
        assert report.has_errors

    def test_has_errors_on_dtype(self):
        report = TransferReport(dtype_mismatched=["x: float32 vs float16"])
        assert report.has_errors

    def test_no_errors_clean(self):
        report = TransferReport(loaded=["a"], skipped_excluded=["b"])
        assert not report.has_errors


# ---------------------------------------------------------------------------
# Checkpoint key normalization
# ---------------------------------------------------------------------------


class TestNormalizeCheckpointKeys:
    def test_strips_model_prefix(self):
        state = {"model.tokenizer.weight": torch.zeros(2)}
        result = _normalize_checkpoint_keys(state)
        assert "tokenizer.weight" in result

    def test_strips_compiled_prefix(self):
        state = {"model._orig_mod.tokenizer.weight": torch.zeros(2)}
        result = _normalize_checkpoint_keys(state)
        assert "tokenizer.weight" in result

    def test_ignores_non_model_keys(self):
        state = {
            "model.foo": torch.zeros(2),
            "optimizer_states": torch.zeros(2),
            "_task_losses.x.weight": torch.zeros(2),
        }
        result = _normalize_checkpoint_keys(state)
        assert len(result) == 1
        assert "foo" in result

    def test_ambiguous_collision_raises(self):
        state = {
            "model.tokenizer.weight": torch.zeros(2),
            "model._orig_mod.tokenizer.weight": torch.zeros(2),
        }
        with pytest.raises(PretrainedTransferError, match="Ambiguous"):
            _normalize_checkpoint_keys(state)


# ---------------------------------------------------------------------------
# Validation logic (no mutation)
# ---------------------------------------------------------------------------


class TestValidateTransfer:
    def test_perfect_match(self):
        ckpt = {"tokenizer.weight": torch.zeros(4, 16)}
        target = {"tokenizer.weight": torch.ones(4, 16)}
        mapping, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert "tokenizer.weight" in mapping
        assert len(report.loaded) == 1
        assert not report.has_errors

    def test_missing_in_checkpoint(self):
        ckpt = {}
        target = {"tokenizer.weight": torch.ones(4, 16)}
        _, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert "tokenizer.weight" in report.missing_in_checkpoint
        assert report.has_errors

    def test_unexpected_in_checkpoint(self):
        ckpt = {
            "tokenizer.weight": torch.zeros(4, 16),
            "tokenizer.extra_layer.weight": torch.zeros(4, 16),
        }
        target = {"tokenizer.weight": torch.ones(4, 16)}
        _, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert "tokenizer.extra_layer.weight" in report.unexpected_in_checkpoint

    def test_shape_mismatch(self):
        ckpt = {"tokenizer.weight": torch.zeros(4, 32)}
        target = {"tokenizer.weight": torch.ones(4, 16)}
        mapping, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert len(mapping) == 0
        assert len(report.shape_mismatched) == 1
        assert report.has_errors

    def test_dtype_mismatch(self):
        ckpt = {"tokenizer.weight": torch.zeros(4, 16, dtype=torch.float16)}
        target = {"tokenizer.weight": torch.ones(4, 16, dtype=torch.float32)}
        mapping, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert len(mapping) == 0
        assert len(report.dtype_mismatched) == 1
        assert report.has_errors

    def test_excluded_keys_reported(self):
        ckpt = {
            "tokenizer.weight": torch.zeros(4, 16),
            "session_emb.weight": torch.zeros(4, 16),
        }
        target = {"tokenizer.weight": torch.ones(4, 16)}
        _, report = _validate_transfer(ckpt, target, ("tokenizer",))
        assert "session_emb.weight" in report.skipped_excluded


# ---------------------------------------------------------------------------
# load_pretrained_weights – strict mode
# ---------------------------------------------------------------------------


class TestLoadPretrainedStrict:
    def test_transfers_shared_components(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt)

        for prefix in ("tokenizer.", "backbone.", "rotary_emb.", "latent_emb."):
            matching = [k for k in report.loaded if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys transferred for {prefix}"

        for key in report.loaded:
            src_val = dict(src.named_parameters())[key]
            dst_val = dict(dst.named_parameters())[key]
            assert torch.equal(src_val, dst_val), f"Mismatch for {key}"

    def test_skips_dataset_specific_keys(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        dst = _DummyModel(embed_dim=16)
        report = load_pretrained_weights(dst, ckpt)

        for prefix in ("channel_emb.", "session_emb.", "task_emb.", "router."):
            matching = [
                k for k in report.skipped_excluded if k.startswith(prefix)
            ]
            assert len(matching) > 0, f"Expected {prefix} keys to be excluded"

    def test_compiled_checkpoint(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "compiled.ckpt"
        _save_lightning_ckpt(src, ckpt, compiled=True)

        report = load_pretrained_weights(dst, ckpt)

        for prefix in ("tokenizer.", "backbone.", "rotary_emb.", "latent_emb."):
            matching = [k for k in report.loaded if k.startswith(prefix)]
            assert len(matching) > 0

        for key in report.loaded:
            src_val = dict(src.named_parameters())[key]
            dst_val = dict(dst.named_parameters())[key]
            assert torch.equal(src_val, dst_val)

    def test_shape_mismatch_raises_strict(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=32)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        with pytest.raises(PretrainedTransferError, match="Shape mismatch"):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

    def test_missing_checkpoint_raises(self, tmp_path):
        dst = _DummyModel(embed_dim=16)
        with pytest.raises(FileNotFoundError):
            load_pretrained_weights(dst, tmp_path / "nonexistent.ckpt")

    def test_missing_state_dict_raises(self, tmp_path):
        ckpt = tmp_path / "bad.ckpt"
        torch.save({"epoch": 5}, ckpt)
        dst = _DummyModel(embed_dim=16)
        with pytest.raises(PretrainedTransferError, match="no 'state_dict'"):
            load_pretrained_weights(dst, ckpt)

    def test_no_model_keys_raises(self, tmp_path):
        ckpt = tmp_path / "empty.ckpt"
        torch.save({"state_dict": {"optimizer.lr": torch.tensor(0.01)}}, ckpt)
        dst = _DummyModel(embed_dim=16)
        with pytest.raises(PretrainedTransferError, match="No model keys"):
            load_pretrained_weights(dst, ckpt)

    def test_no_transferable_components_raises(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)
        dst = _NoTransferModel()
        with pytest.raises(
            PretrainedTransferError, match="transferable_components"
        ):
            load_pretrained_weights(dst, ckpt)


# ---------------------------------------------------------------------------
# Atomicity: model state unchanged after validation failure
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_shape_mismatch_leaves_model_unchanged(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=32)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        before = _snapshot_state(dst)

        with pytest.raises(PretrainedTransferError):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

        _assert_states_equal(
            before, _snapshot_state(dst), "after shape mismatch"
        )

    def test_missing_key_leaves_model_unchanged(self, tmp_path):
        """If checkpoint is missing expected keys, no partial load occurs."""
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)

        state = {"state_dict": {}}
        for k, v in src.state_dict().items():
            if not k.startswith("backbone."):
                state["state_dict"][f"model.{k}"] = v
        ckpt = tmp_path / "partial.ckpt"
        torch.save(state, ckpt)

        before = _snapshot_state(dst)

        with pytest.raises(PretrainedTransferError, match="Missing"):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

        _assert_states_equal(before, _snapshot_state(dst), "after missing key")

    def test_dtype_mismatch_leaves_model_unchanged(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)

        state_dict = {}
        for k, v in src.state_dict().items():
            state_dict[f"model.{k}"] = v.half()
        ckpt = tmp_path / "dtype.ckpt"
        torch.save({"state_dict": state_dict}, ckpt)

        before = _snapshot_state(dst)

        with pytest.raises(PretrainedTransferError):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

        _assert_states_equal(
            before, _snapshot_state(dst), "after dtype mismatch"
        )


# ---------------------------------------------------------------------------
# Freeze behavior
# ---------------------------------------------------------------------------


class TestFreeze:
    def test_freeze_only_transferred(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt, freeze=True)

        transferred = set(report.loaded)
        for name, param in dst.named_parameters():
            if name in transferred:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_no_freeze_by_default(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        load_pretrained_weights(dst, ckpt, freeze=False)

        for _name, param in dst.named_parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Permissive mode
# ---------------------------------------------------------------------------


class TestPermissiveMode:
    def test_shape_mismatch_loads_compatible(self, tmp_path):
        """Permissive mode skips mismatches but transfers compatible keys."""
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)

        # Corrupt one component's shape in the checkpoint
        state_dict = {}
        for k, v in src.state_dict().items():
            if k.startswith("backbone."):
                state_dict[f"model.{k}"] = torch.randn(99, 99)
            else:
                state_dict[f"model.{k}"] = v
        ckpt = tmp_path / "partial.ckpt"
        torch.save({"state_dict": state_dict}, ckpt)

        report = load_pretrained_weights(
            dst, ckpt, mode=TransferMode.PERMISSIVE
        )

        # Backbone keys should be in shape_mismatched
        assert len(report.shape_mismatched) > 0
        # Other components should still have loaded
        tokenizer_loaded = [
            k for k in report.loaded if k.startswith("tokenizer.")
        ]
        assert len(tokenizer_loaded) > 0

    def test_missing_key_still_loads_others(self, tmp_path):
        src = _DummyModel(embed_dim=16)

        state = {"state_dict": {}}
        for k, v in src.state_dict().items():
            if not k.startswith("latent_emb."):
                state["state_dict"][f"model.{k}"] = v
        ckpt = tmp_path / "partial.ckpt"
        torch.save(state, ckpt)

        dst = _DummyModel(embed_dim=16)
        report = load_pretrained_weights(
            dst, ckpt, mode=TransferMode.PERMISSIVE
        )

        assert len(report.missing_in_checkpoint) > 0
        assert len(report.loaded) > 0


# ---------------------------------------------------------------------------
# Empty and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_transfer_components(self, tmp_path):
        """Model that declares no transferable components."""

        class _EmptyTransferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def transferable_components(self):
                return ()

        src = _DummyModel(embed_dim=16)
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        dst = _EmptyTransferModel()
        report = load_pretrained_weights(dst, ckpt)

        assert len(report.loaded) == 0
        assert len(report.skipped_excluded) > 0

    def test_checkpoint_with_extra_keys_in_transfer_components(self, tmp_path):
        """Checkpoint has extra keys within transferable component namespace."""

        class _ExtendedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer = nn.Sequential(
                    nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16)
                )
                self.backbone = nn.Linear(16, 16)

            def transferable_components(self):
                return ("tokenizer", "backbone")

        class _SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer = nn.Linear(4, 16)
                self.backbone = nn.Linear(16, 16)

            def transferable_components(self):
                return ("tokenizer", "backbone")

        src = _ExtendedModel()
        dst = _SmallModel()
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(
            dst, ckpt, mode=TransferMode.PERMISSIVE
        )
        assert len(report.unexpected_in_checkpoint) > 0

    def test_buffers_are_transferred(self, tmp_path):
        """Non-parameter buffers (e.g. running stats) should be transferred."""

        class _ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer = nn.BatchNorm1d(16)

            def transferable_components(self):
                return ("tokenizer",)

        src = _ModelWithBuffer()
        # Simulate some running stats
        src.tokenizer.running_mean.fill_(42.0)
        src.tokenizer.running_var.fill_(7.0)

        dst = _ModelWithBuffer()
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        load_pretrained_weights(dst, ckpt)

        assert torch.equal(
            dst.tokenizer.running_mean, src.tokenizer.running_mean
        )
        assert torch.equal(dst.tokenizer.running_var, src.tokenizer.running_var)


# ---------------------------------------------------------------------------
# Real model transfer: MaskedPOYOEEGModel → POYOEEGModel
# ---------------------------------------------------------------------------


def _build_minimal_poyo(embed_dim=64, C_pad=4, N=10, sequence_length=1.0):
    """Build a tiny POYOEEGModel for downstream finetuning."""
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

    downstream_task_configs = {
        "sleep_staging": {
            "name": "sleep_staging",
            "head": {
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 5,
                "num_layers": 2,
            },
            "target_extractor": {
                "_target_": "foundry.tasks.targets.TargetExtractor",
                "target_field": "sleep_staging",
            },
            "loss": {
                "_target_": "torch.nn.CrossEntropyLoss",
            },
        },
    }

    model = POYOEEGModel(
        tokenizer=tokenizer,
        task_configs=downstream_task_configs,
        embed_dim=embed_dim,
        sequence_length=sequence_length,
        latent_step=0.5,
        num_latents_per_step=2,
        depth=1,
        dim_head=32,
        cross_heads=2,
        self_heads=2,
    )

    model.initialize_vocabs(
        {
            "session_ids": ["downstream_sess_0"],
            "channel_ids": [f"ch_{i}" for i in range(C_pad)],
        }
    )
    return model


def _build_minimal_masked_model(
    embed_dim=64, C_pad=4, N=10, sequence_length=1.0, mask_ratio=0.5
):
    """Build a tiny MaskedPOYOEEGModel for pretraining."""
    from foundry.models.embeddings import PerChannelStrategy
    from foundry.models.embeddings.temporal.resample_cnn import (
        ResampleCNNEmbedding,
    )
    from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel
    from foundry.models.tokenizer import EEGTokenizer
    from foundry.tasks.masking import RandomTokenMasking

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

    model.initialize_vocabs(
        {
            "session_ids": ["pretrain_sess_0", "pretrain_sess_1"],
            "channel_ids": [f"ch_{i}" for i in range(C_pad)],
        }
    )
    return model


class TestRealModelTransfer:
    """Transfer from real MaskedPOYOEEGModel → real POYOEEGModel."""

    def test_transfer_shared_components(self, tmp_path):
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt)

        assert len(report.loaded) > 0
        assert not report.has_errors

        for prefix in ("tokenizer.", "backbone.", "rotary_emb.", "latent_emb."):
            matching = [k for k in report.loaded if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys transferred for {prefix}"

        src_state = dict(src.named_parameters())
        src_buffers = dict(src.named_buffers())
        src_all = {**src_state, **src_buffers}
        dst_state = dict(dst.named_parameters())
        dst_buffers = dict(dst.named_buffers())
        dst_all = {**dst_state, **dst_buffers}

        for key in report.loaded:
            assert torch.equal(src_all[key], dst_all[key]), (
                f"Transferred weight mismatch for {key}"
            )

    def test_dataset_specific_not_transferred(self, tmp_path):
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()

        dst_before = _snapshot_state(dst)

        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt)

        for excluded_prefix in (
            "channel_emb.",
            "session_emb.",
            "task_emb.",
            "router.",
        ):
            matching_excluded = [
                k
                for k in report.skipped_excluded
                if k.startswith(excluded_prefix)
            ]
            assert len(matching_excluded) > 0, (
                f"Expected {excluded_prefix} to be excluded"
            )

            for key in dst_before:
                if key.startswith(excluded_prefix):
                    dst_val = dict(dst.state_dict())[key]
                    if isinstance(dst_val, torch.Tensor):
                        assert torch.equal(dst_before[key], dst_val), (
                            f"Dataset-specific param {key} should be unchanged"
                        )
                    else:
                        assert dst_before[key] == dst_val, (
                            f"Dataset-specific state {key} should be unchanged"
                        )

    def test_masking_specific_excluded(self, tmp_path):
        """MaskedPOYOEEGModel-only modules (recon_channel_proj, masking-task
        router heads) should be excluded from transfer.  Note that
        RandomTokenMasking has no parameters, so we check recon_channel_proj
        and the masked_reconstruction router head."""
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt)

        ssl_specific_keys = [
            k
            for k in report.skipped_excluded
            if k.startswith("recon_channel_proj.")
            or "masked_reconstruction" in k
        ]
        assert len(ssl_specific_keys) > 0, (
            f"Expected SSL-specific keys in excluded, got: {report.skipped_excluded}"
        )

    def test_compiled_real_model(self, tmp_path):
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()
        ckpt = tmp_path / "compiled.ckpt"
        _save_lightning_ckpt(src, ckpt, compiled=True)

        report = load_pretrained_weights(dst, ckpt)

        assert len(report.loaded) > 0
        assert not report.has_errors

    def test_freeze_real_model(self, tmp_path):
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()
        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        report = load_pretrained_weights(dst, ckpt, freeze=True)

        transferred = set(report.loaded)
        for name, param in dst.named_parameters():
            if name in transferred:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_mismatched_tokenizer_fails_strict(self, tmp_path):
        """Different embed_dim between src and dst should fail in strict mode."""
        src = _build_minimal_masked_model(embed_dim=64)
        dst = _build_minimal_poyo(embed_dim=32)

        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        with pytest.raises(PretrainedTransferError, match="Shape mismatch"):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

    def test_mismatched_tokenizer_atomicity(self, tmp_path):
        """After a failed strict transfer, the dst model is completely unchanged."""
        src = _build_minimal_masked_model(embed_dim=64)
        dst = _build_minimal_poyo(embed_dim=32)

        before = _snapshot_state(dst)

        ckpt = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt)

        with pytest.raises(PretrainedTransferError):
            load_pretrained_weights(dst, ckpt, mode=TransferMode.STRICT)

        _assert_states_equal(
            before, _snapshot_state(dst), "after mismatched tokenizer"
        )

    def test_transferable_components_matches_architecture(self):
        """The declared transferable_components exist as real submodules."""
        model = _build_minimal_poyo()
        components = model.transferable_components()

        for comp in components:
            assert hasattr(model, comp), (
                f"Declared component '{comp}' not found on model"
            )
            assert isinstance(getattr(model, comp), nn.Module), (
                f"Declared component '{comp}' is not an nn.Module"
            )

    def test_transferable_components_cover_shared_params(self):
        """Every parameter under a transferable component prefix is
        actually covered by the transfer policy."""
        src = _build_minimal_masked_model()
        dst = _build_minimal_poyo()

        components = dst.transferable_components()

        src_shared = set()
        for name in dict(src.named_parameters()):
            if any(name == c or name.startswith(c + ".") for c in components):
                src_shared.add(name)

        dst_shared = set()
        for name in dict(dst.named_parameters()):
            if any(name == c or name.startswith(c + ".") for c in components):
                dst_shared.add(name)

        assert src_shared == dst_shared, (
            f"Component parameter sets differ:\n"
            f"  Only in src: {src_shared - dst_shared}\n"
            f"  Only in dst: {dst_shared - src_shared}"
        )


# ---------------------------------------------------------------------------
# TransferMode enum
# ---------------------------------------------------------------------------


class TestTransferMode:
    def test_strict_is_default(self):
        assert TransferMode.STRICT.value == "strict"

    def test_from_string(self):
        assert TransferMode("strict") == TransferMode.STRICT
        assert TransferMode("permissive") == TransferMode.PERMISSIVE

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            TransferMode("invalid")

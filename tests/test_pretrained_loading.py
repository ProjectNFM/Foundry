"""Tests for pretrained weight loading (pretrain → finetune transfer)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from foundry.training.pretrained import (
    TRANSFER_PREFIXES,
    load_pretrained_weights,
)


class _DummyModel(nn.Module):
    """Minimal model with the same sub-module names as POYOEEGModel."""

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


def _save_lightning_ckpt(
    model: nn.Module, path: Path, compiled: bool = False
) -> None:
    prefix = "model._orig_mod." if compiled else "model."
    state_dict = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, path)


class TestLoadPretrainedWeights:
    def test_transfers_shared_components(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)

        ckpt_path = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt_path)

        loaded, skipped = load_pretrained_weights(dst, ckpt_path)

        for key in loaded:
            assert any(key.startswith(p) for p in TRANSFER_PREFIXES), (
                f"Loaded key {key} not in TRANSFER_PREFIXES"
            )

        for prefix in ("tokenizer.", "backbone.", "rotary_emb.", "latent_emb."):
            matching = [k for k in loaded if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys transferred for {prefix}"

        for key in loaded:
            src_val = dict(src.named_parameters())[key]
            dst_val = dict(dst.named_parameters())[key]
            assert torch.allclose(src_val, dst_val), (
                f"Weight mismatch for {key}"
            )

    def test_skips_dataset_specific_keys(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        ckpt_path = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt_path)

        dst = _DummyModel(embed_dim=16)
        _, skipped = load_pretrained_weights(dst, ckpt_path)

        skip_prefixes = ("channel_emb.", "session_emb.", "task_emb.", "router.")
        for prefix in skip_prefixes:
            matching = [k for k in skipped if k.startswith(prefix)]
            assert len(matching) > 0, f"Expected {prefix} keys to be skipped"

    def test_freeze_pretrained(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        ckpt_path = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt_path)

        dst = _DummyModel(embed_dim=16)
        loaded, _ = load_pretrained_weights(dst, ckpt_path, freeze=True)

        transferred_set = set(loaded)
        for name, param in dst.named_parameters():
            if name in transferred_set:
                assert not param.requires_grad, f"Param {name} should be frozen"
            else:
                assert param.requires_grad, (
                    f"Param {name} should remain trainable"
                )

    def test_shape_mismatch_skipped(self, tmp_path):
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=32)

        ckpt_path = tmp_path / "pretrained.ckpt"
        _save_lightning_ckpt(src, ckpt_path)

        loaded, skipped = load_pretrained_weights(dst, ckpt_path)
        assert len(loaded) == 0, "No keys should load with mismatched shapes"

    def test_compiled_checkpoint_strips_orig_mod(self, tmp_path):
        """Checkpoints saved from torch.compile'd models have _orig_mod. prefix."""
        src = _DummyModel(embed_dim=16)
        dst = _DummyModel(embed_dim=16)

        ckpt_path = tmp_path / "compiled.ckpt"
        _save_lightning_ckpt(src, ckpt_path, compiled=True)

        loaded, skipped = load_pretrained_weights(dst, ckpt_path)

        for prefix in ("tokenizer.", "backbone.", "rotary_emb.", "latent_emb."):
            matching = [k for k in loaded if k.startswith(prefix)]
            assert len(matching) > 0, (
                f"No keys transferred for {prefix} from compiled checkpoint"
            )

        for key in loaded:
            src_val = dict(src.named_parameters())[key]
            dst_val = dict(dst.named_parameters())[key]
            assert torch.allclose(src_val, dst_val), (
                f"Weight mismatch for {key}"
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        dst = _DummyModel(embed_dim=16)
        with pytest.raises(FileNotFoundError):
            load_pretrained_weights(dst, tmp_path / "nonexistent.ckpt")

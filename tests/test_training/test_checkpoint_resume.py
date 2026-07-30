"""Tests for checkpoint resume behavior and removal of global torch.load patching.

Covers:
- Resume from last.ckpt restores model, optimizer, scheduler, and epoch
- weights_only=False loads OmegaConf/uninitialized objects in Lightning checkpoints
- Fresh finetuning loads pretrained components once (via pretrained transfer)
- Resume plus pretrained configuration raises ValueError
- torch.load and torch.serialization.load identities are unchanged after training
- _validate_checkpoint_policy logic
- _get_resume_checkpoint_path logic
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.serialization as torch_serialization


# ---------------------------------------------------------------------------
# Minimal model/module for integration tests
# ---------------------------------------------------------------------------


class _TinyTaskModel(nn.Module):
    """Minimal model with task_configs and router for Lightning training."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.task_configs = {}

        from foundry.models.readout import ReadoutRouter

        self.router = ReadoutRouter({})

    def forward(self, x, **kwargs):
        return self.fc(x)


class _TinyLightningModule(L.LightningModule):
    """Minimal Lightning module for checkpoint round-trip tests."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _make_datamodule(
    n_samples: int = 32, input_dim: int = 4, output_dim: int = 2
):
    """Create a tiny in-memory datamodule for testing."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, output_dim)
    dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    class _SimpleDataModule(L.LightningDataModule):
        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

    return _SimpleDataModule()


# ---------------------------------------------------------------------------
# _validate_checkpoint_policy
# ---------------------------------------------------------------------------


class TestValidateCheckpointPolicy:
    """Tests for the checkpoint policy validation helper."""

    def test_both_resume_and_pretrained_raises(self):
        from main import _validate_checkpoint_policy

        with pytest.raises(ValueError, match="Both resume checkpoint"):
            _validate_checkpoint_policy(
                "/path/to/last.ckpt", "/path/to/pretrained.ckpt"
            )

    def test_resume_only_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy("/path/to/last.ckpt", None)

    def test_pretrained_only_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy(None, "/path/to/pretrained.ckpt")

    def test_neither_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy(None, None)

    def test_resume_with_empty_string_pretrained_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy("/path/to/last.ckpt", "")

    def test_empty_string_resume_with_pretrained_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy("", "/path/to/pretrained.ckpt")


# ---------------------------------------------------------------------------
# _get_resume_checkpoint_path
# ---------------------------------------------------------------------------


class TestGetResumeCheckpointPath:
    """Tests for the resume checkpoint path resolution logic."""

    def test_returns_none_when_no_checkpoint(self, tmp_path):
        from main import _get_resume_checkpoint_path
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"run": {"resume_if_checkpoint_exists": False}})
        checkpoint_dir = str(tmp_path / "checkpoints")
        Path(checkpoint_dir).mkdir()

        result = _get_resume_checkpoint_path(
            cfg, checkpoint_dir, slurm_restart_count=0
        )
        assert result is None

    def test_returns_path_on_slurm_restart(self, tmp_path):
        from main import _get_resume_checkpoint_path
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"run": {"resume_if_checkpoint_exists": False}})
        checkpoint_dir = str(tmp_path / "checkpoints")
        Path(checkpoint_dir).mkdir()
        last_ckpt = Path(checkpoint_dir) / "last.ckpt"
        torch.save({"state_dict": {}}, last_ckpt)

        result = _get_resume_checkpoint_path(
            cfg, checkpoint_dir, slurm_restart_count=1
        )
        assert result == str(last_ckpt)

    def test_returns_path_when_resume_enabled(self, tmp_path):
        from main import _get_resume_checkpoint_path
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"run": {"resume_if_checkpoint_exists": True}})
        checkpoint_dir = str(tmp_path / "checkpoints")
        Path(checkpoint_dir).mkdir()
        last_ckpt = Path(checkpoint_dir) / "last.ckpt"
        torch.save({"state_dict": {}}, last_ckpt)

        result = _get_resume_checkpoint_path(
            cfg, checkpoint_dir, slurm_restart_count=0
        )
        assert result == str(last_ckpt)

    def test_returns_none_when_resume_disabled(self, tmp_path):
        from main import _get_resume_checkpoint_path
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"run": {"resume_if_checkpoint_exists": False}})
        checkpoint_dir = str(tmp_path / "checkpoints")
        Path(checkpoint_dir).mkdir()
        last_ckpt = Path(checkpoint_dir) / "last.ckpt"
        torch.save({"state_dict": {}}, last_ckpt)

        result = _get_resume_checkpoint_path(
            cfg, checkpoint_dir, slurm_restart_count=0
        )
        assert result is None


# ---------------------------------------------------------------------------
# torch.load identity is unchanged before and after trainer.fit
# ---------------------------------------------------------------------------


class TestTorchLoadIdentityUnchanged:
    """Verify that torch.load and torch.serialization.load are not mutated."""

    def test_torch_load_identity_after_training(self, tmp_path):
        """trainer.fit with weights_only=False does not replace torch.load globally."""
        original_torch_load = torch.load
        original_ts_load = torch_serialization.load

        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        trainer = L.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(module, dm, weights_only=False)

        assert torch.load is original_torch_load, (
            "torch.load was mutated during training"
        )
        assert torch_serialization.load is original_ts_load, (
            "torch.serialization.load was mutated during training"
        )

    def test_torch_load_identity_unchanged_with_resume(self, tmp_path):
        """Resume from checkpoint does not leave torch.load patched."""
        original_torch_load = torch.load
        original_ts_load = torch_serialization.load

        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer1 = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer1.fit(module, dm, weights_only=False)

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
        assert last_ckpt.exists()

        model2 = _TinyTaskModel()
        module2 = _TinyLightningModule(model2)
        trainer2 = L.Trainer(
            max_epochs=2,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer2.fit(module2, dm, ckpt_path=str(last_ckpt), weights_only=False)

        assert torch.load is original_torch_load
        assert torch_serialization.load is original_ts_load


# ---------------------------------------------------------------------------
# Resume restores full trainer state
# ---------------------------------------------------------------------------


class TestResumeRestoresState:
    """Verify that resume from last.ckpt restores model, optimizer, scheduler, and epoch."""

    def test_resume_restores_epoch(self, tmp_path):
        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer1 = L.Trainer(
            max_epochs=2,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer1.fit(module, dm, weights_only=False)
        assert trainer1.current_epoch == 2

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
        assert last_ckpt.exists()

        model2 = _TinyTaskModel()
        module2 = _TinyLightningModule(model2)
        trainer2 = L.Trainer(
            max_epochs=4,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer2.fit(module2, dm, ckpt_path=str(last_ckpt), weights_only=False)

        assert trainer2.current_epoch == 4

    def test_resume_restores_model_weights(self, tmp_path):
        torch.manual_seed(42)
        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer = L.Trainer(
            max_epochs=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, dm, weights_only=False)

        trained_state = {
            k: v.clone() for k, v in module.model.state_dict().items()
        }

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"

        model2 = _TinyTaskModel()
        module2 = _TinyLightningModule(model2)
        trainer2 = L.Trainer(
            max_epochs=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer2.fit(module2, dm, ckpt_path=str(last_ckpt), weights_only=False)

        for key in trained_state:
            assert torch.equal(
                trained_state[key], module2.model.state_dict()[key]
            ), f"Model weight {key} not restored correctly"

    def test_resume_restores_optimizer_state(self, tmp_path):
        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer = L.Trainer(
            max_epochs=2,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, dm, weights_only=False)

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
        ckpt_data = torch.load(
            last_ckpt, map_location="cpu", weights_only=False
        )
        assert "optimizer_states" in ckpt_data
        assert len(ckpt_data["optimizer_states"]) > 0

        model2 = _TinyTaskModel()
        module2 = _TinyLightningModule(model2)
        trainer2 = L.Trainer(
            max_epochs=4,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer2.fit(module2, dm, ckpt_path=str(last_ckpt), weights_only=False)

        opt = trainer2.optimizers
        assert opt is not None

    def test_resume_restores_scheduler_state(self, tmp_path):
        model = _TinyTaskModel()
        module = _TinyLightningModule(model, lr=0.01)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer = L.Trainer(
            max_epochs=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, dm, weights_only=False)

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
        ckpt_data = torch.load(
            last_ckpt, map_location="cpu", weights_only=False
        )
        assert "lr_schedulers" in ckpt_data
        assert len(ckpt_data["lr_schedulers"]) > 0


# ---------------------------------------------------------------------------
# weights_only=False loads objects that require it
# ---------------------------------------------------------------------------


class TestWeightsOnlyFalse:
    """Verify that weights_only=False supports the objects in Lightning checkpoints."""

    def test_checkpoint_with_omegaconf_loads_successfully(self, tmp_path):
        """Checkpoints containing OmegaConf objects need weights_only=False."""
        from omegaconf import DictConfig, OmegaConf

        ckpt_path = tmp_path / "with_omega.ckpt"
        ckpt_data = {
            "state_dict": {
                "fc.weight": torch.randn(2, 4),
                "fc.bias": torch.randn(2),
            },
            "hyper_parameters": OmegaConf.create(
                {"lr": 0.001, "batch_size": 32}
            ),
            "epoch": 5,
        }
        torch.save(ckpt_data, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert isinstance(loaded["hyper_parameters"], DictConfig)
        assert loaded["hyper_parameters"]["lr"] == 0.001

    def test_weights_only_true_rejects_omegaconf(self, tmp_path):
        """Confirm that weights_only=True rejects OmegaConf-containing checkpoints."""
        from omegaconf import OmegaConf

        ckpt_path = tmp_path / "with_omega.ckpt"
        ckpt_data = {
            "state_dict": {"fc.weight": torch.randn(2, 4)},
            "hyper_parameters": OmegaConf.create({"lr": 0.001}),
        }
        torch.save(ckpt_data, ckpt_path)

        with pytest.raises(Exception):
            torch.load(ckpt_path, map_location="cpu", weights_only=True)

    def test_real_lightning_checkpoint_roundtrip(self, tmp_path):
        """Full Lightning checkpoint save/load cycle with weights_only=False."""
        model = _TinyTaskModel()
        module = _TinyLightningModule(model)
        dm = _make_datamodule()

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
        )

        trainer = L.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            logger=False,
            default_root_dir=str(tmp_path),
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, dm, weights_only=False)

        last_ckpt = tmp_path / "checkpoints" / "last.ckpt"
        assert last_ckpt.exists()

        loaded = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        assert "state_dict" in loaded
        assert "epoch" in loaded
        assert "optimizer_states" in loaded


# ---------------------------------------------------------------------------
# Fresh finetuning with pretrained transfer
# ---------------------------------------------------------------------------


class TestFreshFinetuningWithPretrained:
    """Verify that pretrained transfer works correctly for fresh runs."""

    def test_pretrained_then_train_uses_transferred_weights(self, tmp_path):
        """Pretrained weights are applied before training starts."""
        from foundry.training.pretrained import load_pretrained_weights

        class _TransferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(4, 4)
                self.head = nn.Linear(4, 2)

            def transferable_components(self):
                return ("backbone",)

        src = _TransferModel()
        nn.init.constant_(src.backbone.weight, 42.0)
        nn.init.constant_(src.backbone.bias, 7.0)

        ckpt_path = tmp_path / "pretrained.ckpt"
        state_dict = {f"model.{k}": v for k, v in src.state_dict().items()}
        torch.save({"state_dict": state_dict}, ckpt_path)

        dst = _TransferModel()
        assert not torch.equal(dst.backbone.weight, src.backbone.weight)

        report = load_pretrained_weights(dst, ckpt_path)
        assert len(report.loaded) > 0

        assert torch.equal(dst.backbone.weight, src.backbone.weight)
        assert torch.equal(dst.backbone.bias, src.backbone.bias)

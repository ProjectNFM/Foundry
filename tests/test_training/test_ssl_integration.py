"""Tests for SSL (masked pretraining) integration in FoundryModule.

Tests the seam between the model output dict (with injected targets/weights)
and the training module's loss computation + metric update pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.readout import ReadoutRouter
from foundry.tasks.config import TaskConfig
from foundry.training import FoundryModule

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_CONFIG_DIR = REPO_ROOT / "configs" / "tasks"


class _StubSSLModel(nn.Module):
    """Model stub that injects _targets and _weights into output dict.

    Simulates the behavior of MaskedPOYOEEGModel.forward() which injects
    gathered reconstruction targets and validity weights into the output dict.
    """

    def __init__(self, task_configs: dict[str, TaskConfig], embed_dim: int = 8):
        super().__init__()
        self.task_configs = task_configs
        heads = {}
        for name, cfg in task_configs.items():
            head_kwargs = {**cfg.head, "embed_dim": embed_dim}
            if "output_dim" not in head_kwargs:
                head_kwargs["output_dim"] = cfg.output_dim
            heads[name] = instantiate(head_kwargs)
        self.router = ReadoutRouter(heads)
        self._inject_targets = {}
        self._inject_weights = {}

    def set_injected(self, targets: dict, weights: dict):
        self._inject_targets = targets
        self._inject_weights = weights

    def forward(
        self, output_embs=None, task_index=None, unpack_output=False, **kwargs
    ):
        if output_embs is None:
            output_embs = kwargs.get("output_embs")
        flat_task_index = task_index.reshape(-1)
        valid = flat_task_index > 0
        if output_embs.shape[0] != valid.sum():
            flat_embs = output_embs.reshape(-1, output_embs.shape[-1])
            flat_embs = flat_embs[valid]
        else:
            flat_embs = output_embs
        outputs = self.router(flat_embs, (flat_task_index[valid] - 1).long())
        for k, v in self._inject_targets.items():
            outputs[f"{k}_targets"] = v
        for k, v in self._inject_weights.items():
            outputs[f"{k}_weights"] = v
        return outputs


class TestSharedStepTargetExtraction:
    """Test that _shared_step extracts model-side targets from output dict."""

    def test_model_injected_targets_flow_to_loss(self):
        cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
                "num_layers": 2,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
            metrics={
                "_target_": "foundry.tasks.metrics.ssl_metrics",
            },
        )
        task_configs = {cfg.name: cfg}
        model = _StubSSLModel(task_configs, embed_dim=8)

        recon_idx = model.router.get_task_index_by_name("masked_reconstruction")

        targets = torch.randn(6)
        weights = torch.ones(6)
        model.set_injected(
            {"masked_reconstruction": targets},
            {"masked_reconstruction": weights},
        )

        module = FoundryModule(model=model)
        module.to(torch.device("cpu"))

        B, n_out = 2, 3
        embs = torch.randn(B, n_out, 8)
        task_index = torch.full((B, n_out), recon_idx + 1, dtype=torch.long)

        batch = {
            "output_embs": embs,
            "task_index": task_index,
            "target_values": {},
            "target_weights": {},
            "session_id": ["s0", "s1"],
        }

        loss = module._shared_step("train", batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_mixed_supervised_and_ssl_targets(self):
        ssl_cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
                "num_layers": 2,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        sup_cfg = TaskConfig(
            name="supervised_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 3,
            },
            target_extractor={
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        task_configs = {ssl_cfg.name: ssl_cfg, sup_cfg.name: sup_cfg}
        model = _StubSSLModel(task_configs, embed_dim=8)

        recon_idx = model.router.get_task_index_by_name("masked_reconstruction")
        sup_idx = model.router.get_task_index_by_name("supervised_task")

        recon_targets = torch.randn(4)
        recon_weights = torch.ones(4)
        model.set_injected(
            {"masked_reconstruction": recon_targets},
            {"masked_reconstruction": recon_weights},
        )

        module = FoundryModule(model=model)
        module.to(torch.device("cpu"))

        # task_index: 2 recon + 2 supervised per sample, 2 samples
        task_index = torch.tensor(
            [
                [recon_idx + 1, recon_idx + 1, sup_idx + 1, 0],
                [recon_idx + 1, recon_idx + 1, sup_idx + 1, 0],
            ]
        )

        embs = torch.randn(2, 4, 8)

        batch = {
            "output_embs": embs,
            "task_index": task_index,
            "target_values": {
                "supervised_task": torch.tensor([0, 1]),
            },
            "target_weights": {
                "supervised_task": torch.ones(2),
            },
            "session_id": ["s0", "s1"],
        }

        loss = module._shared_step("train", batch)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestWarmupScheduler:
    def test_warmup_creates_sequential_scheduler(self):
        cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        model = _StubSSLModel({cfg.name: cfg})
        module = FoundryModule(
            model=model,
            warmup_epochs=5,
        )

        class _FakeTrainer:
            max_epochs = 100

        module._trainer = _FakeTrainer()
        optim_config = module.configure_optimizers()

        scheduler = optim_config["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    def test_no_warmup_uses_cosine_only(self):
        cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        model = _StubSSLModel({cfg.name: cfg})
        module = FoundryModule(model=model, warmup_epochs=0)

        class _FakeTrainer:
            max_epochs = 100

        module._trainer = _FakeTrainer()
        optim_config = module.configure_optimizers()

        scheduler = optim_config["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_warmup_lr_starts_low(self):
        cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        model = _StubSSLModel({cfg.name: cfg})
        lr = 1e-3
        module = FoundryModule(
            model=model,
            learning_rate=lr,
            warmup_epochs=10,
        )

        class _FakeTrainer:
            max_epochs = 100

        module._trainer = _FakeTrainer()
        optim_config = module.configure_optimizers()

        optimizer = optim_config["optimizer"]
        initial_lr = optimizer.param_groups[0]["lr"]

        assert initial_lr < lr


class TestSSLLossWeighting:
    """Reproduce bugs where model-injected SSL tasks get zero weight in
    _compute_task_losses because their entries are absent from the batch
    task_index (reconstruction indices are created internally by the model)."""

    def test_ssl_only_loss_nonzero_with_empty_task_index(self):
        """Bug 1 (Critical): pure SSL pretraining yields zero total loss.

        In real SSL batches, task_index from tokenize() is empty because
        there are no supervised targets. Reconstruction indices are only
        created inside MaskedPOYOEEGModel.forward(). The weighting in
        _compute_task_losses looks up recon in the batch task_index, finds
        num_sequences=0, and multiplies the loss by 0.
        """
        cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
                "num_layers": 2,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        model = _StubSSLModel({cfg.name: cfg})
        module = FoundryModule(model=model)

        preds = torch.randn(6)
        targets = torch.randn(6)
        weights = torch.ones(6)

        outputs = {"masked_reconstruction": preds}
        target_values = {"masked_reconstruction": targets}
        target_weights = {"masked_reconstruction": weights}

        task_index = torch.zeros((2, 3), dtype=torch.long)

        total_loss, taskwise_loss = module._compute_task_losses(
            outputs, target_values, target_weights, task_index
        )

        assert "masked_reconstruction" in taskwise_loss
        assert taskwise_loss["masked_reconstruction"].item() > 0
        assert total_loss.item() > 0, (
            "Total loss must be nonzero for SSL-only: reconstruction task "
            "is model-injected and absent from batch task_index"
        )

    def test_ssl_contributes_to_mixed_loss(self):
        """Bug 2 (High): in SSL+supervised runs, SSL loss is silently dropped.

        task_index has supervised entries but no reconstruction entries.
        _compute_task_losses computes the recon loss but weights it by
        num_sequences=0, so only supervised loss survives in the total.
        """
        ssl_cfg = TaskConfig(
            name="masked_reconstruction",
            head={
                "_target_": "foundry.tasks.heads.MLPReadoutHead",
                "output_dim": 1,
                "num_layers": 2,
            },
            target_extractor=None,
            loss={"_target_": "foundry.tasks.losses.ReconstructionLoss"},
        )
        sup_cfg = TaskConfig(
            name="supervised_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 3,
            },
            target_extractor={
                "timestamp_key": "t",
                "value_key": "v",
            },
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubSSLModel({ssl_cfg.name: ssl_cfg, sup_cfg.name: sup_cfg})
        module = FoundryModule(model=model)

        sup_idx = model.router.get_task_index_by_name("supervised_task") + 1

        recon_preds = torch.zeros(6)
        recon_targets = torch.ones(6) * 10.0
        sup_preds = torch.randn(2, 3)
        sup_targets = torch.tensor([0, 1])

        outputs = {
            "masked_reconstruction": recon_preds,
            "supervised_task": sup_preds,
        }
        target_values = {
            "masked_reconstruction": recon_targets,
            "supervised_task": sup_targets,
        }
        target_weights = {
            "masked_reconstruction": torch.ones(6),
            "supervised_task": torch.ones(2),
        }

        task_index = torch.tensor([[sup_idx, 0, 0], [sup_idx, 0, 0]])

        total_loss, taskwise_loss = module._compute_task_losses(
            outputs, target_values, target_weights, task_index
        )

        assert "masked_reconstruction" in taskwise_loss
        assert taskwise_loss["masked_reconstruction"].item() > 0
        assert total_loss.item() > taskwise_loss["supervised_task"].item(), (
            "Total loss must include SSL contribution, not just supervised"
        )

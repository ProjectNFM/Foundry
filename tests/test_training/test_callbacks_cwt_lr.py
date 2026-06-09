"""Tests for issue 04: training callbacks and CWT LR support."""

from __future__ import annotations

from unittest.mock import patch

import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from foundry.training import FoundryModule
from foundry.training.callbacks import (
    EffectiveBatchSizeCallback,
    ParameterWatcherCallback,
)


class _CwtStubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.cwt = nn.Linear(4, 2)
        self.task_configs = {}

    def forward(self, **kwargs):
        x = kwargs.get("x", kwargs.get("input"))
        return {"task": self.encoder(x)}


class _SimpleTrainModule(L.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.model(x).sum()


class _FakeDatamodule:
    def __init__(self, batch_size: int = 4, num_workers: int = 2) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._dataset = TensorDataset(torch.randn(32, 4))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class TestEffectiveBatchSizeCallback:
    def test_sets_accumulate_grad_batches(self):
        dm = _FakeDatamodule(batch_size=4, num_workers=2)
        module = _SimpleTrainModule(model=nn.Linear(4, 2))

        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.datamodule = dm

        callback = EffectiveBatchSizeCallback(
            effective_batch_size=16,
            init_val=4,
            max_val=8,
        )
        callback.on_fit_start(trainer, module)

        assert dm.batch_size == 8
        assert dm.num_workers == 2
        assert trainer.accumulate_grad_batches == 2

    def test_oom_stops_search(self):
        dm = _FakeDatamodule()
        module = _SimpleTrainModule(model=nn.Linear(4, 2))

        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.datamodule = dm

        callback = EffectiveBatchSizeCallback(init_val=4, max_val=64)
        with patch.object(
            EffectiveBatchSizeCallback,
            "_try_batch_size",
            side_effect=[True, False],
        ):
            callback.on_fit_start(trainer, module)

        assert dm.batch_size == 4
        assert trainer.accumulate_grad_batches >= 1


class TestParameterWatcherCallback:
    def test_discovers_cwt_params(self):
        model = _CwtStubModel()
        cb = ParameterWatcherCallback(param_patterns=["*cwt*"])
        matched = cb._discover_matched_params(model)
        assert any("cwt" in name for name, _ in matched)

    def test_log_gradients_collects_optimizer_metrics(self):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.full((2,), 0.5)
        optimizer = torch.optim.AdamW([param], lr=0.01)
        optimizer.step()
        optimizer.zero_grad()
        param.grad = torch.full((2,), 0.5)

        cb = ParameterWatcherCallback(
            param_patterns=["*"],
            log_gradients=True,
            individual_value_threshold=8,
        )
        metrics: dict = {}
        cb._collect_gradient_metrics(
            metrics,
            "params/test",
            param,
            optimizer.state,
            optimizer,
        )

        assert "params/test/grad/norm" in metrics
        assert "params/test/grad_to_param_ratio" in metrics
        assert "params/test/optimizer/exp_avg_norm" in metrics
        assert "params/test/optimizer/update_to_param_ratio" in metrics

    def test_trainer_config_instantiates_with_log_gradients(self):
        from hydra.utils import instantiate

        from tests.test_configs.conftest import CONFIGS_ROOT

        trainer_cfg = OmegaConf.load(CONFIGS_ROOT / "trainer" / "default.yaml")
        cb = instantiate(trainer_cfg.callbacks.parameter_watcher)
        assert cb.log_gradients is True


class TestCwtLrMultiplier:
    def test_single_group_when_multiplier_is_one(self):
        model = _CwtStubModel()
        module = FoundryModule(model=model, cwt_lr_multiplier=1.0)
        groups = module._build_param_groups()
        assert len(groups) == 1
        assert groups[0]["lr"] == module.learning_rate

    def test_separate_cwt_param_group(self):
        class _NamedCwt(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.other = nn.Linear(2, 2)
                self.tokenizer = nn.Module()
                self.tokenizer.cwt = nn.Linear(2, 2)
                self.task_configs = {}

            def forward(self, x):
                return x

        named = _NamedCwt()
        module = FoundryModule(
            model=named, learning_rate=1e-2, cwt_lr_multiplier=10.0
        )
        groups = module._build_param_groups()

        assert len(groups) == 2
        assert groups[0]["lr"] == pytest.approx(1e-2)
        assert groups[1]["lr"] == pytest.approx(1e-1)

    def test_module_configs_wire_cwt_lr_multiplier(self):
        from hydra.utils import instantiate

        from tests.test_configs.conftest import (
            CONFIGS_ROOT,
            load_resolved_config,
        )

        from foundry.tasks.config import TaskConfig

        task_cfg = TaskConfig.from_yaml(
            CONFIGS_ROOT / "tasks" / "neurosoft_on_vs_off.yaml"
        )
        stub = _CwtStubModel()
        stub.task_configs = {task_cfg.name: task_cfg}

        cfg = load_resolved_config(CONFIGS_ROOT / "module" / "default.yaml")
        assert cfg.cwt_lr_multiplier == 1.0
        module = instantiate(cfg, model=stub)
        assert module.cwt_lr_multiplier == 1.0

"""GPU integration test for QA plan manual checks (issue 04).

Exercises EffectiveBatchSizeCallback, ParameterWatcherCallback, and
cwt_lr_multiplier end-to-end on GPU with a synthetic model containing a
CWTEmbedding-like submodule.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from foundry.models.embeddings.temporal.cwt import ContinuousCWTLayer
from foundry.training.callbacks import (
    EffectiveBatchSizeCallback,
    ParameterWatcherCallback,
)
from foundry.training.task_modules import BaseMultitaskModule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)


class _CwtModel(nn.Module):
    """Synthetic model with a .cwt. sub-module to exercise param watching."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(16, 16)
        self.cwt = ContinuousCWTLayer(
            num_freqs=4, min_freq=1.0, max_freq=30.0, freq_spacing="log"
        )
        self.head = nn.Linear(16, 2)
        self.readout_specs = {}

    def forward(self, x=None, **kwargs):
        return self.encoder(x)

    def get_watched_params(self):
        return self.cwt.get_watched_params()


class _SimpleGpuModule(L.LightningModule):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        out = self.model.encoder(x)
        return out.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class _CwtTaskModule(BaseMultitaskModule):
    def _build_task_metrics(self, task_name, spec, prefix):
        raise NotImplementedError


class _FakeGpuDatamodule:
    def __init__(self, batch_size: int = 8, num_workers: int = 0) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._dataset = TensorDataset(torch.randn(256, 16))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class TestManualCheck1_EffectiveBatchSizeGPU:
    """Manual check 1: EffectiveBatchSizeCallback on GPU."""

    def test_gpu_batch_size_search_logs_result(self, caplog):
        dm = _FakeGpuDatamodule(batch_size=8, num_workers=0)
        model = _CwtModel()
        module = _SimpleGpuModule(model=model)

        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.datamodule = dm

        callback = EffectiveBatchSizeCallback(
            effective_batch_size=64,
            init_val=8,
            max_val=64,
        )

        with caplog.at_level(logging.INFO):
            callback.on_fit_start(trainer, module.to("cuda"))

        log_output = caplog.text
        assert "EffectiveBatchSize: found max batch_size=" in log_output
        assert dm.num_workers == 0
        assert trainer.accumulate_grad_batches >= 1
        assert dm.batch_size >= 8


class TestManualCheck2_ParameterWatcherGPU:
    """Manual check 2: ParameterWatcher logs CWT params and gradients."""

    def test_discovers_and_logs_cwt_params_on_gpu(self, caplog):
        model = _CwtModel()
        module = _SimpleGpuModule(model=model).to("cuda")

        cb = ParameterWatcherCallback(
            param_patterns=["*cwt*"],
            log_every_n_steps=1,
            log_gradients=True,
        )

        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        with caplog.at_level(logging.INFO):
            cb.on_fit_start(trainer, module)

        assert cb._matched_params is not None
        cwt_names = [n for n, _ in cb._matched_params]
        assert len(cwt_names) > 0
        assert any("cwt" in n for n in cwt_names)
        assert "ParameterWatcherCallback: watching" in caplog.text
        assert "ParameterWatcherCallback: found derived params" in caplog.text

    def test_logs_gradient_metrics_after_step(self):
        model = _CwtModel().to("cuda")

        cwt_params = [p for n, p in model.named_parameters() if "cwt" in n]
        optimizer = torch.optim.AdamW(cwt_params, lr=1e-3)

        # Use CWT forward path so gradients flow through cwt params
        x = torch.randn(2, 4, 128, device="cuda")  # (B, C, T)
        fs = torch.tensor([250.0, 250.0], device="cuda")
        seq_lens = torch.tensor([128, 128], device="cuda")
        out = model.cwt(x, fs, seq_lens, target_time_tokens=10)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second step to build optimizer state
        out2 = model.cwt(x, fs, seq_lens, target_time_tokens=10)
        out2.sum().backward()

        cb = ParameterWatcherCallback(
            param_patterns=["*cwt*"],
            log_every_n_steps=1,
            log_gradients=True,
        )
        cb._matched_params = cb._discover_matched_params(model)

        metrics: dict = {}
        param_dict = dict(model.named_parameters())
        for name, _ in cb._matched_params:
            param = param_dict[name]
            cb._collect_gradient_metrics(
                metrics,
                f"params/{name}",
                param,
                optimizer.state,
                optimizer,
            )

        grad_keys = [k for k in metrics if "/grad/" in k]
        optimizer_keys = [k for k in metrics if "/optimizer/" in k]
        ratio_keys = [k for k in metrics if "update_to_param_ratio" in k]

        assert len(grad_keys) > 0, (
            f"Expected grad metrics, got: {list(metrics.keys())}"
        )
        assert len(optimizer_keys) > 0, (
            f"Expected optimizer metrics, got: {list(metrics.keys())}"
        )
        assert len(ratio_keys) > 0, (
            f"Expected ratio metrics, got: {list(metrics.keys())}"
        )


class TestManualCheck3_CwtLrMultiplier:
    """Manual check 3: cwt_lr_multiplier=10 separates param groups."""

    def test_separate_groups_printed_on_gpu(self, capsys):
        model = _CwtModel()
        module = _CwtTaskModule(
            model=model,
            learning_rate=1e-4,
            cwt_lr_multiplier=10.0,
        )

        groups = module._build_param_groups()

        captured = capsys.readouterr()
        assert len(groups) == 2

        base_group = groups[0]
        cwt_group = groups[1]

        assert base_group["lr"] == pytest.approx(1e-4)
        assert cwt_group["lr"] == pytest.approx(1e-3)

        assert "CWT LR multiplier: 10.0x" in captured.out
        assert "cwt_lr=1.00e-03" in captured.out

        cwt_param_count = sum(p.numel() for p in cwt_group["params"])
        assert cwt_param_count > 0
        assert f"{cwt_param_count} params" in captured.out

    def test_configure_optimizers_uses_separate_groups_on_gpu(self):
        model = _CwtModel()
        module = _CwtTaskModule(
            model=model,
            learning_rate=1e-4,
            cwt_lr_multiplier=10.0,
        ).to("cuda")

        mock_trainer = MagicMock()
        mock_trainer.max_epochs = 100
        module.trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)

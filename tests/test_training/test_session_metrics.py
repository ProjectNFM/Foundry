"""Tests for SessionMetricsCallback and per-session metric accumulation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.readout import ReadoutRouter
from foundry.tasks.config import TaskConfig
from foundry.training.callbacks import SessionMetricsCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_CONFIG_DIR = REPO_ROOT / "configs" / "tasks"


class _StubTaskModel(nn.Module):
    def __init__(self, task_configs: dict[str, TaskConfig], embed_dim: int = 8):
        super().__init__()
        self.task_configs = task_configs
        heads = {
            name: instantiate(
                {
                    **cfg.head,
                    "embed_dim": embed_dim,
                    "output_dim": cfg.output_dim,
                }
            )
            for name, cfg in task_configs.items()
        }
        self.router = ReadoutRouter(heads)

    def forward(
        self, output_embs, task_index=None, unpack_output=False, **kwargs
    ):
        return self.router(output_embs, task_index)


class TestShortenSessionId:
    def test_standard_bids_name(self):
        sid = "sub-03_ses-01_task-AcousStim_acq-LH_desc-raw"
        assert (
            SessionMetricsCallback._shorten_session_id(sid)
            == "sub-03_ses-01_acq-LH"
        )

    def test_anesthesia_session(self):
        sid = "sub-03_ses-07_task-AcousStim_acq-RHanest_desc-raw"
        assert (
            SessionMetricsCallback._shorten_session_id(sid)
            == "sub-03_ses-07_acq-RHanest"
        )

    def test_non_bids_passthrough(self):
        sid = "my_custom_recording"
        assert SessionMetricsCallback._shorten_session_id(sid) == sid


class TestSessionMetricsCallbackLifecycle:
    """SessionMetricsCallback initializes buffers, computes, and resets."""

    def _make_module(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_clf",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 3,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 3,
            },
            metric_summary_modes={"acc": "max"},
        )
        model = _StubTaskModel({cfg.name: cfg})
        module = FoundryModule(model=model)
        return module, cfg

    def test_on_fit_start_creates_buffers(self):
        module, _ = self._make_module()
        callback = SessionMetricsCallback()
        trainer = MagicMock()

        callback.on_fit_start(trainer, module)

        assert hasattr(module, "_val_session_buffers")
        assert module._val_session_buffers == {}

    def test_accumulate_session_preds_groups_by_session(self):
        module, cfg = self._make_module()
        module._val_session_buffers = {}

        task_name = cfg.name
        router_idx = module.model.router.get_task_index_by_name(task_name) + 1

        # 4 batch items: 2 from session A, 2 from session B.
        # Each item has 2 output positions for this task.
        task_index = torch.full((4, 2), router_idx)
        session_id = ["sessA", "sessB", "sessA", "sessB"]

        preds = torch.randn(8, 3)
        target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

        module._accumulate_session_preds(
            task_name, preds, target, task_index, session_id
        )

        buf = module._val_session_buffers[task_name]
        assert set(buf.keys()) == {"sessA", "sessB"}
        assert len(buf["sessA"]["preds"]) == 2
        assert len(buf["sessB"]["preds"]) == 2

        a_preds = torch.cat(buf["sessA"]["preds"])
        b_preds = torch.cat(buf["sessB"]["preds"])
        assert a_preds.shape == (4, 3)
        assert b_preds.shape == (4, 3)

        a_targets = torch.cat(buf["sessA"]["targets"])
        b_targets = torch.cat(buf["sessB"]["targets"])
        # Item 0 (sessA) -> [0,1], Item 2 (sessA) -> [1,2]
        assert torch.equal(a_targets, torch.tensor([0, 1, 1, 2]))
        # Item 1 (sessB) -> [2,0], Item 3 (sessB) -> [0,1]
        assert torch.equal(b_targets, torch.tensor([2, 0, 0, 1]))

    def test_accumulate_skips_empty_items(self):
        module, cfg = self._make_module()
        module._val_session_buffers = {}

        task_name = cfg.name
        router_idx = module.model.router.get_task_index_by_name(task_name) + 1

        # Item 0 has 2 preds, item 1 has 0 preds (padding)
        task_index = torch.tensor([[router_idx, router_idx], [0, 0]])
        session_id = ["sessA", "sessB"]

        preds = torch.randn(2, 3)
        target = torch.tensor([0, 1])

        module._accumulate_session_preds(
            task_name, preds, target, task_index, session_id
        )

        buf = module._val_session_buffers[task_name]
        assert "sessA" in buf
        assert "sessB" not in buf

    def test_on_validation_epoch_end_computes_and_logs(self):
        module, cfg = self._make_module()
        task_name = cfg.name

        module._val_session_buffers = {
            task_name: {
                "sessA": {
                    "preds": [torch.tensor([[5.0, -5.0, -5.0]] * 3)],
                    "targets": [torch.tensor([0, 0, 0])],
                },
                "sessB": {
                    "preds": [torch.tensor([[-5.0, 5.0, -5.0]] * 4)],
                    "targets": [torch.tensor([1, 1, 1, 1])],
                },
            }
        }

        mock_logger = MagicMock()
        trainer = MagicMock()
        trainer.logger = mock_logger
        trainer.current_epoch = 5

        callback = SessionMetricsCallback()
        callback.on_validation_epoch_end(trainer, module)

        mock_logger.log_metrics.assert_called_once()
        logged = mock_logger.log_metrics.call_args[0][0]

        assert any("sessA" in k for k in logged)
        assert any("sessB" in k for k in logged)
        assert any("acc" in k for k in logged)

        for v in logged.values():
            assert isinstance(v, float)

        assert module._val_session_buffers == {}

    def test_on_validation_epoch_end_skips_empty_buffers(self):
        module, _ = self._make_module()
        module._val_session_buffers = {}

        mock_logger = MagicMock()
        trainer = MagicMock()
        trainer.logger = mock_logger
        trainer.current_epoch = 0

        callback = SessionMetricsCallback()
        callback.on_validation_epoch_end(trainer, module)

        mock_logger.log_metrics.assert_not_called()

    def test_on_validation_epoch_end_filters_invalid_targets(self):
        module, cfg = self._make_module()
        task_name = cfg.name

        # Mix valid (0, 1) and invalid (-1) targets
        module._val_session_buffers = {
            task_name: {
                "sessA": {
                    "preds": [
                        torch.tensor(
                            [
                                [5.0, -5.0, -5.0],
                                [-5.0, 5.0, -5.0],
                                [0.0, 0.0, 0.0],
                            ]
                        )
                    ],
                    "targets": [torch.tensor([0, 1, -1])],
                },
            }
        }

        mock_logger = MagicMock()
        trainer = MagicMock()
        trainer.logger = mock_logger
        trainer.current_epoch = 0

        callback = SessionMetricsCallback()
        callback.on_validation_epoch_end(trainer, module)

        mock_logger.log_metrics.assert_called_once()
        logged = mock_logger.log_metrics.call_args[0][0]
        acc_key = [k for k in logged if "acc" in k and "balanced" not in k][0]
        assert logged[acc_key] == pytest.approx(1.0)

    def test_works_without_buffers_attribute(self):
        """Module without _val_session_buffers should not crash."""
        module, _ = self._make_module()
        trainer = MagicMock()

        callback = SessionMetricsCallback()
        callback.on_validation_epoch_end(trainer, module)

        trainer.logger.log_metrics.assert_not_called()


class TestSessionMetricsEndToEnd:
    """Test the full accumulation + callback pipeline with _shared_step."""

    def _make_single_task_module(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_clf",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
            metrics={
                "_target_": "foundry.tasks.metrics.classification_metrics",
                "num_classes": 2,
            },
            metric_summary_modes={"acc": "max"},
        )
        model = _StubTaskModel({cfg.name: cfg}, embed_dim=4)
        module = FoundryModule(model=model)
        module._val_session_buffers = {}
        return module, cfg

    def test_shared_step_accumulates_when_buffers_present(self):
        module, cfg = self._make_single_task_module()
        task_name = cfg.name
        router_idx = module.model.router.get_task_index_by_name(task_name) + 1

        batch = {
            "output_embs": torch.randn(4, 4),
            "task_index": torch.full((2, 2), router_idx),
            "target_values": {task_name: torch.tensor([0, 1, 0, 1])},
            "target_weights": {task_name: 1.0},
            "session_id": ["sess_A", "sess_B"],
        }

        module.log = MagicMock()
        module.log_dict = MagicMock()

        module._shared_step("val", batch)

        assert task_name in module._val_session_buffers
        assert "sess_A" in module._val_session_buffers[task_name]
        assert "sess_B" in module._val_session_buffers[task_name]

    def test_shared_step_does_not_accumulate_on_train(self):
        module, cfg = self._make_single_task_module()
        task_name = cfg.name
        router_idx = module.model.router.get_task_index_by_name(task_name) + 1

        batch = {
            "output_embs": torch.randn(4, 4),
            "task_index": torch.full((2, 2), router_idx),
            "target_values": {task_name: torch.tensor([0, 1, 0, 1])},
            "target_weights": {task_name: 1.0},
            "session_id": ["sess_A", "sess_B"],
        }

        module.log = MagicMock()
        module.log_dict = MagicMock()

        module._shared_step("train", batch)

        assert module._val_session_buffers == {}

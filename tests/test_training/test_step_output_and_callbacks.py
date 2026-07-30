"""Tests for StepOutput contract and callback-owned state (Step 6).

Covers:
- StepOutput construction and field access
- ReconstructionVisualizationCallback: callback-owned buffers, bounded
  buffering, state isolation between instances, state reset at epoch
  boundaries, no logging without W&B, figure cleanup
- SessionMetricsCallback: callback-owned buffers, accumulation from
  StepOutput, state reset
- FoundryModule has no reconstruction/session buffer attributes
- Hydra target resolution after callbacks/ split
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.readout import ReadoutRouter
from foundry.models.ssl_meta import ReconstructionVizMeta
from foundry.tasks.config import TaskConfig
from foundry.training.callbacks import (
    ReconstructionVisualizationCallback,
    SessionMetricsCallback,
)
from foundry.training.step_output import StepOutput, extract_step_output

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"


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


def _make_viz_meta(
    B: int = 2, C: int = 3, N: int = 10
) -> ReconstructionVizMeta:
    """Build a minimal ReconstructionVizMeta for testing."""
    total_tokens = C * N
    mask_count = total_tokens // 4
    mask_indices = torch.stack(
        [torch.randperm(total_tokens)[:mask_count] for _ in range(B)]
    )
    validity_mask = torch.ones(B, mask_count, dtype=torch.bool)
    return ReconstructionVizMeta(
        mask_indices=mask_indices,
        validity_mask=validity_mask,
        num_channels=C,
        num_time_tokens=N,
    )


def _make_step_output(
    B: int = 2,
    C: int = 3,
    N: int = 10,
    with_viz: bool = True,
    session_id: list[str] | None = None,
) -> StepOutput:
    """Build a representative StepOutput for testing."""
    viz_meta = _make_viz_meta(B, C, N) if with_viz else None
    valid_count = viz_meta.validity_mask.sum().item() if viz_meta else 0

    return StepOutput(
        loss=torch.tensor(1.0),
        task_outputs={
            "masked_reconstruction": torch.randn(int(valid_count))
            if viz_meta
            else torch.empty(0),
        },
        target_values={},
        target_weights={},
        task_index=torch.zeros(B, 2, dtype=torch.long),
        session_id=session_id,
        ssl_task_names=set(),
        reconstruction_viz=viz_meta,
        reconstruction_targets=torch.randn(B, C, N) if with_viz else None,
        input_mask=torch.ones(B, C, dtype=torch.bool) if with_viz else None,
    )


# ---------------------------------------------------------------------------
# StepOutput contract
# ---------------------------------------------------------------------------


class TestStepOutput:
    def test_construction_with_all_fields(self):
        so = _make_step_output(session_id=["s0", "s1"])
        assert so.loss.item() == pytest.approx(1.0)
        assert "masked_reconstruction" in so.task_outputs
        assert so.session_id == ["s0", "s1"]
        assert so.reconstruction_viz is not None
        assert so.reconstruction_targets is not None
        assert so.input_mask is not None

    def test_construction_without_optional_fields(self):
        so = _make_step_output(with_viz=False)
        assert so.reconstruction_viz is None
        assert so.reconstruction_targets is None
        assert so.input_mask is None
        assert so.session_id is None
        assert so.ssl_task_names == set()

    def test_extract_step_output_from_dict(self):
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}
        assert extract_step_output(outputs) is so

    def test_extract_step_output_returns_none_for_tensor(self):
        assert extract_step_output(torch.tensor(1.0)) is None

    def test_extract_step_output_returns_none_for_missing_key(self):
        assert extract_step_output({"loss": torch.tensor(1.0)}) is None

    def test_extract_step_output_returns_none_for_none(self):
        assert extract_step_output(None) is None


# ---------------------------------------------------------------------------
# ReconstructionVisualizationCallback
# ---------------------------------------------------------------------------


class TestReconstructionVisualizationCallback:
    def test_buffers_are_on_callback_instance(self):
        cb = ReconstructionVisualizationCallback(num_examples=4)
        assert isinstance(cb._val_buffer, list)
        assert isinstance(cb._train_buffer, list)
        assert len(cb._val_buffer) == 0
        assert len(cb._train_buffer) == 0

    def test_two_instances_do_not_share_state(self):
        cb1 = ReconstructionVisualizationCallback(num_examples=4)
        cb2 = ReconstructionVisualizationCallback(num_examples=4)

        so = _make_step_output()
        cb1._buffer_examples(so, cb1._train_buffer)

        assert len(cb1._train_buffer) > 0
        assert len(cb2._train_buffer) == 0

    def test_bounded_buffering_respects_num_examples(self):
        cb = ReconstructionVisualizationCallback(num_examples=2)
        so = _make_step_output(B=8)

        cb._buffer_examples(so, cb._val_buffer)

        assert len(cb._val_buffer) == 2

    def test_buffer_detaches_and_moves_to_cpu(self):
        cb = ReconstructionVisualizationCallback(num_examples=4)
        so = _make_step_output()
        cb._buffer_examples(so, cb._val_buffer)

        for example in cb._val_buffer:
            for key in (
                "targets",
                "predictions",
                "mask_indices",
                "validity_mask",
                "input_mask",
            ):
                t = example[key]
                assert t.device == torch.device("cpu")
                assert not t.requires_grad

    def test_val_buffer_resets_on_epoch_end(self):
        cb = ReconstructionVisualizationCallback(num_examples=4)
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        trainer.logger = None
        pl_module = MagicMock()

        cb.on_validation_batch_end(trainer, pl_module, outputs, None, 0)
        assert len(cb._val_buffer) > 0

        cb.on_validation_epoch_end(trainer, pl_module)
        assert len(cb._val_buffer) == 0

    def test_train_buffer_resets_after_logging(self):
        cb = ReconstructionVisualizationCallback(
            num_examples=4, log_every_n_steps=1
        )
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        trainer.global_step = 1
        trainer.logger = None
        pl_module = MagicMock()

        cb.on_train_batch_end(trainer, pl_module, outputs, None, 0)
        assert len(cb._train_buffer) == 0

    def test_no_logging_without_wandb(self):
        """No crash and no logging when W&B logger is absent."""
        cb = ReconstructionVisualizationCallback(
            num_examples=4, log_every_n_steps=1
        )
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        trainer.global_step = 1
        trainer.logger = None
        pl_module = MagicMock()

        cb.on_train_batch_end(trainer, pl_module, outputs, None, 0)
        cb.on_validation_batch_end(trainer, pl_module, outputs, None, 0)
        cb.on_validation_epoch_end(trainer, pl_module)

    def test_no_buffering_without_viz_meta(self):
        cb = ReconstructionVisualizationCallback(num_examples=4)
        so = _make_step_output(with_viz=False)
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        pl_module = MagicMock()

        cb.on_validation_batch_end(trainer, pl_module, outputs, None, 0)
        assert len(cb._val_buffer) == 0

    def test_on_train_batch_end_buffers_every_step(self):
        """Buffering happens every step, not just at logging intervals."""
        cb = ReconstructionVisualizationCallback(
            num_examples=4, log_every_n_steps=100
        )
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        trainer.global_step = 5
        trainer.logger = None
        pl_module = MagicMock()

        cb.on_train_batch_end(trainer, pl_module, outputs, None, 0)
        assert len(cb._train_buffer) > 0

    def test_log_every_n_steps_zero_disables_train_logging(self):
        cb = ReconstructionVisualizationCallback(
            num_examples=4, log_every_n_steps=0
        )
        so = _make_step_output()
        outputs = {"loss": so.loss, "step_output": so}

        trainer = MagicMock()
        trainer.global_step = 0
        trainer.logger = None
        pl_module = MagicMock()

        cb.on_train_batch_end(trainer, pl_module, outputs, None, 0)
        assert len(cb._train_buffer) > 0


# ---------------------------------------------------------------------------
# SessionMetricsCallback state isolation
# ---------------------------------------------------------------------------


class TestSessionMetricsCallbackStateIsolation:
    def test_two_instances_do_not_share_buffers(self):
        cb1 = SessionMetricsCallback()
        cb2 = SessionMetricsCallback()
        cb1._val_session_buffers["task_a"] = {
            "sess1": {"preds": [], "targets": []}
        }
        assert cb2._val_session_buffers == {}

    def test_fit_start_resets_buffers(self):
        cb = SessionMetricsCallback()
        cb._val_session_buffers["stale"] = {}

        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_fit_start(trainer, pl_module)

        assert cb._val_session_buffers == {}


# ---------------------------------------------------------------------------
# FoundryModule has no callback-created attributes
# ---------------------------------------------------------------------------


class TestFoundryModuleNoCallbackState:
    def test_no_reconstruction_buffers(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubTaskModel({cfg.name: cfg})
        module = FoundryModule(model=model)

        assert not hasattr(module, "_reconstruction_viz_buffer")
        assert not hasattr(module, "_reconstruction_train_viz_buffer")
        assert not hasattr(module, "_reconstruction_viz_max_examples")

    def test_no_session_buffers(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubTaskModel({cfg.name: cfg})
        module = FoundryModule(model=model)

        assert not hasattr(module, "_val_session_buffers")

    def test_shared_step_returns_step_output(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubTaskModel({cfg.name: cfg}, embed_dim=4)
        module = FoundryModule(model=model)
        module.log = MagicMock()
        module.log_dict = MagicMock()

        router_idx = model.router.get_task_index_by_name("test_task") + 1
        batch = {
            "output_embs": torch.randn(4, 4),
            "task_index": torch.full((2, 2), router_idx),
            "target_values": {"test_task": torch.tensor([0, 1, 0, 1])},
            "target_weights": {"test_task": 1.0},
            "session_id": ["s0", "s1"],
        }

        result = module._shared_step("val", batch)
        assert isinstance(result, StepOutput)
        assert result.loss.dim() == 0
        assert result.session_id == ["s0", "s1"]
        assert "test_task" in result.task_outputs

    def test_training_step_returns_dict_with_loss(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubTaskModel({cfg.name: cfg}, embed_dim=4)
        module = FoundryModule(model=model)
        module.log = MagicMock()
        module.log_dict = MagicMock()

        router_idx = model.router.get_task_index_by_name("test_task") + 1
        batch = {
            "output_embs": torch.randn(4, 4),
            "task_index": torch.full((2, 2), router_idx),
            "target_values": {"test_task": torch.tensor([0, 1, 0, 1])},
            "target_weights": {"test_task": 1.0},
        }

        result = module.training_step(batch, 0)
        assert isinstance(result, dict)
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert "step_output" in result
        assert isinstance(result["step_output"], StepOutput)

    def test_validation_step_returns_dict_with_loss(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig(
            name="test_task",
            head={
                "_target_": "foundry.tasks.heads.ReadoutHead",
                "output_dim": 2,
            },
            target_extractor={"timestamp_key": "t", "value_key": "v"},
            loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
        )
        model = _StubTaskModel({cfg.name: cfg}, embed_dim=4)
        module = FoundryModule(model=model)
        module.log = MagicMock()
        module.log_dict = MagicMock()

        router_idx = model.router.get_task_index_by_name("test_task") + 1
        batch = {
            "output_embs": torch.randn(4, 4),
            "task_index": torch.full((2, 2), router_idx),
            "target_values": {"test_task": torch.tensor([0, 1, 0, 1])},
            "target_weights": {"test_task": 1.0},
        }

        result = module.validation_step(batch, 0)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "step_output" in result


# ---------------------------------------------------------------------------
# Hydra target resolution
# ---------------------------------------------------------------------------


class TestHydraTargetResolution:
    """All existing Hydra _target_ strings resolve after the callbacks/ split."""

    @pytest.mark.parametrize(
        "target",
        [
            "foundry.training.callbacks.VocabInitializerCallback",
            "foundry.training.callbacks.ReconstructionVisualizationCallback",
            "foundry.training.callbacks.ConfusionMatrixCallback",
            "foundry.training.callbacks.SessionMetricsCallback",
            "foundry.training.callbacks.ParameterWatcherCallback",
            "foundry.training.callbacks.DeterministicSamplerCallback",
            "foundry.training.callbacks.EffectiveBatchSizeCallback",
        ],
    )
    def test_target_resolves(self, target: str):
        from hydra._internal.utils import _locate

        cls = _locate(target)
        assert cls is not None, f"Could not resolve {target}"

    def test_default_trainer_config_callbacks_instantiate(self):
        from omegaconf import OmegaConf

        trainer_cfg = OmegaConf.load(CONFIGS_ROOT / "trainer" / "default.yaml")
        for cb_name, cb_cfg in trainer_cfg.callbacks.items():
            target = cb_cfg.get("_target_")
            if target is None:
                continue
            cb = instantiate(cb_cfg)
            assert cb is not None, f"Failed to instantiate callback {cb_name}"

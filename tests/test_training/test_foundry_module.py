"""Tests for FoundryModule (issue 05)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.readout import ReadoutRouter
from foundry.tasks.config import TaskConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_CONFIG_DIR = REPO_ROOT / "configs" / "tasks"


class _StubTaskModel(nn.Module):
    """Minimal model exposing task_configs and router for module tests."""

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

    def forward(
        self, output_embs, task_index=None, unpack_output=False, **kwargs
    ):
        return self.router(output_embs, task_index)


def test_foundry_module_instantiates_losses_from_task_config_yaml():
    from foundry.training import FoundryModule

    clf_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "neurosoft_on_vs_off.yaml"
    )
    reg_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "ajile_pose_estimation.yaml"
    )
    task_configs = {clf_cfg.name: clf_cfg, reg_cfg.name: reg_cfg}
    model = _StubTaskModel(task_configs)

    module = FoundryModule(model=model)

    assert set(module._task_losses.keys()) == set(task_configs.keys())
    assert module.train_metrics[clf_cfg.name] is not None
    assert module.val_metrics[reg_cfg.name] is not None


def test_sequence_weighted_multitask_loss_matches_spec_id_weighting():
    from foundry.training import FoundryModule

    clf_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "neurosoft_on_vs_off.yaml"
    )
    reg_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "ajile_pose_estimation.yaml"
    )
    task_configs = {clf_cfg.name: clf_cfg, reg_cfg.name: reg_cfg}
    model = _StubTaskModel(task_configs, embed_dim=4)
    module = FoundryModule(model=model)
    module.to(torch.device("cpu"))

    clf_name, reg_name = clf_cfg.name, reg_cfg.name
    clf_idx = model.router.get_task_index_by_name(clf_name) + 1
    reg_idx = model.router.get_task_index_by_name(reg_name) + 1

    # 3 clf sequences + 1 reg sequence
    task_index = torch.tensor(
        [
            [clf_idx, clf_idx],
            [clf_idx, reg_idx],
            [clf_idx, clf_idx],
            [reg_idx, reg_idx],
        ]
    )

    outputs = {
        clf_name: torch.tensor([[2.0, -1.0], [1.0, 0.0], [0.5, 0.5]]),
        reg_name: torch.tensor([[1.0, 2.0, 3.0] * 6]),  # (1, 18)
    }
    target_values = {
        clf_name: torch.tensor([1, 0, 1]),
        reg_name: torch.tensor([[1.0] * 18]),
    }
    target_weights = {clf_name: 1.0, reg_name: 1.0}

    total_loss, _ = module._compute_task_losses(
        outputs, target_values, target_weights, task_index
    )

    clf_loss = module._task_losses[clf_name](
        outputs[clf_name], target_values[clf_name], 1.0
    )
    reg_loss = module._task_losses[reg_name](
        outputs[reg_name], target_values[reg_name], 1.0
    )
    num_clf = torch.any(task_index == clf_idx, dim=1).sum()
    num_reg = torch.any(task_index == reg_idx, dim=1).sum()
    expected = (clf_loss * num_clf + reg_loss * num_reg) / (num_clf + num_reg)

    assert total_loss == pytest.approx(expected.item())


@pytest.mark.parametrize(
    ("yaml_name", "preds", "targets", "expected_pred_shape"),
    [
        (
            "neurosoft_on_vs_off",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([1, 0]),
            (2,),
        ),
        (
            "sleep_stage_5class",
            torch.randn(3, 5),
            torch.tensor([0, 2, 4]),
            (3, 5),
        ),
        (
            "ajile_pose_estimation",
            torch.randn(2, 18),
            torch.randn(2, 18),
            (2, 18),
        ),
    ],
)
def test_prepare_for_metrics_uses_task_config_kind(
    yaml_name, preds, targets, expected_pred_shape
):
    from foundry.training import FoundryModule

    cfg = TaskConfig.from_yaml(TASKS_CONFIG_DIR / f"{yaml_name}.yaml")
    model = _StubTaskModel({cfg.name: cfg})
    module = FoundryModule(model=model)

    metric_preds, metric_targets = module._prepare_for_metrics(
        cfg, preds, targets
    )

    assert metric_preds.shape == expected_pred_shape
    assert torch.equal(metric_targets, targets)
    if cfg.kind == "multiclass":
        assert torch.allclose(
            metric_preds.sum(dim=-1), torch.ones(preds.shape[0])
        )
    if cfg.kind == "binary":
        assert torch.all((metric_preds >= 0) & (metric_preds <= 1))


def test_cwt_lr_param_groups_separate_tokenizer_cwt_params():
    from foundry.training import FoundryModule

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


def test_transfer_batch_to_device_converts_float64_to_float32():
    from foundry.training import FoundryModule

    cfg = TaskConfig.from_yaml(TASKS_CONFIG_DIR / "neurosoft_on_vs_off.yaml")
    module = FoundryModule(model=_StubTaskModel({cfg.name: cfg}))

    batch = {
        "output_embs": torch.ones(2, 4, dtype=torch.float64),
        "task_index": torch.zeros(2, 1, dtype=torch.int64),
        "target_values": {cfg.name: torch.tensor([0, 1], dtype=torch.float64)},
        "target_weights": {cfg.name: 1.0},
    }

    moved = module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

    assert moved["output_embs"].dtype == torch.float32
    assert moved["target_values"][cfg.name].dtype == torch.float32


def test_wandb_metric_summaries_use_task_config_modes():
    from unittest.mock import MagicMock

    from foundry.training import FoundryModule

    cfg = TaskConfig.from_yaml(TASKS_CONFIG_DIR / "sleep_stage_5class.yaml")
    model = _StubTaskModel({cfg.name: cfg})
    module = FoundryModule(model=model)

    from lightning.pytorch.loggers import WandbLogger

    experiment = MagicMock()
    wandb_logger = MagicMock(spec=WandbLogger)
    wandb_logger.experiment = experiment
    trainer = MagicMock()
    trainer.logger = wandb_logger
    module._trainer = trainer

    module._configure_wandb_metric_summaries()

    experiment.define_metric.assert_any_call("train/loss", summary="min")
    experiment.define_metric.assert_any_call("val/loss", summary="min")
    experiment.define_metric.assert_any_call(
        f"train/{cfg.name}_loss", summary="min"
    )
    for metric_name, mode in cfg.metric_summary_modes.items():
        if metric_name == "loss":
            continue
        experiment.define_metric.assert_any_call(
            f"train/{cfg.name}_{metric_name}", summary=mode
        )


def test_per_task_metrics_match_classification_and_regression_modules():
    from foundry.training import FoundryModule

    clf_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "neurosoft_on_vs_off.yaml"
    )
    reg_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "ajile_pose_estimation.yaml"
    )
    task_configs = {clf_cfg.name: clf_cfg, reg_cfg.name: reg_cfg}
    module = FoundryModule(model=_StubTaskModel(task_configs))

    clf_preds = torch.tensor([[-1.0, 2.0], [2.0, -1.0]])
    clf_targets = torch.tensor([1, 0])
    reg_preds = torch.tensor([[1.0, 2.0] + [0.0] * 16, [1.0] * 18])
    reg_targets = reg_preds.clone()

    clf_metric_preds, clf_metric_targets = module._prepare_for_metrics(
        clf_cfg, clf_preds, clf_targets
    )
    reg_metric_preds, reg_metric_targets = module._prepare_for_metrics(
        reg_cfg, reg_preds, reg_targets
    )

    module.train_metrics[clf_cfg.name].update(
        clf_metric_preds, clf_metric_targets
    )
    module.train_metrics[reg_cfg.name].update(
        reg_metric_preds, reg_metric_targets
    )

    clf_result = module.train_metrics[clf_cfg.name].compute()
    reg_result = module.train_metrics[reg_cfg.name].compute()

    assert "train/neurosoft_on_vs_off_acc" in clf_result
    assert "train/ajile_pose_estimation_mse" in reg_result
    assert clf_result["train/neurosoft_on_vs_off_acc"] == pytest.approx(1.0)
    assert reg_result["train/ajile_pose_estimation_mse"] == pytest.approx(0.0)


def test_default_module_yaml_instantiates_foundry_module():
    from hydra.utils import instantiate

    from foundry.training import FoundryModule
    from tests.test_configs.conftest import CONFIGS_ROOT, load_resolved_config

    task_cfg = TaskConfig.from_yaml(
        TASKS_CONFIG_DIR / "neurosoft_on_vs_off.yaml"
    )
    module_cfg = load_resolved_config(CONFIGS_ROOT / "module" / "default.yaml")
    module = instantiate(
        module_cfg, model=_StubTaskModel({task_cfg.name: task_cfg})
    )
    assert isinstance(module, FoundryModule)

"""Tests for confusion matrix logging in FoundryModule."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from hydra.utils import instantiate

from foundry.models.readout import ReadoutRouter
from foundry.tasks.classification_mapping import ClassificationMapping
from foundry.tasks.config import TaskConfig
from foundry.training.confusion_matrix import (
    ConfusionMatrixTracker,
    compute_confusion_matrix,
)

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


class TestComputeConfusionMatrix:
    """Verify raw counts and normalization from predictions and targets."""

    def test_perfect_predictions_produce_diagonal_matrix(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        counts, normalized = compute_confusion_matrix(
            preds, targets, num_classes=3
        )

        assert counts.shape == (3, 3)
        assert torch.equal(counts, torch.diag(torch.tensor([2, 2, 2])))
        # Normalized: each row sums to 1
        assert torch.allclose(normalized.sum(dim=1), torch.ones(3))
        assert torch.allclose(normalized, torch.eye(3))

    def test_all_misclassified_produces_off_diagonal(self):
        preds = torch.tensor([1, 0])
        targets = torch.tensor([0, 1])
        counts, normalized = compute_confusion_matrix(
            preds, targets, num_classes=2
        )

        expected_counts = torch.tensor([[0, 1], [1, 0]])
        assert torch.equal(counts, expected_counts)
        expected_norm = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.allclose(normalized, expected_norm)

    def test_empty_row_gets_zero_normalization(self):
        preds = torch.tensor([0, 0, 0])
        targets = torch.tensor([0, 0, 0])
        counts, normalized = compute_confusion_matrix(
            preds, targets, num_classes=3
        )

        assert counts[0, 0] == 3
        assert counts[1].sum() == 0
        assert counts[2].sum() == 0
        # Rows with no samples normalize to 0
        assert normalized[1].sum() == 0.0
        assert normalized[2].sum() == 0.0


class TestConfusionMatrixTracker:
    """Tracks predictions across batches and produces matrix at epoch end."""

    def test_accumulates_predictions_across_updates(self):
        tracker = ConfusionMatrixTracker(num_classes=3)
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        tracker.update(torch.tensor([2, 0]), torch.tensor([2, 2]))

        counts, normalized = tracker.compute()
        assert counts.shape == (3, 3)
        assert counts[0, 0] == 1  # pred=0, target=0
        assert counts[2, 0] == 1  # pred=0, target=2
        assert counts[2, 2] == 1  # pred=2, target=2

    def test_reset_clears_state(self):
        tracker = ConfusionMatrixTracker(num_classes=2)
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        tracker.reset()
        counts, _ = tracker.compute()
        assert counts.sum() == 0

    def test_class_names_attached(self):
        tracker = ConfusionMatrixTracker(
            num_classes=2, class_names=["Wake", "Sleep"]
        )
        assert tracker.class_names == ["Wake", "Sleep"]

    def test_default_class_names(self):
        tracker = ConfusionMatrixTracker(num_classes=3)
        assert tracker.class_names == ["class_0", "class_1", "class_2"]


class TestFoundryModuleConfusionMatrixIntegration:
    """FoundryModule logs confusion matrices at validation epoch end."""

    def _make_module_with_mapping(self):
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
            classification_mapping=ClassificationMapping(
                raw_to_mapped={0: 0, 1: 1, 2: 2},
                names={0: "Wake", 1: "N2", 2: "REM"},
            ),
        )
        model = _StubTaskModel({cfg.name: cfg})
        module = FoundryModule(model=model)
        return module, cfg

    def test_confusion_tracker_created_for_classification_tasks_with_mapping(
        self,
    ):
        module, cfg = self._make_module_with_mapping()
        assert cfg.name in module._val_confusion_trackers

    def test_confusion_tracker_not_created_for_regression(self):
        from foundry.training import FoundryModule

        cfg = TaskConfig.from_yaml(
            TASKS_CONFIG_DIR / "ajile_pose_estimation.yaml"
        )
        model = _StubTaskModel({cfg.name: cfg})
        module = FoundryModule(model=model)
        assert cfg.name not in module._val_confusion_trackers

    def test_confusion_tracker_uses_mapping_class_names(self):
        module, cfg = self._make_module_with_mapping()
        tracker = module._val_confusion_trackers[cfg.name]
        assert tracker.class_names == ["Wake", "N2", "REM"]

    def test_log_confusion_matrices_produces_numeric_payload(self):
        module, cfg = self._make_module_with_mapping()
        mock_logger = MagicMock()

        tracker = module._val_confusion_trackers[cfg.name]
        tracker.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))

        with patch.object(
            type(module),
            "logger",
            new_callable=lambda: property(lambda self: mock_logger),
        ):
            with patch.object(
                type(module),
                "current_epoch",
                new_callable=lambda: property(lambda self: 0),
            ):
                module._log_confusion_matrices()

        mock_logger.log_metrics.assert_called()
        logged = {}
        for c in mock_logger.log_metrics.call_args_list:
            logged.update(c[0][0] if c[0] else c[1].get("metrics", {}))
        assert f"val/{cfg.name}_confusion_counts" in logged

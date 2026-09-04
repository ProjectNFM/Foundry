"""Tests for SourceSessionMetricsCallback (WP3).

Covers per-session supported-class F1, unweighted aggregate mean,
error paths, and metric visibility for early stopping/checkpointing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torchmetrics.classification import F1Score

from foundry.training.callbacks.source_session_metrics import (
    SourceSessionMetricsCallback,
)
from foundry.training.step_output import StepOutput


# ── helpers ──────────────────────────────────────────────────────────────────


def _callback(num_classes=4, metric_key="val/source_session_mean_supported_f1"):
    return SourceSessionMetricsCallback(
        monitor_task="test_task",
        num_classes=num_classes,
        metric_key=metric_key,
    )


def _mock_pl_module(num_classes=4):
    """A minimal LightningModule mock with _prepare_for_metrics returning softmax."""
    module = MagicMock()

    def prepare(cfg, preds, targets):
        return torch.softmax(preds, dim=-1), targets

    module._prepare_for_metrics = prepare
    module.model.task_configs = {"test_task": MagicMock()}
    module.model.router.get_task_index_by_name.return_value = 0
    return module


def _mock_trainer(*, sanity=False):
    trainer = MagicMock()
    trainer.sanity_checking = sanity
    trainer.current_epoch = 0
    trainer.world_size = 1
    trainer.logger = MagicMock()
    return trainer


def _feed_validation_batch(cb, module, session_ids, preds_by_session, targets_by_session):
    """Send one validation batch through the callback's public lifecycle hook."""
    counts = [targets.numel() for targets in targets_by_session]
    task_index = torch.zeros(
        (len(session_ids), max(counts)), dtype=torch.long
    )
    for row, count in enumerate(counts):
        task_index[row, :count] = 1

    step_output = StepOutput(
        loss=torch.tensor(0.0),
        task_outputs={"test_task": torch.cat(preds_by_session)},
        target_values={"test_task": torch.cat(targets_by_session)},
        target_weights={"test_task": 1.0},
        task_index=task_index,
        session_id=list(session_ids),
    )
    cb.on_validation_batch_end(
        _mock_trainer(),
        module,
        outputs={"step_output": step_output},
        batch=None,
        batch_idx=0,
    )


# ── _compute_supported_f1 unit tests ────────────────────────────────────────


class TestComputeSupportedF1:
    """Verify per-session supported-class macro-F1 computation."""

    def test_perfect_predictions(self):
        cb = _callback(num_classes=4)
        preds = torch.tensor([
            [10.0, -10, -10, -10],
            [-10, 10.0, -10, -10],
            [-10, -10, 10.0, -10],
        ])
        targets = torch.tensor([0, 1, 2])
        preds_soft = torch.softmax(preds, dim=-1)

        result = cb._compute_supported_f1(preds_soft, targets)
        assert result is not None
        f1_val, per_class, support, mask = result
        assert f1_val == pytest.approx(1.0, abs=1e-5)
        assert mask.tolist() == [True, True, True, False]

    def test_absent_positive_class_excluded_from_denominator(self):
        """Classes with zero support do not contribute to the macro mean."""
        cb = _callback(num_classes=4)
        preds = torch.tensor([
            [10.0, -10, -10, -10],
            [10.0, -10, -10, -10],
        ])
        targets = torch.tensor([0, 0])
        preds_soft = torch.softmax(preds, dim=-1)

        result = cb._compute_supported_f1(preds_soft, targets)
        assert result is not None
        _, _, support, mask = result
        assert support[0].item() == 2
        assert mask.sum().item() == 1

    def test_no_supported_classes_returns_none(self):
        cb = _callback(num_classes=4)
        preds = torch.empty(0, 4)
        targets = torch.empty(0, dtype=torch.long)

        result = cb._compute_supported_f1(preds, targets)
        assert result is None

    def test_matches_hand_computed_reference(self):
        """Two classes, imperfect predictions: F1 per class then mean."""
        cb = _callback(num_classes=3)
        preds_soft = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ])
        targets = torch.tensor([0, 0, 1, 1])

        f1_metric = F1Score(task="multiclass", num_classes=3, average=None)
        expected_per_class = f1_metric(preds_soft, targets)
        mask = torch.tensor([True, True, False])
        expected_mean = expected_per_class[mask].mean().item()

        result = cb._compute_supported_f1(preds_soft, targets)
        assert result is not None
        actual_f1, _, _, actual_mask = result
        assert actual_mask.tolist() == mask.tolist()
        assert actual_f1 == pytest.approx(expected_mean, abs=1e-6)


# ── unweighted session mean ─────────────────────────────────────────────────


class TestUnweightedSessionMean:
    """The aggregate is the unweighted mean: each session gets equal weight regardless of size."""

    def test_unequal_sizes_get_equal_weight(self):
        cb = _callback(num_classes=3)
        module = _mock_pl_module(num_classes=3)
        trainer = _mock_trainer()

        _feed_validation_batch(
            cb,
            module,
            ["session_A", "session_B"],
            [
                torch.tensor([[10.0, -10, -10]] * 100),
                torch.tensor([[-10, 10.0, -10]] * 5),
            ],
            [
                torch.zeros(100, dtype=torch.long),
                torch.zeros(5, dtype=torch.long),
            ],
        )

        cb.on_validation_epoch_end(trainer, module)

        logged_calls = module.log.call_args_list
        metric_call = next(
            c for c in logged_calls
            if c[0][0] == "val/source_session_mean_supported_f1"
        )
        mean_f1 = metric_call[0][1]

        f1_a = 1.0
        f1_b = 0.0
        expected_mean = (f1_a + f1_b) / 2.0
        assert mean_f1 == pytest.approx(expected_mean, abs=1e-5)

    def test_differs_from_pooled_window_level_f1(self):
        """Unweighted session mean must differ from pooled (window-weighted) F1
        when sessions have different sizes and accuracies."""
        cb = _callback(num_classes=2)
        module = _mock_pl_module(num_classes=2)
        trainer = _mock_trainer()

        _feed_validation_batch(
            cb,
            module,
            ["sess_big", "sess_small"],
            [
                torch.tensor([[10.0, -10]] * 100),
                torch.tensor([[-10, 10.0]] * 3),
            ],
            [
                torch.zeros(100, dtype=torch.long),
                torch.zeros(3, dtype=torch.long),
            ],
        )

        cb.on_validation_epoch_end(trainer, module)

        metric_call = next(
            c for c in module.log.call_args_list
            if c[0][0] == "val/source_session_mean_supported_f1"
        )
        unweighted_mean = metric_call[0][1]

        all_preds = torch.softmax(
            torch.cat([
                torch.tensor([[10.0, -10]] * 100),
                torch.tensor([[-10, 10.0]] * 3),
            ]),
            dim=-1,
        )
        all_targets = torch.cat([
            torch.zeros(100, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
        ])
        pooled_f1 = F1Score(task="multiclass", num_classes=2, average="macro")(
            all_preds, all_targets
        ).item()

        assert unweighted_mean == pytest.approx(0.5, abs=1e-5)
        assert pooled_f1 == pytest.approx(0.4926, abs=1e-4)
        assert unweighted_mean != pytest.approx(pooled_f1, abs=1e-4)


# ── metric visibility for early stopping / checkpointing ────────────────────


class TestMetricVisibility:
    """The aggregate metric must be logged via pl_module.log (not logger-only)."""

    def test_aggregate_logged_via_pl_module_log(self):
        cb = _callback(num_classes=2)
        module = _mock_pl_module(num_classes=2)
        trainer = _mock_trainer()

        _feed_validation_batch(
            cb,
            module,
            ["sess"],
            [torch.tensor([[10.0, -10]])],
            [torch.tensor([0])],
        )

        cb.on_validation_epoch_end(trainer, module)

        metric_keys = [c[0][0] for c in module.log.call_args_list]
        assert "val/source_session_mean_supported_f1" in metric_keys
        assert "val/source_session_count" in metric_keys

    def test_per_session_diagnostics_go_to_logger_only(self):
        cb = _callback(num_classes=2)
        module = _mock_pl_module(num_classes=2)
        trainer = _mock_trainer()

        _feed_validation_batch(
            cb,
            module,
            ["minipigs:sub-01_ses-01_task-X_acq-LH_desc-raw"],
            [torch.tensor([[10.0, -10]])],
            [torch.tensor([0])],
        )

        cb.on_validation_epoch_end(trainer, module)

        logger_call = trainer.logger.log_metrics.call_args
        assert logger_call is not None
        logged_keys = logger_call[0][0]
        assert any("val_session/source_session/" in k for k in logged_keys)
        assert not any("val/source_session/" in k for k in logged_keys)
        assert any("supported_f1" in k for k in logged_keys)


# ── error paths ──────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_empty_buffers_raise(self):
        cb = _callback()
        module = _mock_pl_module()
        trainer = _mock_trainer()

        with pytest.raises(RuntimeError, match="no validation predictions"):
            cb.on_validation_epoch_end(trainer, module)

    def test_session_with_no_valid_targets_raises(self):
        cb = _callback(num_classes=2)
        module = _mock_pl_module(num_classes=2)
        trainer = _mock_trainer()

        _feed_validation_batch(
            cb,
            module,
            ["sess"],
            [torch.tensor([[1.0, 0.0], [0.0, 1.0]])],
            [torch.tensor([-1, -1])],
        )

        with pytest.raises(RuntimeError, match="undefined"):
            cb.on_validation_epoch_end(trainer, module)

    def test_sanity_checking_skips_without_error(self):
        cb = _callback()
        module = _mock_pl_module()
        trainer = _mock_trainer(sanity=True)

        cb._val_session_buffers = {"sess": {"preds": [], "targets": []}}
        cb.on_validation_epoch_end(trainer, module)
        assert cb._val_session_buffers == {}

    def test_distributed_validation_rejected(self):
        cb = _callback()
        module = _mock_pl_module()
        trainer = _mock_trainer()
        trainer.world_size = 2

        with pytest.raises(NotImplementedError, match="distributed"):
            cb.on_fit_start(trainer, module)


# ── _shorten_session_id ─────────────────────────────────────────────────────


class TestShortenSessionId:
    def test_canonical_id_with_namespace(self):
        result = SourceSessionMetricsCallback._shorten_session_id(
            "minipigs:sub-01_ses-02_task-AcousStim_acq-LH_desc-raw"
        )
        assert result == "minipigs:sub-01_ses-02_acq-LH"

    def test_raw_bids_id(self):
        result = SourceSessionMetricsCallback._shorten_session_id(
            "sub-03_ses-01_task-AcousStim_acq-RH_desc-raw"
        )
        assert result == "sub-03_ses-01_acq-RH"

    def test_non_bids_passthrough(self):
        result = SourceSessionMetricsCallback._shorten_session_id("sub-03")
        assert "sub-03" in result

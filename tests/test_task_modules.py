from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from foundry.data.datamodules.neurosoft import (
    LOGFREQ_NORMALIZE_MEAN,
    LOGFREQ_NORMALIZE_STD,
)
from foundry.training import ClassificationModule, RegressionModule


class DummyRegressionModel(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.readout_specs = {
            "test_regression": SimpleNamespace(dim=dim, id=30),
        }


class DummyClassificationModel(nn.Module):
    def __init__(self, dim: int = 3):
        super().__init__()
        self.readout_specs = {
            "test_classification": SimpleNamespace(
                dim=dim, id=20, loss_fn=object()
            ),
        }


def test_regression_module_collects_and_clears_validation_predictions():
    module = RegressionModule(model=DummyRegressionModel())

    module._update_validation_task_state(
        "test_regression",
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[1.5], [2.5]]),
    )

    assert len(module.val_prediction_cache["test_regression"]) == 1

    module._on_validation_epoch_end_task("test_regression")

    assert module.val_prediction_cache["test_regression"] == []


def test_source_ids_are_aligned_to_task_output_tokens():
    module = RegressionModule(model=DummyRegressionModel())

    source_ids = module._source_ids_for_task(
        ["minipigs/session_a", "monkeys/session_b"],
        torch.tensor([[30, 0, 0], [30, 30, 0]]),
        "test_regression",
    )

    assert source_ids == ["minipigs", "monkeys", "monkeys"]


def test_explicit_source_ids_are_aligned_for_single_species_sessions():
    module = RegressionModule(model=DummyRegressionModel())

    source_ids = module._source_ids_for_task(
        ["sub-01_ses-01", "sub-02_ses-01"],
        torch.tensor([[30, 0, 0], [30, 30, 0]]),
        "test_regression",
        sequence_source_ids=["minipigs", "minipigs"],
    )

    assert source_ids == ["minipigs", "minipigs", "minipigs"]


def test_regression_module_collects_source_specific_validation_predictions():
    module = RegressionModule(model=DummyRegressionModel())

    module._update_validation_task_state(
        "test_regression",
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[1.5], [2.5], [3.5]]),
        source_ids=["minipigs", "monkeys", "minipigs"],
    )

    source_cache = module.val_source_prediction_cache["test_regression"]
    minipig_predictions = torch.cat(
        [predictions for predictions, _ in source_cache["minipigs"]]
    )
    monkey_predictions = torch.cat(
        [predictions for predictions, _ in source_cache["monkeys"]]
    )

    torch.testing.assert_close(
        minipig_predictions, torch.tensor([[1.0], [3.0]])
    )
    torch.testing.assert_close(monkey_predictions, torch.tensor([[2.0]]))


def test_regression_module_logs_source_specific_validation_metrics(monkeypatch):
    module = RegressionModule(model=DummyRegressionModel())
    logged = {}

    def capture_log(name, value, **kwargs):
        logged[name] = value

    monkeypatch.setattr(module, "log", capture_log)
    module._update_validation_task_state(
        "test_regression",
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        torch.tensor([[1.0], [2.5], [2.0], [4.5]]),
        source_ids=["minipigs", "minipigs", "monkeys", "monkeys"],
    )

    module._on_validation_epoch_end_task("test_regression")

    assert "val/minipigs/test_regression_mse" in logged
    assert "val/minipigs/test_regression_mae" in logged
    assert "val/minipigs/test_regression_r2" in logged
    assert "val/monkeys/test_regression_mse" in logged
    assert "val/monkeys/test_regression_mae" in logged
    assert "val/monkeys/test_regression_r2" in logged
    assert module.val_source_prediction_cache["test_regression"] == {}


def test_regression_prediction_plot_includes_r2_score():
    module = RegressionModule(model=DummyRegressionModel())

    fig = module._plot_regression_predictions(
        predictions=torch.tensor([1.0, 2.0]),
        targets=torch.tensor([1.0, 3.0]),
        task_name="test_regression",
        r2_score=0.5,
    )

    assert "R2=0.500" in fig.axes[0].get_title()
    plt.close(fig)


def test_log_frequency_regression_values_are_denormalized_for_reporting():
    module = RegressionModule(model=DummyRegressionModel())

    predictions, targets = module._denormalize_regression_values(
        "neurosoft_acoustic_stim_logfreq",
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )

    assert predictions == torch.tensor([LOGFREQ_NORMALIZE_MEAN])
    assert targets == torch.tensor(
        [LOGFREQ_NORMALIZE_MEAN + LOGFREQ_NORMALIZE_STD]
    )


def test_log_frequency_regression_metrics_use_denormalized_values():
    module = RegressionModule(model=DummyRegressionModel())

    predictions, targets = module._prepare_metric_inputs(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        "neurosoft_acoustic_stim_logfreq",
    )

    assert predictions == torch.tensor([LOGFREQ_NORMALIZE_MEAN])
    assert targets == torch.tensor(
        [LOGFREQ_NORMALIZE_MEAN + LOGFREQ_NORMALIZE_STD]
    )


def test_classification_module_collects_source_specific_validation_predictions():
    module = ClassificationModule(model=DummyClassificationModel())
    predictions = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    targets = torch.tensor([0, 1, 2])

    module._update_validation_task_state(
        "test_classification",
        predictions,
        targets,
        source_ids=["minipigs", "monkeys", "minipigs"],
    )

    source_cache = module.val_source_prediction_cache["test_classification"]
    minipig_predictions = torch.cat(
        [preds for preds, _ in source_cache["minipigs"]]
    )
    monkey_predictions = torch.cat(
        [preds for preds, _ in source_cache["monkeys"]]
    )

    torch.testing.assert_close(
        minipig_predictions,
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
    )
    torch.testing.assert_close(
        monkey_predictions, torch.tensor([[0.2, 0.7, 0.1]])
    )


def test_classification_module_logs_source_specific_validation_metrics(
    monkeypatch,
):
    module = ClassificationModule(model=DummyClassificationModel())
    logged = {}

    def capture_log_dict(values, **kwargs):
        logged.update(values)

    monkeypatch.setattr(module, "log_dict", capture_log_dict)
    module._update_validation_task_state(
        "test_classification",
        torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
            ]
        ),
        torch.tensor([0, 1, 2, 0, 1, 2]),
        source_ids=[
            "minipigs",
            "minipigs",
            "minipigs",
            "monkeys",
            "monkeys",
            "monkeys",
        ],
    )

    module._on_validation_epoch_end_task("test_classification")

    assert "val/minipigs/test_classification_acc" in logged
    assert "val/minipigs/test_classification_f1" in logged
    assert "val/minipigs/test_classification_balanced_acc" in logged
    assert "val/monkeys/test_classification_acc" in logged
    assert "val/monkeys/test_classification_f1" in logged
    assert "val/monkeys/test_classification_balanced_acc" in logged
    assert module.val_source_prediction_cache["test_classification"] == {}

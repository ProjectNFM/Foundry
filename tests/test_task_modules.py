from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from foundry.data.datamodules.neurosoft import (
    LOGFREQ_NORMALIZE_MEAN,
    LOGFREQ_NORMALIZE_STD,
)
from foundry.training import RegressionModule


class DummyRegressionModel(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.readout_specs = {
            "test_regression": SimpleNamespace(dim=dim),
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

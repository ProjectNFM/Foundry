import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from foundry.tasks.metrics import (
    classification_metrics,
    regression_metrics,
    ssl_metrics,
)


class TestClassificationMetrics:
    def test_returns_expected_metric_names_multiclass(self):
        metrics = classification_metrics(num_classes=5)

        assert isinstance(metrics, MetricCollection)
        assert set(metrics.keys()) == {
            "acc",
            "f1",
            "auroc",
            "precision",
            "recall",
            "balanced_acc",
            "cohen_kappa",
        }
        for name in metrics.keys():
            assert isinstance(metrics[name], Metric)

        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        metrics.update(preds, targets)
        computed = metrics.compute()
        assert set(computed.keys()) == set(metrics.keys())

    def test_binary_metrics_accept_two_class_inputs(self):
        metrics = classification_metrics(num_classes=2)

        preds = torch.sigmoid(torch.randn(8))
        targets = torch.randint(0, 2, (8,))
        metrics.update(preds, targets)
        computed = metrics.compute()

        assert "acc" in computed
        assert computed["acc"].ndim == 0
        assert type(metrics["acc"]).__name__ == "BinaryAccuracy"


class TestRegressionMetrics:
    def test_returns_expected_metric_names_and_types(self):
        metrics = regression_metrics()

        assert isinstance(metrics, MetricCollection)
        assert set(metrics.keys()) == {"mse", "mae", "r2"}
        assert isinstance(metrics["mse"], MeanSquaredError)
        assert isinstance(metrics["mae"], MeanAbsoluteError)
        assert isinstance(metrics["r2"], R2Score)


class TestSSLMetrics:
    def test_returns_recon_mse(self):
        metrics = ssl_metrics()

        assert isinstance(metrics, MetricCollection)
        assert set(metrics.keys()) == {"recon_mse"}
        assert isinstance(metrics["recon_mse"], MeanSquaredError)

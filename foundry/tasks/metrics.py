from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def classification_metrics(num_classes: int) -> MetricCollection:
    # Always use "multiclass" so that average="macro" is respected for
    # balanced_acc (binary Accuracy ignores the average parameter).
    return MetricCollection(
        {
            "acc": Accuracy(
                task="multiclass",
                num_classes=num_classes,
            ),
            "f1": F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "auroc": AUROC(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
            ),
            "precision": Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "balanced_acc": Accuracy(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "cohen_kappa": CohenKappa(
                task="multiclass",
                num_classes=num_classes,
            ),
        }
    )


def regression_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(multioutput="uniform_average"),
        }
    )


def ssl_metrics() -> MetricCollection:
    return MetricCollection({"recon_mse": MeanSquaredError()})

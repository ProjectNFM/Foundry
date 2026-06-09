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
    task_type = "binary" if num_classes == 2 else "multiclass"
    return MetricCollection(
        {
            "acc": Accuracy(task=task_type, num_classes=num_classes),
            "f1": F1Score(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "auroc": AUROC(task=task_type, num_classes=num_classes),
            "precision": Precision(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "balanced_acc": Accuracy(
                task=task_type, num_classes=num_classes, average="macro"
            ),
            "cohen_kappa": CohenKappa(task=task_type, num_classes=num_classes),
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

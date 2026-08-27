import torch
from torchmetrics import Metric, MetricCollection
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


class _SupportedMeanMetric(Metric):
    """Compute mean of per-class metric values over classes with support > 0.

    All *num_classes* logits are kept; predictions of absent classes are
    penalised because they produce false negatives for the actual class.
    Only the undefined metric term for classes with zero support is excluded.
    """

    def __init__(
        self,
        base_metric_class,
        num_classes: int,
        **base_kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._base = base_metric_class(
            task="multiclass",
            num_classes=num_classes,
            average=None,
            **base_kwargs,
        )
        # AUROC does not expose a confusion matrix or TP/FN state, so support
        # must be tracked independently instead of inferred from metric
        # internals. Summation preserves support across distributed workers.
        self.add_state(
            "target_support",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._base.update(preds, target)
        self.target_support += torch.bincount(
            target.reshape(-1), minlength=self.num_classes
        ).to(self.target_support)

    def compute(self) -> torch.Tensor:
        per_class = self._base.compute()
        support = self.target_support
        mask = support > 0
        if not mask.any():
            return torch.tensor(0.0, device=per_class.device)
        return per_class[mask].mean()

    def _target_support(self) -> torch.Tensor:
        """Compatibility accessor for the explicitly tracked target support."""
        return self.target_support

    def reset(self) -> None:
        super().reset()
        self._base.reset()


class _NumPresentClasses(Metric):
    """Count target classes with at least one sample (support > 0)."""

    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "class_seen",
            default=torch.zeros(num_classes, dtype=torch.bool),
            dist_reduce_fx="max",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for c in range(self.num_classes):
            if (target == c).any():
                self.class_seen[c] = True

    def compute(self) -> torch.Tensor:
        return self.class_seen.sum().float()


def supported_classification_metrics(num_classes: int) -> MetricCollection:
    """Metrics that average only over target classes with support > 0.

    Retains all eight logits so that predicting an absent class is still
    penalised (it creates a false negative for the true class).  Only the
    undefined metric term for zero-support classes is excluded.
    """
    return MetricCollection(
        {
            **classification_metrics(num_classes),
            "supported_f1": _SupportedMeanMetric(
                F1Score, num_classes=num_classes
            ),
            "supported_auroc": _SupportedMeanMetric(
                AUROC, num_classes=num_classes
            ),
            "supported_precision": _SupportedMeanMetric(
                Precision, num_classes=num_classes
            ),
            "supported_recall": _SupportedMeanMetric(
                Recall, num_classes=num_classes
            ),
            "supported_balanced_acc": _SupportedMeanMetric(
                Recall, num_classes=num_classes
            ),
            "num_present_classes": _NumPresentClasses(num_classes=num_classes),
        },
        # The supported metrics keep their actual TorchMetrics state in the
        # nested ``_base`` metric.  Automatic compute grouping only compares
        # the outer ``target_support`` state, so it incorrectly groups all of
        # these wrappers and updates just one of their bases.  This can freeze
        # supported F1/precision/recall after sanity validation.
        compute_groups=False,
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

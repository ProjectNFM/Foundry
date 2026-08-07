"""Task loss functions for the training layer.

Each loss is an ``nn.Module`` with a uniform signature::

    (predictions, targets, sample_weights) -> scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyTaskLoss(nn.Module):
    """Cross-entropy loss for classification tasks.

    Wraps :func:`torch.nn.functional.cross_entropy` with per-sample weighting.
    Class weights and label smoothing are configured at construction time so they
    can be set from YAML and serialized in checkpoints.

    Args:
        label_smoothing: Smoothing factor passed to cross-entropy. ``0.0``
            disables smoothing.
        class_weights: Per-class weights of length ``num_classes``. Registered
            as a buffer when provided.
        ignore_index: Target value that should be ignored in loss computation.
            Defaults to ``-1`` as a safety net for unmapped labels that leak
            through upstream filtering.

    Shape:
        - ``predictions``: ``(N, num_classes)`` unnormalized logits.
        - ``targets``: ``(N,)`` integer class indices.
        - ``sample_weights``: scalar or ``(N,)`` tensor; multiplied per sample
            before the batch mean. A scalar is a no-op.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            predictions,
            targets.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = targets != self.ignore_index
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights
        return loss[valid].mean() if valid.any() else loss.sum() * 0.0


class MSETaskLoss(nn.Module):
    """Mean squared error loss for regression tasks.

    Computes element-wise MSE between predictions and targets, optionally
    weighting each sample before averaging over the full batch.

    Shape:
        - ``predictions``: ``(N, D)`` predicted values.
        - ``targets``: ``(N, D)`` ground-truth values (same shape as predictions).
        - ``sample_weights``: scalar or ``(N,)`` tensor; broadcast across
            target dimensions when a tensor. A scalar is a no-op.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.mse_loss(predictions, targets, reduction="none")
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights.unsqueeze(-1)
        return loss.mean()


class FocalTaskLoss(nn.Module):
    """Focal loss for classification tasks with class imbalance.

    Applies focal loss to address class imbalance by down-weighting easy examples.
    With hard labels (``label_smoothing=0``):

        ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    With soft labels (``label_smoothing > 0``), targets are mixed with a uniform
    prior ``y' = (1 - ε) y + ε / C`` and the per-class form is used:

        ``FL = -sum_k y'_k * alpha_k * (1 - p_k)^gamma * log(p_k)``

    which reduces to the hard-label formula when ``ε = 0``.

    Class weighting must be specified via ``alpha`` (not ``class_weights``).
    Do NOT combine FocalTaskLoss with ``class_weights.mode: auto`` in the config;
    use ``alpha: auto`` instead to auto-compute per-class weights, or provide an
    explicit ``alpha`` list.

    Args:
        alpha: Per-class weights as a list of length ``num_classes``. Scalar values
            are not allowed; use an explicit list. In YAML config, ``alpha: auto``
            is supported and will be resolved to inverse-frequency weights by the
            training script, using ``alpha_smoothing`` as the exponent.
        gamma: Exponent of the modulating factor ``(1 - p)^gamma`` to focus on
            hard examples. Defaults to ``2.0``.
        label_smoothing: Soft-target mixing factor ``ε``. ``0.0`` keeps hard
            one-hot targets; typical values are ``0.05``–``0.1``. Defaults to
            ``0.0``. Unrelated to ``alpha_smoothing``.
        alpha_smoothing: Exponent for inverse-frequency weights when
            ``alpha: auto``. ``1.0`` is full inverse-frequency; lower values
            soften class reweighting. Stored for config/checkpoint fidelity;
            applied by the training script before instantiation. Defaults to
            ``1.0``.
        ignore_index: Target value that should be ignored in loss computation.
            Defaults to ``-1`` as a safety net for unmapped labels that leak
            through upstream filtering.

    Shape:
        - ``predictions``: ``(N, num_classes)`` unnormalized logits.
        - ``targets``: ``(N,)`` integer class indices.
        - ``sample_weights``: scalar or ``(N,)`` tensor; multiplied per sample
            before the batch mean. A scalar is a no-op.
    """

    def __init__(
        self,
        alpha: list[float],
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        alpha_smoothing: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha_smoothing = alpha_smoothing
        self.ignore_index = ignore_index

        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        targets_long = targets.long()
        valid = targets_long != self.ignore_index

        # Placeholder index for ignored rows so gather/scatter stay in-range;
        # those rows are excluded from the mean below.
        safe_targets = targets_long.clone()
        safe_targets[~valid] = 0

        log_probs = F.log_softmax(predictions, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing == 0.0:
            # Hard-label path: O(N) gather instead of dense (N, C) soft targets.
            idx = safe_targets.unsqueeze(1)
            log_pt = log_probs.gather(1, idx).squeeze(1)
            pt = probs.gather(1, idx).squeeze(1)
            loss = (
                -self.alpha[safe_targets] * (1.0 - pt).pow(self.gamma) * log_pt
            )
        else:
            num_classes = predictions.size(-1)
            soft_targets = torch.full_like(
                log_probs, self.label_smoothing / num_classes
            )
            soft_targets.scatter_(
                1,
                safe_targets.unsqueeze(1),
                1.0 - self.label_smoothing + self.label_smoothing / num_classes,
            )
            loss = -(
                soft_targets
                * self.alpha
                * (1.0 - probs).pow(self.gamma)
                * log_probs
            ).sum(dim=-1)

        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights

        return loss[valid].mean() if valid.any() else loss.sum() * 0.0


class ReconstructionLoss(nn.Module):
    """MSE loss for signal reconstruction with validity-mask weighting.

    Expects targets to be pre-normalized (z-scored per channel during
    tokenization). The ``sample_weights`` tensor encodes the validity mask:
    positions from padded channels have weight=0 and do not contribute.

    Follows the uniform ``(predictions, targets, sample_weights) → scalar``
    signature used by all task losses.

    Scalar weights are broadcast: ``0.0`` produces a differentiable zero loss,
    ``0.5`` scales the MSE by that factor, and ``1.0`` is equivalent to plain
    MSE. Zero-valid-target batches return ``predictions.sum() * 0.0`` to
    preserve dtype, device, and gradient graph.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        if isinstance(sample_weights, (int, float)):
            if sample_weights == 0.0:
                return predictions.sum() * 0.0
            return F.mse_loss(predictions, targets) * sample_weights

        valid = sample_weights > 0
        if not valid.any():
            return predictions.sum() * 0.0

        loss = F.mse_loss(predictions[valid], targets[valid], reduction="none")
        if loss.dim() > 1:
            loss = loss.mean(dim=-1)

        weights_valid = sample_weights[valid]
        return (loss * weights_valid).sum() / weights_valid.sum()

import pytest
import torch
import torch.nn.functional as F

from foundry.tasks.losses import (
    CrossEntropyTaskLoss,
    FocalTaskLoss,
    MSETaskLoss,
    ReconstructionLoss,
)


class TestCrossEntropyTaskLoss:
    def test_matches_cross_entropy_without_smoothing_or_weights(self):
        torch.manual_seed(0)
        predictions = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        loss_fn = CrossEntropyTaskLoss()
        expected = F.cross_entropy(predictions, targets)

        assert torch.allclose(loss_fn(predictions, targets), expected)

    def test_with_class_weights_and_label_smoothing(self):
        torch.manual_seed(1)
        predictions = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        class_weights = [1.0, 2.0, 0.5]
        label_smoothing = 0.1

        loss_fn = CrossEntropyTaskLoss(
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
        expected = F.cross_entropy(
            predictions,
            targets,
            weight=torch.tensor(class_weights, dtype=torch.float32),
            label_smoothing=label_smoothing,
            reduction="none",
        ).mean()

        assert torch.allclose(loss_fn(predictions, targets), expected)

    def test_tensor_sample_weights(self):
        torch.manual_seed(2)
        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

        loss_fn = CrossEntropyTaskLoss()
        per_sample = F.cross_entropy(predictions, targets, reduction="none")
        expected = (per_sample * sample_weights).mean()

        assert torch.allclose(
            loss_fn(predictions, targets, sample_weights), expected
        )

    def test_scalar_sample_weights(self):
        torch.manual_seed(3)
        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss_fn = CrossEntropyTaskLoss()
        expected = F.cross_entropy(predictions, targets)

        assert torch.allclose(
            loss_fn(predictions, targets, sample_weights=1.0), expected
        )


class TestMSETaskLoss:
    def test_matches_mse_loss(self):
        torch.manual_seed(4)
        predictions = torch.randn(5, 4)
        targets = torch.randn(5, 4)

        loss_fn = MSETaskLoss()
        expected = F.mse_loss(predictions, targets)

        assert torch.allclose(loss_fn(predictions, targets), expected)

    def test_tensor_sample_weights(self):
        torch.manual_seed(5)
        predictions = torch.randn(3, 2)
        targets = torch.randn(3, 2)
        sample_weights = torch.tensor([1.0, 0.0, 2.0])

        loss_fn = MSETaskLoss()
        per_sample = F.mse_loss(predictions, targets, reduction="none")
        expected = (per_sample * sample_weights.unsqueeze(-1)).mean()

        assert torch.allclose(
            loss_fn(predictions, targets, sample_weights), expected
        )


class TestFocalTaskLoss:
    def test_per_class_alpha_required(self):
        """Focal loss requires per-class alpha weights as a list."""
        torch.manual_seed(6)
        predictions = torch.randn(8, 8)
        targets = torch.randint(0, 8, (8,))

        alpha_weights = [1.0] * 8
        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        result = loss_fn(predictions, targets)

        assert result.dim() == 0
        assert not torch.isnan(result)

    def test_focal_weight_reduces_easy_examples(self):
        """Focal loss should down-weight easy examples with high confidence."""
        torch.manual_seed(7)

        batch_size = 4
        num_classes = 3
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.tensor([0, 1, 2, 0])

        alpha_weights = [1.0] * num_classes
        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        focal_loss = loss_fn(predictions, targets)

        assert torch.is_tensor(focal_loss)
        assert focal_loss.dim() == 0

    def test_gamma_zero_approximates_cross_entropy(self):
        """With gamma=0, focal loss should approximate cross-entropy."""
        torch.manual_seed(8)
        predictions = torch.randn(6, 4)
        targets = torch.randint(0, 4, (6,))

        alpha_weights = [1.0] * 4
        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=0.0)
        result = loss_fn(predictions, targets)

        ce_loss_fn = CrossEntropyTaskLoss()
        ce_result = ce_loss_fn(predictions, targets)

        assert torch.allclose(result, ce_result, atol=1e-5)

    def test_per_class_alpha_weights(self):
        """Focal loss should support per-class alpha weights."""
        torch.manual_seed(9)
        predictions = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        alpha_weights = [1.0, 2.0, 0.5]

        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        result = loss_fn(predictions, targets)

        assert result.dim() == 0
        assert not torch.isnan(result)

    def test_ignore_index_excludes_targets(self):
        """Loss for ignored indices should not contribute to the result."""
        torch.manual_seed(10)
        predictions = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, -1, -1, 2])

        alpha_weights = [1.0] * 3
        loss_fn = FocalTaskLoss(alpha=alpha_weights, ignore_index=-1)
        result = loss_fn(predictions, targets)

        valid = targets != -1
        expected = loss_fn(predictions[valid], targets[valid])

        assert torch.allclose(result, expected, atol=1e-5)

    def test_tensor_sample_weights(self):
        """Focal loss should support per-sample weights."""
        torch.manual_seed(11)
        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

        alpha_weights = [1.0] * 3
        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        result = loss_fn(predictions, targets, sample_weights)

        assert result.dim() == 0
        assert not torch.isnan(result)

    def test_per_class_alpha_balances_classes(self):
        """Focal loss should apply per-class alpha weights."""
        torch.manual_seed(12)
        predictions = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        alpha_weights = [1.0, 2.0, 0.5]

        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        result = loss_fn(predictions, targets)

        assert result.dim() == 0
        assert not torch.isnan(result)

    def test_all_invalid_returns_zero(self):
        """All invalid targets should return zero loss."""
        predictions = torch.randn(4, 3)
        targets = torch.full((4,), -1, dtype=torch.long)

        alpha_weights = [1.0] * 3
        loss_fn = FocalTaskLoss(alpha=alpha_weights, ignore_index=-1)
        result = loss_fn(predictions, targets)

        assert result.item() == 0.0

    def test_gradient_computation(self):
        """Focal loss should allow backpropagation."""
        torch.manual_seed(13)
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))

        alpha_weights = [1.0] * 3
        loss_fn = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)
        result = loss_fn(predictions, targets)
        result.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_higher_gamma_increases_focusing(self):
        """Higher gamma should increase focus on hard examples."""
        torch.manual_seed(14)
        predictions = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        alpha_weights = [1.0] * 4
        loss_gamma_0 = FocalTaskLoss(alpha=alpha_weights, gamma=0.0)(
            predictions, targets
        )
        loss_gamma_2 = FocalTaskLoss(alpha=alpha_weights, gamma=2.0)(
            predictions, targets
        )

        assert loss_gamma_0.item() > loss_gamma_2.item()

    def test_label_smoothing_zero_matches_hard_target(self):
        """ε=0 soft-target path matches the hard one-hot focal loss formula."""
        torch.manual_seed(15)
        predictions = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        alpha_weights = [1.2, 0.8, 1.0, 1.5, 0.7]

        loss_fn = FocalTaskLoss(
            alpha=alpha_weights, gamma=2.0, label_smoothing=0.0
        )
        result = loss_fn(predictions, targets)

        log_probs = F.log_softmax(predictions, dim=-1)
        probs = log_probs.exp()
        idx = torch.arange(8)
        expected = (
            -torch.tensor(alpha_weights)[targets]
            * (1.0 - probs[idx, targets]).pow(2.0)
            * log_probs[idx, targets]
        ).mean()

        assert torch.allclose(result, expected, atol=1e-6)

    def test_label_smoothing_gamma_zero_matches_smoothed_ce(self):
        """With gamma=0 and uniform alpha, soft focal matches label-smoothed CE."""
        torch.manual_seed(16)
        predictions = torch.randn(6, 4)
        targets = torch.randint(0, 4, (6,))
        label_smoothing = 0.1

        loss_fn = FocalTaskLoss(
            alpha=[1.0] * 4,
            gamma=0.0,
            label_smoothing=label_smoothing,
        )
        expected = F.cross_entropy(
            predictions, targets, label_smoothing=label_smoothing
        )

        assert torch.allclose(
            loss_fn(predictions, targets), expected, atol=1e-5
        )

    def test_label_smoothing_changes_loss(self):
        """Non-zero smoothing should change the loss vs hard targets."""
        torch.manual_seed(17)
        predictions = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        alpha_weights = [1.0] * 4

        hard = FocalTaskLoss(
            alpha=alpha_weights, gamma=2.0, label_smoothing=0.0
        )
        soft = FocalTaskLoss(
            alpha=alpha_weights, gamma=2.0, label_smoothing=0.1
        )

        assert not torch.allclose(
            hard(predictions, targets), soft(predictions, targets)
        )

    def test_label_smoothing_with_ignore_index(self):
        """Soft-target path still excludes ignore_index samples."""
        torch.manual_seed(18)
        predictions = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, -1, -1, 2])

        loss_fn = FocalTaskLoss(
            alpha=[1.0] * 3,
            gamma=2.0,
            label_smoothing=0.1,
            ignore_index=-1,
        )
        result = loss_fn(predictions, targets)

        valid = targets != -1
        expected = loss_fn(predictions[valid], targets[valid])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_invalid_label_smoothing_raises(self):
        with pytest.raises(ValueError, match="label_smoothing"):
            FocalTaskLoss(alpha=[1.0, 1.0], label_smoothing=1.0)
        with pytest.raises(ValueError, match="label_smoothing"):
            FocalTaskLoss(alpha=[1.0, 1.0], label_smoothing=-0.1)

    def test_conforms_to_loss_interface(self):
        """FocalTaskLoss has the same (pred, target, weights) -> scalar signature."""
        alpha_weights = [1.0] * 4
        loss_fn = FocalTaskLoss(alpha=alpha_weights)
        assert hasattr(loss_fn, "forward")
        result = loss_fn(torch.randn(3, 4), torch.randint(0, 4, (3,)))
        assert result.dim() == 0


class TestReconstructionLoss:
    def test_scalar_weights_matches_plain_mse(self):
        torch.manual_seed(10)
        predictions = torch.randn(8, 1)
        targets = torch.randn(8, 1)

        loss_fn = ReconstructionLoss()
        expected = F.mse_loss(predictions, targets)

        assert torch.allclose(loss_fn(predictions, targets, 1.0), expected)

    def test_validity_mask_excludes_padded_positions(self):
        torch.manual_seed(11)
        predictions = torch.randn(6, 1)
        targets = torch.randn(6, 1)
        weights = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        loss_fn = ReconstructionLoss()
        result = loss_fn(predictions, targets, weights)

        valid_pred = predictions[:3]
        valid_targ = targets[:3]
        expected = F.mse_loss(valid_pred, valid_targ)

        assert torch.allclose(result, expected)

    def test_weighted_average_over_valid_positions(self):
        predictions = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[0.0], [0.0], [0.0]])
        weights = torch.tensor([1.0, 2.0, 0.0])

        loss_fn = ReconstructionLoss()
        result = loss_fn(predictions, targets, weights)

        # valid entries: pred=[1,2], targ=[0,0], weights=[1,2]
        # MSE per sample: [1, 4]
        # weighted: (1*1 + 4*2) / (1+2) = 9/3 = 3.0
        assert torch.allclose(result, torch.tensor(3.0))

    def test_all_invalid_returns_zero(self):
        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        weights = torch.zeros(4)

        loss_fn = ReconstructionLoss()
        result = loss_fn(predictions, targets, weights)

        assert result.item() == 0.0

    def test_multidim_output_reduces_last_dim(self):
        torch.manual_seed(12)
        predictions = torch.randn(5, 3)
        targets = torch.randn(5, 3)
        weights = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])

        loss_fn = ReconstructionLoss()
        result = loss_fn(predictions, targets, weights)

        valid = weights > 0
        per_sample = F.mse_loss(
            predictions[valid], targets[valid], reduction="none"
        ).mean(dim=-1)
        expected = (per_sample * weights[valid]).sum() / weights[valid].sum()

        assert torch.allclose(result, expected)

    def test_conforms_to_loss_interface(self):
        """ReconstructionLoss has the same (pred, target, weights) -> scalar signature."""
        loss_fn = ReconstructionLoss()
        assert hasattr(loss_fn, "forward")
        result = loss_fn(torch.randn(3, 1), torch.randn(3, 1))
        assert result.dim() == 0

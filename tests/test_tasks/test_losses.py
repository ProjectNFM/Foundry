import torch
import torch.nn.functional as F

from foundry.tasks.losses import (
    CrossEntropyTaskLoss,
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

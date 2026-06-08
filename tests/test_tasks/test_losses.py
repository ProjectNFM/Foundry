import torch
import torch.nn.functional as F

from foundry.tasks.losses import (
    CrossEntropyTaskLoss,
    FocalTaskLoss,
    MSETaskLoss,
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
    def test_gamma_zero_reduces_to_cross_entropy(self):
        torch.manual_seed(6)
        predictions = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        focal = FocalTaskLoss(gamma=0.0)
        expected = F.cross_entropy(predictions, targets)

        assert torch.allclose(focal(predictions, targets), expected)

    def test_tensor_sample_weights(self):
        torch.manual_seed(7)
        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

        loss_fn = FocalTaskLoss(gamma=2.0)
        ce = F.cross_entropy(predictions, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        expected = ((1 - pt) ** 2.0 * ce * sample_weights).mean()

        assert torch.allclose(
            loss_fn(predictions, targets, sample_weights), expected
        )

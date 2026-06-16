from types import SimpleNamespace

import torch
import torch.nn as nn

from foundry.training import ClassificationModule


class DummyClassificationModel(nn.Module):
    def __init__(self, dim: int = 3):
        super().__init__()
        self.readout_specs = {
            "test_classification": SimpleNamespace(
                dim=dim, id=20, loss_fn=object()
            ),
        }


def test_source_ids_are_aligned_to_task_output_tokens():
    module = ClassificationModule(model=DummyClassificationModel())

    source_ids = module._source_ids_for_task(
        ["minipigs/session_a", "monkeys/session_b"],
        torch.tensor([[20, 0, 0], [20, 20, 0]]),
        "test_classification",
    )

    assert source_ids == ["minipigs", "monkeys", "monkeys"]


def test_explicit_source_ids_are_aligned_for_single_species_sessions():
    module = ClassificationModule(model=DummyClassificationModel())

    source_ids = module._source_ids_for_task(
        ["sub-01_ses-01", "sub-02_ses-01"],
        torch.tensor([[20, 0, 0], [20, 20, 0]]),
        "test_classification",
        sequence_source_ids=["minipigs", "minipigs"],
    )

    assert source_ids == ["minipigs", "minipigs", "minipigs"]


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

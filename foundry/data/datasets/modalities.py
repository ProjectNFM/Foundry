import torch
from torch_brain.registry import register_modality, DataType
from torch_brain.nn.loss import CrossEntropyLoss, Loss


class MappedCrossEntropyLoss(Loss):
    """CrossEntropyLoss with label remapping support.

    Maps arbitrary label IDs to consecutive class indices [0, num_classes-1]
    before computing cross-entropy loss.

    Args:
        mapping: Dictionary mapping original label IDs (keys) to target class
                 indices (values). Values should be in range [0, num_classes-1].
    """

    def __init__(self, mapping: dict[int, int]):
        super().__init__()
        keys = list(mapping.keys())
        values = list(mapping.values())

        self.register_buffer("_keys", torch.tensor(keys, dtype=torch.long))
        self.register_buffer("_values", torch.tensor(values, dtype=torch.long))
        self.base_loss = CrossEntropyLoss()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply label mapping and compute cross-entropy loss."""
        mapped_target = torch.zeros_like(target)
        for i, key in enumerate(self._keys):
            mask = target == key
            mapped_target[mask] = self._values[i]

        return self.base_loss(input, mapped_target, weights)


SLEEP_STAGE_5CLASS = register_modality(
    "sleep_stage_5class",
    dim=5,
    type=DataType.MULTINOMIAL,
    timestamp_key="sleep_stages.timestamps",
    value_key="sleep_stages.values",
    loss_fn=CrossEntropyLoss(),
)

P300_TARGET = register_modality(
    "p300_target",
    dim=2,
    type=DataType.BINARY,
    timestamp_key="p300_trials.timestamps",
    value_key="p300_trials.target",
    loss_fn=CrossEntropyLoss(),
)

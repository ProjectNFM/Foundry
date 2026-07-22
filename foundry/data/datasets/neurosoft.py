from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
from torch_brain.data import Data


class NeurosoftMinipigs2026(_AuditoryNeurosoftMinipigs2026):
    """Foundry wrapper for Neurosoft minipig data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)


class NeurosoftMonkeys2026(_AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)

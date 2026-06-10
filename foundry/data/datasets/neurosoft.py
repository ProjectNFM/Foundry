from pathlib import Path

from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
from torch_brain.data import Data

from foundry.tasks.config import TaskConfig

from .mixins import TaskMixin

_TASKS_DIR = Path(__file__).resolve().parents[3] / "configs" / "tasks"

_NEUROSOFT_TASKS = {
    "neurosoft_on_vs_off": TaskConfig.from_yaml(
        _TASKS_DIR / "neurosoft_on_vs_off.yaml"
    ),
    "neurosoft_acoustic_stim": TaskConfig.from_yaml(
        _TASKS_DIR / "neurosoft_acoustic_stim.yaml"
    ),
}


class _NeurosoftTaskMixin(TaskMixin):
    """Shared task registration and recording hook for Neurosoft datasets."""

    AVAILABLE_TASKS = _NEUROSOFT_TASKS

    def get_recording_hook(self, data: Data):
        # Skip auditorydecoding's multitask_readout hook; targets come from TaskConfig.
        super(NeurosoftDataset, self).get_recording_hook(data)


class NeurosoftMinipigs2026(
    _NeurosoftTaskMixin, _AuditoryNeurosoftMinipigs2026
):
    """Foundry wrapper for Neurosoft minipig data with task-config registration."""


class NeurosoftMonkeys2026(_NeurosoftTaskMixin, _AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data with task-config registration."""

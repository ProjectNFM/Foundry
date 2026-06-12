from pathlib import Path

from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
from auditorydecoding.data.neurosoft_pipeline import (
    STIM_FREQUENCY_TO_ID,
)
from torch_brain.data import Data

from foundry.tasks.config import TaskConfig
from foundry.tasks.adaptation import TaskClassSchema

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

# Frequency grouping presets for acoustic stimulation classification
FREQ_GROUPINGS = {
    "3band": {
        "stim_500Hz": "low",
        "stim_800Hz": "low",
        "stim_1000Hz": "medium",
        "stim_2000Hz": "medium",
        "stim_5000Hz": "high",
        "stim_8000Hz": "high",
    },
    "2band": {
        "stim_500Hz": "low",
        "stim_800Hz": "low",
        "stim_5000Hz": "high",
        "stim_8000Hz": "high",
    },
    "2band_rnd_a": {
        "stim_500Hz": "class1",
        "stim_800Hz": "class2",
        "stim_5000Hz": "class1",
        "stim_8000Hz": "class2",
    },
    "2band_rnd_b": {
        "stim_500Hz": "class1",
        "stim_800Hz": "class2",
        "stim_5000Hz": "class2",
        "stim_8000Hz": "class1",
    },
}

FREQ_GROUP_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "class1": 0,
    "class2": 1,
}


def format_acoustic_stim_display_names(names):
    """Strip stim_ prefix from frequency names for display in confusion matrices.
    
    Args:
        names: List of frequency strings (e.g., ["stim_500Hz", "stim_800Hz"])
               or band names (e.g., ["low", "medium", "high"])
    
    Returns:
        List with stim_ prefix removed from frequency names, band names unchanged
        (e.g., ["500Hz", "800Hz"] or ["low", "medium", "high"])
    """
    return [n.removeprefix("stim_") if n.startswith("stim_") else n for n in names]


class _NeurosoftTaskMixin(TaskMixin):
    """Shared task registration and recording hook for Neurosoft datasets."""

    AVAILABLE_TASKS = _NEUROSOFT_TASKS
    TASK_TO_READOUT = {
        "on_vs_off": ["neurosoft_on_vs_off"],
        "acoustic_stim": ["neurosoft_acoustic_stim"],
    }
    TASK_CLASS_SCHEMAS = {
        "neurosoft_acoustic_stim": TaskClassSchema(
            vocabulary=STIM_FREQUENCY_TO_ID,
            interval_filter_field="behavior_labels",
            interval_filter_mode="names",
            grouping_presets=FREQ_GROUPINGS,
            group_order=FREQ_GROUP_ORDER,
            display_name_formatter=format_acoustic_stim_display_names,
        ),
        # on_vs_off: no schema (not filterable)
    }

    def get_recording_hook(self, data: Data):
        # Skip auditorydecoding's multitask_readout hook; targets come from TaskConfig.
        super(NeurosoftDataset, self).get_recording_hook(data)


class NeurosoftMinipigs2026(
    _NeurosoftTaskMixin, _AuditoryNeurosoftMinipigs2026
):
    """Foundry wrapper for Neurosoft minipig data with task-config registration."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)


class NeurosoftMonkeys2026(_NeurosoftTaskMixin, _AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data with task-config registration."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

from __future__ import annotations

from pathlib import Path

from torch_brain.datasets import KempSleepEDF2013 as _TorchBrainKempSleepEDF2013

from foundry.tasks.config import TaskConfig

from .mixins import TaskMixin

_TASKS_DIR = Path(__file__).resolve().parents[3] / "configs" / "tasks"

_KEMP_TASKS = {
    "sleep_stage_5class": TaskConfig.from_yaml(
        _TASKS_DIR / "sleep_stage_5class.yaml"
    ),
}


class KempSleepEDF2013(TaskMixin, _TorchBrainKempSleepEDF2013):
    """Foundry wrapper for Kemp Sleep-EDF 2013 with task-config registration.

    Translates Foundry's ``fold`` / ``split_type`` data-module conventions to
    torch_brain's ``fold_number`` / ``fold_type``.  The ``task_type`` keyword
    accepted by :class:`~foundry.data.datamodules.NeuralDataModule` is consumed
    here and not forwarded to the underlying dataset.
    """

    AVAILABLE_TASKS = _KEMP_TASKS
    TASK_TO_READOUT = {
        "sleep_stage": ["sleep_stage_5class"],
    }

    def __init__(
        self,
        *,
        root,
        fold: int = 0,
        split_type: str = "intrasession",
        task_type: str
        | None = None,  # consumed by NeuralDataModule; not used here
        **kwargs,
    ):
        super().__init__(
            root=root,
            fold_number=fold,
            fold_type=split_type,
            **kwargs,
        )

    @classmethod
    def get_required_transforms(cls, task_type: str) -> list:
        if task_type == "sleep_stage":
            from foundry.data.transforms import (
                SelectEEGChannels,
                PrepareSleepStages,
            )

            return [SelectEEGChannels(), PrepareSleepStages()]
        return []

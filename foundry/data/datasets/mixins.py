import numpy as np
from torch_brain.data import Data

from torch_brain.utils import np_string_prefix

from foundry.tasks.config import TaskConfig


class EEGDatasetMixin:
    """
    Mixin class for :class:`torch_brain.datasets.Dataset` subclasses containing EEG data.

    Provides:
        - ``get_channel_ids()`` for retrieving IDs of all included channels.
        - If the class attribute ``eeg_dataset_mixin_uniquify_channel_ids`` is set to ``True``,
          channel IDs will be made unique across recordings by prefixing each channel ID with the
          corresponding ``session.id``. This helps avoid collisions when combining data from
          multiple sessions. (default: ``False``)
    """

    eeg_dataset_mixin_uniquify_channel_ids: bool = False

    def get_recording_hook(self, data: Data):
        if self.eeg_dataset_mixin_uniquify_channel_ids:
            data.channels.id = np_string_prefix(
                f"{data.session.id}/",
                data.channels.id.astype(str),
            )
        super().get_recording_hook(data)

    def get_channel_ids(self) -> list[str]:
        """Return a sorted list of all channel IDs across all recordings in the dataset."""
        ans = [
            self.get_recording(rid).channels.id for rid in self.recording_ids
        ]
        return np.sort(np.concatenate(ans)).tolist()

    def get_recording_ids(self) -> list[str]:
        """Return a sorted list of all recording IDs in the dataset."""
        return np.sort(np.array(self.recording_ids)).tolist()


class TaskMixin:
    """Mixin for datasets that declare which tasks they support."""

    AVAILABLE_TASKS: dict[str, TaskConfig] = {}

    @classmethod
    def get_task(cls, name: str) -> TaskConfig:
        if name not in cls.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{name}'. Available: {list(cls.AVAILABLE_TASKS)}"
            )
        return cls.AVAILABLE_TASKS[name]

    @classmethod
    def get_tasks(cls, names: list[str] | None = None) -> dict[str, TaskConfig]:
        if names is None:
            return dict(cls.AVAILABLE_TASKS)
        return {n: cls.get_task(n) for n in names}

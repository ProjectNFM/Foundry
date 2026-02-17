from typing import TYPE_CHECKING

import numpy as np
from temporaldata import Data

from torch_brain.utils import np_string_prefix

if TYPE_CHECKING:
    from torch_brain.registry import ModalitySpec


class EEGDatasetMixin:
    """
    Mixin class for :class:`torch_brain.dataset.Dataset` subclasses containing EEG data.

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


class ModalityMixin:
    """Mixin for datasets that provide modality specifications."""

    MODALITIES: dict[str, int] = {}

    @classmethod
    def get_modality_specs(cls) -> dict[str, "ModalitySpec"]:
        """Return ModalitySpec objects for this dataset's modalities keyed by name."""
        from torch_brain.registry import MODALITY_REGISTRY

        return {name: MODALITY_REGISTRY[name] for name in cls.MODALITIES.keys()}

    @classmethod
    def get_modality_names(cls) -> list[str]:
        """Return names of available modalities."""
        return list(cls.MODALITIES.keys())

    @classmethod
    def get_modality(cls, name: str) -> "ModalitySpec":
        """Get a single modality spec by name."""
        if name not in cls.MODALITIES:
            available = cls.get_modality_names()
            raise ValueError(
                f"Unknown modality '{name}'. Available: {available}"
            )
        from torch_brain.registry import MODALITY_REGISTRY

        return MODALITY_REGISTRY[name]

    @classmethod
    def get_modalities(
        cls, names: list[str] | None = None
    ) -> list["ModalitySpec"]:
        """Get modality specs as list. If names is None, returns all available."""
        if names is None:
            names = list(cls.MODALITIES.keys())
        return [cls.get_modality(name) for name in names]

    def get_recording_hook(self, data: Data):
        """Set multitask_readout config based on dataset's supported modalities."""
        if not hasattr(data, "config") or data.config is None:
            data.config = {}
        data.config["multitask_readout"] = [
            {"readout_id": name} for name in self.MODALITIES.keys()
        ]
        super().get_recording_hook(data)


def combine_modalities(
    *dataset_tasks: tuple[type, list[str]],
) -> list["ModalitySpec"]:
    """Combine modalities from multiple datasets.

    Args:
        *dataset_tasks: Variable number of (dataset_class, modality_names) tuples

    Returns:
        List of ModalitySpec objects from all specified datasets

    Example:
        combine_modalities(
            (SchalkWolpawPhysionet2009, ["motor_imagery_5class"]),
            (KempSleepEDF2013, ["sleep_stage_5class"]),
        )
    """
    specs = []
    for dataset_cls, task_names in dataset_tasks:
        specs.extend(dataset_cls.get_modalities(task_names))
    return specs
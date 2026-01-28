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

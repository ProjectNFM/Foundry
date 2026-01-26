import numpy as np
from temporaldata import Data

from torch_brain.utils import np_string_prefix


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

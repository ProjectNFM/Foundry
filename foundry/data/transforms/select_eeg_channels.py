from __future__ import annotations

import numpy as np
from torch_brain.data import Data, RegularTimeSeries

EEG_CHANNEL_SUFFIXES = ("EEG Fpz-Cz", "EEG Pz-Oz")


class SelectEEGChannels:
    """Reduce all sessions to the two standard EEG derivations.

    Keeps channels whose IDs end with ``EEG Fpz-Cz`` or ``EEG Pz-Oz``.
    This normalizes cassette recordings (which have exactly two EEG-typed
    channels) and telemetry recordings (which carry an additional EEG-typed
    ``Marker`` channel) to a uniform two-channel layout.
    """

    def __call__(self, data: Data) -> Data:
        channel_ids = np.asarray(data.channels.id)
        mask = np.array(
            [
                any(
                    str(cid).endswith(suffix) for suffix in EEG_CHANNEL_SUFFIXES
                )
                for cid in channel_ids
            ]
        )

        if not mask.any():
            raise ValueError(
                f"No EEG channels matching suffixes {EEG_CHANNEL_SUFFIXES} "
                f"found. Available channels: {list(channel_ids)}"
            )

        new_signal = data.eeg.signal[:, mask]
        data.eeg = RegularTimeSeries(
            signal=new_signal,
            sampling_rate=data.eeg.sampling_rate,
            domain_start=float(data.eeg.domain.start[0]),
        )
        data.channels = data.channels.select_by_mask(mask)
        return data

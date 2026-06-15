from __future__ import annotations

import numpy as np

from torch_brain.datasets import KempSleepEDF2013 as _TorchBrainKempSleepEDF2013


class KempSleepEDF2013(_TorchBrainKempSleepEDF2013):
    """Foundry wrapper for Kemp Sleep-EDF 2013.

    Translates Foundry's ``fold`` / ``split_type`` data-module conventions to
    torch_brain's ``fold_number`` / ``fold_type``.  The ``task_type`` keyword
    accepted by :class:`~foundry.data.datamodules.NeuralDataModule` is consumed
    here and not forwarded to the underlying dataset.
    """

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

    def get_channel_ids(self) -> list[str]:
        all_ids: set[str] = set()
        for rec_id in self.recording_ids:
            rec = self.get_recording(rec_id, "")
            all_ids.update(str(c) for c in np.asarray(rec.channels.id))
        return sorted(all_ids)

    @classmethod
    def get_required_transforms(cls, task_type: str) -> list:
        if task_type == "sleep_stage":
            from foundry.data.transforms import (
                SelectEEGChannels,
                PrepareSleepStages,
            )

            return [SelectEEGChannels(), PrepareSleepStages()]
        return []

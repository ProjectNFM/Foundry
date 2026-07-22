from __future__ import annotations

from typing import Literal

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

    def get_sampling_intervals(
        self,
        split: Literal["train", "valid", "test"] | None = None,
    ):
        """Return sampling intervals restricted to annotated stage periods.

        The upstream implementation uses ``rec.domain`` for intersubject /
        intersession splits, which spans the full recording including large
        unannotated regions. Sampling from those regions yields windows with
        no stage annotations, breaking target extraction. We override to
        return ``rec.stages`` instead, which carries the ``names`` attribute
        needed by ``_filter_intervals`` for class-mapping filtering.
        """
        if split is not None and self.fold_type in (
            "intersubject",
            "intersession",
        ):
            key = f"splits.{self.fold_type}_fold_{self.fold_number}_assignment"
            fallback_key = f"splits.fold_{self.fold_number}_assignment"
            result = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                try:
                    assignment = str(rec.get_nested_attribute(key))
                except (AttributeError, KeyError):
                    assignment = str(rec.get_nested_attribute(fallback_key))
                if assignment == split:
                    result[rid] = rec.stages
            return result

        return super().get_sampling_intervals(split=split)

    def get_channel_ids(self) -> list[str]:
        all_ids: set[str] = set()
        for rec_id in self.recording_ids:
            rec = self.get_recording(rec_id, "")
            all_ids.update(str(c) for c in np.asarray(rec.channels.id))
        return sorted(all_ids)

    @classmethod
    def get_required_transforms(cls, task_type: str) -> list:
        if task_type == "sleep_stage":
            from foundry.data.transforms import SelectEEGChannels

            return [SelectEEGChannels()]
        return []

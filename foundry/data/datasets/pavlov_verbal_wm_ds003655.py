from __future__ import annotations

from collections.abc import Callable

from torch_brain.datasets import OpenNeuroDataset, OpenNeuroSplitType


class PavlovVerbalWmDS003655(OpenNeuroDataset):
    """Foundry wrapper for the Pavlov Verbal Working Memory dataset (ds003655).

    156 subjects performing a modified Sternberg verbal working memory task
    with 19-channel EEG (+ 2 EOG) at 500 Hz.  EEGLAB format, single session
    per subject.

    Reference:
        Pavlov & Kotchoubey (2021). Scientific Reports.
        DOI: 10.1038/s41598-020-72940-5

    Args:
        root: Root directory containing processed dataset artifacts.
        split_type: Split strategy (intrasession, intersubject, intersession).
        recording_ids: Explicit recording IDs to load, or None for all.
        transform: Optional sample transform.
        **kwargs: Forwarded to :class:`OpenNeuroDataset`.
    """

    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType = "intrasession",
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            root=root,
            dataset_dir="pavlov_verbal_wm_ds003655",
            split_type=split_type,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

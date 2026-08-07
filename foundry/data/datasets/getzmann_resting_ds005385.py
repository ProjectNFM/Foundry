from __future__ import annotations

from collections.abc import Callable

from torch_brain.datasets import OpenNeuroDataset, OpenNeuroSplitType


class GetzmannRestingDS005385(OpenNeuroDataset):
    """Foundry wrapper for the Getzmann Resting-State EEG dataset (ds005385).

    608 subjects (baseline) with 64-channel EEG at 1000 Hz. Four conditions
    per session: eyes-closed/eyes-open crossed with pre/post cognitive battery.
    EDF format, up to 2 sessions per subject.

    Reference:
        Getzmann et al. (2024). Scientific Data.
        DOI: 10.1038/s41597-024-03797-w

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
            dataset_dir="getzmann_resting_ds005385",
            split_type=split_type,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

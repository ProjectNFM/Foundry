from __future__ import annotations

import numpy as np
from torch_brain.datasets import (
    KlinzingSleepDS005555,
    KochiVisualNamingDS006914,
    NestedDataset,
    ShiraziHBNR1DS005505,
)

OPENNEURO_BRAINSET_REGISTRY: dict[str, type] = {
    "klinzing_sleep_ds005555": KlinzingSleepDS005555,
    "shirazi_hbnr1_ds005505": ShiraziHBNR1DS005505,
    "kochi_visualnaming_ds006914": KochiVisualNamingDS006914,
}


class OpenNeuroMultiBrainset(NestedDataset):
    """Multi-brainset wrapper around OpenNeuro datasets.

    Instantiates one or more OpenNeuro brainsets by name and exposes them
    through the :class:`NestedDataset` interface so that
    :class:`~foundry.data.datamodules.NeuralDataModule` can drive training
    without any special-casing.
    """

    def __init__(
        self,
        root: str,
        brainsets: list[str],
        split_type: str = "intrasession",
        recording_ids: dict[str, list[str]] | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform=None,
        task_type: str | None = None,
        **kwargs,
    ):
        recording_ids = recording_ids or {}

        if not brainsets:
            raise ValueError(
                "brainsets must be a non-empty list. "
                f"Available: {sorted(OPENNEURO_BRAINSET_REGISTRY)}"
            )

        unknown = set(brainsets) - OPENNEURO_BRAINSET_REGISTRY.keys()
        if unknown:
            raise ValueError(
                f"Unknown brainset(s): {unknown}. "
                f"Available: {sorted(OPENNEURO_BRAINSET_REGISTRY)}"
            )

        datasets: dict[str, object] = {}
        for name in brainsets:
            cls = OPENNEURO_BRAINSET_REGISTRY[name]
            datasets[name] = cls(
                root=root,
                split_type=split_type,
                recording_ids=recording_ids.get(name),
                split_ratios=split_ratios,
            )

        # NestedDataset.__init__ uses np_string_prefix on each child's
        # recording_ids.  An empty Python list is cast to float64 by numpy,
        # which breaks the string-add operation.  Ensure empty lists are
        # proper string-typed numpy arrays.
        for ds in datasets.values():
            if (
                isinstance(ds._recording_ids, list)
                and len(ds._recording_ids) == 0
            ):
                ds._recording_ids = np.array([], dtype=str)

        super().__init__(datasets=datasets, transform=transform)

    # ------------------------------------------------------------------
    # NeuralDataModule passes split="valid" but OpenNeuroDataset expects "val"
    # ------------------------------------------------------------------
    def get_sampling_intervals(self, split=None):
        if split == "valid":
            split = "val"
        return super().get_sampling_intervals(split=split)

    # ------------------------------------------------------------------
    # NestedDataset does not provide get_channel_ids; aggregate from children
    # ------------------------------------------------------------------
    def get_channel_ids(self) -> list[str]:
        all_ids: list[str] = []
        for ds in self.datasets.values():
            if len(ds.recording_ids) == 0:
                continue
            all_ids.extend(ds.get_channel_ids())
        return sorted(set(all_ids))

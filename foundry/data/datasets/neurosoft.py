from __future__ import annotations

import numpy as np
from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
from torch_brain.data import Data
from torch_brain.datasets import NestedDataset


class NeurosoftMinipigs2026(_AuditoryNeurosoftMinipigs2026):
    """Foundry wrapper for Neurosoft minipig data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)


class NeurosoftMonkeys2026(_AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)


NEUROSOFT_BRAINSET_REGISTRY: dict[str, type] = {
    "neurosoft_minipigs_2026": NeurosoftMinipigs2026,
    "neurosoft_monkeys_2026": NeurosoftMonkeys2026,
}


class NeurosoftMultiBrainset(NestedDataset):
    """Multi-brainset wrapper around Neurosoft datasets.

    Instantiates one or more Neurosoft brainsets by name and exposes them
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
        fold: int = 0,
        transform=None,
        task_type: str | None = None,
        **kwargs,
    ):
        recording_ids = recording_ids or {}

        if not brainsets:
            raise ValueError(
                "brainsets must be a non-empty list. "
                f"Available: {sorted(NEUROSOFT_BRAINSET_REGISTRY)}"
            )

        unknown = set(brainsets) - NEUROSOFT_BRAINSET_REGISTRY.keys()
        if unknown:
            raise ValueError(
                f"Unknown brainset(s): {unknown}. "
                f"Available: {sorted(NEUROSOFT_BRAINSET_REGISTRY)}"
            )

        datasets: dict[str, object] = {}
        for name in brainsets:
            cls = NEUROSOFT_BRAINSET_REGISTRY[name]
            child_kwargs = {
                "root": root,
                "split_type": split_type,
                "recording_ids": recording_ids.get(name),
                "fold": fold,
                **kwargs,
            }
            if task_type is not None:
                child_kwargs["task_type"] = task_type
            datasets[name] = cls(**child_kwargs)

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
    # NestedDataset does not provide get_channel_ids; aggregate from children
    # with the dataset-name prefix that _namespace adds during __getitem__
    # ------------------------------------------------------------------
    def get_channel_ids(self) -> list[str]:
        all_ids: list[str] = []
        for name, ds in self.datasets.items():
            if len(ds.recording_ids) == 0:
                continue
            all_ids.extend(f"{name}/{ch}" for ch in ds.get_channel_ids())
        return sorted(set(all_ids))

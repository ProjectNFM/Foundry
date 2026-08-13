import logging

from auditorydecoding import (
    NeurosoftDataset,
    NeurosoftMinipigs2026 as _AuditoryNeurosoftMinipigs2026,
    NeurosoftMonkeys2026 as _AuditoryNeurosoftMonkeys2026,
)
import numpy as np
from torch_brain.data import Data
from torch_brain.datasets import DatasetIndex


NEURAL_CHANNEL_TYPES = {"eeg", "ecog", "seeg", "ieeg"}
logger = logging.getLogger(__name__)


class NeurosoftMinipigs2026(_AuditoryNeurosoftMinipigs2026):
    """Foundry wrapper for Neurosoft minipig data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)
        data.source_id = "minipigs"


class NeurosoftMonkeys2026(_AuditoryNeurosoftMonkeys2026):
    """Foundry wrapper for Neurosoft monkey data."""

    def __init__(self, *, fold=0, **kwargs):
        super().__init__(fold_num=fold, **kwargs)

    def get_recording_hook(self, data: Data):
        super(NeurosoftDataset, self).get_recording_hook(data)
        data.source_id = "monkeys"


def _recording_channel_count(data: Data) -> int:
    if not hasattr(data.channels, "type"):
        return len(data.channels.id)
    channel_types = np.char.lower(np.asarray(data.channels.type).astype(str))
    return int(np.isin(channel_types, list(NEURAL_CHANNEL_TYPES)).sum())


class NeurosoftMinipigsMonkeys2026:
    """Combined Neurosoft dataset with source-namespaced identities."""

    SOURCES = {
        "minipigs": NeurosoftMinipigs2026,
        "monkeys": NeurosoftMonkeys2026,
    }

    def __init__(
        self,
        root: str,
        transform=None,
        fold: int = 0,
        split_type: str | None = None,
        task_type: str = "on_vs_off",
        minipigs_recording_ids: list[str] | None = None,
        monkeys_recording_ids: list[str] | None = None,
        min_channels: int | None = None,
    ):
        recording_ids_by_source = {
            "minipigs": minipigs_recording_ids,
            "monkeys": monkeys_recording_ids,
        }
        self.datasets = {}
        for source, dataset_class in self.SOURCES.items():
            source_recording_ids = recording_ids_by_source[source]
            if min_channels is not None:
                probe_dataset = dataset_class(
                    root=root,
                    transform=None,
                    fold=fold,
                    split_type=split_type,
                    task_type=task_type,
                    recording_ids=source_recording_ids,
                    keep_files_open=False,
                )
                source_recording_ids = self._filter_recording_ids(
                    probe_dataset, source, min_channels
                )

            self.datasets[source] = dataset_class(
                root=root,
                transform=transform,
                fold=fold,
                split_type=split_type,
                task_type=task_type,
                recording_ids=source_recording_ids,
            )

        self.recording_ids = sorted(
            self._join_recording_id(source, recording_id)
            for source, dataset in self.datasets.items()
            for recording_id in dataset.recording_ids
        )

    @staticmethod
    def _join_recording_id(source: str, recording_id: str) -> str:
        return f"{source}/{recording_id}"

    def _split_recording_id(self, recording_id: str) -> tuple[str, str]:
        source, separator, inner_recording_id = recording_id.partition("/")
        if not separator or source not in self.datasets:
            raise KeyError(f"Unknown Neurosoft recording id '{recording_id}'")
        return source, inner_recording_id

    @staticmethod
    def _filter_recording_ids(dataset, source: str, min_channels: int):
        min_channels = int(min_channels)
        if min_channels <= 0:
            return list(dataset.recording_ids)

        channel_counts = {
            recording_id: _recording_channel_count(
                dataset.get_recording(recording_id)
            )
            for recording_id in dataset.recording_ids
        }
        kept = [
            recording_id
            for recording_id, count in channel_counts.items()
            if count >= min_channels
        ]
        if not kept:
            raise ValueError(
                f"min_channels={min_channels} filtered out all {source} "
                "recordings"
            )
        dropped = {
            recording_id: count
            for recording_id, count in channel_counts.items()
            if count < min_channels
        }
        if dropped:
            logger.warning(
                "Filtered %d %s recording(s) with fewer than %d channels: %s",
                len(dropped),
                source,
                min_channels,
                ", ".join(
                    f"{recording_id} ({count})"
                    for recording_id, count in dropped.items()
                ),
            )
        return kept

    def __getitem__(self, index: DatasetIndex):
        source, recording_id = self._split_recording_id(index.recording_id)
        sample = self.datasets[source][
            DatasetIndex(
                recording_id,
                index.start,
                index.end,
                _namespace=source,
            )
        ]
        if isinstance(sample, dict):
            sample["source_id"] = source
        else:
            sample.source_id = source
        return sample

    def get_recording(self, recording_id: str, _namespace: str = ""):
        source, inner_recording_id = self._split_recording_id(recording_id)
        namespace = source if not _namespace else f"{_namespace}/{source}"
        return self.datasets[source].get_recording(
            inner_recording_id, _namespace=namespace
        )

    def get_sampling_intervals(self, split=None):
        return {
            self._join_recording_id(source, recording_id): interval
            for source, dataset in self.datasets.items()
            for recording_id, interval in dataset.get_sampling_intervals(
                split=split
            ).items()
        }

    def get_channel_ids(self) -> list[str]:
        return sorted(
            {
                f"{source}/{channel_id}"
                for source, dataset in self.datasets.items()
                for channel_id in dataset.get_channel_ids()
            }
        )

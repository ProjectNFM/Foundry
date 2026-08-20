"""NeuralSetAdapter: bridge NeuralSet output to torch_brain Data.

Wraps a NeuralBench SegmentDataset and converts each sample into a
``torch_brain.Data`` object compatible with Foundry's tokenization pipeline.
NeuralBench provides preprocessed (C, T) EEG tensors with one-hot labels;
this adapter translates them into Foundry's signal representation
(``RegularTimeSeries``), label representation (``Interval`` with string
targets), and identity metadata (session, subject, channel IDs).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.utils.data
from torch_brain.data import Data, Interval, RegularTimeSeries

logger = logging.getLogger(__name__)

P3_LABEL_MAP: dict[int, str] = {0: "NonTarget", 1: "Target"}


class NeuralSetAdapter(torch.utils.data.Dataset):
    """Wraps a NeuralBench split dataset, converting samples to torch_brain Data.

    Each NeuralBench sample is a dict with ``neuro`` (1, C, T) float32,
    ``target`` (1, num_classes) int64 one-hot, ``subject_id`` (1, 1) int64,
    and ``channel_positions`` (1, C, 3) float32.  The adapter converts this
    into a ``Data`` object that Foundry's tokenizers can consume.

    Args:
        nb_dataset: NeuralBench split dataset (``loader.dataset``).
        channel_names: Ordered list of EEG channel names for this task.
        sampling_rate: Signal sampling rate in Hz.
        split: Split name (``"train"``, ``"val"``, or ``"test"``).
        label_map: Mapping from one-hot argmax index to string label name.
        label_attr: Name of the ``Interval`` attribute holding the label.
        interval_name: Name of the ``Interval`` on the ``Data`` object.
        session_prefix: Prefix for synthetic session IDs.
        identity_fn: Callable to extract ``(subject_key, session_key)`` from
            a sample. Defaults to extracting NeuralSet segment metadata.
        transform: Optional callable applied after Data construction
            (typically ``model.tokenize``).
    """

    def __init__(
        self,
        nb_dataset,
        channel_names: list[str],
        sampling_rate: float,
        split: str,
        *,
        label_map: dict[int, str],
        label_attr: str = "targets",
        interval_name: str = "p300_trials",
        session_prefix: str = "nb/p3",
        identity_fn: Callable | None = None,
        transform: Callable | None = None,
    ):
        self.nb_dataset = nb_dataset
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        self.split = split
        self.label_map = label_map
        self.label_attr = label_attr
        self.interval_name = interval_name
        self.session_prefix = session_prefix
        self._identity_fn = identity_fn or _extract_identity
        self.transform = transform

        if not self.channel_names:
            raise ValueError("channel_names must not be empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            seen, dupes = set(), set()
            for n in self.channel_names:
                (dupes if n in seen else seen).add(n)
            raise ValueError(f"Duplicate channel names: {sorted(dupes)}")

    def __len__(self) -> int:
        return len(self.nb_dataset)

    def __getitem__(self, idx: int) -> Data:
        data = self._to_torch_brain_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    def _get_sample_data(self, idx: int) -> tuple:
        """Return ``(raw_sample, data_dict)`` from the NeuralBench dataset."""
        sample = self.nb_dataset[idx]
        if hasattr(sample, "data") and isinstance(sample.data, dict):
            return sample, sample.data
        if isinstance(sample, dict):
            return sample, sample
        return sample, sample

    def _to_torch_brain_data(self, idx: int) -> Data:
        sample, sample_data = self._get_sample_data(idx)

        # --- EEG signal: (1, C, T) → (T, C) float32 ---
        neuro = sample_data["neuro"]
        if isinstance(neuro, torch.Tensor):
            neuro = neuro.numpy()
        signal = np.ascontiguousarray(
            neuro.squeeze(0).T, dtype=np.float32
        )  # (T, C)
        n_timepoints, n_channels = signal.shape
        if n_channels != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, "
                f"got {n_channels}"
            )
        duration = n_timepoints / self.sampling_rate

        # --- Label: one-hot (1, K) int64 → string class name ---
        target = sample_data["target"]
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        class_idx = int(np.argmax(target.flatten()))
        if class_idx not in self.label_map:
            raise ValueError(
                f"One-hot argmax {class_idx} not in label_map "
                f"{self.label_map}"
            )
        label_name = self.label_map[class_idx]

        # --- Subject / session identity ---
        subject_key, session_key = self._identity_fn(sample, sample_data)
        subject_id = f"{self.session_prefix}/{subject_key}"
        session_id = f"{self.session_prefix}/{session_key}"

        # --- Channel IDs (uniquified per session) ---
        channel_ids = np.array(
            [f"{session_id}/{ch}" for ch in self.channel_names]
        )
        channel_types = np.array(["EEG"] * n_channels)

        # --- Build torch_brain Data ---
        interval_kwargs = {
            self.label_attr: np.array([label_name]),
        }
        trial_interval = Interval(
            start=np.array([0.0]),
            end=np.array([duration]),
            **interval_kwargs,
        )

        data = Data(
            domain=Interval(np.array([0.0]), np.array([duration])),
            eeg=RegularTimeSeries(
                sampling_rate=self.sampling_rate, signal=signal
            ),
            **{self.interval_name: trial_interval},
        )
        data.channels = Data(id=channel_ids, type=channel_types)
        data.session = Data(id=session_id)
        data.subject = Data(id=subject_id)

        return data


# ------------------------------------------------------------------
# Subject key extraction
# ------------------------------------------------------------------


def _extract_identity(sample, sample_data: dict) -> tuple[str, str]:
    """Extract stable subject and recording/session identifiers.

    A NeuralSet timeline identifies a recording (including run/session),
    while ``subject`` identifies its participant.  Keeping those identities
    separate is necessary for session embeddings and per-session channel
    vocabularies.  The integer ``subject_id`` fallback has no recording
    metadata, so it intentionally uses the subject as its session key.
    """
    # Path 1: segment trigger event metadata
    segments = getattr(sample, "segments", None)
    if segments:
        seg = segments[0]
        ns_events = getattr(seg, "ns_events", None)
        if ns_events:
            for ev in ns_events:
                extra = getattr(ev, "extra", None)
                if not isinstance(extra, dict) or "subject" not in extra:
                    continue
                subject = str(extra["subject"])
                timeline = getattr(ev, "timeline", None)
                if timeline:
                    return subject, str(timeline)
                parts = [subject]
                for key in ("session", "run"):
                    if extra.get(key) not in (None, ""):
                        parts.append(f"{key}={extra[key]}")
                return subject, "/".join(parts)

    # Path 2: integer subject_id tensor
    sid = sample_data.get("subject_id")
    if sid is not None:
        if isinstance(sid, torch.Tensor):
            sid = sid.item()
        elif isinstance(sid, np.ndarray):
            sid = int(sid.flat[0])
        subject = f"sub-{int(sid)}"
        return subject, subject

    raise ValueError(
        "Cannot determine subject identity: no segment metadata and "
        "no 'subject_id' in sample data"
    )

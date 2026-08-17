"""Central adapters for stable sample metadata.

The visualization pipeline must not infer dataset or subject identity from a
session-name convention.  This module is the single boundary that translates
``torch_brain.data.Data`` descriptors into explicit, serializable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch_brain.data import Data


UNKNOWN_DATASET_ID = "__unknown_dataset__"
UNKNOWN_SUBJECT_ID = "__unknown_subject__"
UNKNOWN_SESSION_ID = "__unknown_session__"


@dataclass(frozen=True)
class WindowMetadata:
    """Stable identity fields for one sampled input window."""

    dataset_id: str
    subject_id: str
    session_id: str
    absolute_start: float
    window_duration: float


def _descriptor_id(data: Data, name: str, fallback: str) -> str:
    descriptor = getattr(data, name, None)
    value = getattr(descriptor, "id", None)
    if value is None:
        return fallback
    return str(value)


def extract_window_metadata(data: Data) -> WindowMetadata:
    """Resolve explicit window metadata without dataset-specific parsing.

    ``brainset.id``, ``subject.id``, and ``session.id`` are the authoritative
    descriptors used by torch-brain datasets.  Older/custom datasets that omit
    a descriptor receive a visible sentinel rather than a guessed ID.  Dataset
    adapters should populate the missing descriptor when those observations
    need to participate in grouping analyses.
    """

    domain = getattr(data, "domain", None)
    if domain is None:
        duration = 0.0
    else:
        starts = np.asarray(domain.start, dtype=np.float64)
        ends = np.asarray(domain.end, dtype=np.float64)
        duration = float(np.max(ends) - np.min(starts))

    absolute_start = getattr(data, "absolute_start", None)
    return WindowMetadata(
        dataset_id=_descriptor_id(data, "brainset", UNKNOWN_DATASET_ID),
        subject_id=_descriptor_id(data, "subject", UNKNOWN_SUBJECT_ID),
        session_id=_descriptor_id(data, "session", UNKNOWN_SESSION_ID),
        absolute_start=float(absolute_start or 0.0),
        window_duration=duration,
    )


__all__ = [
    "UNKNOWN_DATASET_ID",
    "UNKNOWN_SUBJECT_ID",
    "UNKNOWN_SESSION_ID",
    "WindowMetadata",
    "extract_window_metadata",
]

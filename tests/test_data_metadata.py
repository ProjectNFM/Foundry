from __future__ import annotations

from types import SimpleNamespace

from torch_brain.data import Data, Interval

from foundry.data.metadata import (
    UNKNOWN_DATASET_ID,
    UNKNOWN_SUBJECT_ID,
    extract_window_metadata,
)


def test_extract_window_metadata_uses_explicit_descriptors():
    data = Data(domain=Interval(0.0, 2.5))
    data.brainset = SimpleNamespace(id="dataset/subnamespace")
    data.subject = SimpleNamespace(id="dataset/sub-01")
    data.session = SimpleNamespace(id="dataset/sub-01/session-02")
    data._absolute_start = 12.25

    metadata = extract_window_metadata(data)

    assert metadata.dataset_id == "dataset/subnamespace"
    assert metadata.subject_id == "dataset/sub-01"
    assert metadata.session_id == "dataset/sub-01/session-02"
    assert metadata.absolute_start == 12.25
    assert metadata.window_duration == 2.5


def test_extract_window_metadata_uses_visible_sentinels_not_string_parsing():
    data = Data(domain=Interval(0.0, 1.0))
    data.session = SimpleNamespace(id="dataset/sub-01/session-02")

    metadata = extract_window_metadata(data)

    assert metadata.dataset_id == UNKNOWN_DATASET_ID
    assert metadata.subject_id == UNKNOWN_SUBJECT_ID

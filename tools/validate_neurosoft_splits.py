"""Data-backed QA for the NeuroSoft 8-band baseline split protocols.

Run before submitting a new processed NeuroSoft artifact or changing the
recording manifests::

    uv run python tools/validate_neurosoft_splits.py \
        --data-root "$SCRATCH/brainsets/processed"

The script verifies every intrasession block fold and every validation-only
LOSO held-out subject.  It exits non-zero on an empty or overlapping split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from foundry.data.datasets import NeurosoftMinipigs2026, NeurosoftMonkeys2026


REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "minipigs": (
        NeurosoftMinipigs2026,
        REPO_ROOT / "configs/data/neurosoft_minipigs/multisess_raw.yaml",
    ),
    "monkeys": (
        NeurosoftMonkeys2026,
        REPO_ROOT / "configs/data/neurosoft_monkeys/multisess_raw.yaml",
    ),
}


def _recording_ids(config_path: Path) -> list[str]:
    with config_path.open() as handle:
        return yaml.safe_load(handle)["dataset_kwargs"]["recording_ids"]


def _subject(recording_id: str) -> str:
    return recording_id.split("_", 1)[0]


def _intervals_overlap(first, second) -> bool:
    """Check whether any intervals overlap, including partial overlaps."""
    if not len(first) or not len(second):
        return False
    first_start = np.asarray(first.start)
    first_end = np.asarray(first.end)
    second_start = np.asarray(second.start)
    second_end = np.asarray(second.end)
    return bool(
        ((first_start[:, None] < second_end) & (second_start < first_end[:, None])).any()
    )


def _active_recordings(intervals: dict) -> set[str]:
    return {recording_id for recording_id, interval in intervals.items() if len(interval)}


def _make_dataset(dataset_class, root: str, recording_ids: list[str], **kwargs):
    return dataset_class(
        root=root,
        recording_ids=recording_ids,
        task_type="acoustic_stim",
        **kwargs,
    )


def _audit_intrasession(dataset_class, root: str, recording_ids: list[str]) -> None:
    expected = set(recording_ids)
    for fold in range(3):
        dataset = _make_dataset(
            dataset_class,
            root,
            recording_ids,
            split_type="intrasession-block",
            fold=fold,
        )
        partitions = {
            split: dataset.get_sampling_intervals(split)
            for split in ("train", "valid", "test")
        }
        for split, intervals in partitions.items():
            active = _active_recordings(intervals)
            if active != expected:
                missing = sorted(expected - active)
                raise AssertionError(
                    f"intrasession fold {fold} {split}: {len(missing)} recordings "
                    f"have no intervals: {missing}"
                )
        for recording_id in recording_ids:
            for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
                if _intervals_overlap(
                    partitions[first][recording_id], partitions[second][recording_id]
                ):
                    raise AssertionError(
                        f"intrasession fold {fold} {recording_id}: {first}/{second} overlap"
                    )
        print(f"intrasession fold {fold}: {len(expected)} recordings, disjoint train/valid/test")


def _audit_loso(dataset_class, root: str, recording_ids: list[str]) -> None:
    expected = set(recording_ids)
    subjects = sorted({_subject(recording_id) for recording_id in recording_ids})
    for held_out_subject in subjects:
        dataset = _make_dataset(
            dataset_class,
            root,
            recording_ids,
            split_type="loso",
            held_out_subject=held_out_subject,
        )
        train = _active_recordings(dataset.get_sampling_intervals("train"))
        valid = _active_recordings(dataset.get_sampling_intervals("valid"))
        test = _active_recordings(dataset.get_sampling_intervals("test"))
        expected_valid = {rid for rid in expected if _subject(rid) == held_out_subject}
        if valid != expected_valid:
            raise AssertionError(
                f"LOSO {held_out_subject}: validation recordings are not exactly "
                "the held-out subject"
            )
        if train != expected - expected_valid or train & valid:
            raise AssertionError(
                f"LOSO {held_out_subject}: train/validation subject leakage"
            )
        if test:
            raise AssertionError(f"LOSO {held_out_subject}: test must be empty")
        print(
            f"LOSO {held_out_subject}: {len(train)} train recordings, "
            f"{len(valid)} validation recordings"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    for species, (dataset_class, config_path) in SPECS.items():
        recording_ids = _recording_ids(config_path)
        print(f"{species}: {len(recording_ids)} recordings")
        _audit_intrasession(dataset_class, args.data_root, recording_ids)
        _audit_loso(dataset_class, args.data_root, recording_ids)
    print("All NeuroSoft intrasession and validation-only LOSO splits passed.")


if __name__ == "__main__":
    main()

"""Data-backed validation for all NeuroSoft split protocols.

The Phase 0 checks include exact recording coverage, non-empty partitions,
pairwise disjointness, chronological ordering for the causal protocol, and
held-out-subject isolation for LOSO.
"""

from __future__ import annotations

import argparse
import json
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
    if not len(first) or not len(second):
        return False
    first_start = np.asarray(first.start)
    first_end = np.asarray(first.end)
    second_start = np.asarray(second.start)
    second_end = np.asarray(second.end)
    return bool(
        (
            (first_start[:, None] < second_end)
            & (second_start < first_end[:, None])
        ).any()
    )


def _active_recordings(intervals: dict) -> set[str]:
    return {
        recording_id
        for recording_id, interval in intervals.items()
        if len(interval)
    }


def _make_dataset(dataset_class, root: str, recording_ids: list[str], **kwargs):
    return dataset_class(
        root=root,
        recording_ids=recording_ids,
        task_type="acoustic_stim",
        **kwargs,
    )


def _assert_exact_coverage(
    protocol: str,
    expected: set[str],
    partitions: dict[str, dict],
) -> None:
    for split, intervals in partitions.items():
        keys = set(intervals)
        if keys != expected:
            raise AssertionError(
                f"{protocol} {split}: recording keys differ; "
                f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
            )
        active = _active_recordings(intervals)
        if active != expected:
            raise AssertionError(
                f"{protocol} {split}: empty recordings="
                f"{sorted(expected - active)}"
            )


def _assert_disjoint(
    protocol: str,
    recording_ids: list[str],
    partitions: dict[str, dict],
) -> None:
    for recording_id in recording_ids:
        for first, second in (
            ("train", "valid"),
            ("train", "test"),
            ("valid", "test"),
        ):
            if _intervals_overlap(
                partitions[first][recording_id],
                partitions[second][recording_id],
            ):
                raise AssertionError(
                    f"{protocol} {recording_id}: {first}/{second} overlap"
                )


def _partition_counts(partitions: dict[str, dict]) -> dict[str, int]:
    return {
        split: sum(len(intervals) for intervals in values.values())
        for split, values in partitions.items()
    }


def _audit_intrasession(
    dataset_class, root: str, recording_ids: list[str]
) -> list[dict]:
    expected = set(recording_ids)
    results = []
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
        protocol = f"intrasession-block fold {fold}"
        _assert_exact_coverage(protocol, expected, partitions)
        _assert_disjoint(protocol, recording_ids, partitions)
        counts = _partition_counts(partitions)
        results.append({"fold": fold, "interval_counts": counts})
        print(
            f"{protocol}: {len(expected)} recordings, disjoint; counts={counts}"
        )
    return results


def _audit_loso(
    dataset_class, root: str, recording_ids: list[str]
) -> list[dict]:
    expected = set(recording_ids)
    subjects = sorted(
        {_subject(recording_id) for recording_id in recording_ids}
    )
    results = []
    for held_out_subject in subjects:
        dataset = _make_dataset(
            dataset_class,
            root,
            recording_ids,
            split_type="loso",
            held_out_subject=held_out_subject,
        )
        partitions = {
            split: dataset.get_sampling_intervals(split)
            for split in ("train", "valid", "test")
        }
        for split, intervals in partitions.items():
            if set(intervals) != expected:
                raise AssertionError(
                    f"LOSO {held_out_subject} {split}: recording keys differ"
                )
        train = _active_recordings(partitions["train"])
        valid = _active_recordings(partitions["valid"])
        test = _active_recordings(partitions["test"])
        expected_valid = {
            recording_id
            for recording_id in expected
            if _subject(recording_id) == held_out_subject
        }
        if valid != expected_valid:
            raise AssertionError(
                f"LOSO {held_out_subject}: valid is not exactly held-out subject"
            )
        if train != expected - expected_valid or train & valid:
            raise AssertionError(
                f"LOSO {held_out_subject}: train/validation subject leakage"
            )
        if test:
            raise AssertionError(f"LOSO {held_out_subject}: test must be empty")
        results.append(
            {
                "held_out_subject": held_out_subject,
                "train_recordings": len(train),
                "valid_recordings": len(valid),
                "target_leakage": [],
            }
        )
        print(
            f"LOSO {held_out_subject}: {len(train)} train recordings, "
            f"{len(valid)} held-out validation recordings"
        )
    return results


def _audit_intrasession_causal(
    dataset_class, root: str, recording_ids: list[str]
) -> dict:
    expected = set(recording_ids)
    dataset = _make_dataset(
        dataset_class,
        root,
        recording_ids,
        split_type="intrasession-causal",
    )
    partitions = {
        split: dataset.get_sampling_intervals(split)
        for split in ("train", "valid", "test")
    }
    protocol = "intrasession-causal"
    _assert_exact_coverage(protocol, expected, partitions)
    _assert_disjoint(protocol, recording_ids, partitions)

    # The artifact is causal within each disjoint recording-domain segment.
    # It is intentionally not one global cut: frequency blocks recur across a
    # session, so global train/valid/test extrema can overlap.
    checked_segments = 0
    for recording_id in recording_ids:
        domain = dataset.get_recording(recording_id).domain
        for segment_start, segment_end in zip(domain.start, domain.end):
            ranked_intervals = []
            for rank, split in enumerate(("train", "valid", "test")):
                intervals = partitions[split][recording_id]
                starts = np.asarray(intervals.start)
                ends = np.asarray(intervals.end)
                inside = (starts >= segment_start) & (ends <= segment_end)
                ranked_intervals.extend(
                    (float(start), rank) for start in starts[inside]
                )
            if not ranked_intervals:
                continue
            checked_segments += 1
            ranks = [rank for _, rank in sorted(ranked_intervals)]
            if any(first > second for first, second in zip(ranks, ranks[1:])):
                raise AssertionError(
                    f"{protocol} {recording_id}: train/valid/test are not "
                    f"chronological within domain segment "
                    f"[{segment_start}, {segment_end}]"
                )

    counts = _partition_counts(partitions)
    print(
        f"{protocol}: {len(expected)} recordings, disjoint and chronological "
        f"within {checked_segments} domain segments; counts={counts}"
    )
    return {
        "recordings": len(expected),
        "domain_segments_checked": checked_segments,
        "interval_counts": counts,
    }


def validate_splits(root: str) -> dict:
    result = {"status": "passed", "species": {}}
    for species, (dataset_class, config_path) in SPECS.items():
        recording_ids = _recording_ids(config_path)
        print(f"{species}: {len(recording_ids)} configured recordings")
        result["species"][species] = {
            "configured_recordings": len(recording_ids),
            "intrasession_block": _audit_intrasession(
                dataset_class, root, recording_ids
            ),
            "intrasession_causal": _audit_intrasession_causal(
                dataset_class, root, recording_ids
            ),
            "loso": _audit_loso(dataset_class, root, recording_ids),
        }
    print(
        "All NeuroSoft block, causal, and validation-only LOSO split checks passed."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = validate_splits(args.data_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Structured output: {args.output_json}")


if __name__ == "__main__":
    main()

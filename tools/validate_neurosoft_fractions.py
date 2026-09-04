"""Validate NeuroSoft nested fraction manifests against processed data.

The validator checks deterministic selection, nesting, immutable interval
identities, per-class support, and non-mutation of validation/test splits. It
reports unsupported scientific cells explicitly instead of treating them as a
software failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
MIN_PRESENT_CLASSES = 6


def _recording_ids(config_path: Path) -> list[str]:
    with config_path.open() as handle:
        return yaml.safe_load(handle)["dataset_kwargs"]["recording_ids"]


def _load_class_mapping():
    from foundry.tasks.classification_mapping import ClassificationMapping

    task_path = REPO_ROOT / "configs/tasks/neurosoft_acoustic_stim_8band.yaml"
    with task_path.open() as handle:
        config = yaml.safe_load(handle)
    return ClassificationMapping.from_dict(config["class_mapping"])


def _make_dataset(dataset_class, root: str, recording_ids: list[str]):
    return dataset_class(
        root=root,
        recording_ids=recording_ids,
        task_type="acoustic_stim",
        split_type="intrasession-causal",
    )


def _interval_signature(intervals) -> str:
    """Hash all selection-relevant interval fields."""
    payload = {
        "start": [float(value).hex() for value in np.asarray(intervals.start)],
        "end": [float(value).hex() for value in np.asarray(intervals.end)],
        "behavior_labels": [
            str(value) for value in np.asarray(intervals.behavior_labels)
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _recording_eligible(
    train,
    valid,
    test,
    class_mapping,
    min_class_support: int,
    min_present_classes: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not len(train):
        reasons.append("empty causal-train split")
    else:
        mapped, _ = class_mapping.filter_and_remap(
            np.asarray(train.behavior_labels)
        )
        counts = np.bincount(mapped, minlength=class_mapping.num_classes)
        present_class_count = int((counts > 0).sum())
        if present_class_count < min_present_classes:
            reasons.append(
                f"represented classes {present_class_count} < "
                f"{min_present_classes}"
            )
        for class_name, count in zip(class_mapping.class_names, counts):
            if 0 < count < min_class_support:
                reasons.append(
                    f"{class_name}: train support {int(count)} < "
                    f"{min_class_support}"
                )
    if not len(valid):
        reasons.append("empty causal-valid split")
    if not len(test):
        reasons.append("empty causal-test split")
    if len(train) and len(valid) and len(test):
        class_sets = []
        for intervals in (train, valid, test):
            mapped, _ = class_mapping.filter_and_remap(
                np.asarray(intervals.behavior_labels)
            )
            class_sets.append(frozenset(mapped.tolist()))
        if len(set(class_sets)) != 1:
            reasons.append("represented class set differs across splits")
    return not reasons, reasons


def validate_all(
    root: str, min_class_support: int, min_present_classes: int
) -> dict:
    """Run the full validation and return a JSON-serializable summary."""
    from foundry.data.datasets import (
        NeurosoftMinipigs2026,
        NeurosoftMonkeys2026,
    )
    from foundry.data.fraction_manifest import FractionManifestBuilder

    class_mapping = _load_class_mapping()
    specs = {
        "minipigs": (
            NeurosoftMinipigs2026,
            REPO_ROOT / "configs/data/neurosoft_minipigs/multisess_raw.yaml",
        ),
        "monkeys": (
            NeurosoftMonkeys2026,
            REPO_ROOT / "configs/data/neurosoft_monkeys/multisess_raw.yaml",
        ),
    }

    failures: list[str] = []
    unavailable_cells: list[dict] = []
    species_results: dict[str, dict] = {}
    available_cells = 0
    eligible_recordings = 0

    for species, (dataset_class, config_path) in specs.items():
        recording_ids = _recording_ids(config_path)
        dataset = _make_dataset(dataset_class, root, recording_ids)
        partitions = {
            split: dataset.get_sampling_intervals(split=split)
            for split in ("train", "valid", "test")
        }
        species_eligible = 0
        species_unavailable = 0

        print(f"\n{'─' * 72}")
        print(f"  {species}: {len(recording_ids)} configured recordings")
        print(f"{'─' * 72}")

        for recording_id in recording_ids:
            train = partitions["train"][recording_id]
            valid = partitions["valid"][recording_id]
            test = partitions["test"][recording_id]
            eligible, reasons = _recording_eligible(
                train,
                valid,
                test,
                class_mapping,
                min_class_support,
                min_present_classes,
            )
            if not eligible:
                print(f"  INELIGIBLE {recording_id}: " + "; ".join(reasons))
                continue

            eligible_recordings += 1
            species_eligible += 1
            split_signatures_before = {
                "valid": _interval_signature(valid),
                "test": _interval_signature(test),
            }

            for seed in SEEDS:
                builder = FractionManifestBuilder(
                    recording_id=recording_id,
                    train_intervals=train,
                    class_mapping=class_mapping,
                    seed=seed,
                    min_class_support=min_class_support,
                    min_present_classes=min_present_classes,
                )
                manifests = builder.build_all_fractions()
                repeated = FractionManifestBuilder(
                    recording_id=recording_id,
                    train_intervals=train,
                    class_mapping=class_mapping,
                    seed=seed,
                    min_class_support=min_class_support,
                    min_present_classes=min_present_classes,
                ).build_all_fractions()

                if not builder.validate_nesting():
                    failures.append(
                        f"nesting failed: {recording_id}, seed={seed}"
                    )
                if [m.to_dict() for m in manifests] != [
                    m.to_dict() for m in repeated
                ]:
                    failures.append(
                        f"determinism failed: {recording_id}, seed={seed}"
                    )

                full = manifests[-1]
                if full.requested_fraction != 1.0:
                    failures.append(
                        f"fraction grid does not end at 1.0: {recording_id}"
                    )
                if len(full.selected_indices) != full.total_intervals:
                    failures.append(
                        f"full fraction omits mapped intervals: {recording_id}"
                    )
                if len(set(full.selected_interval_ids)) != full.total_intervals:
                    failures.append(
                        f"interval identities are not unique: {recording_id}"
                    )

                for manifest in manifests:
                    cell = {
                        "species": species,
                        "recording_id": recording_id,
                        "seed": seed,
                        "requested_fraction": manifest.requested_fraction,
                        "realized_fraction": manifest.realized_fraction,
                        "per_class_counts": manifest.per_class_counts,
                        "failure_reason": manifest.failure_reason,
                    }
                    if manifest.available:
                        available_cells += 1
                    else:
                        unavailable_cells.append(cell)
                        species_unavailable += 1

            split_signatures_after = {
                "valid": _interval_signature(valid),
                "test": _interval_signature(test),
            }
            if split_signatures_before != split_signatures_after:
                failures.append(
                    f"fraction building mutated valid/test: {recording_id}"
                )

        species_results[species] = {
            "configured_recordings": len(recording_ids),
            "eligible_recordings": species_eligible,
            "unavailable_scientific_cells": species_unavailable,
        }
        print(
            f"  {species}: {species_eligible} eligible recordings; "
            f"{species_unavailable} unavailable fraction/seed cells"
        )

    if not eligible_recordings:
        failures.append("no eligible recordings were found")

    result = {
        "status": "passed" if not failures else "failed",
        "seeds": list(SEEDS),
        "min_class_support": min_class_support,
        "min_present_classes": min_present_classes,
        "species": species_results,
        "eligible_recordings": eligible_recordings,
        "available_scientific_cells": available_cells,
        "unavailable_scientific_cells": unavailable_cells,
        "software_failures": failures,
    }

    print(f"\n{'=' * 72}")
    print("  FRACTION VALIDATION SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Eligible recordings: {eligible_recordings}")
    print(f"  Available recording/fraction/seed cells: {available_cells}")
    print(f"  Explicitly unavailable cells: {len(unavailable_cells)}")
    for cell in unavailable_cells:
        print(
            "  UNAVAILABLE "
            f"{cell['species']}/{cell['recording_id']} "
            f"fraction={cell['requested_fraction']:.2f} "
            f"seed={cell['seed']}: {cell['failure_reason']}"
        )
    if failures:
        print("  Software validation failures:")
        for failure in failures:
            print(f"    - {failure}")
    else:
        print(
            "  Determinism, nesting, identities, and split invariance passed."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NeuroSoft nested fraction manifests."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--min-class-support", type=int, default=3)
    parser.add_argument(
        "--min-present-classes", type=int, default=MIN_PRESENT_CLASSES
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.min_class_support < 1:
        parser.error("--min-class-support must be at least 1")
    if not 1 <= args.min_present_classes <= 8:
        parser.error("--min-present-classes must be between 1 and 8")

    result = validate_all(
        args.data_root, args.min_class_support, args.min_present_classes
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  Structured output: {args.output_json}")
    sys.exit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

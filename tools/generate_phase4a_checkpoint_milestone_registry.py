"""Build the hash-pinned early-checkpoint registry from verified Mila sources.

The Phase 4A source runs published immutable manifests for each requested
compute milestone.  This tool deliberately starts from the selected Mila
source registry, so every derived checkpoint retains the parent experiment's
target exclusion and paired source-selection/model-seed provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGISTRY = ROOT / "launch/checkpoint_sets/phase4a-mila-best.jsonl"
MILESTONES = {
    500: "milestone-1pct-step500",
    1500: "milestone-3pct-step1500",
    5000: "milestone-10pct-step5000",
    15000: "milestone-30pct-step15000",
}
SET_ID = "phase4a-mila-early-checkpoints-v1"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_hash")
    actual = canonical_digest(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_hash"
        }
    )
    if expected != actual:
        raise ValueError(f"{path}: manifest self-hash mismatch")
    return manifest


def records() -> list[dict[str, Any]]:
    source_rows = [
        json.loads(line)
        for line in SOURCE_REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: list[dict[str, Any]] = []
    for source in source_rows:
        best_path = Path(source["manifest_path"])
        for step, stem in MILESTONES.items():
            path = best_path.parent / f"{stem}.json"
            manifest = load_verified_manifest(path)
            checkpoint = manifest["checkpoint"]
            trained_on = manifest["trained_on"]
            if trained_on["excluded_target"]["species"] != source["species"]:
                raise ValueError(f"{path}: species drift from selected source")
            if (
                trained_on["excluded_target"]["subject"]
                != source["excluded_target_subject"]
            ):
                raise ValueError(
                    f"{path}: target-exclusion drift from selected source"
                )
            if checkpoint["kind"] != stem.rsplit("-step", 1)[0]:
                raise ValueError(f"{path}: unexpected checkpoint kind")
            output.append(
                {
                    "checkpoint_id": (
                        f"fullpool-{source['species']}-"
                        f"{source['excluded_target_subject']}-"
                        f"sel{source['source_selection_seed']}-"
                        f"model{source['source_model_seed']}-step{step}"
                    ),
                    "checkpoint_set_id": SET_ID,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "condition": {
                        "checkpoint_kind": checkpoint["kind"],
                        "label": f"fullpool-f1.00-step{step}",
                        "milestone": f"step{step}",
                        "milestone_step": step,
                        "source_fraction": 1.0,
                        "source_mixture": "same_species_target_excluded_fullpool",
                    },
                    "excluded_target_subject": source[
                        "excluded_target_subject"
                    ],
                    "manifest_hash": manifest["manifest_hash"],
                    "manifest_path": str(path),
                    "source_model_seed": source["source_model_seed"],
                    "source_selection_seed": source["source_selection_seed"],
                    "species": source["species"],
                }
            )
    expected = {"minipigs": 84, "monkeys": 60}
    observed = {
        species: sum(row["species"] == species for row in output)
        for species in expected
    }
    if observed != expected:
        raise ValueError(f"Unexpected registry counts: {observed}")
    return sorted(output, key=lambda row: row["checkpoint_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "launch/checkpoint_sets/phase4a-mila-early-checkpoints.jsonl",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    rows = records()
    payload = b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    lock = {
        "schema": "foundry-checkpoint-registry-lock",
        "version": 1,
        "checkpoint_set_id": SET_ID,
        "source_registry": str(SOURCE_REGISTRY.relative_to(ROOT)),
        "source_registry_sha256": file_digest(SOURCE_REGISTRY),
        "registry": str(output.relative_to(ROOT)),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "milestone_steps": sorted(MILESTONES),
        "records": len(rows),
        "records_by_species": {
            species: sum(row["species"] == species for row in rows)
            for species in ("minipigs", "monkeys")
        },
    }
    lock_path = output.with_suffix(".lock.json")
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} registry records: {output.relative_to(ROOT)}")
    print(f"registry SHA-256: {lock['registry_sha256']}")


if __name__ == "__main__":
    main()

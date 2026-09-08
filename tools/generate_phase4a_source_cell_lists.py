"""Generate and validate the coupled Phase-4A source cell vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (42, 43, 44)
TARGETS = {"minipigs": range(1, 8), "monkeys": range(1, 6)}


def build_records(repo_root: Path, species: str) -> list[dict]:
    records = []
    for target_number in TARGETS[species]:
        target_subject = f"sub-{target_number:02d}"
        target_dir = f"target-sub-{target_number:02d}"
        for seed in SEEDS:
            relative = (
                Path("manifests/neurosoft_supervised/v1/source_volume")
                / species
                / target_dir
                / "fraction-1.00"
                / f"selection-{seed}.json"
            )
            manifest_path = repo_root / relative
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("target_species") != species:
                raise ValueError(f"{manifest_path}: target_species mismatch")
            if manifest.get("target_subject") != target_subject:
                raise ValueError(f"{manifest_path}: target_subject mismatch")
            if (
                manifest.get("condition", {}).get("source_selection_seed")
                != seed
            ):
                raise ValueError(f"{manifest_path}: source seed mismatch")

            records.append(
                {
                    "cell_id": f"{species}_{target_dir}_s{seed}",
                    "species": species,
                    "target_subject": target_subject,
                    "source_seed": seed,
                    "source_manifest": relative.as_posix(),
                    "overrides": [
                        f"source_manifest={relative.as_posix()}",
                        f"run.source_target_subject={target_subject}",
                        f"run.seed={seed}",
                    ],
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "launch").resolve()

    for species in ("minipigs", "monkeys"):
        records = build_records(repo_root, species)
        expected = 21 if species == "minipigs" else 15
        if len(records) != expected:
            raise AssertionError(
                f"{species}: expected {expected}, got {len(records)}"
            )
        output = output_dir / f"phase4a-source-{species}.jsonl"
        write_jsonl(output, records)
        print(f"{species}: {len(records)} cells -> {output}")


if __name__ == "__main__":
    main()

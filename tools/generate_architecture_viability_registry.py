#!/usr/bin/env python3
"""Build hash-verified fixed-milestone registries for architecture sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from foundry.training.checkpoint_manifest import load_checkpoint_manifest
from tools.generate_architecture_viability_source_cells import (
    CONDITIONS,
    GROUPS,
    TARGETS,
)


ROOT = Path(__file__).resolve().parents[1]
MILESTONES = {
    100: "milestone-1pct",
    300: "milestone-3pct",
    1000: "milestone-10pct",
    3000: "milestone-30pct",
    10000: "milestone-100pct",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_registry(
    run_root: Path, checkpoint_root: Path, condition_id: str
) -> list[dict[str, Any]]:
    expected_model = CONDITIONS[condition_id]
    rows: list[dict[str, Any]] = []
    for species, numbers in TARGETS.items():
        for number in numbers:
            subject = f"sub-{number:02d}"
            run_name = f"arch-{condition_id}-{species}-{subject}-s42-m42"
            manifest_dir = run_root / GROUPS[species] / run_name / "manifests"
            for step, kind in MILESTONES.items():
                matches = sorted(manifest_dir.glob(f"{kind}-step{step}.json"))
                if len(matches) != 1:
                    raise ValueError(
                        f"{manifest_dir}: expected one {kind}-step{step} manifest, "
                        f"found {len(matches)}"
                    )
                path = matches[0].resolve()
                manifest = load_checkpoint_manifest(path)
                trained = manifest.get("trained_on", {})
                checkpoint = manifest.get("checkpoint", {})
                metadata = manifest.get("recipe", {}).get("model_metadata", {})
                checks = {
                    "source_condition": condition_id,
                    "input_adapter_mode": expected_model["input_adapter_mode"],
                    "input_adapter_bias": expected_model["input_adapter_bias"],
                    "shared_input_channels": expected_model[
                        "shared_input_channels"
                    ],
                    "temporal_channels": expected_model["temporal_channels"],
                    "gru_hidden_size": expected_model["gru_hidden_size"],
                    "transferable_parameter_count": expected_model[
                        "transferable_parameter_count"
                    ],
                }
                mismatches = {
                    key: (metadata.get(key), value)
                    for key, value in checks.items()
                    if metadata.get(key) != value
                }
                if mismatches:
                    raise ValueError(
                        f"{path}: model metadata mismatch: {mismatches}"
                    )
                observed = (
                    trained.get("excluded_target", {}).get("species"),
                    trained.get("excluded_target", {}).get("subject"),
                    trained.get("source_selection_seed"),
                    trained.get("source_model_seed"),
                    trained.get("optimizer_steps"),
                    checkpoint.get("kind"),
                )
                expected = (species, subject, 42, 42, step, kind)
                if observed != expected:
                    raise ValueError(
                        f"{path}: provenance {observed!r} != {expected!r}"
                    )
                if not manifest.get("git_sha") or not manifest.get(
                    "snapshot_bundle"
                ):
                    raise ValueError(f"{path}: missing Git/snapshot provenance")
                relative = Path(str(checkpoint.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"{path}: unsafe checkpoint path")
                checkpoint_path = checkpoint_root / relative
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                if _file_digest(checkpoint_path) != checkpoint.get("sha256"):
                    raise ValueError(f"{path}: checkpoint SHA-256 mismatch")
                rows.append(
                    {
                        "checkpoint_set_id": f"architecture-{condition_id}-fixed-v1",
                        "checkpoint_id": (
                            f"architecture-{condition_id}-{species}-{subject}-step{step}"
                        ),
                        "checkpoint_sha256": checkpoint["sha256"],
                        "manifest_path": str(path),
                        "manifest_hash": manifest["manifest_hash"],
                        "species": species,
                        "excluded_target_subject": subject,
                        "source_selection_seed": 42,
                        "source_model_seed": 42,
                        "source_git_sha": manifest["git_sha"],
                        "snapshot_bundle": manifest["snapshot_bundle"],
                        "wandb": manifest.get("wandb", {}),
                        "model_metadata": metadata,
                        "condition": {
                            "checkpoint_kind": kind,
                            "label": f"{condition_id}-fixed-step-{step}",
                            "milestone": f"step{step}",
                            "milestone_step": step,
                            "source_condition": condition_id,
                            "source_fraction": 1.0,
                            "source_mixture": "same_species_target_excluded_fullpool",
                        },
                    }
                )
    return sorted(rows, key=lambda row: row["checkpoint_id"])


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    lock = {
        "schema": "foundry-checkpoint-registry-lock",
        "version": 1,
        "checkpoint_set_id": rows[0]["checkpoint_set_id"],
        "registry": path.as_posix(),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(rows),
        "records_by_species": dict(
            sorted(Counter(r["species"] for r in rows).items())
        ),
        "records_by_step": dict(
            sorted(
                Counter(
                    str(r["condition"]["milestone_step"]) for r in rows
                ).items()
            )
        ),
    }
    lock["lock_payload_sha256"] = _canonical_digest(lock)
    path.with_suffix(".lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "launch/checkpoint_sets"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registries = {
        condition: build_registry(
            args.run_root, args.checkpoint_root, condition
        )
        for condition in CONDITIONS
    }
    counts = {condition: len(rows) for condition, rows in registries.items()}
    if set(counts.values()) != {60} or sum(counts.values()) != 300:
        raise AssertionError(f"Unexpected registry counts: {counts}")
    if not args.check:
        for condition, rows in registries.items():
            write_registry(
                args.output_dir / f"architecture-{condition}-fixed.jsonl", rows
            )
    print(json.dumps({"counts": counts, "total": 240}, sort_keys=True))


if __name__ == "__main__":
    main()

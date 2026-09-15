#!/usr/bin/env python
"""Build the audited Phase 4F registry from Phase 4E milestone manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from foundry.training.checkpoint_manifest import load_checkpoint_manifest


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
TARGETS = {"minipigs": range(1, 8), "monkeys": range(1, 6)}
GROUPS = {
    "minipigs": "PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS",
    "monkeys": "PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS",
}
MILESTONES = {
    100: "milestone-1pct",
    300: "milestone-3pct",
    1000: "milestone-10pct",
    3000: "milestone-30pct",
    10000: "milestone-100pct",
}
CHECKPOINT_SET_ID = "phase4e-fixed-milestones-v1"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_name(species: str, subject: str, seed: int) -> str:
    prefix = "src_mp" if species == "minipigs" else "src_mk"
    return f"{prefix}_{subject}_s{seed}_m{seed}"


def build_registry(
    run_root: Path, checkpoint_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_git_shas: set[str] = set()
    for species, numbers in TARGETS.items():
        for number in numbers:
            subject = f"sub-{number:02d}"
            for seed in SEEDS:
                run_name = _run_name(species, subject, seed)
                manifest_dir = (
                    run_root / GROUPS[species] / run_name / "manifests"
                )
                for step, kind in MILESTONES.items():
                    matches = sorted(
                        manifest_dir.glob(f"{kind}-step{step}.json")
                    )
                    if len(matches) != 1:
                        raise ValueError(
                            f"{manifest_dir}: expected one {kind} step-{step} "
                            f"manifest, found {len(matches)}"
                        )
                    path = matches[0].resolve()
                    manifest = load_checkpoint_manifest(path)
                    trained_on = manifest.get("trained_on", {})
                    checkpoint = manifest.get("checkpoint", {})
                    observed = {
                        "species": trained_on.get("excluded_target", {}).get(
                            "species"
                        ),
                        "subject": trained_on.get("excluded_target", {}).get(
                            "subject"
                        ),
                        "selection_seed": trained_on.get(
                            "source_selection_seed"
                        ),
                        "model_seed": trained_on.get("source_model_seed"),
                        "step": trained_on.get("optimizer_steps"),
                        "kind": checkpoint.get("kind"),
                    }
                    expected = {
                        "species": species,
                        "subject": subject,
                        "selection_seed": seed,
                        "model_seed": seed,
                        "step": step,
                        "kind": kind,
                    }
                    if observed != expected:
                        raise ValueError(
                            f"{path}: provenance mismatch: "
                            f"{observed!r} != {expected!r}"
                        )
                    relative = Path(str(checkpoint.get("path", "")))
                    if (
                        not relative.parts
                        or relative.is_absolute()
                        or ".." in relative.parts
                    ):
                        raise ValueError(f"{path}: unsafe checkpoint path")
                    checkpoint_path = checkpoint_root / relative
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(checkpoint_path)
                    if _file_digest(checkpoint_path) != checkpoint.get(
                        "sha256"
                    ):
                        raise ValueError(f"{path}: checkpoint SHA-256 mismatch")
                    git_sha = str(manifest.get("git_sha") or "")
                    if not git_sha:
                        raise ValueError(f"{path}: missing source Git SHA")
                    source_git_shas.add(git_sha)
                    rows.append(
                        {
                            "checkpoint_id": (
                                f"phase4e-{species}-{subject}-sel{seed}-"
                                f"model{seed}-step{step}"
                            ),
                            "checkpoint_set_id": CHECKPOINT_SET_ID,
                            "checkpoint_sha256": checkpoint["sha256"],
                            "condition": {
                                "checkpoint_kind": kind,
                                "label": (
                                    f"same-species-fullpool-fixed-step-{step}"
                                ),
                                "milestone": f"step{step}",
                                "milestone_step": step,
                                "source_fraction": 1.0,
                                "source_mixture": (
                                    "same_species_target_excluded_fullpool"
                                ),
                            },
                            "excluded_target_subject": subject,
                            "manifest_hash": manifest["manifest_hash"],
                            "manifest_path": str(path),
                            "source_git_sha": git_sha,
                            "source_model_seed": seed,
                            "source_selection_seed": seed,
                            "species": species,
                        }
                    )
    if len(source_git_shas) != 1:
        raise ValueError(
            f"Expected one Phase 4E source Git SHA, found {sorted(source_git_shas)}"
        )
    return sorted(rows, key=lambda row: row["checkpoint_id"])


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(
            row, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    species_counts = Counter(str(row["species"]) for row in rows)
    step_counts = Counter(
        int(row["condition"]["milestone_step"]) for row in rows
    )
    lock = {
        "schema": "foundry-checkpoint-registry-lock",
        "version": 1,
        "checkpoint_set_id": CHECKPOINT_SET_ID,
        "registry": path.as_posix(),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(rows),
        "records_by_species": dict(sorted(species_counts.items())),
        "records_by_step": {
            str(key): value for key, value in sorted(step_counts.items())
        },
        "source_git_sha": rows[0]["source_git_sha"],
    }
    lock["lock_payload_sha256"] = _canonical_digest(lock)
    path.with_suffix(".lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/network/scratch/s/sobralm/runs"),
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "launch/checkpoint_sets/phase4e-fixed-milestones.jsonl",
    )
    args = parser.parse_args()
    rows = build_registry(args.run_root, args.checkpoint_root)
    write_registry(args.output, rows)
    counts = Counter(str(row["species"]) for row in rows)
    print(f"wrote {len(rows)} milestone records to {args.output}")
    print(f"records by species: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

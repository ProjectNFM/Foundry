#!/usr/bin/env python
"""Build Phase 4E source cell lists and loss-selected checkpoint registry.

The default mode writes the exact 36 source-pretraining cells.  ``--registry``
is deliberately a post-source-run operation: it verifies each published
loss-selected manifest and writes the 36-row registry consumed by the
downstream compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
CHECKPOINT_SET_ID = "phase4e-validation-loss-v1"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest(species: str, subject: str, seed: int) -> Path:
    return (
        Path("manifests/neurosoft_supervised/v1/source_volume")
        / species
        / f"target-{subject}"
        / "fraction-1.00"
        / f"selection-{seed}.json"
    )


def build_source_cells(repo_root: Path, species: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for number in TARGETS[species]:
        subject = f"sub-{number:02d}"
        for seed in SEEDS:
            manifest = _source_manifest(species, subject, seed)
            path = repo_root / manifest
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("target_species") != species:
                raise ValueError(f"{path}: target_species mismatch")
            if payload.get("target_subject") != subject:
                raise ValueError(f"{path}: target_subject mismatch")
            if (
                payload.get("condition", {}).get("source_selection_seed")
                != seed
            ):
                raise ValueError(f"{path}: source selection seed mismatch")
            cells.append(
                {
                    "cell_id": f"phase4e_source__{species}__{subject}__s{seed}",
                    "species": species,
                    "target_subject": subject,
                    "source_selection_seed": seed,
                    "source_model_seed": seed,
                    "source_manifest": manifest.as_posix(),
                    "wandb_group": GROUPS[species],
                    "overrides": [
                        f"source_manifest={manifest.as_posix()}",
                        f"run.source_target_subject={subject}",
                        f"run.seed={seed}",
                        f"run.group={GROUPS[species]}",
                        "trainer.max_steps=10000",
                        "trainer.val_check_interval=100",
                        "trainer.callbacks.early_stopping=null",
                        "trainer.callbacks.model_checkpoint.monitor=val/loss",
                        "trainer.callbacks.model_checkpoint.mode=min",
                        "trainer.callbacks.compute_tracking.monitor=val/loss",
                        "trainer.callbacks.compute_tracking.mode=min",
                    ],
                }
            )
    return cells


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _expected_run_name(species: str, subject: str, seed: int) -> str:
    prefix = "src_mp" if species == "minipigs" else "src_mk"
    return f"{prefix}_{subject}_s{seed}_m{seed}"


def build_checkpoint_registry(
    run_root: Path, checkpoint_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, numbers in TARGETS.items():
        for number in numbers:
            subject = f"sub-{number:02d}"
            for seed in SEEDS:
                run_name = _expected_run_name(species, subject, seed)
                manifest_path = (
                    run_root / GROUPS[species] / run_name / "manifests"
                )
                manifests = sorted(manifest_path.glob("best*.json"))
                if len(manifests) != 1:
                    raise ValueError(
                        f"{manifest_path}: expected exactly one loss-selected best manifest, found {len(manifests)}"
                    )
                path = manifests[0].resolve()
                manifest = load_checkpoint_manifest(path)
                trained_on = manifest.get("trained_on", {})
                selection = manifest.get("selection", {})
                checkpoint = manifest.get("checkpoint", {})
                checks = {
                    "species": trained_on.get("excluded_target", {}).get(
                        "species"
                    ),
                    "subject": trained_on.get("excluded_target", {}).get(
                        "subject"
                    ),
                    "selection_seed": trained_on.get("source_selection_seed"),
                    "model_seed": trained_on.get("source_model_seed"),
                    "monitor": selection.get("monitor"),
                    "kind": checkpoint.get("kind"),
                }
                expected = {
                    "species": species,
                    "subject": subject,
                    "selection_seed": seed,
                    "model_seed": seed,
                    "monitor": "val/loss",
                    "kind": "best",
                }
                if checks != expected:
                    raise ValueError(
                        f"{path}: provenance mismatch: {checks!r} != {expected!r}"
                    )
                relative = Path(str(checkpoint.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"{path}: unsafe checkpoint path")
                checkpoint_path = checkpoint_root / relative
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                if _file_digest(checkpoint_path) != checkpoint.get("sha256"):
                    raise ValueError(f"{path}: checkpoint SHA-256 mismatch")
                value = selection.get("monitor_value")
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(
                        f"{path}: invalid val/loss selection value"
                    )
                rows.append(
                    {
                        "checkpoint_id": f"phase4e-{species}-{subject}-sel{seed}-model{seed}-best-loss",
                        "checkpoint_set_id": CHECKPOINT_SET_ID,
                        "checkpoint_sha256": checkpoint["sha256"],
                        "condition": {
                            "checkpoint_kind": "best",
                            "label": "same-species-fullpool-validation-loss-best",
                            "milestone": "validation_loss_selected",
                            "source_fraction": 1.0,
                            "source_mixture": "same_species_target_excluded_fullpool",
                        },
                        "excluded_target_subject": subject,
                        "manifest_hash": manifest["manifest_hash"],
                        "manifest_path": str(path),
                        "source_model_seed": seed,
                        "source_selection_seed": seed,
                        "species": species,
                    }
                )
    return sorted(rows, key=lambda row: row["checkpoint_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "launch/phase4e"
    )
    parser.add_argument("--registry", action="store_true")
    parser.add_argument(
        "--run-root", type=Path, default=Path("/network/scratch/s/sobralm/runs")
    )
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    args = parser.parse_args()
    if args.registry:
        if args.checkpoint_root is None:
            parser.error("--registry requires --checkpoint-root")
        rows = build_checkpoint_registry(args.run_root, args.checkpoint_root)
        output = args.output_dir / "phase4e-validation-loss.jsonl"
        _write_jsonl(output, rows)
        print(f"wrote {len(rows)} loss-selected checkpoint records: {output}")
        return
    for species in TARGETS:
        rows = build_source_cells(ROOT, species)
        output = args.output_dir / f"phase4e-source-{species}.jsonl"
        _write_jsonl(output, rows)
        print(f"wrote {len(rows)} source cells: {output}")


if __name__ == "__main__":
    main()

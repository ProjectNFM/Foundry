"""JSON/Markdown checkpoint manifests for NeuroSoft source pretraining."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from foundry.data.fraction_manifest import _canonical_hash

CHECKPOINT_MANIFEST_SCHEMA = "neurosoft-pretraining-checkpoint"
CHECKPOINT_MANIFEST_VERSION = 1

_BLOCK_SIZE = 1024 * 1024


class CheckpointManifestError(RuntimeError):
    """Raised when a checkpoint manifest fails validation or integrity checks."""


class CheckpointManifestWriter:
    """Write, load, and verify hash-backed pretraining checkpoint manifests."""

    schema = CHECKPOINT_MANIFEST_SCHEMA
    version = CHECKPOINT_MANIFEST_VERSION

    @staticmethod
    def write(
        checkpoint_path: str | Path,
        manifest_dir: str | Path,
        *,
        kind: str,
        trained_on: dict[str, Any],
        selection: dict[str, Any],
        compute: dict[str, Any],
        recipe: dict[str, Any],
        normalization_artifact_hashes: dict[str, str],
        git_sha: str,
        snapshot_bundle: str,
        slurm_job_id: str,
        wandb_info: dict[str, str],
    ) -> tuple[Path, Path]:
        return write_checkpoint_manifest(
            checkpoint_path,
            manifest_dir,
            kind=kind,
            trained_on=trained_on,
            selection=selection,
            compute=compute,
            recipe=recipe,
            normalization_artifact_hashes=normalization_artifact_hashes,
            git_sha=git_sha,
            snapshot_bundle=snapshot_bundle,
            slurm_job_id=slurm_job_id,
            wandb_info=wandb_info,
        )

    @staticmethod
    def load(manifest_path: str | Path) -> dict[str, Any]:
        return load_checkpoint_manifest(manifest_path)

    @staticmethod
    def verify_integrity(manifest: dict[str, Any], checkpoint_root: str) -> None:
        verify_checkpoint_integrity(manifest, checkpoint_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _compute_manifest_hash(payload: dict[str, Any]) -> str:
    hash_payload = dict(payload)
    hash_payload.pop("manifest_hash", None)
    return _canonical_hash(hash_payload)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp = Path(temp_path)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    manifest_dir: str | Path,
    *,
    kind: str,
    trained_on: dict[str, Any],
    selection: dict[str, Any],
    compute: dict[str, Any],
    recipe: dict[str, Any],
    normalization_artifact_hashes: dict[str, str],
    git_sha: str,
    snapshot_bundle: str,
    slurm_job_id: str,
    wandb_info: dict[str, str],
) -> tuple[Path, Path]:
    """Write JSON and Markdown checkpoint manifests atomically.

    Returns:
        Tuple of ``(json_path, md_path)``.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise CheckpointManifestError(f"Checkpoint file not found: {checkpoint}")

    destination_dir = Path(manifest_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    sha256 = _sha256_file(checkpoint)
    size_bytes = checkpoint.stat().st_size
    stem = checkpoint.stem
    if checkpoint.is_absolute():
        manifest_checkpoint_path = f"checkpoints/{checkpoint.name}"
    else:
        manifest_checkpoint_path = checkpoint.as_posix()

    manifest: dict[str, Any] = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "version": CHECKPOINT_MANIFEST_VERSION,
        "checkpoint": {
            "kind": kind,
            "path": manifest_checkpoint_path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        },
        "trained_on": dict(trained_on),
        "selection": dict(selection),
        "compute": dict(compute),
        "recipe": dict(recipe),
        "normalization_artifact_hashes": dict(normalization_artifact_hashes),
        "git_sha": git_sha,
        "snapshot_bundle": snapshot_bundle,
        "slurm_job_id": slurm_job_id,
        "wandb": dict(wandb_info),
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)

    json_path = destination_dir / f"{stem}.json"
    md_path = destination_dir / f"{stem}.md"

    json_text = json.dumps(
        manifest,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    _atomic_write_text(json_path, json_text)
    _atomic_write_text(md_path, generate_checkpoint_markdown(manifest))

    return json_path, md_path


def generate_checkpoint_markdown(manifest: dict[str, Any]) -> str:
    """Generate a human-readable Markdown summary from a checkpoint manifest."""
    checkpoint = manifest["checkpoint"]
    trained_on = manifest["trained_on"]
    selection = manifest["selection"]
    compute = manifest["compute"]
    excluded = trained_on["excluded_target"]
    wandb_info = manifest.get("wandb", {})

    subjects = trained_on.get("subjects", [])
    recordings = trained_on.get("recordings", [])
    class_union = trained_on.get("class_union", [])
    class_intersection = trained_on.get("class_intersection", [])
    session_scores = selection.get("source_session_scores", {})

    lines = [
        "# NeuroSoft Pretraining Checkpoint Manifest",
        "",
        "## Checkpoint",
        "",
        f"- **Kind:** {checkpoint['kind']}",
        f"- **Monitor:** {selection.get('monitor', 'n/a')}",
        f"- **Monitor value:** {selection.get('monitor_value', 'n/a')}",
        f"- **Path:** `{checkpoint['path']}`",
        f"- **SHA-256:** `{checkpoint['sha256']}`",
        f"- **Size (bytes):** {checkpoint['size_bytes']}",
        "",
        "## Target Excluded From Pretraining",
        "",
        f"- **Species:** {excluded['species']}",
        f"- **Subject:** {excluded['subject']}",
        "",
        "## Source Data",
        "",
        f"- **Selection ID:** {trained_on.get('source_selection_id', 'n/a')}",
        f"- **Source manifest:** `{trained_on.get('source_manifest_path', 'n/a')}`",
        f"- **Source manifest hash:** `{trained_on.get('source_manifest_hash', 'n/a')}`",
        f"- **Subjects ({len(subjects)}):**",
    ]
    for subject in subjects:
        lines.append(f"  - `{subject}`")
    lines.extend(
        [
            f"- **Recordings ({len(recordings)}):**",
        ]
    )
    for recording in recordings:
        lines.append(f"  - `{recording}`")
    lines.extend(
        [
            f"- **Class union ({len(class_union)}):** {', '.join(class_union) or 'n/a'}",
            f"- **Class intersection ({len(class_intersection)}):** "
            f"{', '.join(class_intersection) or 'n/a'}",
            "",
            "## Training Accounting",
            "",
            f"- **Selected train examples:** "
            f"{trained_on.get('selected_train_examples', 'n/a')}",
            f"- **Available train windows:** "
            f"{trained_on.get('available_train_windows', 'n/a')}",
            f"- **Realized train windows / epoch:** "
            f"{trained_on.get('realized_train_windows_per_epoch', 'n/a')}",
            f"- **Processed windows:** {trained_on.get('processed_windows', 'n/a')}",
            f"- **Completed effective epochs:** "
            f"{trained_on.get('completed_effective_epochs', 'n/a')}",
            f"- **Optimizer steps:** {trained_on.get('optimizer_steps', 'n/a')}",
            f"- **Signal seconds:** {compute.get('signal_seconds', 'n/a')}",
            "",
            "## Compute",
            "",
            f"- **Cumulative FLOPs:** {compute.get('cumulative_flops', 'n/a')}",
            f"- **FLOP method:** {compute.get('flop_method', 'n/a')}",
            f"- **Wall time (seconds):** {compute.get('wall_time_seconds', 'n/a')}",
            f"- **GPU:** {compute.get('gpu', 'n/a')}",
            f"- **Precision:** {compute.get('precision', 'n/a')}",
            "",
            "## Source Validation Scores",
            "",
        ]
    )
    if session_scores:
        for session_id, score in sorted(session_scores.items()):
            lines.append(f"- `{session_id}`: {score}")
    else:
        lines.append("- (none recorded)")
    lines.extend(
        [
            "",
            "## Recipe",
            "",
            "```json",
            json.dumps(
                manifest.get("recipe", {}),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Normalization Artifact Hashes",
            "",
            "```json",
            json.dumps(
                manifest.get("normalization_artifact_hashes", {}),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Provenance",
            "",
            f"- **Git SHA:** `{manifest.get('git_sha', 'n/a')}`",
            f"- **Snapshot bundle:** `{manifest.get('snapshot_bundle', 'n/a')}`",
            f"- **Slurm job ID:** `{manifest.get('slurm_job_id', 'n/a')}`",
            f"- **W&B project:** `{wandb_info.get('project', 'n/a')}`",
            f"- **W&B group:** `{wandb_info.get('group', 'n/a')}`",
            f"- **W&B run ID:** `{wandb_info.get('run_id', 'n/a')}`",
            f"- **Manifest hash:** `{manifest.get('manifest_hash', 'n/a')}`",
            "",
        ]
    )
    return "\n".join(lines)


def load_checkpoint_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load, schema-check, and hash-verify a checkpoint manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise CheckpointManifestError(f"Manifest file not found: {path}")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointManifestError(
            f"Invalid JSON in checkpoint manifest {path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise CheckpointManifestError(
            f"Checkpoint manifest root must be an object: {path}"
        )

    schema = manifest.get("schema")
    if schema != CHECKPOINT_MANIFEST_SCHEMA:
        raise CheckpointManifestError(
            f"Unsupported schema: expected {CHECKPOINT_MANIFEST_SCHEMA!r}, "
            f"got {schema!r}"
        )

    version = manifest.get("version")
    if version != CHECKPOINT_MANIFEST_VERSION:
        raise CheckpointManifestError(
            f"Unsupported version: expected {CHECKPOINT_MANIFEST_VERSION}, "
            f"got {version!r}"
        )

    recorded_hash = manifest.get("manifest_hash")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise CheckpointManifestError("Manifest is missing manifest_hash")

    actual_hash = _compute_manifest_hash(manifest)
    if actual_hash != recorded_hash:
        raise CheckpointManifestError(
            "Manifest hash mismatch.\n"
            f"  Expected: {recorded_hash}\n"
            f"  Actual  : {actual_hash}"
        )

    return manifest


def verify_checkpoint_integrity(
    manifest: dict[str, Any], checkpoint_root: str
) -> None:
    """Verify that the manifest checkpoint exists and matches its SHA-256."""
    checkpoint_info = manifest.get("checkpoint")
    if not isinstance(checkpoint_info, dict):
        raise CheckpointManifestError("Manifest is missing checkpoint section")

    rel_path = checkpoint_info.get("path")
    expected_hash = checkpoint_info.get("sha256")
    if not isinstance(rel_path, str) or not rel_path:
        raise CheckpointManifestError("Manifest checkpoint.path must be a string")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise CheckpointManifestError("Manifest checkpoint.sha256 must be a string")

    relative_path = Path(rel_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CheckpointManifestError(
            "Manifest checkpoint.path must be a relative path contained in "
            "checkpoint_root"
        )

    checkpoint_path = Path(checkpoint_root) / relative_path
    if not checkpoint_path.is_file():
        raise CheckpointManifestError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    actual_hash = _sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise CheckpointManifestError(
            "Checkpoint SHA-256 mismatch.\n"
            f"  Path    : {checkpoint_path}\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual  : {actual_hash}"
        )

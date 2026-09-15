#!/usr/bin/env python3
"""Local, single-process evaluator for adapter-bias perturbations.

The evaluator deliberately keeps experiment-specific policy here rather than in
``main.py``.  It reconstructs one source datamodule, holds the requested source
checkpoints resident, and fans each validation batch out to every condition.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from foundry.config_resolvers import register_resolvers
from foundry.data.source_manifest import SourceSelectionManifest
from foundry.training.checkpoint_manifest import load_checkpoint_manifest


LOG = logging.getLogger("adapter_bias_evaluator")
REPO_ROOT = Path(__file__).resolve().parents[1]
STEM = "20260915-MS-adapter-bias-perturbation"
PROJECT = "neurosoft_supervised_pretraining"
MILESTONE_STEPS = (100, 300, 1_000, 3_000, 10_000)
ALL_CONDITIONS = (
    "intact",
    "zero",
    "mean",
    "derangement_0",
    "derangement_1",
    "derangement_2",
    "derangement_3",
    "derangement_4",
)
DEFAULT_ROOTS = {
    "minipigs": Path(
        "/network/scratch/s/sobralm/runs/"
        "PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS"
    ),
    "monkeys": Path(
        "/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS"
    ),
}
REGISTRIES = {
    "minipigs": REPO_ROOT / "launch/phase4e/phase4e-source-minipigs.jsonl",
    "monkeys": REPO_ROOT / "launch/phase4e/phase4e-source-monkeys.jsonl",
}
RUN_PATTERNS = {
    "minipigs": re.compile(r"^src_mp_(sub-\d+)_s(\d+)_m(\d+)$"),
    "monkeys": re.compile(r"^src_mk_(sub-\d+)_s(\d+)_m(\d+)$"),
}
RESULT_FIELDS = (
    "species",
    "source_run_name",
    "excluded_target_subject",
    "source_selection_seed",
    "source_model_seed",
    "source_manifest_path",
    "source_manifest_hash",
    "checkpoint_kind",
    "checkpoint_step",
    "checkpoint_global_step",
    "checkpoint_manifest_path",
    "checkpoint_manifest_hash",
    "checkpoint_path",
    "checkpoint_sha256",
    "condition",
    "derangement_index",
    "derangement_mapping_hash",
    "cross_entropy_sum",
    "target_count",
    "cross_entropy",
    "pooled_supported_f1",
    "recording_mean_supported_f1",
    "recording_count",
    "window_count",
    "batch_count",
    "elapsed_seconds",
    "git_sha",
    "device_name",
    "effective_precision",
    "evaluation_signature",
    "diagnostic_max_batches",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_duration(seconds: float) -> str:
    """Format a duration compactly for progress and ETA logging."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def atomic_write_text(path: Path, text: str) -> None:
    """Replace *path* atomically, retaining an existing file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def deterministic_derangement(
    canonical_ids: Sequence[str], stable_identifier: str, index: int
) -> dict[str, str]:
    """Return a deterministic Sattolo derangement over sorted IDs."""
    ids = sorted(canonical_ids)
    if len(ids) < 2:
        raise ValueError(
            "A derangement requires at least two active recordings"
        )
    seed_bytes = hashlib.sha256(
        f"{stable_identifier}|derangement={index}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:16], "big"))
    permuted = ids.copy()
    for position in range(len(permuted) - 1, 0, -1):
        swap = rng.randrange(position)
        permuted[position], permuted[swap] = (
            permuted[swap],
            permuted[position],
        )
    mapping = dict(zip(ids, permuted))
    if any(source == target for source, target in mapping.items()):
        raise AssertionError("Sattolo generation produced a fixed point")
    return mapping


def generate_derangements(
    canonical_ids: Sequence[str], stable_identifier: str, count: int = 5
) -> list[dict[str, str]]:
    mappings: list[dict[str, str]] = []
    hashes: set[str] = set()
    for index in range(count):
        mapping = deterministic_derangement(
            canonical_ids, stable_identifier, index
        )
        mapping_hash = stable_json_hash(mapping)
        # Distinct cycles are overwhelmingly likely.  Deterministically search
        # farther if a small active set collides.
        retry_index = index
        attempts = 0
        while (
            mapping_hash in hashes and len(canonical_ids) > 2 and attempts < 100
        ):
            retry_index += count
            attempts += 1
            mapping = deterministic_derangement(
                canonical_ids, stable_identifier, retry_index
            )
            mapping_hash = stable_json_hash(mapping)
        mappings.append(mapping)
        hashes.add(mapping_hash)
    return mappings


def active_biases(
    model: torch.nn.Module, canonical_ids: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Copy only manifest-active adapter biases and validate the contract."""
    ids = sorted(canonical_ids)
    if len(ids) < 2:
        raise ValueError(
            "At least two active canonical recordings are required"
        )
    layers = model.session_adapter.layers
    missing = [
        recording_id for recording_id in ids if recording_id not in layers
    ]
    if missing:
        raise KeyError(f"Missing active canonical adapter(s): {missing}")
    result = {
        recording_id: layers[recording_id].bias.detach().clone()
        for recording_id in ids
    }
    invalid = {
        recording_id: tuple(bias.shape)
        for recording_id, bias in result.items()
        if bias.ndim != 1 or bias.numel() != 64
    }
    if invalid:
        raise ValueError(
            f"Active adapter biases must be 64-dimensional: {invalid}"
        )
    return result


def condition_biases(
    learned: Mapping[str, torch.Tensor],
    condition: str,
    derangements: Sequence[Mapping[str, str]],
) -> dict[str, torch.Tensor]:
    ids = sorted(learned)
    if condition == "intact":
        return {recording_id: learned[recording_id] for recording_id in ids}
    if condition == "zero":
        return {
            recording_id: torch.zeros_like(learned[recording_id])
            for recording_id in ids
        }
    if condition == "mean":
        mean = torch.stack(
            [learned[recording_id] for recording_id in ids]
        ).mean(0)
        return {recording_id: mean for recording_id in ids}
    if condition.startswith("derangement_"):
        index = int(condition.rsplit("_", 1)[1])
        mapping = derangements[index]
        if set(mapping) != set(ids) or set(mapping.values()) != set(ids):
            raise ValueError("Derangement does not exactly cover active IDs")
        if any(source == target for source, target in mapping.items()):
            raise ValueError("Derangement contains a fixed point")
        return {source: learned[target] for source, target in mapping.items()}
    raise ValueError(f"Unknown condition: {condition}")


def apply_bias_condition(
    model: torch.nn.Module,
    learned: Mapping[str, torch.Tensor],
    condition: str,
    derangements: Sequence[Mapping[str, str]],
) -> None:
    """Explicitly overwrite every active bias for a condition."""
    desired = condition_biases(learned, condition, derangements)
    with torch.no_grad():
        for recording_id in sorted(learned):
            model.session_adapter.layers[recording_id].bias.copy_(
                desired[recording_id]
            )


def confusion_matrix(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> np.ndarray:
    flat = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return (
        torch.bincount(flat, minlength=num_classes * num_classes)
        .reshape(num_classes, num_classes)
        .cpu()
        .numpy()
    )


def supported_macro_f1(matrix: np.ndarray) -> float:
    """Macro-F1 over target-supported classes only."""
    support = matrix.sum(axis=1)
    mask = support > 0
    if not np.any(mask):
        return float("nan")
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = support - tp
    denominator = 2 * tp[mask] + fp[mask] + fn[mask]
    f1 = 2 * tp[mask] / denominator
    return float(f1.mean())


class MetricAccumulator:
    """Sufficient statistics for exact CE and pooled/per-recording F1."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.cross_entropy_sum = 0.0
        self.target_count = 0
        self.pooled = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.by_recording: dict[str, np.ndarray] = {}
        self.window_count = 0
        self.batch_count = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        counts: Sequence[int],
        recording_ids: Sequence[str],
    ) -> None:
        if sum(counts) != len(targets) or len(counts) != len(recording_ids):
            raise ValueError("Task routing counts do not align with targets")
        self.batch_count += 1
        self.window_count += len(recording_ids)
        offset = 0
        for count, recording_id in zip(counts, recording_ids):
            item_logits = logits[offset : offset + count]
            item_targets = targets[offset : offset + count]
            offset += count
            valid = item_targets >= 0
            item_logits = item_logits[valid]
            item_targets = item_targets[valid].long()
            if item_targets.numel() == 0:
                continue
            self.cross_entropy_sum += float(
                F.cross_entropy(
                    item_logits.float(), item_targets, reduction="sum"
                ).item()
            )
            self.target_count += int(item_targets.numel())
            predictions = item_logits.argmax(dim=-1)
            matrix = confusion_matrix(
                predictions, item_targets, self.num_classes
            )
            self.pooled += matrix
            self.by_recording.setdefault(
                str(recording_id),
                np.zeros_like(self.pooled),
            )
            self.by_recording[str(recording_id)] += matrix

    def summary(self) -> dict[str, Any]:
        if self.target_count == 0 or not self.by_recording:
            raise RuntimeError(
                "No valid labeled validation targets accumulated"
            )
        per_recording = [
            supported_macro_f1(matrix) for matrix in self.by_recording.values()
        ]
        return {
            "cross_entropy_sum": self.cross_entropy_sum,
            "target_count": self.target_count,
            "cross_entropy": self.cross_entropy_sum / self.target_count,
            "pooled_supported_f1": supported_macro_f1(self.pooled),
            "recording_mean_supported_f1": float(np.mean(per_recording)),
            "recording_count": len(per_recording),
            "window_count": self.window_count,
            "batch_count": self.batch_count,
        }


def _expected_run_names(species: str) -> set[str]:
    prefix = "src_mp" if species == "minipigs" else "src_mk"
    names: set[str] = set()
    with REGISTRIES[species].open(encoding="utf-8") as stream:
        for line in stream:
            cell = json.loads(line)
            names.add(
                f"{prefix}_{cell['target_subject']}_"
                f"s{cell['source_selection_seed']}_m{cell['source_model_seed']}"
            )
    return names


def infer_species(root: Path) -> str:
    upper = root.name.upper()
    if "MINIPIG" in upper:
        return "minipigs"
    if "MONKEY" in upper:
        return "monkeys"
    raise ValueError(f"Cannot infer species from source root: {root}")


def resolve_checkpoint_path(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    name = Path(str(manifest["checkpoint"]["path"])).name
    path = run_dir / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def discover_checkpoints(
    run_dir: Path, selected: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Validate and return five milestone manifests and one best manifest."""
    manifest_dir = run_dir / "manifests"
    paths = sorted(manifest_dir.glob("milestone-*.json"))
    best_paths = sorted(manifest_dir.glob("best-*.json"))
    if len(best_paths) != 1:
        raise RuntimeError(
            f"{run_dir.name}: expected one best manifest, found {len(best_paths)}"
        )
    records: list[dict[str, Any]] = []
    milestone_steps: set[int] = set()
    for path in paths + best_paths:
        manifest = load_checkpoint_manifest(path)
        checkpoint = resolve_checkpoint_path(run_dir, manifest)
        actual_sha = sha256_file(checkpoint)
        expected_sha = manifest["checkpoint"]["sha256"]
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch for {checkpoint}: "
                f"{actual_sha} != {expected_sha}"
            )
        is_best = path.name.startswith("best-")
        global_step = manifest.get("trained_on", {}).get("optimizer_steps")
        step = None if is_best else int(global_step)
        if step is not None:
            milestone_steps.add(step)
        records.append(
            {
                "kind": "best" if is_best else "milestone",
                "step": step,
                "global_step": global_step,
                "manifest_path": path,
                "manifest_hash": manifest["manifest_hash"],
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": expected_sha,
                "manifest": manifest,
            }
        )
    if milestone_steps != set(MILESTONE_STEPS) or len(paths) != 5:
        raise RuntimeError(
            f"{run_dir.name}: expected milestones {list(MILESTONE_STEPS)}, "
            f"found {sorted(milestone_steps)}"
        )
    records.sort(key=lambda item: (item["kind"] == "best", item["step"] or 0))
    if selected is None:
        return records
    wanted = set(selected)
    return [
        record
        for record in records
        if (record["kind"] == "best" and "best" in wanted)
        or str(record["step"]) in wanted
    ]


def discover_source_runs(
    roots: Sequence[Path], species_choice: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_species: set[str] = set()
    for root in roots:
        species = infer_species(root)
        if species_choice != "all" and species != species_choice:
            continue
        seen_species.add(species)
        expected = _expected_run_names(species)
        actual = {
            path.name
            for path in root.iterdir()
            if path.is_dir() and RUN_PATTERNS[species].match(path.name)
        }
        if actual != expected:
            raise RuntimeError(
                f"{species}: expected {len(expected)} source runs but found "
                f"{len(actual)}; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for name in sorted(actual):
            match = RUN_PATTERNS[species].match(name)
            assert match is not None
            records.append(
                {
                    "species": species,
                    "run_dir": root / name,
                    "source_run_name": name,
                    "excluded_target_subject": match.group(1),
                    "source_selection_seed": int(match.group(2)),
                    "source_model_seed": int(match.group(3)),
                }
            )
    required = (
        {species_choice} if species_choice != "all" else set(DEFAULT_ROOTS)
    )
    if seen_species != required:
        raise RuntimeError(
            f"Missing source root(s) for {sorted(required - seen_species)}"
        )
    return sorted(
        records, key=lambda item: (item["species"], item["source_run_name"])
    )


def source_manifest_path(cfg: Any) -> Path:
    configured = Path(str(OmegaConf.select(cfg, "source_manifest")))
    return configured if configured.is_absolute() else REPO_ROOT / configured


def validate_source_identity(
    record: Mapping[str, Any], checkpoints: Sequence[Mapping[str, Any]]
) -> tuple[Path, SourceSelectionManifest]:
    cfg = OmegaConf.load(record["run_dir"] / ".hydra/config.yaml")
    path = source_manifest_path(cfg)
    source = SourceSelectionManifest.load(path)
    for checkpoint in checkpoints:
        trained = checkpoint["manifest"].get("trained_on", {})
        if trained.get("source_manifest_hash") != source.manifest_hash:
            raise RuntimeError(
                f"{record['source_run_name']}: source manifest identity mismatch"
            )
    return path, source


def source_complete(
    existing_rows: Sequence[Mapping[str, Any]],
    source_run_name: str,
    checkpoints: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    evaluation_signature: str,
) -> bool:
    rows = [
        row
        for row in existing_rows
        if row.get("source_run_name") == source_run_name
    ]
    expected = {
        (
            checkpoint["checkpoint_sha256"],
            checkpoint["manifest_hash"],
            condition,
        )
        for checkpoint in checkpoints
        for condition in conditions
    }
    actual = {
        (
            row.get("checkpoint_sha256"),
            row.get("checkpoint_manifest_hash"),
            row.get("condition"),
        )
        for row in rows
        if row.get("evaluation_signature") == evaluation_signature
    }
    return len(rows) == len(expected) and actual == expected


def replace_source_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    source_run_name: str,
    new_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    retained = [
        dict(row)
        for row in existing_rows
        if row.get("source_run_name") != source_run_name
    ]
    return retained + [dict(row) for row in new_rows]


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "sha": run("git", "rev-parse", "HEAD") or None,
        "status_short": run("git", "status", "--short").splitlines(),
    }


def _parse_csv_option(value: str, allowed: Iterable[str]) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(result) - set(allowed)
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            f"invalid values {sorted(invalid)}; allowed: {sorted(allowed)}"
        )
    return result


def _parse_seed_option(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "source model seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError(
            "at least one source model seed is required"
        )
    return seeds


def _prepare_normalization_cache(run_dir: Path, output_dir: Path) -> Path:
    manifest_path = run_dir / "input_normalization_manifest.json"
    stats_path = run_dir / "input_normalization_stats.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_key = manifest.get("provenance", {}).get("cache_key")
    if not cache_key:
        raise RuntimeError(f"Missing normalization cache key: {manifest_path}")
    destination = output_dir / ".normalization_cache" / cache_key
    destination.mkdir(parents=True, exist_ok=True)
    for source in (manifest_path, stats_path):
        target = destination / source.name
        if not target.exists() or sha256_file(target) != sha256_file(source):
            shutil.copy2(source, target)
    return destination.parent


def _validate_normalization_identity(
    run_dir: Path, checkpoints: Sequence[Mapping[str, Any]]
) -> None:
    stats_path = run_dir / "input_normalization_stats.npz"
    stats_sha = sha256_file(stats_path)
    identities = []
    for checkpoint in checkpoints:
        identity = checkpoint["manifest"].get(
            "normalization_artifact_hashes", {}
        )
        if identity.get("stats_sha256") != stats_sha:
            raise RuntimeError(
                f"Normalization stats identity mismatch for "
                f"{checkpoint['manifest_path']}"
            )
        identities.append(
            (
                identity.get("stats_sha256"),
                identity.get("train_interval_hash"),
                identity.get("cache_key"),
            )
        )
    if len(set(identities)) != 1:
        raise RuntimeError("Checkpoints disagree on normalization identity")


def _build_models_and_data(
    record: Mapping[str, Any],
    source: SourceSelectionManifest,
    checkpoints: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[torch.nn.Module], Any, str, list[str], int]:
    import main as production_main

    run_dir = Path(record["run_dir"])
    _validate_normalization_identity(run_dir, checkpoints)
    payloads = [
        torch.load(
            checkpoint["checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        for checkpoint in checkpoints
    ]
    adapter_prefix = "model.session_adapter.layers."
    adapter_suffix = ".weight"

    def checkpoint_session_configs(
        payload: Mapping[str, Any],
    ) -> dict[str, int]:
        configs = {}
        for key, tensor in payload["state_dict"].items():
            if key.startswith(adapter_prefix) and key.endswith(adapter_suffix):
                recording_id = key[len(adapter_prefix) : -len(adapter_suffix)]
                configs[recording_id] = int(tensor.shape[1])
        return configs

    session_configs = checkpoint_session_configs(payloads[0])
    if not session_configs:
        raise RuntimeError("Checkpoint contains no session-adapter weights")
    for payload in payloads[1:]:
        if checkpoint_session_configs(payload) != session_configs:
            raise RuntimeError(
                "Requested checkpoints have different adapter ID sets"
            )
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    OmegaConf.update(cfg, "source_manifest", str(source_manifest_path(cfg)))
    if args.data_root:
        OmegaConf.update(cfg, "data.root", str(args.data_root))
    OmegaConf.update(cfg, "data.batch_size", args.batch_size)
    OmegaConf.update(cfg, "hyperparameters.batch_size", args.batch_size)
    OmegaConf.update(cfg, "data.num_workers", args.num_workers)
    OmegaConf.update(cfg, "hyperparameters.num_workers", args.num_workers)
    OmegaConf.update(cfg, "data.pin_memory", device.type == "cuda")
    cache_root = _prepare_normalization_cache(run_dir, args.output_dir)
    OmegaConf.update(
        cfg, "data.input_normalization.cache.directory", str(cache_root)
    )
    canonical_session_configs = {
        recording.canonical_recording_id: recording.supported_channel_count
        for recording in source.recordings
    }
    for recording_id, channel_count in canonical_session_configs.items():
        if session_configs.get(recording_id) != channel_count:
            raise RuntimeError(
                "Checkpoint adapter shape disagrees with source manifest for "
                f"{recording_id}"
            )
    OmegaConf.update(
        cfg, "hyperparameters.session_configs", session_configs, force_add=True
    )
    OmegaConf.update(
        cfg,
        "hyperparameters.num_channels",
        max(session_configs.values()),
        force_add=True,
    )
    model, datamodule = production_main._build_source_model_and_data(cfg)
    template = production_main._build_lightning_module(cfg, model, datamodule)
    models: list[torch.nn.Module] = []
    canonical_ids = sorted(datamodule.source_canonical_recording_ids or [])
    if canonical_ids != sorted(canonical_session_configs):
        raise RuntimeError(
            "Datamodule active canonical IDs differ from source manifest"
        )
    task_names = [
        name
        for name, task_cfg in model.task_configs.items()
        if task_cfg.kind in ("multiclass", "binary")
    ]
    if len(task_names) != 1:
        raise RuntimeError(
            f"Expected one classification task, found {task_names}"
        )
    task_name = task_names[0]
    num_classes = model.task_configs[task_name].output_dim
    for checkpoint, payload in zip(checkpoints, payloads):
        module = copy.deepcopy(template)
        module.load_state_dict(payload["state_dict"], strict=True)
        if payload.get("global_step") != checkpoint["global_step"]:
            raise RuntimeError(
                "Checkpoint global_step disagrees with manifest: "
                f"{checkpoint['checkpoint_path']}"
            )
        checkpoint["loaded_global_step"] = payload.get("global_step")
        module.model.to(device).eval()
        module.model.requires_grad_(False)
        active_biases(module.model, canonical_ids)
        models.append(module.model)
    return models, datamodule, task_name, canonical_ids, num_classes


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.float64:
            value = value.float()
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def evaluate_source_run(
    record: Mapping[str, Any],
    source_path: Path,
    source: SourceSelectionManifest,
    checkpoints: list[dict[str, Any]],
    conditions: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    effective_precision: str,
    evaluation_signature: str,
    git_sha: str | None,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, float]]:
    start = time.perf_counter()
    models, datamodule, task_name, canonical_ids, num_classes = (
        _build_models_and_data(record, source, checkpoints, args, device)
    )
    stable_identifier = (
        f"species={record['species']}|excluded={record['excluded_target_subject']}|"
        f"selection={record['source_selection_seed']}|"
        f"model={record['source_model_seed']}"
    )
    derangements = generate_derangements(canonical_ids, stable_identifier)
    mapping_hashes = {
        f"derangement_{index}": stable_json_hash(mapping)
        for index, mapping in enumerate(derangements)
    }
    learned_by_checkpoint = [
        active_biases(model, canonical_ids) for model in models
    ]
    accumulators = {
        (checkpoint_index, condition): MetricAccumulator(num_classes)
        for checkpoint_index in range(len(checkpoints))
        for condition in conditions
    }
    elapsed_by_key = {key: 0.0 for key in accumulators}
    autocast_enabled = device.type == "cuda" and effective_precision == "fp16"
    dataloader = datamodule.val_dataloader()
    available_batches = len(dataloader)
    planned_batches = (
        min(available_batches, args.max_batches)
        if args.max_batches is not None
        else available_batches
    )
    setup_seconds = time.perf_counter() - start
    logical_per_batch = len(checkpoints) * len(conditions)
    LOG.info(
        "%s prepared in %s: %d validation batches, %d checkpoint/condition "
        "evaluations per batch (%d total batch-evaluations)",
        record["source_run_name"],
        format_duration(setup_seconds),
        planned_batches,
        logical_per_batch,
        planned_batches * logical_per_batch,
    )
    traversal_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(dataloader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = _to_device(cpu_batch, device)
            task_index = batch["task_index"]
            targets = batch["target_values"][task_name]
            recording_ids = list(batch["session_id"])
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key
                not in {
                    "target_values",
                    "target_weights",
                    "session_id",
                    "absolute_start",
                    "eval_mask",
                }
            }
            for checkpoint_index, model in enumerate(models):
                router_index = (
                    model.router.get_task_index_by_name(task_name) + 1
                )
                counts = (task_index == router_index).sum(dim=1).tolist()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                intact_start = time.perf_counter()
                apply_bias_condition(
                    model,
                    learned_by_checkpoint[checkpoint_index],
                    "intact",
                    derangements,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    first_intact = model(**model_inputs)[task_name]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                intact_forward_seconds = time.perf_counter() - intact_start
                for condition in conditions:
                    condition_start = time.perf_counter()
                    apply_bias_condition(
                        model,
                        learned_by_checkpoint[checkpoint_index],
                        condition,
                        derangements,
                    )
                    if condition == "intact":
                        logits = first_intact
                    else:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=autocast_enabled,
                        ):
                            logits = model(**model_inputs)[task_name]
                    accumulators[(checkpoint_index, condition)].update(
                        logits, targets, counts, recording_ids
                    )
                    elapsed_by_key[(checkpoint_index, condition)] += (
                        time.perf_counter() - condition_start
                    )
                    if condition == "intact":
                        elapsed_by_key[(checkpoint_index, condition)] += (
                            intact_forward_seconds
                        )
                if batch_index == 0:
                    apply_bias_condition(
                        model,
                        learned_by_checkpoint[checkpoint_index],
                        "intact",
                        derangements,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=autocast_enabled,
                    ):
                        restored = model(**model_inputs)[task_name]
                    tolerance = 1e-3 if autocast_enabled else 1e-6
                    torch.testing.assert_close(
                        first_intact, restored, rtol=0.0, atol=tolerance
                    )
            completed_batches = batch_index + 1
            if (
                completed_batches == 1
                or completed_batches % args.progress_every_batches == 0
                or completed_batches == planned_batches
            ):
                traversal_elapsed = time.perf_counter() - traversal_start
                seconds_per_batch = traversal_elapsed / completed_batches
                eta_seconds = seconds_per_batch * (
                    planned_batches - completed_batches
                )
                LOG.info(
                    "%s progress: %d/%d batches (%.1f%%), %d/%d logical "
                    "batch-evaluations; elapsed %s, ETA %s",
                    record["source_run_name"],
                    completed_batches,
                    planned_batches,
                    100 * completed_batches / planned_batches,
                    completed_batches * logical_per_batch,
                    planned_batches * logical_per_batch,
                    format_duration(traversal_elapsed),
                    format_duration(eta_seconds),
                )
    rows: list[dict[str, Any]] = []
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU"
    )
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for condition in conditions:
            derangement_index = (
                int(condition.rsplit("_", 1)[1])
                if condition.startswith("derangement_")
                else None
            )
            row = {
                **record,
                "source_manifest_path": str(source_path),
                "source_manifest_hash": source.manifest_hash,
                "checkpoint_kind": checkpoint["kind"],
                "checkpoint_step": checkpoint["step"],
                "checkpoint_global_step": checkpoint.get(
                    "loaded_global_step", checkpoint["global_step"]
                ),
                "checkpoint_manifest_path": str(checkpoint["manifest_path"]),
                "checkpoint_manifest_hash": checkpoint["manifest_hash"],
                "checkpoint_path": str(checkpoint["checkpoint_path"]),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "condition": condition,
                "derangement_index": derangement_index,
                "derangement_mapping_hash": mapping_hashes.get(condition),
                **accumulators[(checkpoint_index, condition)].summary(),
                "elapsed_seconds": elapsed_by_key[
                    (checkpoint_index, condition)
                ],
                "git_sha": git_sha,
                "device_name": device_name,
                "effective_precision": effective_precision,
                "evaluation_signature": evaluation_signature,
                "diagnostic_max_batches": args.max_batches,
            }
            row.pop("run_dir", None)
            rows.append(row)
    if args.max_batches is None:
        for checkpoint in checkpoints:
            if checkpoint["kind"] != "best" or "intact" not in conditions:
                continue
            expected_loss = (
                checkpoint["manifest"].get("selection", {}).get("monitor_value")
            )
            actual_loss = next(
                row["cross_entropy"]
                for row in rows
                if row["checkpoint_kind"] == "best"
                and row["condition"] == "intact"
            )
            if expected_loss is not None and not np.isclose(
                actual_loss, float(expected_loss), rtol=5e-3, atol=1e-2
            ):
                raise RuntimeError(
                    "Intact best-checkpoint validation loss failed the "
                    f"correctness canary: computed={actual_loss:.8f}, "
                    f"manifest={float(expected_loss):.8f}; tolerance is "
                    "rtol=0.005, atol=0.01"
                )
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0.0
    )
    return (
        rows,
        mapping_hashes,
        {
            "elapsed_seconds": time.perf_counter() - start,
            "peak_cuda_memory_bytes": peak_memory,
            "windows": float(next(iter(accumulators.values())).window_count),
        },
    )


def _write_status(
    output_dir: Path,
    completed: Sequence[str],
    planned: int,
    state: str,
) -> None:
    atomic_write_json(
        output_dir / "status.json",
        {
            "state": state,
            "completed_source_runs": list(completed),
            "completed_count": len(completed),
            "planned_source_runs": planned,
            "updated_unix_time": time.time(),
        },
    )


def upload_wandb(
    args: argparse.Namespace,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=PROJECT,
        group="PHASE4E_ADAPTER_BIAS_PERTURBATION",
        name=STEM,
        mode=args.wandb_mode,
        config={key: str(value) for key, value in vars(args).items()},
    )
    table = wandb.Table(columns=list(RESULT_FIELDS))
    for row in rows:
        table.add_data(*(row.get(field) for field in RESULT_FIELDS))
    run.log({"results": table})
    artifact = wandb.Artifact(f"{STEM}-results", type="evaluation")
    for name in ("results.csv", "provenance.json", "status.json", "run.log"):
        path = output_dir / name
        if path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()
    return run.id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        help="Phase 4E species root; repeat for both (defaults to both roots)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=os.environ.get("FOUNDRY_DATA_ROOT")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / STEM,
    )
    parser.add_argument(
        "--species", choices=("minipigs", "monkeys", "all"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=5,
        help="Log within-source progress every N validation batches",
    )
    parser.add_argument("--precision", choices=("fp16", "fp32"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-source-runs", type=int)
    parser.add_argument(
        "--source-model-seeds",
        type=_parse_seed_option,
        help="Comma-separated source model seeds, e.g. 42 or 42,43",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="DIAGNOSTIC ONLY: truncate each validation traversal",
    )
    parser.add_argument("--checkpoints", default="100,300,1000,3000,10000,best")
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    register_resolvers()
    roots = args.source_root or list(DEFAULT_ROOTS.values())
    checkpoints_selected = _parse_csv_option(
        args.checkpoints, [str(step) for step in MILESTONE_STEPS] + ["best"]
    )
    conditions = _parse_csv_option(args.conditions, ALL_CONDITIONS)
    source_runs = discover_source_runs(roots, args.species)
    if args.source_model_seeds is not None:
        selected_seeds = set(args.source_model_seeds)
        source_runs = [
            record
            for record in source_runs
            if record["source_model_seed"] in selected_seeds
        ]
        if not source_runs:
            raise RuntimeError(
                f"No source runs match model seeds {sorted(selected_seeds)}"
            )
    if args.max_source_runs is not None:
        source_runs = source_runs[: args.max_source_runs]
    discovered: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, tuple[Path, SourceSelectionManifest]] = {}
    for record in source_runs:
        all_checkpoints = discover_checkpoints(record["run_dir"])
        _validate_normalization_identity(record["run_dir"], all_checkpoints)
        selected = [
            checkpoint
            for checkpoint in all_checkpoints
            if str(checkpoint["step"]) in checkpoints_selected
            or (checkpoint["kind"] == "best" and "best" in checkpoints_selected)
        ]
        discovered[record["source_run_name"]] = selected
        sources[record["source_run_name"]] = validate_source_identity(
            record, all_checkpoints
        )
    logical = sum(len(value) * len(conditions) for value in discovered.values())
    print(
        f"Discovered {len(source_runs)} source runs, "
        f"{sum(len(v) for v in discovered.values())} selected checkpoints, "
        f"{logical} logical evaluations."
    )
    if args.dry_run:
        counts: dict[str, int] = {}
        for record in source_runs:
            counts[record["species"]] = counts.get(record["species"], 0) + 1
        print(f"Source counts: {json.dumps(counts, sort_keys=True)}")
        print(
            "Dry run complete; no model/data evaluation or output writes performed."
        )
        return 0

    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.progress_every_batches < 1
    ):
        raise ValueError(
            "batch-size and progress-every-batches must be positive, and "
            "num-workers must be non-negative"
        )
    device_name = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    effective_precision = args.precision or (
        "fp16" if device.type == "cuda" else "fp32"
    )
    if device.type == "cpu" and effective_precision == "fp16":
        raise ValueError("fp16 evaluation is only supported on CUDA")
    args.output_dir = args.output_dir.resolve()
    results_path = args.output_dir / "results.csv"
    if args.overwrite:
        existing_rows: list[dict[str, str]] = []
    else:
        existing_rows = read_results(results_path)
    git = git_info()
    signature_payload = {
        "tool_sha256": sha256_file(Path(__file__)),
        "git_sha": git["sha"],
        "device_type": device.type,
        "precision": effective_precision,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "conditions": conditions,
        "checkpoints": checkpoints_selected,
        "sources": {
            name: [
                (item["manifest_hash"], item["checkpoint_sha256"])
                for item in values
            ]
            for name, values in discovered.items()
        },
    }
    evaluation_signature = stable_json_hash(signature_payload)
    incompatible = {
        row.get("evaluation_signature")
        for row in existing_rows
        if row.get("evaluation_signature") != evaluation_signature
    }
    if incompatible:
        raise RuntimeError(
            "Existing results have incompatible provenance; use a different "
            "output directory or --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        atomic_write_csv(results_path, [])
    log_path = args.output_dir / "run.log"
    handler = logging.FileHandler(
        log_path, mode="w" if args.overwrite else "a", encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), handler],
        force=True,
    )
    old_provenance_path = args.output_dir / "provenance.json"
    old_provenance = (
        json.loads(old_provenance_path.read_text(encoding="utf-8"))
        if old_provenance_path.exists() and not args.overwrite
        else {}
    )
    provenance = {
        "experiment": STEM,
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "source_roots": [str(root) for root in roots],
        "git": git,
        "device": device_name,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU",
        "effective_precision": effective_precision,
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "evaluation_signature": evaluation_signature,
        "signature_payload": signature_payload,
        "derangement_mapping_hashes": old_provenance.get(
            "derangement_mapping_hashes", {}
        ),
    }
    atomic_write_json(args.output_dir / "provenance.json", provenance)
    completed: list[str] = []
    _write_status(args.output_dir, completed, len(source_runs), "running")
    benchmarks: list[dict[str, Any]] = []
    complete_at_start = {
        record["source_run_name"]
        for record in source_runs
        if source_complete(
            existing_rows,
            record["source_run_name"],
            discovered[record["source_run_name"]],
            conditions,
            evaluation_signature,
        )
    }
    pending_names = {
        record["source_run_name"]
        for record in source_runs
        if record["source_run_name"] not in complete_at_start
    }
    pending_work = {
        record["source_run_name"]: (
            sum(
                recording.available_validation_windows
                for recording in sources[record["source_run_name"]][
                    1
                ].recordings
            )
            * len(discovered[record["source_run_name"]])
            * len(conditions)
        )
        for record in source_runs
        if record["source_run_name"] in pending_names
    }
    pending_start = time.perf_counter()
    completed_pending_work = 0
    LOG.info(
        "Resume plan: %d/%d source runs already complete; %d pending",
        len(complete_at_start),
        len(source_runs),
        len(pending_names),
    )
    for index, record in enumerate(source_runs, start=1):
        name = record["source_run_name"]
        checkpoints = discovered[name]
        if source_complete(
            existing_rows,
            name,
            checkpoints,
            conditions,
            evaluation_signature,
        ):
            LOG.info("Skipping complete source run %s", name)
            completed.append(name)
            continue
        LOG.info(
            "Evaluating source run %d/%d: %s", index, len(source_runs), name
        )
        source_path, source = sources[name]
        try:
            rows, mapping_hashes, benchmark = evaluate_source_run(
                record,
                source_path,
                source,
                checkpoints,
                conditions,
                args,
                device,
                effective_precision,
                evaluation_signature,
                git["sha"],
            )
        except BaseException:
            _write_status(
                args.output_dir, completed, len(source_runs), "failed"
            )
            LOG.exception("Source run failed: %s", name)
            raise
        existing_rows = replace_source_rows(existing_rows, name, rows)
        atomic_write_csv(results_path, existing_rows)
        provenance["derangement_mapping_hashes"][name] = mapping_hashes
        atomic_write_json(args.output_dir / "provenance.json", provenance)
        completed.append(name)
        _write_status(args.output_dir, completed, len(source_runs), "running")
        windows_per_second = benchmark["windows"] / benchmark["elapsed_seconds"]
        projected = (
            benchmark["elapsed_seconds"] * 36
            if args.max_batches is None
            else None
        )
        benchmark.update(
            {
                "source_run_name": name,
                "windows_per_second": windows_per_second,
                "projected_full_matrix_seconds": projected,
            }
        )
        benchmarks.append(benchmark)
        completed_pending_work += pending_work[name]
        remaining_work = sum(pending_work.values()) - completed_pending_work
        pending_elapsed = time.perf_counter() - pending_start
        overall_eta = (
            pending_elapsed * remaining_work / completed_pending_work
            if completed_pending_work
            else 0.0
        )
        LOG.info(
            "Overall progress: %d/%d source runs complete (%d/%d newly "
            "evaluated); elapsed this launch %s, rolling ETA %s",
            len(completed),
            len(source_runs),
            len(benchmarks),
            len(pending_names),
            format_duration(pending_elapsed),
            format_duration(overall_eta),
        )
        if projected is None:
            LOG.info(
                "Diagnostic %s: %.2fs, %.2f windows/s over %d truncated "
                "batch(es), peak CUDA %.3f GiB; no full-runtime projection",
                name,
                benchmark["elapsed_seconds"],
                windows_per_second,
                args.max_batches,
                benchmark["peak_cuda_memory_bytes"] / 2**30,
            )
        else:
            LOG.info(
                "Benchmark %s: %.2fs, %.2f windows/s, projected 36-run "
                "%.2fh, peak CUDA %.3f GiB",
                name,
                benchmark["elapsed_seconds"],
                windows_per_second,
                projected / 3600,
                benchmark["peak_cuda_memory_bytes"] / 2**30,
            )
    _write_status(args.output_dir, completed, len(source_runs), "complete")
    provenance["benchmarks"] = benchmarks
    provenance["wandb_run_id"] = upload_wandb(
        args, args.output_dir, existing_rows
    )
    atomic_write_json(args.output_dir / "provenance.json", provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

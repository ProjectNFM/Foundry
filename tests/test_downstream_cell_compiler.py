"""Focused tests for checkpoint-to-downstream cell fan-out."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import main
from foundry.training.checkpoint_manifest import write_checkpoint_manifest
from hydra_plugins.foundry_launcher.packed_launcher import _expand_cell_list
from hydra_plugins.foundry_launcher.launch_snapshot import prepare_snapshot
from tools.compile_downstream_cells import compile_cells, write_outputs


REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4A_REGISTRY = REPO_ROOT / "launch/checkpoint_sets/phase4a-mila-best.jsonl"
PHASE4A_RECIPE = (
    REPO_ROOT / "configs/downstream_recipes/phase4a_full_finetuning.yaml"
)
PHASE4A_AUDIT = REPO_ROOT / "docs/neurosoft-phase0-audit.json"
CHECKPOINT_ROOT = Path("/network/scratch/s/sobralm/foundry-checkpoints")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint_root = tmp_path / "checkpoint-root"
    checkpoint = checkpoint_root / "source" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest_dir = tmp_path / "source-run" / "manifests"
    manifest_path, _ = write_checkpoint_manifest(
        checkpoint,
        manifest_dir,
        kind="best",
        trained_on={
            "source_selection_id": "volume_minipigs_target-sub-01_f1.00_sel42",
            "source_selection_seed": 42,
            "source_model_seed": 43,
            "excluded_target": {"species": "minipigs", "subject": "sub-01"},
        },
        selection={"monitor": "val/f1", "monitor_value": 0.5},
        compute={},
        recipe={},
        normalization_artifact_hashes={},
        git_sha="test",
        snapshot_bundle="test",
        slurm_job_id="test",
        wandb_info={"project": "test", "group": "test", "run_id": "abcdefgh"},
        checkpoint_relative_path="source/best.ckpt",
    )
    manifest = json.loads(manifest_path.read_text())
    registry_record = {
        "checkpoint_set_id": "synthetic-set",
        "checkpoint_id": "synthetic-minipigs-sub-01-sel42-model43-best",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_hash": manifest["manifest_hash"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "species": "minipigs",
        "excluded_target_subject": "sub-01",
        "source_selection_seed": 42,
        "source_model_seed": 43,
        "condition": {
            "source_mixture": "synthetic",
            "source_fraction": 0.5,
            "milestone": "validation_selected",
            "checkpoint_kind": "best",
            "label": "synthetic-best",
        },
    }
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps(registry_record) + "\n")
    audit_payload = {
        "recordings": [
            {
                "recording_id": "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw",
                "subject": "sub-01",
                "species": "minipigs",
                "eligible": True,
                "fraction_availability": {"0.50": {"available": True}},
            }
        ]
    }
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {**audit_payload, "artifact_sha256": _canonical_hash(audit_payload)}
        )
    )
    recipe_payload = {
        "schema": "foundry-downstream-recipe",
        "version": 1,
        "recipe_id": "synthetic",
        "audit_path": str(audit),
        "eligible_session_selection": {"eligible": True},
        "base_experiments": {"minipigs": "auditory_decoding/example"},
        "transfer_regimes": ["full_finetuning", "frozen_representation"],
        "target_training_fractions": [0.5],
        "target_finetuning_seeds": [7, 8],
        "evaluate_test": True,
        "species": {
            "minipigs": {
                "wandb_group": "SYNTHETIC",
                "wandb_tags": ["test"],
                "expected_checkpoints": 1,
                "expected_cells": 4,
            }
        },
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(recipe_payload, sort_keys=False))
    return registry, recipe, audit, checkpoint_root


def test_synthetic_end_to_end_exact_compilation(tmp_path: Path) -> None:
    registry, recipe, audit, root = _fixture(tmp_path)
    cells, metadata = compile_cells(registry, recipe, audit, root)
    assert metadata["counts"] == {"minipigs": 4}
    assert len(cells["minipigs"]) == 4
    first = cells["minipigs"][0]
    assert (
        first["checkpoint_id"] == "synthetic-minipigs-sub-01-sel42-model43-best"
    )
    assert (
        "data.dataset_kwargs.recording_ids=[sub-01_ses-01_task-AcousStim_acq-LH_desc-raw]"
        in first["overrides"]
    )
    assert "run.evaluate_test=true" in first["overrides"]
    expanded = _expand_cell_list(
        [["experiment=auditory_decoding/example"]],
        write_outputs(tmp_path / "out", cells, metadata)[0],
    )
    assert len(expanded) == 4
    assert all(
        vector[0] == "experiment=auditory_decoding/example"
        for vector in expanded
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('snapshot')\n")
    (project / "pyproject.toml").write_text(
        "[project]\nname='test'\nversion='0'\n"
    )
    for directory in ("foundry", "hydra_plugins", "configs"):
        path = project / directory
        path.mkdir()
        (path / ".keep").write_text("")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=project, check=True
    )
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=project, check=True)
    snapshot = prepare_snapshot(
        project_root=project,
        snapshot_root=tmp_path / "snapshots",
        sweep_name="synthetic",
        job_overrides=expanded,
        hydra_cfg=OmegaConf.create({"run": {"group": "SYNTHETIC"}}),
    )
    task_configs = Path(snapshot.bundle_dir) / "task-configs"
    assert len(list(task_configs.glob("task_*.json"))) == 4


@pytest.mark.parametrize("field", ["manifest_hash", "checkpoint_sha256"])
def test_registry_hash_disagreement_is_rejected(
    tmp_path: Path, field: str
) -> None:
    registry, recipe, audit, root = _fixture(tmp_path)
    record = json.loads(registry.read_text())
    record[field] = "0" * 64
    registry.write_text(json.dumps(record) + "\n")
    with pytest.raises((ValueError, RuntimeError), match="disagrees"):
        compile_cells(registry, recipe, audit, root)


def test_duplicate_scientific_checkpoint_identity_is_rejected(
    tmp_path: Path,
) -> None:
    registry, recipe, audit, root = _fixture(tmp_path)
    first = json.loads(registry.read_text())
    checkpoint = root / "source" / "duplicate.ckpt"
    checkpoint.write_bytes(b"different checkpoint content")
    manifest_dir = tmp_path / "duplicate-source-run" / "manifests"
    manifest_path, _ = write_checkpoint_manifest(
        checkpoint,
        manifest_dir,
        kind="best",
        trained_on={
            "source_selection_id": "volume_minipigs_target-sub-01_f1.00_sel42",
            "source_selection_seed": 42,
            "source_model_seed": 43,
            "excluded_target": {"species": "minipigs", "subject": "sub-01"},
        },
        selection={"monitor": "val/f1", "monitor_value": 0.6},
        compute={},
        recipe={},
        normalization_artifact_hashes={},
        git_sha="test",
        snapshot_bundle="test",
        slurm_job_id="test",
        wandb_info={"project": "test", "group": "test", "run_id": "ijklmnop"},
        checkpoint_relative_path="source/duplicate.ckpt",
    )
    manifest = json.loads(manifest_path.read_text())
    duplicate = {
        **first,
        "checkpoint_id": "different-id-same-scientific-cell",
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
    }
    registry.write_text(json.dumps(first) + "\n" + json.dumps(duplicate) + "\n")
    with pytest.raises(
        ValueError, match="Duplicate scientific checkpoint identity"
    ):
        compile_cells(registry, recipe, audit, root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("excluded_target_subject", "sub-02"), ("species", "monkeys")],
)
def test_registry_manifest_species_subject_mismatch_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    registry, recipe, audit, root = _fixture(tmp_path)
    record = json.loads(registry.read_text())
    record[field] = value
    registry.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=field):
        compile_cells(registry, recipe, audit, root)


def test_output_and_lock_are_deterministic(tmp_path: Path) -> None:
    registry, recipe, audit, root = _fixture(tmp_path)
    cells, metadata = compile_cells(registry, recipe, audit, root)
    first = write_outputs(tmp_path / "first", cells, metadata)[0]
    second = write_outputs(tmp_path / "second", cells, metadata)[0]
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(".lock.json").read_bytes()
        == second.with_suffix(".lock.json").read_bytes()
    )


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(), reason="Mila checkpoint storage unavailable"
)
def test_actual_phase4a_registry_compiles_exact_matrix() -> None:
    cells, metadata = compile_cells(
        PHASE4A_REGISTRY, PHASE4A_RECIPE, PHASE4A_AUDIT, CHECKPOINT_ROOT
    )
    assert metadata["counts"] == {"minipigs": 360, "monkeys": 117}
    rows = cells["minipigs"] + cells["monkeys"]
    assert len(rows) == 477
    assert len({row["cell_id"] for row in rows}) == 477
    assert len({row["run_name"] for row in rows}) == 477
    assert len({(row["wandb_group"], row["run_name"]) for row in rows}) == 477
    assert (
        len({(row["wandb_group"], row["wandb_run_id"]) for row in rows}) == 477
    )


@pytest.mark.skipif(
    not CHECKPOINT_ROOT.is_dir(), reason="Mila checkpoint storage unavailable"
)
@pytest.mark.parametrize("species", ["minipigs", "monkeys"])
def test_phase4a_cell_resolves_transfer_config(
    tmp_path: Path, species: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells, _ = compile_cells(
        PHASE4A_REGISTRY, PHASE4A_RECIPE, PHASE4A_AUDIT, CHECKPOINT_ROOT
    )
    row = cells[species][0]
    monkeypatch.setenv("FOUNDRY_DATA_ROOT", "/shared/processed")
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(REPO_ROOT / "configs"), version_base=None
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"experiment=auditory_decoding/neurosoft_conv_bigru_transfer_{species}",
                    *row["overrides"],
                ],
            )
        OmegaConf.resolve(cfg)
        assert cfg.run.name == row["cell_id"]
        assert cfg.run.checkpoint_id == row["checkpoint_id"]
        assert cfg.run.resume_wandb_if_name_matches is True
        assert cfg.run.unsupported_bf16_fallback == "16-mixed"
        assert str(cfg.data.root) == "/shared/processed"
        assert cfg.data.training_fraction_seed == row["target_finetuning_seed"]
    finally:
        GlobalHydra.instance().clear()


def _resume_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "run": {
                "cell_id": "cell-a",
                "name": "cell-a",
                "group": "group-a",
                "tags": ["transfer", "test"],
                "checkpoint_set_id": "set-a",
                "checkpoint_id": "checkpoint-a",
                "source_selection_seed": 42,
                "source_model_seed": 43,
                "source_condition": "fullpool",
                "target_species": "minipigs",
                "target_subject": "sub-01",
                "seed": 44,
                "pretrained_checkpoint_manifest": "/manifest.json",
                "pretrained_checkpoint_manifest_hash": "a" * 64,
                "pretrained_checkpoint_sha256": "b" * 64,
                "pretrained_transfer_regime": "full_finetuning",
            },
            "data": {
                "training_fraction": 1.0,
                "dataset_kwargs": {"recording_ids": ["sub-01_ses-01"]},
            },
            "logger": {"project": "project-a", "id": "wandbid1"},
        }
    )


def test_new_run_validates_manifest_and_resume_skips_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _resume_cfg()
    calls: list[str] = []
    monkeypatch.setattr(
        main,
        "_load_and_validate_checkpoint_manifest",
        lambda cfg, dm: calls.append("load") or {"manifest_hash": "a" * 64},
    )
    manifest = main._prepare_manifest_transfer_for_run(
        cfg, object(), str(tmp_path), None
    )
    assert manifest is not None and calls == ["load"]
    resumed = main._prepare_manifest_transfer_for_run(
        cfg, object(), str(tmp_path), str(tmp_path / "last.ckpt")
    )
    assert resumed is None and calls == ["load"]


def test_resume_refuses_different_compiled_cell(tmp_path: Path) -> None:
    cfg = _resume_cfg()
    main._write_or_validate_cell_provenance(cfg, str(tmp_path), resume=False)
    changed = copy.deepcopy(cfg)
    changed.run.seed = 99
    with pytest.raises(RuntimeError, match="different compiled cell"):
        main._write_or_validate_cell_provenance(
            changed, str(tmp_path), resume=True
        )

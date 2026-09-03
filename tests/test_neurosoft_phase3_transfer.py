"""WP5 gate tests: strict manifest-based transfer for NeuroSoft Phase 3.

Covers:
- Source adapter keys are excluded in both full_finetuning and frozen_representation.
- Target adapter parameters are bitwise unchanged after loading.
- Full finetuning loads frontend/GRU/router and all target parameters are trainable.
- Frozen representation loads/freezes frontend/GRU, leaves router fresh, and
  trains only the router plus target adapter.
- Tampered checkpoint SHA-256 or manifest hash fails atomically.
- Mismatched target species or subject fails before any transfer.
- Simultaneous pretrained_checkpoint and pretrained_checkpoint_manifest is rejected.
- Transfer report JSON and Markdown are written and contain the expected fields.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from foundry.models import NeurosoftConvBiGRU
from foundry.tasks.config import TaskConfig
from foundry.training.pretrained import (
    TransferMode,
    load_pretrained_weights,
)
from foundry.training.checkpoint_manifest import (
    CheckpointManifestError,
    load_checkpoint_manifest,
    verify_checkpoint_integrity,
    write_checkpoint_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_configs():
    return {
        "neurosoft": TaskConfig.from_dict(
            {
                "name": "neurosoft",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 8,
                },
                "target_extractor": {
                    "_target_": "foundry.tasks.targets.TargetExtractor",
                    "timestamp_key": "neurosoft.timestamps",
                    "value_key": "neurosoft.values",
                },
                "loss": {
                    "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"
                },
                "class_names": [str(i) for i in range(8)],
            }
        )
    }


def _build_source_model(
    session_configs: dict[str, int] | None = None,
    id_aliases: dict | None = None,
):
    """Build a NeurosoftConvBiGRU as if it were a source-pretraining model."""
    if session_configs is None:
        session_configs = {
            "minipigs:sub-02_ses-01_task-AcousStim_acq-LH_desc-raw": 18,
            "minipigs:sub-03_ses-06_task-AcousStim_acq-LH_desc-raw": 20,
        }
    return NeurosoftConvBiGRU(
        task_configs=_make_task_configs(),
        session_configs=session_configs,
        num_samples=1000,
        adapter_dim=64,
        temporal_channels=128,
        temporal_kernel_samples=64,
        temporal_stride=4,
        conv_depth=1,
        dropout_rate=0.3,
        gru_hidden_size=128,
        gru_num_layers=2,
        gru_bidirectional=True,
        gru_dropout=0.0,
        id_aliases=id_aliases,
    )


def _build_target_model(
    session_configs: dict[str, int] | None = None,
):
    """Build a NeurosoftConvBiGRU as if it were a downstream target model."""
    if session_configs is None:
        session_configs = {
            "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw": 22,
        }
    return NeurosoftConvBiGRU(
        task_configs=_make_task_configs(),
        session_configs=session_configs,
        num_samples=1000,
        adapter_dim=64,
        temporal_channels=128,
        temporal_kernel_samples=64,
        temporal_stride=4,
        conv_depth=1,
        dropout_rate=0.3,
        gru_hidden_size=128,
        gru_num_layers=2,
        gru_bidirectional=True,
        gru_dropout=0.0,
    )


def _save_lightning_ckpt(model: nn.Module, path: Path) -> None:
    prefix = "model."
    state_dict = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, path)


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def _write_test_manifest(
    checkpoint_path: Path,
    manifest_dir: Path,
    kind: str = "best",
    excluded_species: str = "minipigs",
    excluded_subject: str = "sub-06",
) -> tuple[Path, Path]:
    """Write a minimal checkpoint manifest and return (json_path, md_path)."""
    return write_checkpoint_manifest(
        checkpoint_path,
        manifest_dir,
        kind=kind,
        trained_on={
            "source_selection_id": "smoke_minipigs_target-sub-06",
            "source_manifest_path": "phase3_smoke/minipigs/target-sub-06.json",
            "source_manifest_hash": "abc123",
            "excluded_target": {
                "species": excluded_species,
                "subject": excluded_subject,
            },
            "subjects": ["minipigs:sub-02", "minipigs:sub-03"],
            "recordings": [
                "minipigs:sub-02_ses-01_task-AcousStim_acq-LH_desc-raw",
                "minipigs:sub-03_ses-06_task-AcousStim_acq-LH_desc-raw",
            ],
            "selected_train_examples": 1533,
            "available_train_windows": 1533,
            "realized_train_windows_per_epoch": 1520,
            "processed_windows": 5000,
            "completed_effective_epochs": 3.29,
            "optimizer_steps": 500,
            "class_union": ["0", "1", "2", "3", "4", "5", "6", "7"],
            "class_intersection": ["0", "1", "2", "3", "4", "5"],
        },
        selection={
            "monitor": "val/source_session_mean_supported_f1",
            "monitor_value": 0.35,
            "source_session_scores": {
                "minipigs:sub-02_ses-01_task-AcousStim_acq-LH_desc-raw": 0.33,
                "minipigs:sub-03_ses-06_task-AcousStim_acq-LH_desc-raw": 0.37,
            },
        },
        compute={
            "cumulative_flops": 384049152000,
            "flop_method": "torch-flop-counter-v1",
            "signal_seconds": 2500.0,
            "wall_time_seconds": 120.0,
            "gpu": "test-gpu",
            "precision": "bf16-mixed",
        },
        recipe={
            "model": {"adapter_dim": 64, "temporal_channels": 128},
            "hyperparameters": {"batch_size": 16, "learning_rate": 0.0015},
        },
        normalization_artifact_hashes={},
        git_sha="test123",
        snapshot_bundle="test-bundle",
        slurm_job_id="12345",
        wandb_info={"project": "test", "group": "test", "run_id": "test"},
    )


# ---------------------------------------------------------------------------
# Transfer regime component selection
# ---------------------------------------------------------------------------


class TestTransferRegimeSelection:
    """NeurosoftConvBiGRU declares the correct components for each regime."""

    def test_full_finetuning_includes_frontend_gru_router(self):
        model = _build_target_model()
        components = model.transferable_components_for_mode("full_finetuning")
        assert components == ("temporal_frontend", "gru", "router")

    def test_frozen_representation_excludes_router(self):
        model = _build_target_model()
        components = model.transferable_components_for_mode(
            "frozen_representation"
        )
        assert components == ("temporal_frontend", "gru")

    def test_invalid_regime_raises(self):
        model = _build_target_model()
        with pytest.raises(ValueError, match="mode must be"):
            model.transferable_components_for_mode("unknown_mode")


# ---------------------------------------------------------------------------
# Source adapter exclusion
# ---------------------------------------------------------------------------


class TestSourceAdapterExclusion:
    """Source adapter keys must never load into any target model."""

    def test_full_finetuning_excludes_source_adapters(self, tmp_path):
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode("full_finetuning")
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        adapter_loaded = [k for k in report.loaded if "session_adapter" in k]
        assert adapter_loaded == [], (
            f"Source adapter keys should never load: {adapter_loaded}"
        )

        adapter_excluded = [
            k for k in report.skipped_excluded if "session_adapter" in k
        ]
        assert len(adapter_excluded) > 0, (
            "Source adapter keys must appear in skipped_excluded"
        )

    def test_frozen_representation_excludes_source_adapters(self, tmp_path):
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        adapter_loaded = [k for k in report.loaded if "session_adapter" in k]
        assert adapter_loaded == [], (
            f"Source adapter keys should never load: {adapter_loaded}"
        )

    def test_full_finetuning_excludes_router_keys_from_excluded(
        self, tmp_path
    ):
        """In full_finetuning, router IS transferred, not excluded."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode("full_finetuning")
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        router_loaded = [k for k in report.loaded if k.startswith("router.")]
        assert len(router_loaded) > 0, "Router keys must be loaded"

    def test_frozen_representation_excludes_router_from_transfer(
        self, tmp_path
    ):
        """In frozen_representation, router IS excluded from transfer."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        router_loaded = [k for k in report.loaded if k.startswith("router.")]
        assert router_loaded == [], (
            "Router keys must not be loaded in frozen_representation"
        )
        router_excluded = [
            k for k in report.skipped_excluded if k.startswith("router.")
        ]
        assert len(router_excluded) > 0, (
            "Router keys must appear in skipped_excluded"
        )


# ---------------------------------------------------------------------------
# Target adapter remains bitwise fresh
# ---------------------------------------------------------------------------


class TestTargetAdapterFreshness:
    """Target adapter parameters must be bitwise identical before and after transfer."""

    def _adapter_state(
        self, model: nn.Module
    ) -> dict[str, torch.Tensor]:
        return {
            k: v.clone()
            for k, v in model.state_dict().items()
            if "session_adapter" in k
        }

    def test_full_finetuning_preserves_target_adapter(self, tmp_path):
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        before = self._adapter_state(dst)

        components = dst.transferable_components_for_mode("full_finetuning")
        load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        after = self._adapter_state(dst)

        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), (
                f"Target adapter param {key} was modified by loading!"
            )

    def test_frozen_representation_preserves_target_adapter(self, tmp_path):
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        before = self._adapter_state(dst)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        after = self._adapter_state(dst)

        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), (
                f"Target adapter param {key} was modified by loading!"
            )

    def test_frozen_representation_preserves_router(self, tmp_path):
        """Frozen representation must leave the router bitwise fresh."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        router_before = {
            k: v.clone()
            for k, v in dst.state_dict().items()
            if k.startswith("router.")
        }

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        router_after = {
            k: v.clone()
            for k, v in dst.state_dict().items()
            if k.startswith("router.")
        }

        for key in router_before:
            assert torch.equal(router_before[key], router_after[key]), (
                f"Router param {key} was modified in frozen_representation!"
            )


# ---------------------------------------------------------------------------
# Full finetuning: all parameters trainable
# ---------------------------------------------------------------------------


class TestFullFinetuningTrainability:
    def test_all_target_params_trainable(self, tmp_path):
        """After full_finetuning transfer, every parameter is trainable."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode("full_finetuning")
        load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        for name, param in dst.named_parameters():
            assert param.requires_grad, (
                f"Parameter {name} should be trainable in full_finetuning"
            )

    def test_shared_components_loaded(self, tmp_path):
        """Full finetuning loads frontend, GRU, and router."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode("full_finetuning")
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        for prefix in ("temporal_frontend.", "gru.", "router."):
            matched = [k for k in report.loaded if k.startswith(prefix)]
            assert len(matched) > 0, (
                f"No keys loaded for {prefix}"
            )


# ---------------------------------------------------------------------------
# Frozen representation: correct freeze/train split
# ---------------------------------------------------------------------------


class TestFrozenRepresentationTrainability:
    def test_frontend_gru_frozen(self, tmp_path):
        """Frontend and GRU are frozen after frozen_representation transfer."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        load_pretrained_weights(
            dst,
            ckpt,
            components=components,
            mode=TransferMode.STRICT,
            freeze=True,
        )

        for name, param in dst.named_parameters():
            if name.startswith("temporal_frontend.") or name.startswith("gru."):
                assert not param.requires_grad, (
                    f"Parameter {name} should be frozen"
                )

    def test_router_and_adapter_trainable(self, tmp_path):
        """Router and session adapter are trainable after frozen_representation."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        load_pretrained_weights(
            dst,
            ckpt,
            components=components,
            mode=TransferMode.STRICT,
            freeze=True,
        )

        for name, param in dst.named_parameters():
            if name.startswith("router.") or name.startswith(
                "session_adapter."
            ):
                assert param.requires_grad, (
                    f"Parameter {name} should be trainable"
                )

    def test_trainable_count_correct(self, tmp_path):
        """Only router + adapter params are trainable in frozen_representation."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        load_pretrained_weights(
            dst,
            ckpt,
            components=components,
            mode=TransferMode.STRICT,
            freeze=True,
        )

        trainable_names = {
            name
            for name, param in dst.named_parameters()
            if param.requires_grad
        }
        for name in trainable_names:
            assert name.startswith("router.") or name.startswith(
                "session_adapter."
            ), f"Unexpected trainable parameter: {name}"


# ---------------------------------------------------------------------------
# Weights are actually loaded (not just reported)
# ---------------------------------------------------------------------------


class TestWeightsActuallyLoaded:
    def test_full_finetuning_changes_shared_weights(self, tmp_path):
        """After transfer, shared component tensors match the source."""
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode("full_finetuning")
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        src_state = dict(src.named_parameters())
        src_buffers = dict(src.named_buffers())
        src_all = {**src_state, **src_buffers}

        dst_state = dict(dst.named_parameters())
        dst_buffers = dict(dst.named_buffers())
        dst_all = {**dst_state, **dst_buffers}

        for key in report.loaded:
            assert torch.equal(src_all[key], dst_all[key]), (
                f"Loaded key {key} does not match source"
            )

    def test_frozen_representation_changes_frontend_gru(self, tmp_path):
        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "source.ckpt"
        _save_lightning_ckpt(src, ckpt)

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        src_all = {
            **dict(src.named_parameters()),
            **dict(src.named_buffers()),
        }
        dst_all = {
            **dict(dst.named_parameters()),
            **dict(dst.named_buffers()),
        }

        for key in report.loaded:
            assert torch.equal(src_all[key], dst_all[key]), (
                f"Loaded key {key} does not match source"
            )


# ---------------------------------------------------------------------------
# Tampered checkpoint / manifest detection
# ---------------------------------------------------------------------------


class TestTamperDetection:
    def test_tampered_checkpoint_sha256_fails(self, tmp_path):
        """Corrupting the checkpoint file must cause a SHA-256 mismatch."""
        src = _build_source_model()
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_test_manifest(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)

        with open(ckpt, "ab") as f:
            f.write(b"TAMPERED")

        with pytest.raises(CheckpointManifestError, match="SHA-256 mismatch"):
            verify_checkpoint_integrity(manifest, str(tmp_path))

    def test_tampered_manifest_hash_fails(self, tmp_path):
        """Modifying manifest JSON must cause a manifest hash mismatch."""
        src = _build_source_model()
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_test_manifest(ckpt, manifest_dir)

        data = json.loads(json_path.read_text())
        data["selection"]["monitor_value"] = 0.99
        json_path.write_text(json.dumps(data, indent=2))

        with pytest.raises(CheckpointManifestError, match="hash mismatch"):
            load_checkpoint_manifest(json_path)

    def test_missing_checkpoint_file_fails(self, tmp_path):
        """A missing checkpoint must fail integrity verification."""
        src = _build_source_model()
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_test_manifest(ckpt, manifest_dir)
        manifest = load_checkpoint_manifest(json_path)

        ckpt.unlink()

        with pytest.raises(CheckpointManifestError, match="not found"):
            verify_checkpoint_integrity(manifest, str(tmp_path))


# ---------------------------------------------------------------------------
# Target mismatch detection
# ---------------------------------------------------------------------------


class TestTargetMismatch:
    def test_wrong_species_fails(self, tmp_path):
        from main import _validate_manifest_target

        manifest = {
            "trained_on": {
                "excluded_target": {
                    "species": "monkeys",
                    "subject": "sub-06",
                }
            }
        }

        class _FakeDataModule:
            dataset_class = type("FakeMinipigs", (), {"__name__": "NeurosoftMinipigs2026"})
            dataset = type("FakeDataset", (), {"recording_ids": ["sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"]})()

        with pytest.raises(ValueError, match="does not match"):
            _validate_manifest_target(manifest, _FakeDataModule())

    def test_wrong_subject_fails(self, tmp_path):
        from main import _validate_manifest_target

        manifest = {
            "trained_on": {
                "excluded_target": {
                    "species": "minipigs",
                    "subject": "sub-01",
                }
            }
        }

        class _FakeDataModule:
            dataset_class = type("FakeMinipigs", (), {"__name__": "NeurosoftMinipigs2026"})
            dataset = type("FakeDataset", (), {"recording_ids": ["sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"]})()

        with pytest.raises(ValueError, match="does not match"):
            _validate_manifest_target(manifest, _FakeDataModule())

    def test_correct_target_passes(self, tmp_path):
        from main import _validate_manifest_target

        manifest = {
            "trained_on": {
                "excluded_target": {
                    "species": "minipigs",
                    "subject": "sub-06",
                }
            }
        }

        class _FakeDataModule:
            dataset_class = type("FakeMinipigs", (), {"__name__": "NeurosoftMinipigs2026"})
            dataset = type("FakeDataset", (), {"recording_ids": ["sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"]})()

        _validate_manifest_target(manifest, _FakeDataModule())

    def test_missing_excluded_target_fails(self, tmp_path):
        from main import _validate_manifest_target

        manifest = {"trained_on": {}}

        class _FakeDataModule:
            dataset_class = type("FakeMinipigs", (), {"__name__": "NeurosoftMinipigs2026"})
            dataset = type("FakeDataset", (), {"recording_ids": ["sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"]})()

        with pytest.raises(ValueError, match="missing"):
            _validate_manifest_target(manifest, _FakeDataModule())


# ---------------------------------------------------------------------------
# Mutual exclusion of checkpoint paths
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    def test_both_checkpoint_paths_raises(self):
        from main import _validate_checkpoint_policy

        with pytest.raises(ValueError, match="Both resume checkpoint"):
            _validate_checkpoint_policy(
                "/path/to/resume.ckpt", "/path/to/pretrained.ckpt"
            )

    def test_resume_only_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy("/path/to/resume.ckpt", None)

    def test_pretrained_only_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy(None, "/path/to/pretrained.ckpt")

    def test_neither_passes(self):
        from main import _validate_checkpoint_policy

        _validate_checkpoint_policy(None, None)


# ---------------------------------------------------------------------------
# Transfer report persistence
# ---------------------------------------------------------------------------


class TestTransferReportPersistence:
    def test_transfer_report_json_written(self, tmp_path):
        """_apply_manifest_transfer writes transfer-report.json."""
        from main import _apply_manifest_transfer
        from omegaconf import OmegaConf

        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_test_manifest(ckpt, manifest_dir)
        manifest = load_checkpoint_manifest(json_path)

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        cfg = OmegaConf.create({
            "run": {
                "pretrained_checkpoint_manifest": str(json_path),
                "pretrained_transfer_regime": "full_finetuning",
            }
        })

        os.environ["FOUNDRY_CHECKPOINT_ROOT"] = str(tmp_path)
        try:
            _apply_manifest_transfer(dst, manifest, cfg, output_dir)
        finally:
            del os.environ["FOUNDRY_CHECKPOINT_ROOT"]

        report_json = Path(output_dir) / "transfer-report.json"
        assert report_json.exists(), "transfer-report.json must be written"

        data = json.loads(report_json.read_text())
        assert data["transfer_regime"] == "full_finetuning"
        assert "loaded" in data
        assert "skipped_excluded" in data
        assert len(data["loaded"]) > 0

    def test_transfer_report_md_written(self, tmp_path):
        """_apply_manifest_transfer writes transfer-report.md."""
        from main import _apply_manifest_transfer
        from omegaconf import OmegaConf

        src = _build_source_model()
        dst = _build_target_model()
        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, _ = _write_test_manifest(ckpt, manifest_dir)
        manifest = load_checkpoint_manifest(json_path)

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        cfg = OmegaConf.create({
            "run": {
                "pretrained_checkpoint_manifest": str(json_path),
                "pretrained_transfer_regime": "full_finetuning",
            }
        })

        os.environ["FOUNDRY_CHECKPOINT_ROOT"] = str(tmp_path)
        try:
            _apply_manifest_transfer(dst, manifest, cfg, output_dir)
        finally:
            del os.environ["FOUNDRY_CHECKPOINT_ROOT"]

        report_md = Path(output_dir) / "transfer-report.md"
        assert report_md.exists(), "transfer-report.md must be written"

        md_text = report_md.read_text()
        assert "full_finetuning" in md_text
        assert "Loaded" in md_text
        assert "Excluded" in md_text


# ---------------------------------------------------------------------------
# Manifest schema and version validation
# ---------------------------------------------------------------------------


class TestManifestSchemaValidation:
    def test_wrong_schema_rejected(self, tmp_path):
        bad_manifest = tmp_path / "bad_schema.json"
        bad_manifest.write_text(json.dumps({
            "schema": "wrong-schema",
            "version": 1,
            "manifest_hash": "abc",
        }))

        with pytest.raises(CheckpointManifestError, match="Unsupported schema"):
            load_checkpoint_manifest(bad_manifest)

    def test_wrong_version_rejected(self, tmp_path):
        bad_manifest = tmp_path / "bad_version.json"
        bad_manifest.write_text(json.dumps({
            "schema": "neurosoft-pretraining-checkpoint",
            "version": 99,
            "manifest_hash": "abc",
        }))

        with pytest.raises(
            CheckpointManifestError, match="Unsupported version"
        ):
            load_checkpoint_manifest(bad_manifest)

    def test_missing_manifest_hash_rejected(self, tmp_path):
        bad_manifest = tmp_path / "no_hash.json"
        bad_manifest.write_text(json.dumps({
            "schema": "neurosoft-pretraining-checkpoint",
            "version": 1,
        }))

        with pytest.raises(CheckpointManifestError, match="manifest_hash"):
            load_checkpoint_manifest(bad_manifest)


# ---------------------------------------------------------------------------
# End-to-end: manifest write -> load -> verify -> transfer
# ---------------------------------------------------------------------------


class TestEndToEndManifestTransfer:
    def test_full_pipeline_full_finetuning(self, tmp_path):
        """Full manifest pipeline: write, load, verify, transfer."""
        src = _build_source_model()
        dst = _build_target_model()
        target_before = _snapshot_state(dst)

        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_test_manifest(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)
        verify_checkpoint_integrity(manifest, str(tmp_path))

        components = dst.transferable_components_for_mode("full_finetuning")
        report = load_pretrained_weights(
            dst, ckpt, components=components, mode=TransferMode.STRICT
        )

        assert len(report.loaded) > 0
        assert not report.has_errors

        adapter_loaded = [k for k in report.loaded if "session_adapter" in k]
        assert adapter_loaded == []

        for key in target_before:
            if "session_adapter" in key:
                assert torch.equal(
                    target_before[key], dst.state_dict()[key]
                )

    def test_full_pipeline_frozen_representation(self, tmp_path):
        """Full manifest pipeline with frozen representation."""
        src = _build_source_model()
        dst = _build_target_model()
        target_adapter_before = {
            k: v.clone()
            for k, v in dst.state_dict().items()
            if "session_adapter" in k
        }
        router_before = {
            k: v.clone()
            for k, v in dst.state_dict().items()
            if k.startswith("router.")
        }

        ckpt = tmp_path / "checkpoints" / "best.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        _save_lightning_ckpt(src, ckpt)

        manifest_dir = tmp_path / "manifests"
        json_path, md_path = _write_test_manifest(ckpt, manifest_dir)

        manifest = load_checkpoint_manifest(json_path)
        verify_checkpoint_integrity(manifest, str(tmp_path))

        components = dst.transferable_components_for_mode(
            "frozen_representation"
        )
        report = load_pretrained_weights(
            dst,
            ckpt,
            components=components,
            mode=TransferMode.STRICT,
            freeze=True,
        )

        assert len(report.loaded) > 0
        assert not report.has_errors

        for key in target_adapter_before:
            assert torch.equal(
                target_adapter_before[key], dst.state_dict()[key]
            )

        for key in router_before:
            assert torch.equal(router_before[key], dst.state_dict()[key])

        for name, param in dst.named_parameters():
            if name.startswith("temporal_frontend.") or name.startswith("gru."):
                assert not param.requires_grad
            elif name.startswith("router.") or name.startswith(
                "session_adapter."
            ):
                assert param.requires_grad

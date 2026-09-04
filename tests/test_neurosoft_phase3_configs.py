"""WP6 gate tests: Hydra config composition for NeuroSoft Phase 3.

Covers:
- Source pretraining configs compose for both species without error.
- Transfer configs compose for both species without error.
- Source configs use the correct monitor for checkpoint/early-stopping.
- Transfer configs require pretrained_checkpoint_manifest and transfer_regime.
- Transfer configs support full_finetuning and frozen_representation.
- Index resolvers find smoke manifests by ID.
- Index resolvers generate sweeps by family/species.
- Config composition does not hardcode recording lists.
- Configs resolve to the intended manifest/checkpoint policies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from foundry.config_resolvers import register_resolvers

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = str(REPO_ROOT / "configs")
INDEX_PATH = str(
    REPO_ROOT / "manifests" / "neurosoft_supervised" / "v1" / "index.json"
)
SMOKE_MANIFEST_MINIPIGS = str(
    REPO_ROOT
    / "manifests"
    / "neurosoft_supervised"
    / "v1"
    / "phase3_smoke"
    / "minipigs"
    / "target-sub-06.json"
)
SMOKE_MANIFEST_MONKEYS = str(
    REPO_ROOT
    / "manifests"
    / "neurosoft_supervised"
    / "v1"
    / "phase3_smoke"
    / "monkeys"
    / "target-sub-01.json"
)


@pytest.fixture(autouse=True)
def _register_resolvers():
    register_resolvers()
    yield


@pytest.fixture()
def _hydra_context():
    """Provide a fresh Hydra global context for each test."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        yield
    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# Source pretraining config composition
# ---------------------------------------------------------------------------


class TestSourcePretrainingConfigs:
    @pytest.mark.skipif(
        not os.path.isfile(SMOKE_MANIFEST_MINIPIGS),
        reason="Manifests not generated",
    )
    def test_minipigs_source_pretraining_composes(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        assert OmegaConf.select(cfg, "data.role") == "source_pretraining"
        assert OmegaConf.select(cfg, "data.source_test_policy") == "forbidden"
        assert OmegaConf.select(cfg, "run.evaluate_test") is False

    @pytest.mark.skipif(
        not os.path.isfile(SMOKE_MANIFEST_MONKEYS),
        reason="Manifests not generated",
    )
    def test_monkeys_source_pretraining_composes(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_monkeys",
                f"source_manifest={SMOKE_MANIFEST_MONKEYS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        assert OmegaConf.select(cfg, "data.role") == "source_pretraining"
        assert OmegaConf.select(cfg, "data.source_test_policy") == "forbidden"

    def test_source_config_monitors_source_session_mean(self, _hydra_context):
        """Source configs must monitor source_session_mean_supported_f1."""
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        es_monitor = OmegaConf.select(
            cfg, "trainer.callbacks.early_stopping.monitor"
        )
        ckpt_monitor = OmegaConf.select(
            cfg, "trainer.callbacks.model_checkpoint.monitor"
        )
        assert es_monitor == "val/source_session_mean_supported_f1"
        assert ckpt_monitor == "val/source_session_mean_supported_f1"

    def test_source_config_has_milestone_callback(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        milestone_cfg = OmegaConf.select(
            cfg, "trainer.callbacks.compute_milestones"
        )
        assert milestone_cfg is not None
        target = OmegaConf.select(milestone_cfg, "_target_")
        assert "ComputeMilestoneCheckpointCallback" in target

    def test_source_config_has_session_metrics_callback(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        ssm_cfg = OmegaConf.select(
            cfg, "trainer.callbacks.source_session_metrics"
        )
        assert ssm_cfg is not None
        target = OmegaConf.select(ssm_cfg, "_target_")
        assert "SourceSessionMetricsCallback" in target

    def test_source_config_uses_step_budget(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )
        assert cfg.trainer.max_steps == 500
        assert cfg.trainer.max_epochs == -1

    def test_source_config_no_hardcoded_recordings(self, _hydra_context):
        """Source configs must not hardcode recording_ids — they derive from manifest."""
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )
        recording_ids = OmegaConf.select(
            cfg, "data.dataset_kwargs.recording_ids", default=None
        )
        assert recording_ids is None, (
            "Source pretraining configs must not hardcode recording_ids"
        )


# ---------------------------------------------------------------------------
# Transfer config composition
# ---------------------------------------------------------------------------


class TestTransferConfigs:
    def test_minipigs_transfer_composes_full_finetuning(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        assert (
            OmegaConf.select(cfg, "run.pretrained_transfer_regime")
            == "full_finetuning"
        )
        assert (
            OmegaConf.select(cfg, "run.pretrained_checkpoint_manifest")
            == "/fake/manifest.json"
        )
        assert OmegaConf.select(cfg, "run.pretrained_checkpoint") is None

    def test_minipigs_transfer_composes_frozen_representation(
        self, _hydra_context
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=frozen_representation",
            ],
        )
        assert (
            OmegaConf.select(cfg, "run.pretrained_transfer_regime")
            == "frozen_representation"
        )

    def test_monkeys_transfer_composes_full_finetuning(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys",
                "data.dataset_kwargs.recording_ids=[sub-01_ses-04_task-AcousStim_acq-RH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        assert (
            OmegaConf.select(cfg, "run.pretrained_transfer_regime")
            == "full_finetuning"
        )

    def test_monkeys_transfer_composes_frozen_representation(
        self, _hydra_context
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys",
                "data.dataset_kwargs.recording_ids=[sub-01_ses-04_task-AcousStim_acq-RH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=frozen_representation",
            ],
        )
        assert (
            OmegaConf.select(cfg, "run.pretrained_transfer_regime")
            == "frozen_representation"
        )

    def test_transfer_config_monitors_target_validation(self, _hydra_context):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        es_monitor = OmegaConf.select(
            cfg, "trainer.callbacks.early_stopping.monitor"
        )
        assert es_monitor == "val/neurosoft_acoustic_stim_8band_supported_f1"

    def test_transfer_config_uses_epoch_budget(self, _hydra_context):
        """Transfer configs use max_epochs, not max_steps."""
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        assert cfg.trainer.max_epochs == 200

    def test_transfer_config_recipe_matches_phase2(self, _hydra_context):
        """Transfer configs use the Phase 2 recipe."""
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        assert cfg.model.adapter_dim == 64
        assert cfg.model.temporal_channels == 128
        assert cfg.model.gru_hidden_size == 128
        assert cfg.hyperparameters.learning_rate == 0.00025
        assert cfg.hyperparameters.batch_size == 16

    def test_transfer_config_seeds(self, _hydra_context):
        """Transfer configs use seed 42 by default."""
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                "run.pretrained_transfer_regime=full_finetuning",
            ],
        )
        assert cfg.run.seed == 42


# ---------------------------------------------------------------------------
# Index resolver tests
# ---------------------------------------------------------------------------


class TestIndexResolvers:
    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_smoke_manifest_by_id_minipigs(self):
        from foundry.config_resolvers import _source_manifest_by_id

        path = _source_manifest_by_id(
            INDEX_PATH, "smoke_minipigs_target-sub-06"
        )
        assert os.path.isfile(path)
        assert "phase3_smoke" in path
        assert "minipigs" in path

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_smoke_manifest_by_id_monkeys(self):
        from foundry.config_resolvers import _source_manifest_by_id

        path = _source_manifest_by_id(INDEX_PATH, "smoke_monkeys_target-sub-01")
        assert os.path.isfile(path)
        assert "phase3_smoke" in path
        assert "monkeys" in path

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_volume_sweep_minipigs(self):
        from foundry.config_resolvers import _source_manifest_sweep

        sweep = _source_manifest_sweep(
            INDEX_PATH, "source_volume", species="minipigs"
        )
        paths = sweep.split(",")
        assert len(paths) > 0
        for p in paths:
            assert "source_volume" in p

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_volume_sweep_monkeys(self):
        from foundry.config_resolvers import _source_manifest_sweep

        sweep = _source_manifest_sweep(
            INDEX_PATH, "source_volume", species="monkeys"
        )
        paths = sweep.split(",")
        assert len(paths) > 0

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_volume_sweep_target_filter(self):
        from foundry.config_resolvers import _source_manifest_sweep

        sweep = _source_manifest_sweep(
            INDEX_PATH,
            "source_volume",
            species="minipigs",
            target_subject="sub-06",
        )
        paths = sweep.split(",")
        for p in paths:
            assert "target-sub-06" in p

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_nonexistent_id_raises(self):
        from foundry.config_resolvers import _source_manifest_by_id

        with pytest.raises(ValueError, match="not found"):
            _source_manifest_by_id(INDEX_PATH, "nonexistent_id")

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_path_stem_resolver(self):
        from foundry.config_resolvers import _path_stem

        stem = _path_stem(
            "manifests/neurosoft_supervised/v1/phase3_smoke/minipigs/target-sub-06.json"
        )
        assert stem == "target-sub-06"


# ---------------------------------------------------------------------------
# Index completeness
# ---------------------------------------------------------------------------


class TestIndexCompleteness:
    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_expected_family_counts(self):
        """Index must contain the expected number of manifests per family."""
        with open(INDEX_PATH) as f:
            index = json.load(f)

        entries = index["entries"]
        families: dict[str, int] = {}
        for entry in entries:
            fam = entry["family"]
            families[fam] = families.get(fam, 0) + 1

        assert families.get("phase3_smoke", 0) == 2
        assert families.get("source_pool", 0) == 12
        assert families.get("source_volume", 0) == 144
        assert families.get("species_composition", 0) == 108

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_no_duplicate_selection_ids(self):
        with open(INDEX_PATH) as f:
            index = json.load(f)

        ids = [e["selection_id"] for e in index["entries"]]
        assert len(ids) == len(set(ids)), (
            "Index contains duplicate selection IDs"
        )

    @pytest.mark.skipif(
        not os.path.isfile(INDEX_PATH), reason="Index not generated"
    )
    def test_all_entries_eligible(self):
        with open(INDEX_PATH) as f:
            index = json.load(f)

        ineligible = [
            e["selection_id"]
            for e in index["entries"]
            if not e.get("eligible", True)
        ]
        assert ineligible == [], (
            f"Some entries are ineligible: {ineligible[:5]}"
        )


# ---------------------------------------------------------------------------
# Recipe consistency across configs
# ---------------------------------------------------------------------------


class TestRecipeConsistency:
    def test_source_and_transfer_use_same_model_recipe(self, _hydra_context):
        """Source pretraining and transfer configs use the same model recipe."""
        src_cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            tgt_cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                    "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                    "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                    "run.pretrained_transfer_regime=full_finetuning",
                ],
            )

        for key in [
            "adapter_dim",
            "temporal_channels",
            "temporal_kernel_samples",
            "temporal_stride",
            "conv_depth",
            "dropout_rate",
            "gru_hidden_size",
            "gru_num_layers",
            "gru_bidirectional",
            "gru_dropout",
        ]:
            src_val = OmegaConf.select(src_cfg, f"model.{key}")
            tgt_val = OmegaConf.select(tgt_cfg, f"model.{key}")
            assert src_val == tgt_val, (
                f"model.{key} mismatch: source={src_val}, target={tgt_val}"
            )

    def test_source_and_transfer_use_same_hyperparameters(self, _hydra_context):
        """Source and transfer use the same optimizer hyperparameters."""
        src_cfg = compose(
            config_name="config",
            overrides=[
                "experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs",
                f"source_manifest={SMOKE_MANIFEST_MINIPIGS}",
                "run.seed=42",
                "trainer.max_steps=500",
                "trainer.val_check_interval=100",
            ],
        )

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            tgt_cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs",
                    "data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw]",
                    "run.pretrained_checkpoint_manifest=/fake/manifest.json",
                    "run.pretrained_transfer_regime=full_finetuning",
                ],
            )

        for key in ["learning_rate", "weight_decay", "batch_size"]:
            src_val = OmegaConf.select(src_cfg, f"hyperparameters.{key}")
            tgt_val = OmegaConf.select(tgt_cfg, f"hyperparameters.{key}")
            assert src_val == tgt_val, (
                f"hyperparameters.{key}: source={src_val}, target={tgt_val}"
            )

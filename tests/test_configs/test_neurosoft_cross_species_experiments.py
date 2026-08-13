from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from foundry.config_resolvers import register_resolvers


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose_experiment(name: str, *overrides: str):
    register_resolvers()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment=auditory_decoding/{name}", *overrides],
        )
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    ("experiment_name", "expected_sources"),
    [
        ("poyo_neurosoft_freqband8_joint_causal", None),
        ("poyo_neurosoft_freqband8_minipigs_causal", ["minipigs"]),
        ("poyo_neurosoft_freqband8_monkeys_causal", ["monkeys"]),
    ],
)
def test_baseline_experiments_share_model_and_data_settings(
    experiment_name,
    expected_sources,
):
    cfg = _compose_experiment(experiment_name)

    assert cfg.data._target_.endswith("NeurosoftMultispeciesDataModule")
    assert cfg.data.dataset_kwargs.get("sources") == expected_sources
    assert cfg.data.dataset_kwargs.min_channels == 8
    assert cfg.data.split_type == "intrasession-causal"
    assert cfg.model.use_encoder_session_embedding is False
    assert list(cfg.model.decoder_source_ids) == ["minipigs", "monkeys"]
    assert cfg.model.tokenizer.temporal_embedding.kernel_size == 200
    assert cfg.model.tokenizer.temporal_embedding.stride == 200
    assert cfg.trainer.callbacks.early_stopping.monitor == "val/loss"
    assert cfg.trainer.callbacks.early_stopping.patience == 20


def test_one_band_experiment_filters_only_auxiliary_source():
    cfg = _compose_experiment(
        "poyo_neurosoft_freqband8_monkeys_plus_minipig_band_causal",
        "auxiliary_band=low_mids",
    )

    assert list(cfg.data.train_band_ids_by_source.minipigs) == ["low_mids"]
    assert cfg.data.train_uniform_band_total_count_by_source is None


def test_uniform_control_resolves_band_matched_trial_count():
    cfg = _compose_experiment(
        "poyo_neurosoft_freqband8_minipigs_plus_monkey_uniform_causal",
        "auxiliary_band=high_treble",
    )

    assert cfg.data.train_band_ids_by_source is None
    assert cfg.data.train_uniform_band_total_count_by_source.monkeys == 3645
    assert "uniform_n3645" in cfg.run.name

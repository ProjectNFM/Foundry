"""Pre-training entry point for LaBraM (Stages 1 and 2).

Similar structure to main.py but without multitask training and task configs.
Supports two stages:
- Stage 1: VQ-NSP neural tokenizer training
- Stage 2: Masked EEG modeling with frozen tokenizer
"""

import hashlib
import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.tools.stage_data import stage_data

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_ROOT = "../scratch/brainsets/processed"
DEFAULT_COMPRESSED_ROOT = "../scratch/brainsets/compressed"

def setup_logging(log_level: str):
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            logging.StreamHandler()
        ],
        force=True,
    )


def _configure_output_paths(cfg: DictConfig) -> tuple[str, str]:
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if OmegaConf.select(cfg, "trainer.callbacks.model_checkpoint") is not None:
        OmegaConf.update(
            cfg, "trainer.callbacks.model_checkpoint.dirpath", checkpoint_dir
        )
    OmegaConf.update(cfg, "trainer.default_root_dir", output_dir)

    return output_dir, checkpoint_dir


def _configure_wandb(cfg: DictConfig, output_dir: str) -> None:
    """Configure WandB run identity."""
    if "WandbLogger" not in OmegaConf.select(
        cfg, "logger._target_", default=""
    ):
        return

    OmegaConf.update(cfg, "logger.save_dir", output_dir)
    if OmegaConf.select(cfg, "logger.id") is not None:
        return

    resume_wandb_if_name_matches = OmegaConf.select(
        cfg, "run.resume_wandb_if_name_matches", default=False
    )
    if resume_wandb_if_name_matches:
        wandb_run_id = hashlib.md5(cfg.run.name.encode()).hexdigest()[:8]
        OmegaConf.update(cfg, "logger.id", wandb_run_id)


def _is_wandb_logger_enabled(cfg: DictConfig) -> bool:
    return "WandbLogger" in OmegaConf.select(cfg, "logger._target_", default="")


def _populate_data_driven_hyperparams(cfg: DictConfig):
    """Auto-populate num_channels and num_samples if missing."""
    if OmegaConf.select(cfg, "model.num_channels") is None:
        dataset_class = get_class(cfg.data.dataset_class)
        root = OmegaConf.select(cfg, "data.root", default="./data/processed/")
        try:
            num_channels = dataset_class.get_max_channels(root)
            OmegaConf.update(cfg, "model.num_channels", num_channels)
            logger.info(f"Auto-set num_channels={num_channels}")
        except Exception as e:
            logger.warning(f"Could not auto-populate num_channels: {e}")

    if OmegaConf.select(cfg, "model.num_samples") is None:
        seq_len = OmegaConf.select(cfg, "hyperparameters.sequence_length", default=8.0)
        sr = OmegaConf.select(cfg, "hyperparameters.sampling_rate", default=200)
        num_samples = int(seq_len * sr)
        OmegaConf.update(cfg, "model.num_samples", num_samples)
        logger.info(f"Auto-set num_samples={num_samples} (seq_len={seq_len}s @ {sr}Hz)")


def _build_pretrain_pipeline(cfg: DictConfig) -> tuple:
    """Build model and datamodule for pre-training (no task_configs)."""
    _populate_data_driven_hyperparams(cfg)
    
    normalize_data_config(cfg.data)
    datamodule = instantiate(cfg.data, tokenizer=None)
    
    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }
    model_kwargs["num_channels"] = cfg.model.num_channels
    model_kwargs["num_samples"] = cfg.model.num_samples
    model = ModelClass(**model_kwargs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    normalize_data_config(cfg.data)
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)

    return model, datamodule


def _stage_data_if_needed(cfg: DictConfig) -> None:
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        return

    stage_cfg = OmegaConf.to_container(
        cfg.get("stage", OmegaConf.create({})), resolve=True
    )
    if stage_cfg.get("skip", False):
        return

    new_root = stage_data(
        data_cfg=cfg.data,
        source_root=stage_cfg.get("source_root", DEFAULT_SOURCE_ROOT),
        compressed_root=stage_cfg.get(
            "compressed_root", DEFAULT_COMPRESSED_ROOT
        ),
        dest_root=slurm_tmpdir,
        compress=stage_cfg.get("compress", False),
    )
    OmegaConf.update(cfg, "data.root", new_root)
    logger.info("Data staged to %s", new_root)


@hydra_main_wrapper
@hydra.main(config_path="configs", config_name="pretrain_config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg.run.log_level)
    set_seed(cfg.run.seed)

    output_dir, checkpoint_dir = _configure_output_paths(cfg)
    _configure_wandb(cfg, output_dir)

    _stage_data_if_needed(cfg)

    model, datamodule = _build_pretrain_pipeline(cfg)

    lightning_module = instantiate(cfg.module, model=model)

    trainer = instantiate(cfg.trainer)

    logger.info(f"Starting pre-training on {cfg.data.dataset_class}")
    trainer.fit(lightning_module, datamodule)

    logger.info(f"Pre-training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

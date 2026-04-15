import hashlib
import logging
import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
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
            RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        ],
        force=True,
    )


def _get_slurm_restart_count() -> int:
    restart_count_raw = os.environ.get("SLURM_RESTART_COUNT", "0")
    try:
        return int(restart_count_raw)
    except ValueError:
        logger.warning(
            "Invalid SLURM_RESTART_COUNT=%r; treating as 0.",
            restart_count_raw,
        )
        return 0


# -- Config patching -------------------------------------------------------


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
    """Configure WandB run identity and resume behavior."""
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


def _finish_active_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    logger.info(
        "Finishing lingering WandB run id=%s name=%s before continuing.",
        wandb.run.id,
        wandb.run.name,
    )
    wandb.finish()


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


# -- Component construction ------------------------------------------------


def _populate_data_driven_hyperparams(cfg: DictConfig) -> None:
    """Auto-derive session_configs and num_channels from the dataset when missing."""
    session_configs = OmegaConf.select(
        cfg, "hyperparameters.session_configs", default=None
    )
    num_channels = OmegaConf.select(
        cfg, "hyperparameters.num_channels", default=None
    )

    if session_configs is not None and num_channels is not None:
        return

    dm = instantiate(cfg.data, tokenizer=None)
    dm.setup("fit")

    if session_configs is None:
        from foundry.data.utils import get_session_configs

        session_configs = get_session_configs(dm.dataset)
        OmegaConf.update(
            cfg,
            "hyperparameters.session_configs",
            session_configs,
            force_add=True,
        )
        logger.info(
            "Auto-populated hyperparameters.session_configs from dataset"
            " (%d sessions).",
            len(session_configs),
        )

    if num_channels is None:
        from foundry.data.utils import get_max_channels

        num_channels = get_max_channels(dm.dataset)
        OmegaConf.update(
            cfg,
            "hyperparameters.num_channels",
            num_channels,
            force_add=True,
        )
        logger.info(
            "Auto-populated hyperparameters.num_channels=%d from dataset.",
            num_channels,
        )


def _build_model_and_data(cfg: DictConfig):
    _populate_data_driven_hyperparams(cfg)

    DataModuleClass = get_class(cfg.data._target_)
    readout_specs = DataModuleClass.get_readout_specs_for_task(
        cfg.data.task_type
    )

    model = instantiate(cfg.model, readout_specs=readout_specs)
    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)

    return model, datamodule


def _compute_class_weights(cfg: DictConfig, datamodule):
    if cfg.module.class_weights != "auto":
        return None

    datamodule.setup("fit")
    smoothing = OmegaConf.select(
        cfg, "module.class_weight_smoothing", default=1.0
    )
    return datamodule.compute_class_weights(smoothing=smoothing)


def _build_lightning_module(cfg: DictConfig, model, datamodule):
    class_weights = _compute_class_weights(cfg, datamodule)

    if cfg.module.class_weights in (None, "none"):
        OmegaConf.update(cfg, "module.class_weights", None)

    return instantiate(
        cfg.module,
        model=model,
        class_names=datamodule.get_class_names_for_task(cfg.data.task_type),
        class_weights=class_weights,
    )


def _build_trainer(cfg: DictConfig):
    if OmegaConf.is_dict(cfg.trainer.get("callbacks")):
        cfg.trainer.callbacks = list(cfg.trainer.callbacks.values())
    return instantiate(cfg.trainer)


# -- Checkpointing ---------------------------------------------------------


def _get_resume_checkpoint_path(
    cfg: DictConfig,
    checkpoint_dir: str,
    slurm_restart_count: int,
) -> str | None:
    last_ckpt = Path(checkpoint_dir) / "last.ckpt"
    if not last_ckpt.exists():
        if slurm_restart_count > 0:
            logger.warning(
                "SLURM restart detected but checkpoint %s is missing; "
                "starting from scratch.",
                last_ckpt,
            )
        return None

    if slurm_restart_count > 0:
        ckpt_path = str(last_ckpt)
        logger.info(
            "SLURM restart detected (restart_count=%s). Resuming from %s.",
            slurm_restart_count,
            ckpt_path,
        )
        return ckpt_path

    resume_if_checkpoint_exists = OmegaConf.select(
        cfg,
        "run.resume_if_checkpoint_exists",
        default=False,
    )
    if resume_if_checkpoint_exists:
        ckpt_path = str(last_ckpt)
        logger.info(
            "run.resume_if_checkpoint_exists=true. Resuming from %s.",
            ckpt_path,
        )
        return ckpt_path

    logger.info(
        "Found checkpoint %s but run.resume_if_checkpoint_exists=false; "
        "starting from scratch.",
        last_ckpt,
    )
    return None


# -- WandB -----------------------------------------------------------------


def _log_config_to_wandb(trainer, cfg: DictConfig):
    if not isinstance(trainer.logger, WandbLogger):
        return

    loggable_keys = [
        "run",
        "hyperparameters",
        "model",
        "data",
        "module",
        "trainer",
    ]
    config_to_log = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in loggable_keys
        if key in cfg
    }
    trainer.logger.experiment.config.update(
        config_to_log, allow_val_change=True
    )


# -- Entry point ------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    setup_logging(cfg.run.log_level)
    torch.set_float32_matmul_precision(
        str(
            OmegaConf.select(
                cfg, "run.float32_matmul_precision", default="high"
            )
        )
    )
    seed_everything(cfg.run.seed, workers=True)
    logger.info("Starting training: %s", cfg.run.name)

    slurm_restart_count = _get_slurm_restart_count()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        logger.info(
            "SLURM job_id=%s array_task=%s restart_count=%s",
            slurm_job_id,
            os.environ.get("SLURM_ARRAY_TASK_ID"),
            slurm_restart_count,
        )

    using_wandb_logger = _is_wandb_logger_enabled(cfg)
    if using_wandb_logger:
        _finish_active_wandb_run()

    output_dir, checkpoint_dir = _configure_output_paths(cfg)
    _configure_wandb(cfg, output_dir)
    _stage_data_if_needed(cfg)

    model, datamodule = _build_model_and_data(cfg)
    lightning_module = _build_lightning_module(cfg, model, datamodule)
    trainer = _build_trainer(cfg)

    _log_config_to_wandb(trainer, cfg)

    ckpt_path = _get_resume_checkpoint_path(
        cfg, checkpoint_dir, slurm_restart_count
    )
    try:
        trainer.fit(
            lightning_module,
            datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    finally:
        if using_wandb_logger:
            _finish_active_wandb_run()


if __name__ == "__main__":
    register_resolvers()
    main()

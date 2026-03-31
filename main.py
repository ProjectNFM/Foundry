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


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    setup_logging(cfg.run.log_level)
    matmul_precision = OmegaConf.select(
        cfg, "run.float32_matmul_precision", default="high"
    )
    torch.set_float32_matmul_precision(str(matmul_precision))
    logger.info(
        "Set torch float32 matmul precision to '%s'.", matmul_precision
    )
    seed_everything(cfg.run.seed, workers=True)
    logger.info(f"Starting training: {cfg.run.name}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_restart_count = _get_slurm_restart_count()
    if slurm_job_id:
        logger.info(
            "SLURM job_id=%s array_task=%s restart_count=%s",
            slurm_job_id,
            os.environ.get("SLURM_ARRAY_TASK_ID"),
            slurm_restart_count,
        )

    # Hydra does not chdir by default (version_base=None), so resolve all
    # output paths as absolute to avoid writing into the project root.
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if OmegaConf.select(cfg, "trainer.callbacks.model_checkpoint") is not None:
        OmegaConf.update(
            cfg, "trainer.callbacks.model_checkpoint.dirpath", checkpoint_dir
        )
    OmegaConf.update(cfg, "trainer.default_root_dir", output_dir)

    # Deterministic WandB run ID so preempted jobs resume the same run
    if "WandbLogger" in OmegaConf.select(cfg, "logger._target_", default=""):
        OmegaConf.update(cfg, "logger.save_dir", output_dir)
        if OmegaConf.select(cfg, "logger.id") is None:
            wandb_run_id = hashlib.md5(cfg.run.name.encode()).hexdigest()[:8]
            OmegaConf.update(cfg, "logger.id", wandb_run_id)

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    stage_cfg = OmegaConf.to_container(
        cfg.get("stage", OmegaConf.create({})), resolve=True
    )
    if slurm_tmpdir and not stage_cfg.get("skip", False):
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

    DataModuleClass = get_class(cfg.data._target_)
    readout_specs = DataModuleClass.get_readout_specs_for_task(
        cfg.data.task_type
    )

    model = instantiate(cfg.model, readout_specs=readout_specs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)

    if cfg.module.class_weights == "auto":
        datamodule.setup("fit")
        smoothing = OmegaConf.select(
            cfg, "module.class_weight_smoothing", default=1.0
        )
        class_weights = datamodule.compute_class_weights(smoothing=smoothing)
    else:
        class_weights = None

    if cfg.module.class_weights in (None, "none"):
        OmegaConf.update(cfg, "module.class_weights", None)

    lightning_module = instantiate(
        cfg.module,
        model=model,
        class_names=datamodule.get_class_names_for_task(cfg.data.task_type),
        class_weights=class_weights,
    )

    if OmegaConf.is_dict(cfg.trainer.get("callbacks")):
        cfg.trainer.callbacks = list(cfg.trainer.callbacks.values())
    trainer = instantiate(cfg.trainer)
    _log_config_to_wandb(trainer, cfg)

    ckpt_path = _get_resume_checkpoint_path(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        slurm_restart_count=slurm_restart_count,
    )

    trainer.fit(
        lightning_module, datamodule, ckpt_path=ckpt_path, weights_only=False
    )


if __name__ == "__main__":
    register_resolvers()
    main()

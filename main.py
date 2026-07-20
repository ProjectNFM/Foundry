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
from rich.logging import RichHandler

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.tools.stage_data import stage_data
from foundry.training.pretrained import TransferMode, load_pretrained_weights

logger = logging.getLogger(__name__)

os.environ.setdefault("SLURM_TMPDIR", "/tmp")

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


def _log_output_destinations(
    cfg: DictConfig,
    output_dir: str,
    checkpoint_dir: str,
    using_wandb: bool,
) -> None:
    """Print a concise summary of where artifacts and metrics will be stored."""
    lines = [
        f"  Hydra output dir : {output_dir}",
        f"  Checkpoints      : {checkpoint_dir}",
    ]
    if using_wandb:
        project = OmegaConf.select(cfg, "logger.project", default="(default)")
        lines.append(f"  WandB project    : {project}")
        lines.append(
            f"  WandB save dir   : {OmegaConf.select(cfg, 'logger.save_dir', default=output_dir)}"
        )
    else:
        lines.append(
            "  Logger           : (no WandB — metrics to console only)"
        )
    logger.info("Output destinations:\n%s", "\n".join(lines))


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
    slurm_tmpdir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    if not slurm_tmpdir.exists():
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

    normalize_data_config(cfg.data)
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


def _resolve_dataset_class(cfg: DictConfig):
    dataset_class = cfg.data.dataset_class
    if isinstance(dataset_class, str):
        dataset_class = get_class(dataset_class)
    return dataset_class


_TASKS_DIR = Path(__file__).resolve().parent / "configs" / "tasks"


def _load_task_configs(cfg: DictConfig) -> dict:
    from foundry.tasks.config import TaskConfig

    names = OmegaConf.to_container(cfg.task_configs, resolve=True)
    configs = {}
    for name in names:
        path = _TASKS_DIR / f"{name}.yaml"
        tc = TaskConfig.from_yaml(path)
        configs[tc.name] = tc
    return configs


def _apply_auto_class_weights(
    cfg: DictConfig, datamodule, task_configs: dict
) -> dict:
    class_weights_cfg = OmegaConf.select(cfg, "class_weights", default=None)
    if class_weights_cfg is None:
        return task_configs

    mode = class_weights_cfg.get("mode", None)
    if mode != "auto":
        return task_configs

    datamodule.setup("fit")
    smoothing = class_weights_cfg.get("smoothing", 1.0)
    weights = datamodule.compute_class_weights(smoothing=smoothing)
    for name, class_weights in weights.items():
        task_configs[name].loss["class_weights"] = class_weights
    return task_configs


def _build_model_and_data(cfg: DictConfig):
    _populate_data_driven_hyperparams(cfg)

    task_configs = _load_task_configs(cfg)
    normalize_data_config(cfg.data)
    datamodule = instantiate(cfg.data, tokenizer=None)
    datamodule._task_configs = task_configs
    task_configs = _apply_auto_class_weights(cfg, datamodule, task_configs)

    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }
    model = ModelClass(task_configs=task_configs, **model_kwargs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule.set_tokenizer(tokenizer)

    return model, datamodule


def _build_lightning_module(cfg: DictConfig, model, datamodule):
    return instantiate(cfg.module, model=model)


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


def _validate_checkpoint_policy(
    resume_path: str | None,
    pretrained_path: str | None,
) -> None:
    """Validate that resume and pretrained checkpoints don't conflict.

    When resuming from a checkpoint, all trainer state (model weights,
    optimizer, scheduler, epoch) is restored.  Pretrained transfer should
    only apply when starting a *new* run, since resume already restores
    the model weights that include previously transferred pretrained state.

    Raises:
        ValueError: If both resume and pretrained paths are specified.
    """
    if resume_path and pretrained_path:
        raise ValueError(
            f"Both resume checkpoint ({resume_path}) and pretrained checkpoint "
            f"({pretrained_path}) are specified.  When resuming, all model "
            f"state is restored from the resume checkpoint, making pretrained "
            f"transfer redundant and potentially harmful.  Either remove "
            f"run.pretrained_checkpoint when resuming, or remove the resume "
            f"checkpoint to start fresh with pretrained initialization."
        )


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
    set_seed(
        cfg.run.seed,
        deterministic=OmegaConf.select(cfg, "run.deterministic", default=False),
    )
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

    _log_output_destinations(
        cfg, output_dir, checkpoint_dir, using_wandb_logger
    )
    _stage_data_if_needed(cfg)

    # Eagerly resolve cfg.run so that ${data.subject} (and similar
    # interpolation-only keys) are baked in before normalize_data_config
    # strips them from cfg.data.
    OmegaConf.resolve(cfg.run)

    model, datamodule = _build_model_and_data(cfg)

    pretrained_ckpt = OmegaConf.select(
        cfg, "run.pretrained_checkpoint", default=None
    )
    if pretrained_ckpt:
        freeze = OmegaConf.select(cfg, "run.freeze_pretrained", default=False)
        transfer_mode_str = OmegaConf.select(
            cfg, "run.pretrained_transfer_mode", default="strict"
        )
        transfer_mode = TransferMode(transfer_mode_str)
        load_pretrained_weights(
            model, pretrained_ckpt, freeze=freeze, mode=transfer_mode
        )

    compile_mode = OmegaConf.select(cfg, "run.compile", default=False)
    if compile_mode and torch.cuda.is_available():
        logger.info("Compiling model with torch.compile(mode=%r)", compile_mode)
        model = torch.compile(model, mode=str(compile_mode))

    lightning_module = _build_lightning_module(cfg, model, datamodule)
    trainer = _build_trainer(cfg)

    _log_config_to_wandb(trainer, cfg)

    ckpt_path = _get_resume_checkpoint_path(
        cfg, checkpoint_dir, slurm_restart_count
    )

    _validate_checkpoint_policy(ckpt_path, pretrained_ckpt)

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

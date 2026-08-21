import hashlib
import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
import torch
import torch.multiprocessing
from hydra.core.hydra_config import HydraConfig

from hydra.utils import get_class, instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.tools.stage_data import (
    DEFAULT_COMPRESSED_ROOT,
    DEFAULT_SOURCE_ROOT,
    destination_lock,
    stage_data,
)
from foundry.training.pretrained import TransferMode, load_pretrained_weights

torch.multiprocessing.set_sharing_strategy("file_system")
logger = logging.getLogger(__name__)


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

    # Attach to the run created by the wandb sweep agent (WANDB_RUN_ID).
    if _is_sweep_mode():
        sweep_run_id = os.environ.get("WANDB_RUN_ID")
        if sweep_run_id:
            OmegaConf.update(cfg, "logger.id", sweep_run_id)
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


def _clear_torchinductor_cache() -> None:
    """Remove stale TorchInductor/Triton kernel caches.

    Cached kernels compiled by a different PyTorch version can cause
    ImportErrors (e.g. missing ``triton_helpers``) when ``torch.compile``
    tries to reload them.  Clearing the cache forces a clean recompilation.
    """
    import shutil

    cache_dir = os.path.join(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp"),
        f"torchinductor_{os.environ.get('USER', 'unknown')}",
    )
    if os.path.isdir(cache_dir):
        logger.info("Clearing stale TorchInductor cache at %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _finish_active_wandb_run(exit_code: int = 0) -> None:
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
    wandb.finish(exit_code=exit_code)


def _stage_data_if_needed(cfg: DictConfig) -> None:
    mode = str(OmegaConf.select(cfg, "stage.mode", default="node_local"))
    if mode == "direct":
        logger.info(
            "Data staging disabled (stage.mode=direct); using %s", cfg.data.root
        )
        return
    if mode != "node_local":
        raise ValueError(
            f"Unsupported stage.mode={mode!r}; expected 'node_local' or 'direct'."
        )

    dest_root = OmegaConf.select(cfg, "stage.destination_root", default=None)
    if dest_root is None:
        dest_root = os.environ.get("SLURM_TMPDIR")
    if not dest_root:
        logger.info(
            "No node-local staging destination is available; using configured "
            "data.root=%s",
            cfg.data.root,
        )
        return

    source_root = OmegaConf.select(
        cfg, "stage.source_root", default=DEFAULT_SOURCE_ROOT
    )
    compressed_root = OmegaConf.select(
        cfg, "stage.compressed_root", default=DEFAULT_COMPRESSED_ROOT
    )
    compress = bool(OmegaConf.select(cfg, "stage.compress", default=False))

    with destination_lock(dest_root):
        new_root = stage_data(
            data_cfg=cfg.data,
            source_root=source_root,
            compressed_root=compressed_root,
            dest_root=dest_root,
            compress=compress,
        )
    OmegaConf.update(cfg, "data.root", new_root)
    logger.info("Data staged to %s", new_root)


# -- Component construction ------------------------------------------------


def _is_neuralbench_data(cfg: DictConfig) -> bool:
    """Check if the data config targets a NeuralBenchDataModule."""
    target = OmegaConf.select(cfg, "data._target_", default="")
    return "NeuralBenchDataModule" in target


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

    if _is_neuralbench_data(cfg):
        dm = instantiate(cfg.data, tokenizer=None)
    else:
        normalize_data_config(cfg.data)
        dm = instantiate(cfg.data, tokenizer=None)
    dm.setup("fit")

    if session_configs is None:
        if hasattr(dm, "get_session_configs"):
            session_configs = dm.get_session_configs()
        else:
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
        if hasattr(dm, "get_num_channels"):
            num_channels = dm.get_num_channels()
        else:
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


def _validate_and_apply_focal_loss_weights(
    cfg: DictConfig, datamodule, task_configs: dict, setup_done: bool = False
) -> tuple[dict, bool]:
    """Validate FocalTaskLoss configuration and apply auto-alpha if requested.

    Raises an error if FocalTaskLoss is used with class_weights.mode='auto'.
    If alpha='auto' is set, computes inverse-frequency weights using each
    task's ``alpha_smoothing`` (default ``1.0``).

    Returns:
        (task_configs, setup_done) - tuple where setup_done indicates if datamodule.setup("fit")
        was called and doesn't need to be called again.
    """
    class_weights_cfg = OmegaConf.select(cfg, "class_weights", default=None)
    class_weights_mode = (
        class_weights_cfg.get("mode", None) if class_weights_cfg else None
    )

    # (task_name, alpha_smoothing)
    focal_tasks_with_auto_alpha: list[tuple[str, float]] = []

    for name, task_cfg in task_configs.items():
        loss_cfg = task_cfg.loss

        # Handle both dict and DictConfig
        if isinstance(loss_cfg, dict):
            loss_target = loss_cfg.get("_target_", None)
            alpha_val = loss_cfg.get("alpha", None)
            alpha_smoothing = loss_cfg.get("alpha_smoothing", 1.0)
        else:
            loss_target = OmegaConf.select(loss_cfg, "_target_", default=None)
            alpha_val = OmegaConf.select(loss_cfg, "alpha", default=None)
            alpha_smoothing = OmegaConf.select(
                loss_cfg, "alpha_smoothing", default=1.0
            )

        if loss_target is None or "FocalTaskLoss" not in loss_target:
            continue

        if class_weights_mode == "auto":
            raise ValueError(
                f"Task '{name}' uses FocalTaskLoss with class_weights.mode='auto'. "
                "FocalTaskLoss should not use class_weights; use 'alpha' instead. "
                "Set class_weights.mode='none' or remove it, then set alpha='auto' or "
                "provide an explicit per-class alpha list."
            )

        if alpha_val == "auto":
            focal_tasks_with_auto_alpha.append((name, float(alpha_smoothing)))

    if focal_tasks_with_auto_alpha:
        logger.info(
            "FocalTaskLoss alpha='auto' for %s: running datamodule setup + "
            "class-frequency scan (same cost as class_weights.mode='auto'). "
            "Use an explicit alpha list to skip this.",
            [name for name, _ in focal_tasks_with_auto_alpha],
        )
        if not setup_done:
            datamodule.setup("fit")
            setup_done = True

        # Group by smoothing to avoid redundant weight computation
        smoothing_to_tasks: dict[float, list[str]] = {}
        for name, alpha_smoothing in focal_tasks_with_auto_alpha:
            smoothing_to_tasks.setdefault(alpha_smoothing, []).append(name)

        smoothing_to_weights = {
            smoothing: datamodule.compute_class_weights(smoothing=smoothing)
            for smoothing in smoothing_to_tasks
        }

        for name, alpha_smoothing in focal_tasks_with_auto_alpha:
            weights = smoothing_to_weights[alpha_smoothing]
            if name in weights:
                task_configs[name].loss["alpha"] = weights[name]
                logger.info(
                    "Applied auto-computed alpha weights to FocalTaskLoss for "
                    "task %r (alpha_smoothing=%.3f): %s",
                    name,
                    alpha_smoothing,
                    weights[name],
                )
            else:
                logger.warning(
                    "Task %r not found in computed weights. Available tasks: %s",
                    name,
                    list(weights.keys()),
                )

    return task_configs, setup_done


def _apply_auto_class_weights(
    cfg: DictConfig, datamodule, task_configs: dict, setup_done: bool = False
) -> tuple[dict, bool]:
    """Apply auto-computed class weights to losses that support them.

    Returns:
        (task_configs, setup_done) - tuple where setup_done indicates if datamodule.setup("fit")
        was called and doesn't need to be called again.
    """
    class_weights_cfg = OmegaConf.select(cfg, "class_weights", default=None)
    if class_weights_cfg is None:
        return task_configs, setup_done

    mode = class_weights_cfg.get("mode", None)
    if mode != "auto":
        return task_configs, setup_done

    if not setup_done:
        datamodule.setup("fit")
        setup_done = True
    smoothing = class_weights_cfg.get("smoothing", 1.0)
    weights = datamodule.compute_class_weights(smoothing=smoothing)
    for name, class_weights in weights.items():
        task_configs[name].loss["class_weights"] = class_weights
    return task_configs, setup_done


def _build_model_and_data(cfg: DictConfig):
    """Construct the model and data module from the Hydra config.

    Handles hyperparameter auto-population from the dataset, task config
    loading, class weight computation, session embedding config unpacking,
    and context cache attachment for dynamic session embedding mode.

    Returns:
        ``(model, datamodule)`` tuple ready for the Lightning trainer.
    """
    _populate_data_driven_hyperparams(cfg)

    task_configs = _load_task_configs(cfg)
    if not _is_neuralbench_data(cfg):
        normalize_data_config(cfg.data)
    datamodule = instantiate(cfg.data, tokenizer=None)
    datamodule._task_configs = task_configs

    # Resolve FocalTaskLoss alpha='auto' before instantiate; track setup state
    task_configs, setup_done = _validate_and_apply_focal_loss_weights(
        cfg, datamodule, task_configs
    )

    # Apply auto class weights for CrossEntropy; reuse setup state from above
    task_configs, _ = _apply_auto_class_weights(
        cfg, datamodule, task_configs, setup_done=setup_done
    )

    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }
    session_emb_cfg = model_kwargs.pop("session_emb", None)
    if session_emb_cfg is not None:
        if OmegaConf.is_config(session_emb_cfg):
            session_emb_cfg = OmegaConf.to_container(
                session_emb_cfg, resolve=True
            )
        # session_context is consumed later by set_context_cache(); keep it
        # out of the constructor kwargs.
        session_emb_cfg.pop("session_context", None)
        model_kwargs.update(session_emb_cfg)

    model = ModelClass(task_configs=task_configs, **model_kwargs)

    if getattr(model, "session_emb_mode", None) == "dynamic":
        session_context_cfg = OmegaConf.select(
            cfg, "model.session_emb.session_context", default=None
        )
        if session_context_cfg is not None:
            from foundry.models.session_embedding import SessionContextCache

            model.set_context_cache(
                SessionContextCache(
                    num_windows=session_context_cfg.num_context_windows,
                    context_source=session_context_cfg.context_source,
                    context_duration=session_context_cfg.context_duration,
                )
            )

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule.set_tokenizer(tokenizer)

    return model, datamodule


def _build_lightning_module(cfg: DictConfig, model, datamodule):
    """Instantiate the :class:`FoundryModule` Lightning wrapper from config."""
    return instantiate(cfg.module, model=model)


def _build_trainer(cfg: DictConfig):
    """Instantiate the Lightning :class:`Trainer` from config.

    Converts callback dicts to lists when Hydra composes them as a mapping.
    """
    if OmegaConf.is_dict(cfg.trainer.get("callbacks")):
        cfg.trainer.callbacks = list(cfg.trainer.callbacks.values())
    return instantiate(cfg.trainer)


# -- Checkpointing ---------------------------------------------------------


def _get_resume_checkpoint_path(
    cfg: DictConfig,
    checkpoint_dir: str,
    slurm_restart_count: int,
) -> str | None:
    """Resolve the checkpoint path for resuming training, if any.

    Priority: SLURM restart (automatic resume) > config flag
    ``run.resume_if_checkpoint_exists``.  Returns ``None`` when no
    resume is appropriate.
    """
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
    from hydra_plugins.foundry_launcher.launch_snapshot import (
        get_slurm_job_identifiers,
        get_snapshot_provenance_for_wandb,
    )

    config_to_log.update(get_slurm_job_identifiers())
    provenance = get_snapshot_provenance_for_wandb()
    if provenance:
        config_to_log.update(provenance)

    trainer.logger.experiment.config.update(
        config_to_log, allow_val_change=True
    )


def _is_sweep_mode() -> bool:
    """Check if running under WandB sweep."""
    return "WANDB_SWEEP_ID" in os.environ


def _inject_sweep_hyperparams(cfg: DictConfig) -> None:
    """Inject hyperparameters from WandB sweep config into Hydra config.

    When running as a WandB sweep agent, the sweep system populates
    wandb.config with the current trial's hyperparameters. This function
    injects those into the Hydra config so they override defaults/CLI args.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not available; skipping sweep param injection")
        return

    if not _is_sweep_mode():
        return

    if wandb.run is None:
        return

    # Pull all wandb.config values and inject into cfg
    sweep_config = dict(wandb.config)
    logger.info("Injecting %d sweep hyperparameters", len(sweep_config))

    for key, value in sweep_config.items():
        try:
            OmegaConf.update(cfg, key, value, force_add=True)
            logger.debug("Injected sweep param: %s = %s", key, value)
        except Exception as e:
            logger.warning(
                "Failed to inject sweep param %s = %s: %s",
                key,
                value,
                e,
            )


# -- Snapshot provenance ----------------------------------------------------


def _log_snapshot_provenance() -> None:
    """Log snapshot identity at startup and verify imports if active."""
    bundle_dir = os.environ.get("FOUNDRY_SNAPSHOT_BUNDLE_DIR")
    if not bundle_dir:
        return

    git_sha = os.environ.get("FOUNDRY_SNAPSHOT_GIT_SHA", "unknown")
    source_dir = os.environ.get("FOUNDRY_SNAPSHOT_SOURCE_DIR", "unknown")
    bundle_id = os.environ.get("FOUNDRY_SNAPSHOT_BUNDLE_ID", "unknown")
    source_digest = os.environ.get("FOUNDRY_SNAPSHOT_SOURCE_DIGEST", "unknown")
    manifest = os.environ.get("FOUNDRY_SNAPSHOT_MANIFEST", "unknown")

    logger.info(
        "Snapshot provenance:\n"
        "  Bundle ID      : %s\n"
        "  Git SHA        : %s\n"
        "  Source digest  : %s\n"
        "  Source dir     : %s\n"
        "  Manifest       : %s\n"
        "  SLURM job      : %s (task %s, restart %s)",
        bundle_id,
        git_sha,
        source_digest[:16] if len(source_digest) > 16 else source_digest,
        source_dir,
        manifest,
        os.environ.get("SLURM_JOB_ID", "n/a"),
        os.environ.get("SLURM_ARRAY_TASK_ID", "n/a"),
        os.environ.get("SLURM_RESTART_COUNT", "0"),
    )

    from hydra_plugins.foundry_launcher.launch_snapshot import (
        LaunchSnapshot,
        verify_snapshot,
        verify_import_paths,
    )

    if os.environ.get("FOUNDRY_SNAPSHOT_VERIFY_ON_WORKER", "1") == "1":
        descriptor_path = Path(manifest).with_name("snapshot-descriptor.json")
        verify_snapshot(LaunchSnapshot.from_json(descriptor_path.read_text()))
    verify_import_paths(source_dir)


def _write_snapshot_task_provenance(output_dir: str) -> None:
    """Write snapshot provenance after Hydra has created the task output dir."""
    manifest_path = os.environ.get("FOUNDRY_SNAPSHOT_MANIFEST")
    if not manifest_path:
        return

    from hydra_plugins.foundry_launcher.launch_snapshot import (
        LaunchSnapshot,
        write_task_provenance,
    )

    snapshot = LaunchSnapshot.from_json(
        Path(manifest_path).with_name("snapshot-descriptor.json").read_text()
    )
    # Hydra leaves ``job.num`` as a mandatory-but-unresolved value for a
    # one-item local multirun.  That task still maps to the first snapshot
    # config, so use index 0 when no explicit job number is available.
    task_index = int(
        OmegaConf.select(
            HydraConfig.get(), "job.num", default=0, throw_on_missing=False
        )
    )
    task_config_path = (
        Path(snapshot.bundle_dir)
        / "task-configs"
        / f"task_{task_index:04d}.json"
    )
    task_overrides = []
    if task_config_path.is_file():
        import json

        task_overrides = json.loads(task_config_path.read_text())["overrides"]

    write_task_provenance(snapshot, task_index, task_overrides, output_dir)


# -- Entry point ------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    """Hydra entry point: configure, build, and run a training session.

    Orchestrates logging setup, SLURM resume detection, WandB configuration,
    data staging, model/data construction, optional pretrained weight
    transfer, ``torch.compile``, training, and optional best-checkpoint test
    evaluation.
    """
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
    from hydra_plugins.foundry_launcher.launch_snapshot import (
        get_slurm_job_identifiers,
    )

    slurm_ids = get_slurm_job_identifiers()
    if slurm_ids:
        logger.info(
            "SLURM job_id=%s raw_job_id=%s restart_count=%s",
            slurm_ids["slurm_job_id"],
            slurm_ids.get("slurm_raw_job_id"),
            slurm_restart_count,
        )

    _log_snapshot_provenance()

    using_wandb_logger = _is_wandb_logger_enabled(cfg)
    if using_wandb_logger:
        _finish_active_wandb_run()

    output_dir, checkpoint_dir = _configure_output_paths(cfg)
    _write_snapshot_task_provenance(output_dir)
    _configure_wandb(cfg, output_dir)

    # Inject WandB sweep hyperparameters if running under sweep
    _inject_sweep_hyperparams(cfg)

    _log_output_destinations(
        cfg, output_dir, checkpoint_dir, using_wandb_logger
    )
    if not _is_neuralbench_data(cfg):
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
    elif OmegaConf.select(cfg, "run.freeze_backbone", default=False):
        if hasattr(model, "transferable_components"):
            frozen_count = 0
            for comp_name in model.transferable_components():
                component = getattr(model, comp_name, None)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = False
                        frozen_count += 1
            logger.info(
                "Froze %d backbone parameters (freeze_backbone=true, no checkpoint).",
                frozen_count,
            )

    compile_mode = OmegaConf.select(cfg, "run.compile", default=False)
    if compile_mode and torch.cuda.is_available():
        _clear_torchinductor_cache()
        logger.info("Compiling model with torch.compile(mode=%r)", compile_mode)
        model = torch.compile(model, mode=str(compile_mode))

    lightning_module = _build_lightning_module(cfg, model, datamodule)
    trainer = _build_trainer(cfg)

    _log_config_to_wandb(trainer, cfg)

    ckpt_path = _get_resume_checkpoint_path(
        cfg, checkpoint_dir, slurm_restart_count
    )

    _validate_checkpoint_policy(ckpt_path, pretrained_ckpt)

    run_failed = False
    try:
        trainer.fit(
            lightning_module,
            datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        if OmegaConf.select(cfg, "run.evaluate_test", default=False):
            logger.info(
                "Evaluating the best validation checkpoint on the test split."
            )
            trainer.test(
                lightning_module,
                datamodule=datamodule,
                ckpt_path="best",
            )
    except BaseException:
        run_failed = True
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if using_wandb_logger:
            _finish_active_wandb_run(exit_code=1 if run_failed else 0)


if __name__ == "__main__":
    register_resolvers()
    main()

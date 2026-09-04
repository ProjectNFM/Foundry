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
from foundry.core import VocabManager
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.tools.stage_data import (
    DEFAULT_COMPRESSED_ROOT,
    DEFAULT_SOURCE_ROOT,
    destination_lock,
    stage_data,
)
from foundry.training.pretrained import TransferMode, load_pretrained_weights
from foundry.training.checkpoint_manifest import (
    load_checkpoint_manifest,
    verify_checkpoint_integrity,
)

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
    # Node-pool attempts need a unique, parent-recorded identity even if an
    # experiment config happened to provide a reusable logger ID.
    foundry_run_id = os.environ.get("FOUNDRY_WANDB_RUN_ID")
    if foundry_run_id:
        OmegaConf.update(cfg, "logger.id", foundry_run_id)
        return

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

    # Manifest-backed and training-fraction datamodules validate their
    # intervals against task mappings during setup.  Attach those mappings
    # before the discovery setup rather than waiting for the later model/data
    # construction path to do so.
    task_configs = _load_task_configs(cfg)

    if _is_neuralbench_data(cfg):
        dm = instantiate(cfg.data, tokenizer=None)
    else:
        normalize_data_config(cfg.data)
        dm = instantiate(cfg.data, tokenizer=None)
    dm._task_configs = task_configs
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
    task_configs, setup_done = _apply_auto_class_weights(
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

    # DataLoader workers apply ``model.tokenize`` in their own processes.  Lazy
    # vocabularies must consequently be initialized here, before attaching the
    # tokenizer, rather than waiting for Lightning's on_fit_start callback.
    if isinstance(model, VocabManager) and model.has_lazy_vocabs():
        if not setup_done:
            datamodule.setup("fit")
        vocab_info = {}
        for method_name, key in [
            ("get_recording_ids", "session_ids"),
            ("get_channel_ids", "channel_ids"),
        ]:
            if hasattr(datamodule, method_name):
                vocab_info[key] = getattr(datamodule, method_name)()
        model.initialize_vocabs(vocab_info)

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


def _resolve_pretrained_components(
    model: torch.nn.Module,
    cfg: DictConfig,
) -> tuple[str, ...] | None:
    """Resolve an optional model-declared pretrained transfer regime."""
    regime = OmegaConf.select(
        cfg, "run.pretrained_transfer_regime", default=None
    )
    if regime is None:
        return None

    components_for_mode = getattr(
        model, "transferable_components_for_mode", None
    )
    if not callable(components_for_mode):
        raise ValueError(
            f"Model {type(model).__name__} does not support named pretrained "
            f"transfer regimes; cannot use {regime!r}."
        )
    components = tuple(components_for_mode(str(regime)))
    if not components:
        raise ValueError(
            f"Pretrained transfer regime {regime!r} selected no components."
        )
    return components


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


def _log_normalization_artifacts_to_wandb(
    trainer, artifacts: dict[str, str] | None
) -> None:
    """Upload the immutable input-normalization artifacts when WandB is active."""
    if artifacts is None or not isinstance(trainer.logger, WandbLogger):
        return
    import wandb

    trainer.logger.experiment.config.update(
        {
            "input_normalization/stats_sha256": artifacts["stats_sha256"],
            "input_normalization/train_interval_hash": artifacts[
                "train_interval_hash"
            ],
        },
        allow_val_change=True,
    )
    artifact = wandb.Artifact(
        name=f"input-normalization-{artifacts['stats_sha256'][:16]}",
        type="input-normalization",
    )
    artifact.add_file(artifacts["stats_path"])
    artifact.add_file(artifacts["manifest_path"])
    trainer.logger.experiment.log_artifact(artifact)


def _parse_bids_components(recording_id: str) -> dict[str, str]:
    """Extract BIDS components (sub, ses, acq, etc.) from a recording ID."""
    components = {}
    for part in recording_id.split("_"):
        if "-" in part:
            key, _, val = part.partition("-")
            components[key] = val
    return components


def _derive_species(datamodule) -> str | None:
    """Derive species from the dataset class name."""
    class_name = datamodule.dataset_class.__name__.lower()
    if "minipig" in class_name:
        return "minipigs"
    if "monkey" in class_name:
        return "monkeys"
    return None


def _derive_target_subject(datamodule) -> str:
    """Return the sole BIDS subject in a downstream transfer datamodule."""
    if getattr(datamodule, "dataset", None) is None:
        datamodule.setup("fit")
    recording_ids = list(getattr(datamodule.dataset, "recording_ids", []))
    subjects = {
        f"sub-{subject}"
        for recording_id in recording_ids
        if (subject := _parse_bids_components(str(recording_id)).get("sub"))
        is not None
    }
    if len(subjects) != 1:
        raise ValueError(
            "Manifest-based NeuroSoft transfer requires exactly one target "
            f"subject, got recording IDs {recording_ids!r}"
        )
    return next(iter(subjects))


def _validate_manifest_target(manifest: dict, datamodule) -> None:
    """Require a transfer manifest to exclude this exact downstream target."""
    trained_on = manifest.get("trained_on")
    excluded = (
        trained_on.get("excluded_target")
        if isinstance(trained_on, dict)
        else None
    )
    if not isinstance(excluded, dict):
        raise ValueError(
            "Checkpoint manifest is missing trained_on.excluded_target"
        )
    manifest_species = excluded.get("species")
    manifest_subject = excluded.get("subject")
    species = _derive_species(datamodule)
    subject = _derive_target_subject(datamodule)
    if not isinstance(manifest_species, str) or not isinstance(
        manifest_subject, str
    ):
        raise ValueError(
            "Checkpoint manifest excluded_target must contain string species "
            "and subject fields"
        )
    if species != manifest_species or subject != manifest_subject:
        raise ValueError(
            "Checkpoint manifest target does not match the downstream target: "
            f"manifest={manifest_species}/{manifest_subject}, "
            f"downstream={species}/{subject}"
        )


def _prepare_fraction_provenance(
    cfg: DictConfig,
    datamodule,
    output_dir: str,
) -> dict | None:
    """Build and write fraction provenance JSON if manifests are available.

    Returns the provenance dict for WandB logging, or None.
    """
    import json

    manifests = datamodule.fraction_manifests
    if not manifests:
        return None

    split_hashes = datamodule.fraction_split_hashes
    split_class_counts = datamodule.fraction_split_class_counts
    audit_records = datamodule.fraction_audit_records
    species = _derive_species(datamodule)

    recording_ids = list(manifests.keys())
    if len(recording_ids) != 1:
        logger.warning(
            "Fraction provenance expects single-session; got %d recordings",
            len(recording_ids),
        )

    rid = recording_ids[0]
    manifest = manifests[rid]
    bids = _parse_bids_components(rid)

    from hydra_plugins.foundry_launcher.launch_snapshot import (
        get_snapshot_provenance_for_wandb,
    )

    snapshot_prov = get_snapshot_provenance_for_wandb() or {}

    audit_record = audit_records.get(rid, {})
    audit_hash = datamodule.fraction_audit_artifact_sha256

    provenance = {
        "species": species,
        "subject": bids.get("sub"),
        "session": bids.get("ses"),
        "recording_id": rid,
        "dataset_class": datamodule.dataset_class.__name__,
        "split_type": datamodule.dataset_kwargs.get("split_type"),
        "model_seed": int(cfg.run.seed),
        "fraction_seed": int(
            datamodule.training_fraction_seed
            if datamodule.training_fraction_seed is not None
            else datamodule.seed
        ),
        "training_fraction_requested": manifest.requested_fraction,
        "training_fraction_realized": manifest.realized_fraction,
        "fraction_manifest": manifest.to_dict(),
        "split_hashes": {
            "actual": split_hashes.get(rid, {}),
            "expected": audit_record.get("split_hashes", {}),
        },
        "split_class_counts": {
            "actual": split_class_counts.get(rid, {}),
            "expected": audit_record.get("per_class_counts", {}),
        },
        "split_present_classes": {
            "actual": {
                split: [
                    class_name
                    for class_name, count in counts.items()
                    if count > 0
                ]
                for split, counts in split_class_counts.get(rid, {}).items()
            },
            "expected": {
                split: [
                    class_name
                    for class_name, count in counts.items()
                    if count > 0
                ]
                for split, counts in audit_record.get(
                    "per_class_counts", {}
                ).items()
            },
        },
        "train_present_classes": manifest.present_classes,
        "train_absent_classes": manifest.absent_classes,
        "train_per_class_counts": manifest.per_class_counts,
        "num_present_classes": len(manifest.present_classes),
        "manifest_hash": manifest.manifest_hash,
        "source_intervals_hash": manifest.source_intervals_hash,
        "audit_artifact_sha256": audit_hash,
        "snapshot_provenance": snapshot_prov,
    }

    provenance_path = os.path.join(output_dir, "neurosoft_provenance.json")
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=True)
    logger.info("Wrote fraction provenance to %s", provenance_path)

    return provenance


def _log_neurosoft_provenance_to_wandb(
    trainer, provenance: dict, output_dir: str
) -> None:
    """Upload provenance as WandB artifact and add queryable config fields."""
    if not isinstance(trainer.logger, WandbLogger):
        return

    import wandb

    queryable = {
        "neurosoft/species": provenance.get("species"),
        "neurosoft/subject": provenance.get("subject"),
        "neurosoft/session": provenance.get("session"),
        "neurosoft/recording_id": provenance.get("recording_id"),
        "neurosoft/training_fraction_requested": provenance.get(
            "training_fraction_requested"
        ),
        "neurosoft/training_fraction_realized": provenance.get(
            "training_fraction_realized"
        ),
        "neurosoft/model_seed": provenance.get("model_seed"),
        "neurosoft/fraction_seed": provenance.get("fraction_seed"),
        "neurosoft/split_type": provenance.get("split_type"),
        "neurosoft/present_classes": provenance.get("train_present_classes"),
        "neurosoft/absent_classes": provenance.get("train_absent_classes"),
        "neurosoft/num_present_classes": provenance.get("num_present_classes"),
        "neurosoft/per_class_counts": provenance.get("train_per_class_counts"),
        "neurosoft/manifest_hash": provenance.get("manifest_hash"),
        "neurosoft/source_intervals_hash": provenance.get(
            "source_intervals_hash"
        ),
        "neurosoft/runtime_split_hashes": provenance.get(
            "split_hashes", {}
        ).get("actual"),
        "neurosoft/audit_expected_split_hashes": provenance.get(
            "split_hashes", {}
        ).get("expected"),
        "neurosoft/runtime_split_class_counts": provenance.get(
            "split_class_counts", {}
        ).get("actual"),
        "neurosoft/audit_expected_split_class_counts": provenance.get(
            "split_class_counts", {}
        ).get("expected"),
        "neurosoft/audit_artifact_sha256": provenance.get(
            "audit_artifact_sha256"
        ),
        "neurosoft/eligible": True,
    }
    trainer.logger.experiment.config.update(queryable, allow_val_change=True)

    provenance_path = os.path.join(output_dir, "neurosoft_provenance.json")
    if os.path.isfile(provenance_path):
        artifact = wandb.Artifact(
            name=f"neurosoft-provenance-{provenance.get('recording_id', 'unknown')}",
            type="provenance",
        )
        artifact.add_file(provenance_path)
        trainer.logger.experiment.log_artifact(artifact)
        logger.info("Uploaded neurosoft provenance artifact to WandB")


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
    task_index_raw = os.environ.get("FOUNDRY_SNAPSHOT_TASK_INDEX")
    task_index = int(
        task_index_raw
        if task_index_raw is not None
        else OmegaConf.select(
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


def _resolve_manifest_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a configured manifest from Hydra's transient run directory."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        try:
            from hydra.utils import get_original_cwd

            candidate = Path(get_original_cwd()) / candidate
        except ValueError:
            pass
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint manifest not found: {path}")


def _load_and_validate_checkpoint_manifest(
    cfg: DictConfig,
    datamodule,
) -> dict | None:
    """Load a checkpoint manifest, validate provenance, and return it.

    Returns ``None`` when ``run.pretrained_checkpoint_manifest`` is not set.
    Raises on provenance mismatches so no bad transfer can proceed.
    """
    manifest_path = OmegaConf.select(
        cfg, "run.pretrained_checkpoint_manifest", default=None
    )
    if manifest_path is None:
        return None

    pretrained_ckpt = OmegaConf.select(
        cfg, "run.pretrained_checkpoint", default=None
    )
    if pretrained_ckpt:
        raise ValueError(
            "run.pretrained_checkpoint and run.pretrained_checkpoint_manifest "
            "are mutually exclusive."
        )

    resolved_manifest_path = _resolve_manifest_path(manifest_path)
    manifest = load_checkpoint_manifest(resolved_manifest_path)

    checkpoint_root = os.environ.get("FOUNDRY_CHECKPOINT_ROOT")
    if checkpoint_root:
        verify_checkpoint_integrity(manifest, checkpoint_root)
    else:
        manifest_dir = str(resolved_manifest_path.parent.parent)
        verify_checkpoint_integrity(manifest, manifest_dir)

    _validate_manifest_target(manifest, datamodule)
    trained_on = manifest["trained_on"]
    excluded = trained_on["excluded_target"]

    logger.info(
        "Loaded checkpoint manifest: kind=%s monitor=%s score=%.4f "
        "excluded_target=%s/%s source=%s",
        manifest["checkpoint"]["kind"],
        manifest["selection"]["monitor"],
        manifest["selection"]["monitor_value"],
        excluded.get("species"),
        excluded.get("subject"),
        trained_on.get("source_selection_id"),
    )

    return manifest


def _apply_manifest_transfer(
    model: torch.nn.Module,
    manifest: dict,
    cfg: DictConfig,
    output_dir: str,
) -> None:
    """Apply pretrained weights from a checkpoint manifest.

    Resolves the checkpoint path, loads weights using the existing strict
    transfer pipeline, and persists the transfer report.
    """
    checkpoint_root = os.environ.get("FOUNDRY_CHECKPOINT_ROOT")
    checkpoint_info = manifest["checkpoint"]
    rel_path = checkpoint_info["path"]

    if checkpoint_root:
        ckpt_path = Path(checkpoint_root) / rel_path
    else:
        manifest_path = _resolve_manifest_path(
            OmegaConf.select(cfg, "run.pretrained_checkpoint_manifest")
        )
        ckpt_path = manifest_path.parent.parent / rel_path

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at resolved path: {ckpt_path}"
        )

    regime = OmegaConf.select(
        cfg, "run.pretrained_transfer_regime", default=None
    )
    if regime is None:
        raise ValueError(
            "run.pretrained_transfer_regime must be set when using "
            "pretrained_checkpoint_manifest (e.g. 'full_finetuning' or "
            "'frozen_representation')"
        )

    components = _resolve_pretrained_components(model, cfg)
    freeze = regime == "frozen_representation"

    report = load_pretrained_weights(
        model,
        ckpt_path,
        freeze=freeze,
        mode=TransferMode.STRICT,
        components=components,
    )

    report_dict = {
        "source_checkpoint_manifest": str(
            OmegaConf.select(cfg, "run.pretrained_checkpoint_manifest")
        ),
        "source_checkpoint_path": str(ckpt_path),
        "source_checkpoint_sha256": checkpoint_info["sha256"],
        "transfer_regime": regime,
        "components": list(components) if components else [],
        "loaded": report.loaded,
        "skipped_excluded": report.skipped_excluded,
        "missing_in_checkpoint": report.missing_in_checkpoint,
        "unexpected_in_checkpoint": report.unexpected_in_checkpoint,
        "shape_mismatched": report.shape_mismatched,
        "dtype_mismatched": report.dtype_mismatched,
    }

    import json as _json

    report_json_path = os.path.join(output_dir, "transfer-report.json")
    with open(report_json_path, "w") as f:
        _json.dump(report_dict, f, indent=2, ensure_ascii=True)

    report_md_lines = [
        "# Transfer Report",
        "",
        f"- **Regime:** {regime}",
        f"- **Checkpoint:** `{ckpt_path}`",
        f"- **SHA-256:** `{checkpoint_info['sha256']}`",
        f"- **Loaded:** {len(report.loaded)}",
        f"- **Excluded (by design):** {len(report.skipped_excluded)}",
        f"- **Missing in checkpoint:** {len(report.missing_in_checkpoint)}",
        f"- **Shape mismatched:** {len(report.shape_mismatched)}",
        f"- **Dtype mismatched:** {len(report.dtype_mismatched)}",
        "",
        "## Loaded Parameters",
        "",
    ]
    for key in report.loaded[:50]:
        report_md_lines.append(f"- `{key}`")
    if len(report.loaded) > 50:
        report_md_lines.append(f"- ... and {len(report.loaded) - 50} more")
    report_md_lines.extend(["", "## Excluded Parameters", ""])
    for key in report.skipped_excluded[:50]:
        report_md_lines.append(f"- `{key}`")
    if len(report.skipped_excluded) > 50:
        report_md_lines.append(
            f"- ... and {len(report.skipped_excluded) - 50} more"
        )

    report_md_path = os.path.join(output_dir, "transfer-report.md")
    with open(report_md_path, "w") as f:
        f.write("\n".join(report_md_lines))

    logger.info(
        "Transfer from manifest: loaded=%d excluded=%d regime=%s report=%s",
        len(report.loaded),
        len(report.skipped_excluded),
        regime,
        report_json_path,
    )


def _configure_source_compute_callbacks(
    trainer, datamodule, cfg: DictConfig
) -> None:
    """Set realized_train_windows_per_epoch on the ComputeTrackingCallback.

    Called after the source datamodule is set up so the manifest summary is
    available. This enables effective-epoch computation in compute tracking.
    """
    from foundry.training.callbacks.compute import ComputeTrackingCallback

    manifest = getattr(datamodule, "_source_manifest", None)
    if manifest is None:
        return

    realized_windows = getattr(
        manifest.summary, "realized_train_windows_per_epoch", None
    )
    if realized_windows is None:
        return

    for callback in trainer.callbacks:
        if isinstance(callback, ComputeTrackingCallback):
            callback.realized_train_windows_per_epoch = int(realized_windows)
            logger.info(
                "Configured ComputeTrackingCallback: "
                "realized_train_windows_per_epoch=%d",
                realized_windows,
            )
            break


def _emit_source_checkpoint_manifests(
    trainer,
    cfg: DictConfig,
    datamodule,
    output_dir: str,
    normalization_artifacts: dict | None,
) -> None:
    """Write JSON/Markdown checkpoint manifests for best and milestone checkpoints.

    Called after ``trainer.fit()`` in source-pretraining mode. Gathers metadata
    from callbacks, the source manifest, and run config, then writes manifest
    files for the best checkpoint and each saved milestone.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    from foundry.training.callbacks.compute import ComputeTrackingCallback
    from foundry.training.callbacks.compute_milestone import (
        ComputeMilestoneCheckpointCallback,
    )
    from foundry.training.callbacks.source_session_metrics import (
        SourceSessionMetricsCallback,
    )
    from foundry.training.checkpoint_manifest import write_checkpoint_manifest

    source_manifest = getattr(datamodule, "_source_manifest", None)
    manifest_dir = os.path.join(output_dir, "manifests")

    compute_cb = None
    milestone_cb = None
    session_metrics_cb = None
    model_ckpt_cb = None
    for callback in trainer.callbacks:
        if isinstance(callback, ComputeTrackingCallback):
            compute_cb = callback
        elif isinstance(callback, ComputeMilestoneCheckpointCallback):
            milestone_cb = callback
        elif isinstance(callback, SourceSessionMetricsCallback):
            session_metrics_cb = callback
        elif isinstance(callback, ModelCheckpoint):
            model_ckpt_cb = callback

    git_sha = os.environ.get("FOUNDRY_SNAPSHOT_GIT_SHA", "unknown")
    snapshot_bundle = os.environ.get("FOUNDRY_SNAPSHOT_BUNDLE", "unknown")

    from hydra_plugins.foundry_launcher.launch_snapshot import (
        get_slurm_job_identifiers,
    )

    slurm_ids = get_slurm_job_identifiers()
    slurm_job_id = (
        slurm_ids.get("slurm_job_id", "unknown") if slurm_ids else "unknown"
    )

    wandb_info = {"project": "unknown", "group": "unknown", "run_id": "unknown"}
    if trainer.logger is not None:
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(trainer.logger, WandbLogger):
            exp = trainer.logger.experiment
            if exp is not None:
                wandb_info = {
                    "project": getattr(exp, "project", "unknown"),
                    "group": OmegaConf.select(
                        cfg, "run.group", default="unknown"
                    ),
                    "run_id": getattr(exp, "id", "unknown"),
                }

    norm_hashes: dict[str, str] = {}
    if normalization_artifacts and isinstance(normalization_artifacts, dict):
        for key, value in normalization_artifacts.items():
            if isinstance(value, dict) and "hash" in value:
                norm_hashes[str(key)] = str(value["hash"])
            elif isinstance(value, str):
                norm_hashes[str(key)] = value

    recipe = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "hyperparameters": OmegaConf.to_container(
            cfg.hyperparameters, resolve=True
        ),
        "trainer_precision": str(trainer.precision),
    }

    def _build_trained_on(compute_snap: dict) -> dict:
        trained_on: dict = {}
        if source_manifest is not None:
            trained_on["source_selection_id"] = source_manifest.selection_id
            trained_on["source_manifest_path"] = str(
                getattr(datamodule, "selection_manifest_path", "unknown")
            )
            trained_on["source_manifest_hash"] = source_manifest.manifest_hash
            trained_on["excluded_target"] = {
                "species": source_manifest.target_species,
                "subject": source_manifest.target_subject,
            }
            trained_on["subjects"] = list(source_manifest.subjects)
            trained_on["recordings"] = [
                r.canonical_recording_id for r in source_manifest.recordings
            ]
            trained_on["selected_train_examples"] = (
                source_manifest.summary.selected_train_examples
            )
            trained_on["available_train_windows"] = (
                source_manifest.summary.available_train_windows
            )
            trained_on["realized_train_windows_per_epoch"] = (
                source_manifest.summary.realized_train_windows_per_epoch
            )
            trained_on["class_union"] = list(
                source_manifest.summary.represented_class_union
            )
            trained_on["class_intersection"] = list(
                source_manifest.summary.represented_class_intersection
            )
        else:
            trained_on["excluded_target"] = {
                "species": "unknown",
                "subject": "unknown",
            }

        trained_on["processed_windows"] = compute_snap.get(
            "processed_windows", 0
        )
        ee = compute_snap.get("effective_epochs")
        trained_on["completed_effective_epochs"] = (
            round(ee, 4) if ee is not None else 0.0
        )
        trained_on["optimizer_steps"] = compute_snap.get("optimizer_steps", 0)
        return trained_on

    def _build_selection(compute_snap: dict) -> dict:
        selection: dict = {
            "monitor": (compute_cb.monitor if compute_cb else "unknown"),
            "monitor_value": compute_snap.get("monitor_value", 0.0),
            "source_session_scores": {},
        }
        if session_metrics_cb is not None:
            selection["source_session_scores"] = dict(
                session_metrics_cb._best_session_scores
            )
        return selection

    def _build_compute(compute_snap: dict) -> dict:
        return {
            "cumulative_flops": compute_snap.get("cumulative_flops", 0),
            "flop_method": compute_snap.get("flop_method", "none"),
            "signal_seconds": compute_snap.get("signal_seconds", 0.0),
            "wall_time_seconds": compute_snap.get("wall_time_seconds", 0.0),
            "gpu": compute_snap.get("gpu", "unknown"),
            "precision": compute_snap.get("precision", "unknown"),
        }

    written_manifests: list[str] = []

    if model_ckpt_cb is not None and model_ckpt_cb.best_model_path:
        best_path = model_ckpt_cb.best_model_path
        if os.path.isfile(best_path):
            if compute_cb is not None:
                snap = compute_cb.get_best_compute_snapshot()
                snap["precision"] = str(trainer.precision)
            else:
                snap = {
                    "processed_windows": 0,
                    "optimizer_steps": trainer.global_step,
                    "monitor_value": 0.0,
                }

            try:
                json_path, md_path = write_checkpoint_manifest(
                    best_path,
                    manifest_dir,
                    kind="best",
                    trained_on=_build_trained_on(snap),
                    selection=_build_selection(snap),
                    compute=_build_compute(snap),
                    recipe=recipe,
                    normalization_artifact_hashes=norm_hashes,
                    git_sha=git_sha,
                    snapshot_bundle=snapshot_bundle,
                    slurm_job_id=slurm_job_id,
                    wandb_info=wandb_info,
                )
                written_manifests.append(str(json_path))
                logger.info("Wrote best checkpoint manifest: %s", json_path)
            except Exception:
                logger.warning(
                    "Failed to write best checkpoint manifest",
                    exc_info=True,
                )

    if milestone_cb is not None:
        for step, info in sorted(milestone_cb.get_saved_milestones().items()):
            ckpt_path = info.get("path")
            if not ckpt_path or not os.path.isfile(ckpt_path):
                continue

            snap = info.get("compute_snapshot", {})
            snap.setdefault("precision", str(trainer.precision))
            snap.setdefault("optimizer_steps", step)

            realized_pct = info.get("realized_pct", 0.0)
            kind = f"milestone-{realized_pct:.0f}pct"

            if session_metrics_cb is not None:
                snap["monitor_value"] = (
                    session_metrics_cb._latest_mean_f1 or 0.0
                )

            try:
                json_path, md_path = write_checkpoint_manifest(
                    ckpt_path,
                    manifest_dir,
                    kind=kind,
                    trained_on=_build_trained_on(snap),
                    selection=_build_selection(snap),
                    compute=_build_compute(snap),
                    recipe=recipe,
                    normalization_artifact_hashes=norm_hashes,
                    git_sha=git_sha,
                    snapshot_bundle=snapshot_bundle,
                    slurm_job_id=slurm_job_id,
                    wandb_info=wandb_info,
                )
                written_manifests.append(str(json_path))
                logger.info(
                    "Wrote milestone checkpoint manifest (step %d): %s",
                    step,
                    json_path,
                )
            except Exception:
                logger.warning(
                    "Failed to write milestone manifest for step %d",
                    step,
                    exc_info=True,
                )

    logger.info(
        "Checkpoint manifest emission: %d manifests written",
        len(written_manifests),
    )
    return written_manifests


def _build_source_model_and_data(cfg: DictConfig):
    """Build model and datamodule for source pretraining with canonical session IDs.

    When the datamodule uses a source manifest, session configs and ID aliases
    are derived from the manifest recordings instead of from the dataset's
    default session structure.
    """
    _populate_data_driven_hyperparams(cfg)

    task_configs = _load_task_configs(cfg)
    if not _is_neuralbench_data(cfg):
        normalize_data_config(cfg.data)

    datamodule = instantiate(cfg.data, tokenizer=None)
    datamodule._task_configs = task_configs

    task_configs, setup_done = _validate_and_apply_focal_loss_weights(
        cfg, datamodule, task_configs
    )
    task_configs, setup_done = _apply_auto_class_weights(
        cfg, datamodule, task_configs, setup_done=setup_done
    )

    if not setup_done:
        datamodule.setup("fit")
        setup_done = True

    source_session_configs = getattr(
        datamodule, "get_source_session_configs", lambda: None
    )()
    source_id_aliases = getattr(
        datamodule, "get_source_id_aliases", lambda: None
    )()

    if source_session_configs is not None:
        OmegaConf.update(
            cfg,
            "hyperparameters.session_configs",
            source_session_configs,
            force_add=True,
        )
        logger.info(
            "Source pretraining: derived session_configs from manifest "
            "(%d sessions).",
            len(source_session_configs),
        )

    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }
    model_kwargs.pop("session_emb", None)
    if source_id_aliases is not None:
        model_kwargs["id_aliases"] = source_id_aliases

    model = ModelClass(task_configs=task_configs, **model_kwargs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule.set_tokenizer(tokenizer)

    return model, datamodule


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

    is_source_pretraining = (
        OmegaConf.select(cfg, "data.role", default=None) == "source_pretraining"
    )
    if is_source_pretraining:
        model, datamodule = _build_source_model_and_data(cfg)
    else:
        model, datamodule = _build_model_and_data(cfg)

    # Prepare fraction manifests before WandB logging and training.
    if (
        hasattr(datamodule, "training_fraction")
        and datamodule.training_fraction is not None
    ):
        if datamodule.dataset is None:
            datamodule.setup("fit")
        datamodule.prepare_training_fraction_manifests()

    normalization_artifacts = None
    if getattr(datamodule, "input_normalization_config", None):
        normalization_cfg = datamodule.input_normalization_config
        if normalization_cfg and normalization_cfg.get("mode") != "disabled":
            # Fit before any loader is constructed, then capture the exact
            # frozen artifact used by this run.
            datamodule.setup("fit")
            normalization_artifacts = datamodule.write_normalization_artifacts(
                output_dir,
                git_sha=os.environ.get("FOUNDRY_SNAPSHOT_GIT_SHA"),
            )

    # -- Pretrained weight transfer -------------------------------------------

    pretrained_ckpt = OmegaConf.select(
        cfg, "run.pretrained_checkpoint", default=None
    )
    checkpoint_manifest = _load_and_validate_checkpoint_manifest(
        cfg, datamodule
    )

    if checkpoint_manifest is not None:
        _apply_manifest_transfer(model, checkpoint_manifest, cfg, output_dir)
    elif pretrained_ckpt:
        freeze = OmegaConf.select(cfg, "run.freeze_pretrained", default=False)
        transfer_mode_str = OmegaConf.select(
            cfg, "run.pretrained_transfer_mode", default="strict"
        )
        transfer_mode = TransferMode(transfer_mode_str)
        components = _resolve_pretrained_components(model, cfg)
        load_pretrained_weights(
            model,
            pretrained_ckpt,
            freeze=freeze,
            mode=transfer_mode,
            components=components,
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
    lightning_module.input_normalization_artifacts = normalization_artifacts
    trainer = _build_trainer(cfg)

    if is_source_pretraining:
        _configure_source_compute_callbacks(trainer, datamodule, cfg)

    _log_config_to_wandb(trainer, cfg)
    _log_normalization_artifacts_to_wandb(trainer, normalization_artifacts)

    # Log fraction provenance to WandB if manifests were prepared.
    neurosoft_provenance = _prepare_fraction_provenance(
        cfg, datamodule, output_dir
    )
    if neurosoft_provenance is not None:
        _log_neurosoft_provenance_to_wandb(
            trainer, neurosoft_provenance, output_dir
        )

    ckpt_path = _get_resume_checkpoint_path(
        cfg, checkpoint_dir, slurm_restart_count
    )

    effective_pretrained = pretrained_ckpt or (
        OmegaConf.select(
            cfg, "run.pretrained_checkpoint_manifest", default=None
        )
    )
    _validate_checkpoint_policy(ckpt_path, effective_pretrained)

    run_failed = False
    try:
        trainer.fit(
            lightning_module,
            datamodule,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        if is_source_pretraining:
            _emit_source_checkpoint_manifests(
                trainer,
                cfg,
                datamodule,
                output_dir,
                normalization_artifacts,
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

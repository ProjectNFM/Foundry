import logging
import os

import hydra
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    setup_logging(cfg.run.log_level)
    seed_everything(cfg.run.seed, workers=True)
    logger.info(f"Starting training: {cfg.run.name}")

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

    class_weights_cfg = OmegaConf.select(
        cfg, "module.class_weights", default=None
    )
    class_weights = None
    if class_weights_cfg == "auto":
        datamodule.setup("fit")
        class_weights = datamodule.compute_class_weights()
    elif class_weights_cfg is not None:
        # TODO: Fix this path
        class_weights = OmegaConf.to_container(class_weights_cfg, resolve=True)

    eeg_module = instantiate(
        cfg.module, model=model, class_weights=class_weights
    )

    trainer = instantiate(cfg.trainer)
    _log_config_to_wandb(trainer, cfg)
    trainer.fit(eeg_module, datamodule)


if __name__ == "__main__":
    register_resolvers()
    main()

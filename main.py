import logging
import os

import hydra
from hydra.utils import get_class, instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
from foundry.tools.stage_data import stage_data
from foundry.training import EEGTask

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    setup_logging(cfg.run.log_level)
    seed_everything(cfg.run.seed, workers=True)
    logger.info(f"Starting training: {cfg.run.name}")

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        stage_cfg = OmegaConf.to_container(
            cfg.get("stage", OmegaConf.create({})), resolve=True
        )
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

    task = EEGTask(model=model, **cfg.task)

    trainer = instantiate(cfg.trainer)
    trainer.fit(task, datamodule)


if __name__ == "__main__":
    register_resolvers()
    main()

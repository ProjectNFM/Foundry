import logging

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from rich.logging import RichHandler

from foundry.training import EEGTask

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training entry point using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    setup_logging(cfg.experiment.log_level)
    seed_everything(cfg.experiment.seed, workers=True)
    logger.info(f"Starting training: {cfg.experiment.name}")

    model = instantiate(cfg.model)

    datamodule = instantiate(cfg.data, model=model)

    task = EEGTask(model=model, **cfg.task)

    trainer = instantiate(cfg.trainer)

    trainer.fit(task, datamodule)


if __name__ == "__main__":
    main()

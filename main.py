import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from foundry.training import EEGTask


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training entry point using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    model = instantiate(cfg.model)

    datamodule = instantiate(cfg.data, model=model)

    task = EEGTask(model=model, **cfg.task)

    trainer = instantiate(cfg.trainer)

    trainer.fit(task, datamodule)


if __name__ == "__main__":
    main()

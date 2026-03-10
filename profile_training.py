import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.profiler
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
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


def _build_run_tag(cfg: DictConfig) -> str:
    """Build a human-readable tag from the timestamp and CLI overrides.

    Strips config group prefixes so ``hyperparameters.num_workers=4`` becomes
    ``num_workers-4``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    overrides = HydraConfig.get().overrides.task
    if overrides:
        parts = []
        for o in overrides:
            key, _, value = o.partition("=")
            short_key = key.rsplit(".", 1)[-1]
            parts.append(f"{short_key}-{value}" if value else short_key)
        return f"{timestamp}_{'_'.join(parts)}"
    return f"{timestamp}_{cfg.run.name}"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Training profiling entry point using PyTorch Lightning's PyTorchProfiler.

    Runs a real training loop with torch.profiler recording CPU and GPU activity,
    producing Chrome/Perfetto traces viewable in TensorBoard.
    """
    setup_logging(cfg.run.log_level)
    seed_everything(cfg.run.seed, workers=True)

    run_tag = _build_run_tag(cfg)
    logger.info(f"Starting profiling run: {run_tag}")

    DataModuleClass = get_class(cfg.data._target_)
    readout_specs = DataModuleClass.get_readout_specs_for_task(
        cfg.data.task_type
    )

    model = instantiate(cfg.model, readout_specs=readout_specs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)

    task = EEGTask(model=model, **cfg.task)

    prof_cfg = cfg.profiling
    output_dir = Path(prof_cfg.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Profiler output directory: {output_dir}")
    logger.info(
        f"Profiler schedule: wait={prof_cfg.schedule.wait}, "
        f"warmup={prof_cfg.schedule.warmup}, "
        f"active={prof_cfg.schedule.active}, "
        f"repeat={prof_cfg.schedule.repeat}"
    )

    from lightning.pytorch.profilers import PyTorchProfiler

    profiler = PyTorchProfiler(
        dirpath=str(output_dir),
        filename=run_tag,
        export_to_chrome=True,
        record_module_names=True,
        sort_by_key=prof_cfg.sort_by_key,
        row_limit=prof_cfg.row_limit,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(**prof_cfg.schedule),
        record_shapes=prof_cfg.record_shapes,
        profile_memory=prof_cfg.profile_memory,
        with_stack=prof_cfg.with_stack,
        with_flops=prof_cfg.with_flops,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir)
        ),
    )

    trainer = instantiate(
        cfg.trainer,
        max_epochs=prof_cfg.max_epochs,
        profiler=profiler,
    )

    logger.info(f"Running profiling for {prof_cfg.max_epochs} epoch(s)")
    trainer.fit(task, datamodule)

    logger.info(f"Profiling complete. Traces saved to: {output_dir}/")


if __name__ == "__main__":
    main()

"""Batch-size tuning callback."""

from __future__ import annotations

import logging

import lightning as L
import torch
from lightning import Trainer

log = logging.getLogger(__name__)


class EffectiveBatchSizeCallback(L.Callback):
    """Find the max batch size that fits in GPU memory, then set
    ``accumulate_grad_batches`` so ``batch_size * accum == effective_batch_size``.

    Runs a simple power-of-2 search at the start of training: one forward +
    backward pass per candidate batch size, doubling until OOM.  The search
    uses ``num_workers=0`` to avoid multiprocessing issues with lazy
    vocabulary initialization, then restores the configured ``num_workers``
    for actual training.

    Args:
        effective_batch_size: Target effective batch size to maintain across
            all runs regardless of per-GPU micro-batch size.
        init_val: Starting batch size for the search.
        max_val: Upper bound for the batch size (capped to ``effective_batch_size``
            if not set).
        steps_per_trial: Forward+backward steps to attempt per candidate size.
    """

    def __init__(
        self,
        effective_batch_size: int = 1024,
        init_val: int = 64,
        max_val: int | None = None,
        steps_per_trial: int = 3,
    ):
        super().__init__()
        self.effective_batch_size = effective_batch_size
        self.init_val = init_val
        self.max_val = max_val or effective_batch_size
        self.steps_per_trial = steps_per_trial

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        dm = trainer.datamodule
        device = pl_module.device

        best_bs = self.init_val
        size = self.init_val

        orig_workers = dm.num_workers
        dm.num_workers = 0

        while size <= self.max_val:
            dm.batch_size = size
            dl = dm.train_dataloader()
            if not self._try_batch_size(dl, trainer, pl_module, device):
                break
            best_bs = size
            log.info("BatchSizeFinder: batch_size=%d fits in memory", size)
            size *= 2

        dm.num_workers = orig_workers
        dm.batch_size = best_bs
        accum = max(1, self.effective_batch_size // best_bs)
        trainer.accumulate_grad_batches = accum

        log.info(
            "EffectiveBatchSize: found max batch_size=%d, "
            "accumulate_grad_batches=%d, effective=%d",
            best_bs,
            accum,
            best_bs * accum,
        )

    def _try_batch_size(
        self, dl, trainer: Trainer, pl_module: L.LightningModule, device
    ) -> bool:
        import gc

        _orig_log = pl_module.log
        pl_module.log = lambda *a, **kw: None

        try:
            it = iter(dl)
            for _ in range(self.steps_per_trial):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = trainer.strategy.batch_to_device(batch, device)
                result = pl_module.training_step(batch, 0)
                loss = result["loss"] if isinstance(result, dict) else result
                if loss is not None:
                    loss.backward()
                pl_module.zero_grad(set_to_none=True)

            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.info("BatchSizeFinder: OOM at current batch_size")
                return False
            raise
        finally:
            pl_module.log = _orig_log
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

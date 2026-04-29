"""Lightning callbacks for Foundry model training."""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import Any

import lightning as L
import torch
from lightning import Trainer

from foundry.core import VocabManager

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

        # Power-of-2 search using the datamodule's own dataloader factory
        # with num_workers=0 to avoid multiprocessing issues with
        # lazily-initialized vocabularies.
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

    @staticmethod
    def _try_batch_size(
        dl, trainer: Trainer, pl_module: L.LightningModule, device
    ) -> bool:
        import gc

        try:
            it = iter(dl)
            for _ in range(3):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = trainer.strategy.batch_to_device(batch, device)
                pl_module.training_step(batch, 0)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                log.info("BatchSizeFinder: OOM at current batch_size")
                return False
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VocabInitializerCallback(L.Callback):
    """Callback to initialize model vocabularies from the datamodule.

    This callback handles the initialization of lazy vocabularies (e.g., session and
    channel embeddings) before training begins. It decouples vocab setup from the
    datamodule, allowing models to be reused with different datasets.

    Usage:
        trainer = Trainer(callbacks=[VocabInitializerCallback()])
    """

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        """Initialize vocabularies at the start of training.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module being trained.
        """
        # Check if model implements vocab initialization
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        if not isinstance(model, VocabManager):
            return

        if not model.has_lazy_vocabs():
            return

        # Get datamodule
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError(
                "VocabInitializerCallback requires a datamodule. "
                "Call trainer.fit(module, datamodule=dm) or set trainer.datamodule."
            )

        # Initialize vocabularies from datamodule or its underlying dataset
        vocab_info = {}
        dataset = getattr(datamodule, "dataset", None)

        for method_name, key in [
            ("get_recording_ids", "session_ids"),
            ("get_channel_ids", "channel_ids"),
        ]:
            if hasattr(datamodule, method_name):
                vocab_info[key] = getattr(datamodule, method_name)()
            elif dataset is not None and hasattr(dataset, method_name):
                vocab_info[key] = getattr(dataset, method_name)()

        model.initialize_vocabs(vocab_info)


class ParameterWatcherCallback(L.Callback):
    """Log the evolution of specific model parameters to W&B during training.

    Matches parameters by name pattern (``fnmatch``-style) and logs scalar
    statistics (mean, std, min, max, norm).  For small tensors (numel <=
    ``individual_value_threshold``), each element is also logged as its own
    scalar so you get one W&B line per element.

    Modules anywhere in the model tree may additionally implement a
    ``get_watched_params() -> dict[str, Tensor]`` method to expose *derived*
    values (e.g. human-readable frequencies after a softplus reparametrization).
    These are always collected — no pattern matching required.

    Args:
        param_patterns: Shell-style patterns matched against
            ``model.named_parameters()`` names.  ``"*"`` matches everything
            within a single dotted segment; ``"**"`` is not supported, but you
            can use ``"*cwt*"`` to match any name containing "cwt".
        log_every_n_steps: How often (in global training steps) to log.
        log_histograms: Whether to log W&B histograms in addition to scalars.
        individual_value_threshold: Parameters (and derived values) with at
            most this many elements get each element logged individually.
    """

    def __init__(
        self,
        param_patterns: list[str],
        log_every_n_steps: int = 50,
        log_histograms: bool = False,
        individual_value_threshold: int = 64,
    ):
        super().__init__()
        self.param_patterns = param_patterns
        self.log_every_n_steps = log_every_n_steps
        self.log_histograms = log_histograms
        self.individual_value_threshold = individual_value_threshold
        self._matched_params: list[tuple[str, str]] | None = None

    def _resolve_model(self, pl_module: L.LightningModule) -> torch.nn.Module:
        return pl_module.model if hasattr(pl_module, "model") else pl_module

    def _discover_matched_params(
        self, model: torch.nn.Module
    ) -> list[tuple[str, str]]:
        """Return ``(param_name, pattern_that_matched)`` for all matches."""
        matched = []
        for name, _ in model.named_parameters():
            for pattern in self.param_patterns:
                if fnmatch(name, pattern):
                    matched.append((name, pattern))
                    break
        return matched

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        model = self._resolve_model(pl_module)
        self._matched_params = self._discover_matched_params(model)
        if not self._matched_params:
            log.warning(
                "ParameterWatcherCallback: no parameters matched patterns %s",
                self.param_patterns,
            )
        else:
            names = [n for n, _ in self._matched_params]
            log.info(
                "ParameterWatcherCallback: watching %d parameters: %s",
                len(names),
                names,
            )
        derived_modules = [
            mod_name
            for mod_name, mod in model.named_modules()
            if hasattr(mod, "get_watched_params")
            and callable(mod.get_watched_params)
        ]
        if derived_modules:
            log.info(
                "ParameterWatcherCallback: found derived params on modules: %s",
                derived_modules,
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return

        wandb_experiment = self._get_wandb_experiment(trainer)
        if wandb_experiment is None:
            return

        model = self._resolve_model(pl_module)
        metrics: dict[str, Any] = {}

        if self._matched_params is None:
            self._matched_params = self._discover_matched_params(model)

        param_dict = dict(model.named_parameters())
        for name, _ in self._matched_params:
            param = param_dict.get(name)
            if param is None:
                continue
            data = param.detach().float()
            prefix = f"params/{name}"
            self._collect_tensor_metrics(metrics, prefix, data)

        for mod_name, module in model.named_modules():
            if not hasattr(module, "get_watched_params"):
                continue
            for key, value in module.get_watched_params().items():
                full_key = f"{mod_name}.{key}" if mod_name else key
                prefix = f"params/{full_key}"
                data = value.detach().float()
                self._collect_tensor_metrics(metrics, prefix, data)

        if metrics:
            wandb_experiment.log(metrics, commit=False)

    def _collect_tensor_metrics(
        self, metrics: dict[str, Any], prefix: str, data: torch.Tensor
    ) -> None:
        metrics[f"{prefix}/mean"] = data.mean().item()
        metrics[f"{prefix}/std"] = (
            data.std().item() if data.numel() > 1 else 0.0
        )
        metrics[f"{prefix}/min"] = data.min().item()
        metrics[f"{prefix}/max"] = data.max().item()
        metrics[f"{prefix}/norm"] = data.norm().item()

        if data.numel() <= self.individual_value_threshold:
            flat = data.flatten()
            for i, val in enumerate(flat):
                metrics[f"{prefix}/{i}"] = val.item()

        if self.log_histograms:
            import wandb

            metrics[f"{prefix}/hist"] = wandb.Histogram(
                data.cpu().numpy().flatten()
            )

    @staticmethod
    def _get_wandb_experiment(trainer: Trainer):
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(trainer.logger, WandbLogger):
            return trainer.logger.experiment
        return None

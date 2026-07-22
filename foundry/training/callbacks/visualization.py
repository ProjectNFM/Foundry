"""Reconstruction visualization callback for masked pretraining."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning import Trainer

from foundry.training.step_output import StepOutput, extract_step_output


class ReconstructionVisualizationCallback(L.Callback):
    """Log example masked-reconstruction plots to W&B.

    Works with :class:`~foundry.models.masked_poyo_eeg.MaskedPOYOEEGModel`.
    Buffers a small number of per-sample reconstructions on the callback
    instance (not on the Lightning module).  Training reconstructions are
    logged every ``log_every_n_steps`` global steps; validation
    reconstructions are logged at each validation epoch end.

    Args:
        num_examples: How many samples to visualize per log event.
        num_channels: Maximum number of EEG channels to show per sample.
        log_every_n_steps: How often (in global training steps) to log
            training reconstructions.  Set to 0 to disable training-step
            visualization.
    """

    def __init__(
        self,
        num_examples: int = 4,
        num_channels: int = 8,
        log_every_n_steps: int = 500,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.num_channels = num_channels
        self.log_every_n_steps = log_every_n_steps
        self._val_buffer: list[dict] = []
        self._train_buffer: list[dict] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step_output = extract_step_output(outputs)
        if (
            step_output is not None
            and step_output.reconstruction_viz is not None
        ):
            self._buffer_examples(step_output, self._train_buffer)

        if self.log_every_n_steps <= 0:
            return

        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return

        self._log_reconstructions(self._train_buffer, "train", trainer)
        self._train_buffer = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        step_output = extract_step_output(outputs)
        if (
            step_output is not None
            and step_output.reconstruction_viz is not None
        ):
            self._buffer_examples(step_output, self._val_buffer)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        self._log_reconstructions(self._val_buffer, "val", trainer)
        self._val_buffer = []

    def _buffer_examples(
        self, step_output: StepOutput, buffer: list[dict]
    ) -> None:
        """Store a few per-sample reconstruction examples for visualization."""
        if len(buffer) >= self.num_examples:
            return

        viz_meta = step_output.reconstruction_viz
        recon_preds = step_output.task_outputs.get("masked_reconstruction")
        targets = step_output.reconstruction_targets
        input_mask = step_output.input_mask
        if recon_preds is None or targets is None or input_mask is None:
            return

        per_sample_counts = viz_meta.validity_mask.sum(dim=1)
        per_sample_preds = torch.split(recon_preds, per_sample_counts.tolist())

        B = viz_meta.mask_indices.shape[0]
        for b in range(B):
            if len(buffer) >= self.num_examples:
                break
            buffer.append(
                {
                    "targets": targets[b].detach().cpu(),
                    "predictions": per_sample_preds[b].detach().cpu(),
                    "mask_indices": viz_meta.mask_indices[b].detach().cpu(),
                    "validity_mask": viz_meta.validity_mask[b].detach().cpu(),
                    "input_mask": input_mask[b].detach().cpu(),
                    "num_channels": viz_meta.num_channels,
                    "num_time_tokens": viz_meta.num_time_tokens,
                }
            )

    def _log_reconstructions(
        self,
        buffer: list[dict],
        prefix: str,
        trainer: Trainer,
    ) -> None:
        """Plot reconstructions from *buffer* and log to W&B under *prefix*."""
        if not buffer:
            return

        from foundry.training.callbacks import get_wandb_experiment

        wandb_experiment = get_wandb_experiment(trainer)
        if wandb_experiment is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            import wandb
        except ImportError:
            return

        figures: dict[str, Any] = {}
        for i, example in enumerate(buffer):
            fig = self._plot_reconstruction(example)
            if fig is not None:
                figures[f"{prefix}/reconstruction_example_{i}"] = wandb.Image(
                    fig
                )
                plt.close(fig)

        if figures:
            wandb_experiment.log(figures, commit=False)

    def _plot_reconstruction(self, example: dict):
        import matplotlib.pyplot as plt
        import numpy as np

        targets_flat = example["targets"].float()
        predictions = example["predictions"].float()
        mask_indices = example["mask_indices"]
        validity_mask = example["validity_mask"]
        input_mask = example["input_mask"]
        C_pad: int = example["num_channels"]
        N: int = example["num_time_tokens"]

        if predictions.dim() == 2 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(-1)
        if predictions.dim() == 2 and predictions.shape[1] > 1:
            return None

        targets_2d = targets_flat.reshape(C_pad, N).numpy()

        valid_indices = mask_indices[validity_mask.bool()]
        valid_ch = (valid_indices // N).numpy()
        valid_t = (valid_indices % N).numpy()
        pred_values = predictions.numpy()

        real_channels = input_mask.bool().nonzero(as_tuple=True)[0].numpy()
        num_ch = min(len(real_channels), self.num_channels)
        if num_ch == 0:
            return None
        channels_to_plot = real_channels[:num_ch]

        fig, axes = plt.subplots(
            num_ch, 1, figsize=(14, 2.5 * num_ch), sharex=True
        )
        if num_ch == 1:
            axes = [axes]

        time_axis = np.arange(N)

        for ax_idx, ch in enumerate(channels_to_plot):
            ax = axes[ax_idx]
            gt = targets_2d[ch]

            ax.plot(
                time_axis,
                gt,
                color="steelblue",
                linewidth=0.8,
                alpha=0.7,
                label="Ground truth" if ax_idx == 0 else None,
            )

            ch_all_mask = (mask_indices // N == ch).numpy()
            ch_masked_times = (mask_indices[ch_all_mask] % N).numpy()
            for t in ch_masked_times:
                ax.axvspan(t - 0.5, t + 0.5, alpha=0.12, color="salmon")

            ch_pred_sel = valid_ch == ch
            ch_times = valid_t[ch_pred_sel]
            ch_preds = pred_values[ch_pred_sel]
            if len(ch_times) > 0:
                ax.scatter(
                    ch_times,
                    ch_preds,
                    color="red",
                    s=12,
                    zorder=5,
                    label="Prediction" if ax_idx == 0 else None,
                )

            ax.set_ylabel(f"Ch {ch}", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Token index")
        fig.suptitle("Masked Reconstruction (z-scored)", fontsize=12)
        fig.tight_layout()
        return fig

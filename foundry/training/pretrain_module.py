from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn


class PretrainModule(L.LightningModule):
    """Lightning module for masked reconstruction pretraining.

    Wraps the same ``POYOEEGModel`` used for supervised training but drives
    it via reconstruction loss only.

    Args:
        model: A ``POYOEEGModel`` configured with a ``reconstruction_head``.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        loss_type: ``"mse"`` or ``"smooth_l1"``.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type!r}")

        self.save_hyperparameters(ignore=["model"])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Downcast float64 tensors to float32 and move the batch to *device*."""
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.float() if t.dtype == torch.float64 else t,
        )
        return move_data_to_device(batch, device)

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Delegate to the wrapped model's forward pass."""
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute reconstruction loss for one training batch.

        Args:
            batch: Collated batch dict from the dataloader.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        recon_targets, model_inputs = self._unpack_batch(batch)
        outputs = self.model(**model_inputs, unpack_output=False)
        flat_targets = self._flatten_targets(recon_targets, model_inputs)
        loss = self.loss_fn(outputs["reconstruction"], flat_targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute reconstruction loss for one validation batch.

        Args:
            batch: Collated batch dict from the dataloader.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        recon_targets, model_inputs = self._unpack_batch(batch)
        outputs = self.model(**model_inputs, unpack_output=False)
        flat_targets = self._flatten_targets(recon_targets, model_inputs)
        loss = self.loss_fn(outputs["reconstruction"], flat_targets)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing LR schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _unpack_batch(self, batch: Dict[str, Any]):
        """Separate reconstruction targets from model inputs.

        Pops ``reconstruction_targets`` and any supervised/metadata keys
        that the model's ``forward`` does not accept, leaving only the
        keyword arguments expected by ``POYOEEGModel.forward``.

        Args:
            batch: Mutable batch dict from the dataloader.

        Returns:
            Tuple of ``(recon_targets, model_inputs)`` where
            *recon_targets* is ``(B, num_tokens, D)`` and
            *model_inputs* is the remaining dict.
        """
        recon_targets = batch.pop("reconstruction_targets")
        batch.pop("target_values", None)
        batch.pop("target_weights", None)
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)
        return recon_targets, batch

    def _flatten_targets(
        self,
        recon_targets: torch.Tensor,
        model_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """Extract valid targets at masked positions, flattened across batch.

        ``recon_targets`` is ``(B, num_tokens, D)`` (a full-size tensor
        where only entries at masked positions are non-zero).
        ``masking_mask`` is ``(B, num_tokens)`` boolean.
        The model's reconstruction output is ``(total_masked, D)`` so we
        gather the corresponding targets using the mask.
        """
        masking_mask = model_inputs.get("masking_mask")
        if masking_mask is not None:
            return recon_targets[masking_mask]

        from foundry.models.poyo_eeg import RECON_DECODER_ID

        decoder_idx = model_inputs["output_decoder_index"]
        parts = []
        for b in range(decoder_idx.shape[0]):
            n_recon = (decoder_idx[b] == RECON_DECODER_ID).sum().item()
            parts.append(recon_targets[b, :n_recon])
        return torch.cat(parts, dim=0)


__all__ = ["PretrainModule"]

"""Lightning modules for LaBraM pre-training stages."""

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn as nn

from foundry.models.vqnsp import VQNSPModel
from foundry.models.masked_labram import (
    MaskedLaBram,
    apply_masking,
)
from foundry.tasks.masking import MaskingStrategy


class VQNSPPretrainingModule(L.LightningModule):
    """Lightning module for Stage 1: VQ-NSP neural tokenizer training.

    Trains the vector-quantized tokenizer that encodes EEG patches into
    discrete spectral codes. Codebook parameters are excluded from the optimizer
    and updated via exponential moving average.

    Args:
        model: VQNSPModel instance.
        learning_rate: Learning rate (default: 5e-4).
        weight_decay: Weight decay (default: 1e-4).
    """

    def __init__(
        self,
        model: VQNSPModel,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda tensor: (
                tensor.float() if tensor.dtype == torch.float64 else tensor
            ),
        )
        return move_data_to_device(batch, device)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        input_patches = batch.get("input_patches")

        if input_patches is None:
            raise ValueError("Batch missing 'input_patches'")

        outputs = self.model(input_patches=input_patches)

        loss = outputs["total_loss"]

        self.log("train/vq_loss", outputs["vq_loss"], prog_bar=True)
        self.log("train/amplitude_loss", outputs["amplitude_loss"])
        self.log("train/phase_loss", outputs["phase_loss"])
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        input_patches = batch.get("input_patches")

        if input_patches is None:
            raise ValueError("Batch missing 'input_patches'")

        outputs = self.model(input_patches=input_patches)

        loss = outputs["total_loss"]

        self.log("val/vq_loss", outputs["vq_loss"])
        self.log("val/amplitude_loss", outputs["amplitude_loss"])
        self.log("val/phase_loss", outputs["phase_loss"])
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "codebook" not in n
                ],
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        ]

        optimizer = torch.optim.AdamW(param_groups)
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


class LaBraMPretrainingModule(L.LightningModule):
    """Lightning module for Stage 2: MaskedLaBram pretraining.

    Pre-trains MaskedLaBram to predict VQ-NSP codebook token IDs
    at masked positions in EEG patches. Uses BEiT-v2 symmetric masking
    (two forward passes with complementary masks).

    Args:
        model: MaskedLaBram instance.
        vqnsp_ckpt_path: Path to VQNSPModel checkpoint (for frozen tokenizer).
        masking: MaskingStrategy to use (default: RandomTokenMasking(0.5)).
        symmetric_masking: If True, use symmetric masking (default: True).
        learning_rate: Learning rate (default: 5e-4).
        weight_decay: Weight decay (default: 1e-4).
    """

    def __init__(
        self,
        model: MaskedLaBram,
        vqnsp_ckpt_path: str,
        masking: Optional[MaskingStrategy] = None,
        symmetric_masking: bool = True,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.vqnsp_ckpt_path = vqnsp_ckpt_path
        self.masking = masking
        self.symmetric_masking = symmetric_masking
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model", "masking"])

        self.vqnsp_model: Optional[VQNSPModel] = None
        self.criterion = nn.CrossEntropyLoss()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda tensor: (
                tensor.float() if tensor.dtype == torch.float64 else tensor
            ),
        )
        return move_data_to_device(batch, device)

    def on_fit_start(self):
        """Load frozen VQ-NSP tokenizer."""
        if self.vqnsp_model is None:
            self.vqnsp_model = VQNSPModel(
                num_channels=self.model.num_channels,
                num_samples=self.model.num_samples,
            )

            try:
                state_dict = torch.load(
                    self.vqnsp_ckpt_path, map_location="cpu"
                )
                self.vqnsp_model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load VQ-NSP checkpoint from {self.vqnsp_ckpt_path}: {e}"
                ) from e

            self.vqnsp_model = self.vqnsp_model.to(self.device)
            self.vqnsp_model.eval()

            for param in self.vqnsp_model.parameters():
                param.requires_grad = False

    def forward(self, **kwargs) -> torch.Tensor:
        return self.model(**kwargs)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        input_patches = batch.get("input_patches")

        if input_patches is None:
            raise ValueError("Batch missing 'input_patches'")

        B, C, N, patch_size = input_patches.shape

        with torch.no_grad():
            target_codes = self.vqnsp_model.encode(input_patches)

        masked_patches, bool_mask = apply_masking(
            input_patches,
            masking=self.masking,
            symmetric=self.symmetric_masking,
        )

        token_logits = self.model(
            input_patches=masked_patches, bool_mask=bool_mask
        )

        B_aug = token_logits.shape[0]
        N_seq = token_logits.shape[1]

        target_codes_aug = target_codes.repeat(2, 1, 1) if self.symmetric_masking else target_codes

        target_codes_flat = target_codes_aug.reshape(B_aug * N_seq)
        logits_flat = token_logits.reshape(B_aug * N_seq, -1)

        loss = self.criterion(logits_flat, target_codes_flat)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        input_patches = batch.get("input_patches")

        if input_patches is None:
            raise ValueError("Batch missing 'input_patches'")

        with torch.no_grad():
            target_codes = self.vqnsp_model.encode(input_patches)

            masked_patches, bool_mask = apply_masking(
                input_patches,
                masking=self.masking,
                symmetric=self.symmetric_masking,
            )

            token_logits = self.model(
                input_patches=masked_patches, bool_mask=bool_mask
            )

            B_aug = token_logits.shape[0]
            N_seq = token_logits.shape[1]

            target_codes_aug = target_codes.repeat(2, 1, 1) if self.symmetric_masking else target_codes

            target_codes_flat = target_codes_aug.reshape(B_aug * N_seq)
            logits_flat = token_logits.reshape(B_aug * N_seq, -1)

            loss = self.criterion(logits_flat, target_codes_flat)

        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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

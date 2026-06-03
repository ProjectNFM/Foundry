"""Lightning module for self-supervised MAE pretraining."""

from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.training.masking import build_token_mask, generate_temporal_mask


class SSLModule(L.LightningModule):
    """Self-supervised pretraining module using masked auto-encoding.

    Each step: tokenize clean signal (teacher, stop-grad) → mask a contiguous
    time span across all channels → replace with learnable mask token → run
    Perceiver backbone with decoder queries at masked timestamps → predict
    teacher embeddings via ``ssl_head`` → MSE loss.

    Args:
        model: A :class:`POYOEEGModel` with empty ``readout_specs``.
        mask_ratio: Fraction of time tokens to mask per sample.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        cwt_lr_multiplier: LR multiplier for CWT parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_ratio: float = 0.75,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        cwt_lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cwt_lr_multiplier = cwt_lr_multiplier

        embed_dim = model.embed_dim
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.ssl_head = nn.Linear(embed_dim, embed_dim)

        self.save_hyperparameters(ignore=["model"])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection

        batch = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.float() if t.dtype == torch.float64 else t,
        )
        return move_data_to_device(batch, device)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step("train", batch)

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step("val", batch)

    def _shared_step(self, stage: str, batch: Dict[str, Any]) -> torch.Tensor:
        batch.pop("session_id", None)
        batch.pop("input_session_ids", None)

        input_values = batch["input_values"]
        input_channel_index = batch["input_channel_index"]
        input_mask = batch["input_mask"]
        input_sampling_rate = batch["input_sampling_rate"]
        input_timestamps = batch["input_timestamps"]
        input_session_index = batch["input_session_index"]
        latent_index = batch["latent_index"]
        latent_timestamps = batch["latent_timestamps"]

        model = self.model

        with torch.no_grad():
            teacher_tokens = model.tokenizer(
                input_values,
                input_channel_index=input_channel_index,
                input_mask=input_mask,
                input_sampling_rate=input_sampling_rate,
                input_seq_len=batch.get("input_seq_len"),
                input_session_ids=batch.get("input_session_ids"),
                input_channel_counts=batch.get("input_channel_counts"),
                channel_emb_fn=model.channel_emb,
            )

        num_channels = input_mask.shape[1]
        num_total_tokens = teacher_tokens.shape[1]
        num_time_tokens = num_total_tokens // num_channels

        start, end = generate_temporal_mask(num_time_tokens, self.mask_ratio)
        token_mask = build_token_mask(num_channels, num_time_tokens, start, end)
        token_mask = token_mask.to(teacher_tokens.device)

        # Per-sample mask marking tokens that belong to active (non-padded)
        # channels.  input_mask is (B, C_pad); expand to (B, C_pad * T).
        active_token_mask = (
            input_mask.unsqueeze(2)
            .expand(-1, -1, num_time_tokens)
            .reshape(input_mask.shape[0], -1)
        )
        # Which of the masked positions are valid (active channel)?
        valid_at_masked = active_token_mask[:, token_mask]  # (B, num_masked)

        teacher_targets = teacher_tokens[:, token_mask].detach()

        student_tokens = teacher_tokens.clone()
        student_tokens[:, token_mask] = self.mask_token

        session_emb = model.session_emb(input_session_index).unsqueeze(1)
        student_tokens = student_tokens + session_emb

        input_timestamp_emb = model.rotary_emb(input_timestamps)

        latents = model.latent_emb(latent_index)
        latent_timestamp_emb = model.rotary_emb(latent_timestamps)

        masked_timestamps = input_timestamps[:, token_mask]
        num_masked = masked_timestamps.shape[1]
        decoder_queries = (
            model.session_emb(input_session_index)
            .unsqueeze(1)
            .expand(-1, num_masked, -1)
        )
        decoder_timestamp_emb = model.rotary_emb(masked_timestamps)

        output_latents = model.backbone(
            inputs=student_tokens,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=decoder_queries,
            output_timestamp_emb=decoder_timestamp_emb,
        )

        predictions = self.ssl_head(output_latents)

        per_token_loss = F.mse_loss(
            predictions, teacher_targets, reduction="none"
        )
        per_token_loss = per_token_loss.mean(dim=-1)  # (B, num_masked)
        weight = valid_at_masked.float()
        loss = (per_token_loss * weight).sum() / weight.sum().clamp(min=1)

        self.log(
            f"pretrain/{stage}_loss" if stage == "val" else "pretrain/loss",
            loss,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        if self.cwt_lr_multiplier == 1.0:
            params = [
                {
                    "params": list(self.parameters()),
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            ]
        else:
            cwt_params, other_params = [], []
            for name, param in self.named_parameters():
                if ".cwt." in name:
                    cwt_params.append(param)
                else:
                    other_params.append(param)
            params = [
                {
                    "params": other_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
            ]
            if cwt_params:
                params.append(
                    {
                        "params": cwt_params,
                        "lr": self.learning_rate * self.cwt_lr_multiplier,
                        "weight_decay": self.weight_decay,
                    }
                )

        optimizer = torch.optim.AdamW(params)
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

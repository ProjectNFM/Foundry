"""MAE-style masked pretraining subclass of POYOEEGModel.

Overrides ``forward()`` to split tokens into visible/masked sets, runs the
encoder+processor on visible tokens only, builds reconstruction queries at
masked positions, and gathers reconstruction targets from the output dict.
Overrides ``tokenize()`` to compute per-channel z-scored reconstruction
targets and wrap ``input_values`` in ``pad2d`` for mixed-sr collation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_brain.data import Data
from torch_brain.batching import pad2d

from foundry.models.poyo_eeg import POYOEEGModel
from foundry.models.ssl_meta import ReconstructionVizMeta, SSLTaskMeta
from foundry.tasks.masking import MaskingStrategy


def _compute_visible_indices(
    total_tokens: int,
    mask_indices: torch.LongTensor,
) -> torch.LongTensor:
    """Complement of mask_indices: which tokens are visible.

    Fully vectorized via ``scatter_`` + ``sort`` — no Python loops.

    Args:
        total_tokens: C_pad * N.
        mask_indices: (B, num_masked).

    Returns:
        (B, num_visible) where num_visible = total_tokens - num_masked.
    """
    B, num_masked = mask_indices.shape
    num_visible = total_tokens - num_masked

    is_masked = torch.zeros(
        B, total_tokens, dtype=torch.bool, device=mask_indices.device
    )
    is_masked.scatter_(1, mask_indices, True)

    _, sorted_idx = is_masked.long().sort(dim=1, stable=True)
    return sorted_idx[:, :num_visible]


class MaskedPOYOEEGModel(POYOEEGModel):
    """POYO model with MAE-style masked pretraining.

    Overrides forward() to:
    1. Embed all tokens via GPU tokenizer
    2. Apply masking strategy -> split visible / masked (fixed count)
    3. Encoder + processor on visible tokens only
    4. Build reconstruction queries at masked positions
    5. Concatenate with downstream queries (if any)
    6. Decode combined queries
    7. Route through ReadoutRouter
    8. Gather reconstruction targets at masked positions, return in output dict
    """

    RECONSTRUCTION_TASK_NAME: str = "masked_reconstruction"

    def __init__(self, *args, masking: MaskingStrategy, **kwargs):
        super().__init__(*args, **kwargs)
        self.masking = masking

        assert self.tokenizer.uses_per_channel, (
            "MaskedPOYOEEGModel requires PerChannelStrategy. "
            "SpatialProjectionStrategy is not supported."
        )
        self._variable_time_tokens = (
            not self.tokenizer.temporal_embedding.has_fixed_token_count
        )

        ch_dim = self.tokenizer.channel_emb_dim
        if ch_dim != self.embed_dim:
            self.recon_channel_proj = torch.nn.Linear(ch_dim, self.embed_dim)
        else:
            self.recon_channel_proj = None

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        input_sampling_rate: Optional[torch.Tensor] = None,
        input_seq_len: Optional[torch.Tensor] = None,
        input_session_ids=None,
        input_channel_counts: Optional[torch.Tensor] = None,
        latent_index: torch.Tensor,
        latent_timestamps: torch.Tensor,
        output_session_index: torch.Tensor,
        output_timestamps: torch.Tensor,
        task_index: torch.Tensor,
        reconstruction_targets: Optional[torch.Tensor] = None,
        unpack_output: bool = False,
    ) -> dict:
        del unpack_output
        self._validate_vocab_initialization()

        # 1. GPU tokenization
        inputs = self.tokenizer(
            input_values,
            input_channel_index=input_channel_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            input_session_ids=input_session_ids,
            input_channel_counts=input_channel_counts,
            channel_emb_fn=self.channel_emb,
        )

        session_emb = self.session_emb(input_session_index).unsqueeze(1)
        inputs = inputs + session_emb

        B, num_tokens, D = inputs.shape
        device = inputs.device
        C_pad = input_mask.shape[1]
        N = num_tokens // C_pad

        # Flatten pad2d-collated (B, C_pad, N) → (B, C_pad*N) for
        # variable-length temporal embeddings (e.g. PerTimepointLinear).
        if input_timestamps.ndim == 3:
            input_timestamps = input_timestamps.reshape(B, -1)
        if (
            reconstruction_targets is not None
            and reconstruction_targets.ndim == 3
        ):
            reconstruction_targets = reconstruction_targets.reshape(B, -1)

        # 2. Generate mask from full (C_pad, N) grid
        mask_indices, validity_mask = self.masking(
            num_channels=C_pad,
            num_time_tokens=N,
            channel_mask=input_mask,
            device=device,
        )

        # Exclude time-padded positions from validity when using
        # variable-length temporal tokens (N = T_max across batch).
        if self._variable_time_tokens and input_seq_len is not None:
            time_of_token = mask_indices % N
            validity_mask = validity_mask & (
                time_of_token < input_seq_len.unsqueeze(1)
            )

        visible_indices = _compute_visible_indices(num_tokens, mask_indices)

        # 3. Gather visible tokens
        expand_D = visible_indices.unsqueeze(-1).expand(-1, -1, D)
        visible_inputs = torch.gather(inputs, 1, expand_D)
        visible_ts = torch.gather(input_timestamps, 1, visible_indices)
        visible_ts_emb = self.rotary_emb(visible_ts)

        # 4. Encoder + processor on visible only
        latents = self.latent_emb(latent_index)
        latent_ts_emb = self.rotary_emb(latent_timestamps)

        latents = self.backbone.encoder(
            latents,
            visible_inputs,
            latent_ts_emb,
            visible_ts_emb,
        )
        latents = self.backbone.processor(latents, latent_ts_emb)

        # 5. Build reconstruction queries at masked positions
        recon_task_idx = self.router.get_task_index_by_name(
            self.RECONSTRUCTION_TASK_NAME
        )
        recon_task_emb = self.task_emb(
            torch.full(
                (B, mask_indices.shape[1]),
                recon_task_idx,
                dtype=torch.long,
                device=device,
            )
        )
        recon_session_emb = self.session_emb(input_session_index).unsqueeze(1)
        masked_channel_idx = mask_indices // N
        recon_channel_tokens = torch.gather(
            input_channel_index, 1, masked_channel_idx
        )
        recon_channel_emb = self.channel_emb(recon_channel_tokens)
        if self.recon_channel_proj is not None:
            recon_channel_emb = self.recon_channel_proj(recon_channel_emb)
        recon_queries = recon_session_emb + recon_task_emb + recon_channel_emb

        masked_ts = torch.gather(input_timestamps, 1, mask_indices)
        recon_ts_emb = self.rotary_emb(masked_ts)

        recon_task_index = torch.where(
            validity_mask,
            torch.full_like(mask_indices, recon_task_idx + 1),
            torch.zeros_like(mask_indices),
        )

        # 6. Build downstream queries (if any)
        if output_timestamps.numel() > 0:
            ds_task_ids = (task_index - 1).clamp(min=0)
            downstream_queries = self.session_emb(
                output_session_index
            ) + self.task_emb(ds_task_ids)
            downstream_ts_emb = self.rotary_emb(output_timestamps)
            all_queries = torch.cat([recon_queries, downstream_queries], dim=1)
            all_ts_emb = torch.cat([recon_ts_emb, downstream_ts_emb], dim=1)
            combined_task_index = torch.cat(
                [recon_task_index, task_index], dim=1
            )
        else:
            all_queries = recon_queries
            all_ts_emb = recon_ts_emb
            combined_task_index = recon_task_index

        # 7. Decode
        output_latents = self.backbone.decoder(
            all_queries,
            latents,
            all_ts_emb,
            latent_ts_emb,
        )

        # 8. Route through ReadoutRouter
        B_out, N_out, D_out = output_latents.shape
        flat_embs = output_latents.reshape(B_out * N_out, D_out)
        flat_task_index = combined_task_index.reshape(B_out * N_out)
        valid = flat_task_index > 0
        outputs = self.router(
            flat_embs[valid], (flat_task_index[valid] - 1).long()
        )

        # 9. Gather reconstruction targets and inject via typed contract
        if reconstruction_targets is not None:
            gathered_targets = torch.gather(
                reconstruction_targets, 1, mask_indices
            )
            recon_valid = (recon_task_index > 0).reshape(-1)
            task_name = self.RECONSTRUCTION_TASK_NAME
            outputs["_ssl_meta"] = {
                task_name: SSLTaskMeta(
                    targets=gathered_targets.reshape(-1)[recon_valid],
                    weights=validity_mask.float().reshape(-1)[recon_valid],
                )
            }

        outputs["_reconstruction_viz"] = ReconstructionVizMeta(
            mask_indices=mask_indices,
            validity_mask=validity_mask,
            num_channels=C_pad,
            num_time_tokens=N,
        )

        return outputs

    def tokenize(self, data: Data) -> dict:
        """Tokenize with reconstruction targets for masked pretraining.

        Calls the base ``tokenize()`` then delegates reconstruction target
        computation to the tokenizer and wraps ``input_values`` in ``pad2d``
        for mixed-sr collation.
        """
        result = super().tokenize(data)

        signal, sampling_rate, _ = self._prepare_signal(
            data, normalize_length=True
        )

        C_pad = self.tokenizer.channel_strategy.max_channels
        targets_tensor = self.tokenizer.compute_reconstruction_targets(
            signal, sampling_rate, self.sequence_length
        )

        if self.tokenizer._do_patching:
            result["reconstruction_targets"] = targets_tensor
        elif self.tokenizer.temporal_embedding.has_fixed_token_count:
            result["reconstruction_targets"] = targets_tensor.reshape(-1)
        else:
            result["reconstruction_targets"] = pad2d(targets_tensor)
            N = targets_tensor.shape[1]
            ts = result["input_timestamps"]
            result["input_timestamps"] = pad2d(ts.reshape(C_pad, N))

        result["input_values"] = pad2d(result["input_values"])

        return result

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
from foundry.models.ssl_meta import (
    ModelOutput,
    ReconstructionVizMeta,
    SSLTaskMeta,
)
from foundry.tasks.masking import MaskingStrategy, build_token_validity_mask


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

        if not self.tokenizer.uses_per_channel:
            raise ValueError(
                "MaskedPOYOEEGModel requires PerChannelStrategy. "
                "SpatialProjectionStrategy is not supported."
            )
        self._variable_time_tokens = not self.tokenizer.has_fixed_token_count

        ch_dim = self.tokenizer.channel_emb_dim
        if ch_dim != self.embed_dim:
            self.recon_channel_proj = torch.nn.Linear(ch_dim, self.embed_dim)
        else:
            self.recon_channel_proj = None

        recon_task_idx = self.router.get_task_index_by_name(
            self.RECONSTRUCTION_TASK_NAME
        )
        self._recon_task_idx: int = recon_task_idx

    def _build_reconstruction_queries(
        self,
        session_emb: torch.Tensor,
        mask_indices: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_timestamps: torch.Tensor,
        N: int,
        masked_validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build reconstruction queries at masked positions.

        Owns the channel-embedding projection and reconstruction task-index
        construction so that ``forward()`` does not handle fusion dimensions
        or task-index offset arithmetic for reconstruction.

        Args:
            session_emb: (B, 1, D) session embedding (broadcast-ready).
            mask_indices: (B, num_masked) flat indices into the (C_pad*N) grid.
            input_channel_index: (B, C_pad) channel identity tokens.
            input_timestamps: (B, C_pad*N) flattened timestamps.
            N: Number of time tokens per channel.
            masked_validity: (B, num_masked) boolean validity for masked positions.

        Returns:
            ``(queries, ts_emb, task_index)`` where *task_index* uses the
            padded convention (0 = invalid, ``recon_task_idx + 1`` = valid).
        """
        B = mask_indices.shape[0]
        device = mask_indices.device

        recon_task_emb = self.task_emb(
            torch.full(
                (B, mask_indices.shape[1]),
                self._recon_task_idx,
                dtype=torch.long,
                device=device,
            )
        )

        masked_channel_idx = mask_indices // N
        recon_channel_tokens = torch.gather(
            input_channel_index, 1, masked_channel_idx
        )
        recon_channel_emb = self.channel_emb(recon_channel_tokens)
        if self.recon_channel_proj is not None:
            recon_channel_emb = self.recon_channel_proj(recon_channel_emb)

        queries = session_emb + recon_task_emb + recon_channel_emb

        masked_ts = torch.gather(input_timestamps, 1, mask_indices)
        ts_emb = self.rotary_emb(masked_ts)

        task_index = torch.where(
            masked_validity,
            torch.full_like(mask_indices, self._recon_task_idx + 1),
            torch.zeros_like(mask_indices),
        )

        return queries, ts_emb, task_index

    def _gather_reconstruction_targets(
        self,
        reconstruction_targets: Optional[torch.Tensor],
        mask_indices: torch.Tensor,
        recon_task_index: torch.Tensor,
        masked_validity: torch.Tensor,
    ) -> dict[str, SSLTaskMeta] | None:
        """Gather targets at masked positions and wrap in :class:`SSLTaskMeta`.

        Returns ``None`` when *reconstruction_targets* is not provided
        (inference mode).
        """
        if reconstruction_targets is None:
            return None

        gathered = torch.gather(reconstruction_targets, 1, mask_indices)
        recon_valid = (recon_task_index > 0).reshape(-1)
        return {
            self.RECONSTRUCTION_TASK_NAME: SSLTaskMeta(
                targets=gathered.reshape(-1)[recon_valid],
                weights=masked_validity.float().reshape(-1)[recon_valid],
            )
        }

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        input_channel_index: torch.Tensor,
        input_session_index: torch.Tensor,
        input_mask: torch.Tensor,
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
    ) -> ModelOutput:
        del unpack_output
        self._validate_vocab_initialization()

        # 1. GPU tokenization
        inputs, session_emb = self._tokenize_and_add_session(
            input_values,
            input_channel_index,
            input_session_index,
            input_mask=input_mask,
            input_sampling_rate=input_sampling_rate,
            input_seq_len=input_seq_len,
            input_session_ids=input_session_ids,
            input_channel_counts=input_channel_counts,
        )

        B, num_tokens, D = inputs.shape
        device = inputs.device
        C_pad = input_mask.shape[1]
        N = num_tokens // C_pad

        if input_timestamps.ndim == 3:
            input_timestamps = input_timestamps.reshape(B, -1)
        if (
            reconstruction_targets is not None
            and reconstruction_targets.ndim == 3
        ):
            reconstruction_targets = reconstruction_targets.reshape(B, -1)

        # 2. Build authoritative token validity
        token_validity = build_token_validity_mask(
            input_mask,
            N,
            input_seq_len=input_seq_len if self._variable_time_tokens else None,
        )

        # 3. Generate mask and derive per-position validity
        mask_indices, _channel_validity = self.masking(
            num_channels=C_pad,
            num_time_tokens=N,
            channel_mask=input_mask,
            device=device,
        )
        masked_validity = torch.gather(token_validity, 1, mask_indices)

        visible_indices = _compute_visible_indices(num_tokens, mask_indices)
        visible_validity = torch.gather(token_validity, 1, visible_indices)

        valid_visible_per_sample = visible_validity.sum(dim=1)
        if (valid_visible_per_sample == 0).any():
            raise ValueError(
                "At least one sample has zero valid visible tokens after "
                "masking. This can happen when all real channels are masked "
                "or the sequence is fully padded."
            )

        # 4. Gather visible tokens
        expand_D = visible_indices.unsqueeze(-1).expand(-1, -1, D)
        visible_inputs = torch.gather(inputs, 1, expand_D)
        visible_ts = torch.gather(input_timestamps, 1, visible_indices)
        visible_ts_emb = self.rotary_emb(visible_ts)

        # 5. Encoder + processor on visible tokens (shared helper)
        latents, latent_ts_emb = self._build_latents(
            latent_index, latent_timestamps
        )
        latents = self._encode_and_process(
            visible_inputs,
            visible_ts_emb,
            latents,
            latent_ts_emb,
            input_mask=visible_validity,
        )

        # 6. Build reconstruction queries (masked-model helper)
        recon_queries, recon_ts_emb, recon_task_index = (
            self._build_reconstruction_queries(
                session_emb,
                mask_indices,
                input_channel_index,
                input_timestamps,
                N,
                masked_validity,
            )
        )

        # 7. Combine with downstream queries (shared helper) if present
        if output_timestamps.numel() > 0:
            ds_queries, ds_ts_emb = self._build_downstream_queries(
                output_session_index, task_index, output_timestamps
            )
            all_queries = torch.cat([recon_queries, ds_queries], dim=1)
            all_ts_emb = torch.cat([recon_ts_emb, ds_ts_emb], dim=1)
            combined_task_index = torch.cat(
                [recon_task_index, task_index], dim=1
            )
        else:
            all_queries = recon_queries
            all_ts_emb = recon_ts_emb
            combined_task_index = recon_task_index

        # 8. Decode + route (shared helpers)
        output_latents = self._decode(
            all_queries, latents, all_ts_emb, latent_ts_emb
        )
        task_outputs = self._route(output_latents, combined_task_index)

        # 9. Gather reconstruction targets
        ssl_meta = self._gather_reconstruction_targets(
            reconstruction_targets,
            mask_indices,
            recon_task_index,
            masked_validity,
        )

        viz = ReconstructionVizMeta(
            mask_indices=mask_indices,
            validity_mask=masked_validity,
            num_channels=C_pad,
            num_time_tokens=N,
        )

        return ModelOutput(
            task_outputs=task_outputs, ssl_meta=ssl_meta, viz=viz
        )

    def tokenize(self, data: Data) -> dict:
        """Tokenize with reconstruction targets for masked pretraining.

        Calls the base ``_tokenize_core()`` and reuses the already-prepared
        signal (already length-normalized via :class:`PreparedSignal`) for
        reconstruction targets, ensuring encoder inputs and targets share
        the same signal contract.
        """
        result, prepared = self._tokenize_core(data)

        targets_tensor = self.tokenizer.compute_reconstruction_targets(
            prepared.signal, prepared.sampling_rate, self.sequence_length
        )

        if self.tokenizer.does_patching:
            result["reconstruction_targets"] = targets_tensor
        elif self.tokenizer.has_fixed_token_count:
            result["reconstruction_targets"] = targets_tensor.reshape(-1)
        else:
            C_pad = self.tokenizer.channel_strategy.max_channels
            result["reconstruction_targets"] = pad2d(targets_tensor)
            N = targets_tensor.shape[1]
            ts = result["input_timestamps"]
            result["input_timestamps"] = pad2d(ts.reshape(C_pad, N))

        result["input_values"] = pad2d(result["input_values"])

        return result

"""Utilities for loading pretrained weights into downstream models.

Handles the weight transfer from ``MaskedPOYOEEGModel`` (SSL pretraining)
to ``POYOEEGModel`` (downstream classification/regression).  Only the
shared architectural components are transferred; dataset-specific embeddings
and task-specific heads are left for fresh initialization.

Uses direct ``param.data.copy_()`` instead of ``nn.Module.load_state_dict``
to avoid triggering custom ``_load_from_state_dict`` hooks in modules like
``InfiniteVocabEmbedding`` that expect all their keys to be present.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TRANSFER_PREFIXES = (
    "tokenizer.",
    "backbone.",
    "rotary_emb.",
    "latent_emb.",
)

SKIP_PREFIXES = (
    "channel_emb.",
    "session_emb.",
    "task_emb.",
    "router.",
    "masking.",
    "recon_channel_proj.",
)


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    freeze: bool = False,
) -> tuple[list[str], list[str]]:
    """Load pretrained weights into *model* from a Lightning checkpoint.

    Extracts the ``state_dict`` from the checkpoint, strips the Lightning
    module ``model.`` prefix, and transfers only the shared architectural
    components (tokenizer, backbone, rotary embeddings, latent embeddings).
    Dataset-specific embeddings (channel, session) and task heads (router,
    task_emb) are intentionally skipped so they can be freshly initialized
    for the downstream dataset/task.

    Weights are copied directly via ``param.data.copy_()`` rather than
    ``nn.Module.load_state_dict`` to avoid issues with custom state-dict
    hooks (e.g. ``InfiniteVocabEmbedding``'s vocab-resize hook).

    Args:
        model: Target model (typically :class:`POYOEEGModel`).
        checkpoint_path: Path to a Lightning ``.ckpt`` file.
        freeze: If ``True``, set ``requires_grad = False`` on all
            successfully transferred parameters.

    Returns:
        Tuple of ``(loaded_keys, skipped_keys)`` for logging/debugging.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_state = ckpt["state_dict"]

    def _strip_prefix(key: str) -> str:
        key = key.removeprefix("model.")
        # torch.compile wraps the original module as ``_orig_mod``
        key = key.removeprefix("_orig_mod.")
        return key

    pretrained_state = {
        _strip_prefix(k): v
        for k, v in ckpt_state.items()
        if k.startswith("model.")
    }

    transfer_state = {
        k: v
        for k, v in pretrained_state.items()
        if any(k.startswith(p) for p in TRANSFER_PREFIXES)
    }
    skipped_keys = [k for k in pretrained_state if k not in transfer_state]

    # Build lookup of model parameters and buffers by name.
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

    loaded_keys: list[str] = []
    shape_mismatches: list[str] = []
    not_in_model: list[str] = []

    for key, ckpt_tensor in transfer_state.items():
        target = model_params.get(key)
        if target is None:
            target = model_buffers.get(key)
        if target is None:
            not_in_model.append(key)
            continue
        if target.shape != ckpt_tensor.shape:
            shape_mismatches.append(
                f"  {key}: ckpt {tuple(ckpt_tensor.shape)} "
                f"vs model {tuple(target.shape)}"
            )
            skipped_keys.append(key)
            continue
        target.data.copy_(ckpt_tensor)
        loaded_keys.append(key)

    if shape_mismatches:
        logger.warning(
            "Skipped %d keys due to shape mismatch:\n%s",
            len(shape_mismatches),
            "\n".join(shape_mismatches),
        )
    if not_in_model:
        logger.debug(
            "Ignored %d checkpoint keys not present in target model: %s",
            len(not_in_model),
            not_in_model,
        )

    logger.info(
        "Loaded %d pretrained parameters from %s (skipped %d).",
        len(loaded_keys),
        checkpoint_path.name,
        len(skipped_keys),
    )

    if freeze:
        frozen_count = 0
        transferred_set = set(loaded_keys)
        for name, param in model.named_parameters():
            if name in transferred_set:
                param.requires_grad = False
                frozen_count += 1
        logger.info(
            "Froze %d pretrained parameters (freeze_pretrained=true).",
            frozen_count,
        )

    return loaded_keys, skipped_keys

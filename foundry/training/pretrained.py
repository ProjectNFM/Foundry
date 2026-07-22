"""Validated, atomic pretrained weight transfer.

Loads shared architectural components from a ``MaskedPOYOEEGModel`` (SSL
pretraining) checkpoint into a ``POYOEEGModel`` (downstream) model.
The model declares which components are transferable via
``transferable_components()``, and the loader validates every key before
mutating any target tensor.

Design invariants:

* **Model-owned transfer policy** – the model declares transferable
  components; the loader handles checkpoint mechanics.
* **Validate-then-apply** – shape/dtype validation completes before any
  target tensor changes.  A later mismatch cannot leave partial state.
* **Strict by default** – missing expected keys or shape mismatches raise
  ``PretrainedTransferError``.  Permissive mode is opt-in and produces
  a loud report.
* **Structured report** – :class:`TransferReport` distinguishes loaded,
  intentionally excluded, missing, unexpected, and mismatched keys.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LIGHTNING_PREFIX = "model."
_COMPILE_PREFIX = "_orig_mod."


class TransferMode(enum.Enum):
    """Controls transfer strictness."""

    STRICT = "strict"
    PERMISSIVE = "permissive"


class PretrainedTransferError(RuntimeError):
    """Raised when strict pretrained transfer validation fails."""


@dataclass
class TransferReport:
    """Structured report of a pretrained weight transfer attempt.

    Every key in the checkpoint falls into exactly one category.
    """

    loaded: list[str] = field(default_factory=list)
    skipped_excluded: list[str] = field(default_factory=list)
    missing_in_checkpoint: list[str] = field(default_factory=list)
    unexpected_in_checkpoint: list[str] = field(default_factory=list)
    shape_mismatched: list[str] = field(default_factory=list)
    dtype_mismatched: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  loaded:                 {len(self.loaded)}",
            f"  excluded (by design):   {len(self.skipped_excluded)}",
            f"  missing in checkpoint:  {len(self.missing_in_checkpoint)}",
            f"  unexpected in ckpt:     {len(self.unexpected_in_checkpoint)}",
            f"  shape mismatched:       {len(self.shape_mismatched)}",
            f"  dtype mismatched:       {len(self.dtype_mismatched)}",
        ]
        return "\n".join(lines)

    @property
    def has_errors(self) -> bool:
        return bool(
            self.missing_in_checkpoint
            or self.shape_mismatched
            or self.dtype_mismatched
        )


def _strip_lightning_prefix(key: str) -> str:
    """Strip ``model.`` and optional ``_orig_mod.`` wrappers."""
    key = key.removeprefix(_LIGHTNING_PREFIX)
    key = key.removeprefix(_COMPILE_PREFIX)
    return key


def _normalize_checkpoint_keys(
    ckpt_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract model keys from a Lightning state_dict and normalize prefixes.

    Handles both compiled (``model._orig_mod.``) and uncompiled
    (``model.``) checkpoints.  Raises on ambiguous collisions where two
    differently-prefixed keys map to the same stripped name.
    """
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in ckpt_state.items():
        if not raw_key.startswith(_LIGHTNING_PREFIX):
            continue
        stripped = _strip_lightning_prefix(raw_key)
        if stripped in result:
            raise PretrainedTransferError(
                f"Ambiguous checkpoint: both compiled and uncompiled keys "
                f"map to '{stripped}'"
            )
        result[stripped] = tensor
    return result


def _collect_target_state(
    model: nn.Module,
    component_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Collect all parameters and buffers under the declared transferable components."""
    target: dict[str, torch.Tensor] = {}
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_state = {**model_params, **model_buffers}

    for name, tensor in all_state.items():
        if any(
            name == comp or name.startswith(comp + ".")
            for comp in component_names
        ):
            target[name] = tensor
    return target


def _validate_transfer(
    checkpoint_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    component_names: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    """Validate every key and build a complete transfer mapping.

    Returns the validated mapping (checkpoint key -> tensor) and the report.
    No model tensors are mutated.
    """
    report = TransferReport()

    transfer_ckpt = {
        k: v
        for k, v in checkpoint_state.items()
        if any(
            k == comp or k.startswith(comp + ".") for comp in component_names
        )
    }

    report.skipped_excluded = sorted(
        k for k in checkpoint_state if k not in transfer_ckpt
    )

    report.unexpected_in_checkpoint = sorted(
        k for k in transfer_ckpt if k not in target_state
    )

    report.missing_in_checkpoint = sorted(
        k for k in target_state if k not in transfer_ckpt
    )

    validated_mapping: dict[str, torch.Tensor] = {}
    for key in sorted(set(transfer_ckpt) & set(target_state)):
        ckpt_tensor = transfer_ckpt[key]
        model_tensor = target_state[key]

        if ckpt_tensor.shape != model_tensor.shape:
            report.shape_mismatched.append(
                f"{key}: checkpoint {tuple(ckpt_tensor.shape)} "
                f"vs model {tuple(model_tensor.shape)}"
            )
            continue

        if ckpt_tensor.dtype != model_tensor.dtype:
            report.dtype_mismatched.append(
                f"{key}: checkpoint {ckpt_tensor.dtype} "
                f"vs model {model_tensor.dtype}"
            )
            continue

        validated_mapping[key] = ckpt_tensor
        report.loaded.append(key)

    return validated_mapping, report


def _apply_transfer(
    model: nn.Module,
    validated_mapping: dict[str, torch.Tensor],
) -> None:
    """Apply validated mapping to model under ``torch.no_grad()``.

    Uses ``param.data.copy_()`` rather than ``load_state_dict`` to avoid
    triggering custom ``_load_from_state_dict`` hooks in modules like
    ``InfiniteVocabEmbedding``.
    """
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_state = {**model_params, **model_buffers}

    with torch.no_grad():
        for key, ckpt_tensor in validated_mapping.items():
            all_state[key].data.copy_(ckpt_tensor)


def _freeze_transferred(
    model: nn.Module,
    loaded_keys: list[str],
) -> int:
    """Freeze exactly the parameters that were successfully transferred."""
    transferred_set = set(loaded_keys)
    frozen_count = 0
    for name, param in model.named_parameters():
        if name in transferred_set:
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    freeze: bool = False,
    mode: TransferMode = TransferMode.STRICT,
) -> TransferReport:
    """Load pretrained weights with validated, atomic transfer.

    The model must implement ``transferable_components()`` returning a tuple
    of top-level submodule names whose state should be transferred. Keys
    outside those components are intentionally excluded (dataset/task-specific
    embeddings and heads).

    In :attr:`TransferMode.STRICT` (default), any missing expected keys,
    shape mismatches, or dtype mismatches raise
    :class:`PretrainedTransferError` **before** any model weights change.

    In :attr:`TransferMode.PERMISSIVE`, validation issues are logged as
    warnings but transfer proceeds for all compatible keys.

    Args:
        model: Target model (typically :class:`POYOEEGModel`).
        checkpoint_path: Path to a Lightning ``.ckpt`` file.
        freeze: If ``True``, set ``requires_grad = False`` on all
            successfully transferred parameters.
        mode: Transfer strictness.  Defaults to ``STRICT``.

    Returns:
        :class:`TransferReport` with complete transfer details.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        PretrainedTransferError: If the checkpoint has no ``state_dict``,
            contains ambiguous keys, or (in strict mode) has missing or
            incompatible keys.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise PretrainedTransferError(
            f"Checkpoint at {checkpoint_path} has no 'state_dict' key. "
            f"Available keys: {sorted(ckpt.keys())}"
        )

    checkpoint_state = _normalize_checkpoint_keys(ckpt["state_dict"])
    if not checkpoint_state:
        raise PretrainedTransferError(
            f"No model keys found in checkpoint (expected keys starting with "
            f"'{_LIGHTNING_PREFIX}'). This may not be a Lightning checkpoint."
        )

    if not hasattr(model, "transferable_components"):
        raise PretrainedTransferError(
            f"Model {type(model).__name__} does not implement "
            f"transferable_components(). Cannot determine which components "
            f"to transfer."
        )

    component_names = model.transferable_components()
    target_state = _collect_target_state(model, component_names)

    validated_mapping, report = _validate_transfer(
        checkpoint_state, target_state, component_names
    )

    if report.has_errors and mode == TransferMode.STRICT:
        error_lines = [
            f"Strict pretrained transfer failed for {checkpoint_path.name}:"
        ]
        if report.missing_in_checkpoint:
            error_lines.append(
                f"  Missing in checkpoint ({len(report.missing_in_checkpoint)}):"
            )
            for k in report.missing_in_checkpoint[:10]:
                error_lines.append(f"    - {k}")
            if len(report.missing_in_checkpoint) > 10:
                error_lines.append(
                    f"    ... and {len(report.missing_in_checkpoint) - 10} more"
                )
        if report.shape_mismatched:
            error_lines.append(
                f"  Shape mismatches ({len(report.shape_mismatched)}):"
            )
            for desc in report.shape_mismatched[:10]:
                error_lines.append(f"    - {desc}")
        if report.dtype_mismatched:
            error_lines.append(
                f"  Dtype mismatches ({len(report.dtype_mismatched)}):"
            )
            for desc in report.dtype_mismatched[:10]:
                error_lines.append(f"    - {desc}")
        raise PretrainedTransferError("\n".join(error_lines))

    if report.has_errors and mode == TransferMode.PERMISSIVE:
        logger.warning(
            "Permissive pretrained transfer has issues:\n%s",
            report.summary(),
        )
        if report.missing_in_checkpoint:
            logger.warning(
                "Missing in checkpoint: %s", report.missing_in_checkpoint
            )
        if report.shape_mismatched:
            logger.warning("Shape mismatches: %s", report.shape_mismatched)
        if report.dtype_mismatched:
            logger.warning("Dtype mismatches: %s", report.dtype_mismatched)

    _apply_transfer(model, validated_mapping)

    if freeze and report.loaded:
        frozen_count = _freeze_transferred(model, report.loaded)
        logger.info(
            "Froze %d pretrained parameters (freeze_pretrained=true).",
            frozen_count,
        )

    logger.info(
        "Pretrained transfer from %s (%s mode):\n%s",
        checkpoint_path.name,
        mode.value,
        report.summary(),
    )

    return report

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MaskingStrategy(ABC):
    """Base class for token masking strategies used in pretraining."""

    @abstractmethod
    def generate_mask(self, num_tokens: int) -> torch.BoolTensor:
        """Generate a boolean mask indicating which tokens to mask.

        Args:
            num_tokens: Total number of tokens to consider.

        Returns:
            Boolean tensor of shape ``(num_tokens,)`` where ``True`` means masked.
        """
        ...


class RandomPatchMasking(MaskingStrategy):
    """Independently mask each token with a fixed probability.

    Args:
        mask_ratio: Probability of masking each token.
    """

    def __init__(self, mask_ratio: float = 0.15):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio

    def generate_mask(self, num_tokens: int) -> torch.BoolTensor:
        """Sample an i.i.d. Bernoulli mask, guaranteeing at least one masked token.

        Args:
            num_tokens: Total number of tokens to consider.

        Returns:
            Boolean tensor of shape ``(num_tokens,)`` where ``True`` means masked.
        """
        mask = torch.rand(num_tokens) < self.mask_ratio
        if not mask.any():
            mask[torch.randint(num_tokens, (1,))] = True
        return mask


class ContiguousSpanMasking(MaskingStrategy):
    """Mask contiguous spans of tokens.

    Span lengths are sampled from a geometric distribution and placed
    at random start positions until the target masked ratio is reached.

    Args:
        mask_ratio: Target fraction of tokens to mask.
        mean_span_length: Mean of the geometric distribution for span lengths.
    """

    def __init__(self, mask_ratio: float = 0.15, mean_span_length: int = 3):
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if mean_span_length < 1:
            raise ValueError(
                f"mean_span_length must be >= 1, got {mean_span_length}"
            )
        self.mask_ratio = mask_ratio
        self.mean_span_length = mean_span_length

    def generate_mask(self, num_tokens: int) -> torch.BoolTensor:
        """Place contiguous spans until the target masked count is reached.

        Span lengths are drawn from ``Geometric(1 / mean_span_length)``
        (minimum length 1).  Spans may overlap, so the actual masked
        ratio can be slightly lower than ``mask_ratio``.

        Args:
            num_tokens: Total number of tokens to consider.

        Returns:
            Boolean tensor of shape ``(num_tokens,)`` where ``True`` means masked.
        """
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        target_masked = max(1, round(self.mask_ratio * num_tokens))
        p = 1.0 / self.mean_span_length

        masked_count = 0
        while masked_count < target_masked:
            span_len = (
                int(torch.distributions.Geometric(probs=p).sample().item()) + 1
            )
            span_len = min(span_len, target_masked - masked_count)

            max_start = num_tokens - span_len
            if max_start < 0:
                break
            start = torch.randint(0, max_start + 1, (1,)).item()
            mask[start : start + span_len] = True
            masked_count = mask.sum().item()

        if not mask.any():
            mask[torch.randint(num_tokens, (1,))] = True

        return mask


__all__ = ["MaskingStrategy", "RandomPatchMasking", "ContiguousSpanMasking"]

import torch
import torch.nn as nn

from foundry.models.embeddings.temporal.base import TemporalEmbedding


class PatchLinearEmbedding(TemporalEmbedding):
    """Convert patched EEG signal to embeddings via linear projection.

    Flattens the (channels, time) dimensions and projects to ``embed_dim``.

    Args:
        embed_dim: Output embedding dimension.
        num_input_channels: Number of input channels per patch.
        patch_samples: Number of time samples per patch.
    """

    def __init__(
        self, embed_dim: int, num_input_channels: int, patch_samples: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_input_channels = num_input_channels
        self.patch_samples = patch_samples
        self.projection = nn.Linear(
            num_input_channels * patch_samples, embed_dim
        )
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def get_num_time_tokens(
        self, sequence_length: float, sampling_rate: float
    ) -> int:
        num_samples = round(sampling_rate * sequence_length)
        return max(1, num_samples // self.patch_samples)

    @property
    def has_fixed_token_count(self) -> bool:
        return True

    def forward(self, patches: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            patches: (B, P, C, S)

        Returns:
            (B, P, embed_dim)
        """
        B, P, C, S = patches.shape
        return self.projection(patches.reshape(B, P, C * S))


__all__ = ["PatchLinearEmbedding"]

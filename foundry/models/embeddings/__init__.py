from foundry.models.embeddings.activations import get_activation
from foundry.models.embeddings.channel import (
    ChannelStrategy,
    FixedChannelStrategy,
    LinearSpatialProjector,
    PerChannelStrategy,
    PerceiverSpatialProjector,
    SessionSpatialProjector,
    SpatialProjectionStrategy,
)
from foundry.models.embeddings.patching import (
    compute_patch_timestamps,
    patch_signal,
)
from foundry.models.embeddings.temporal import (
    CWTEmbedding,
    ContinuousCWTLayer,
    PatchCNNEmbedding,
    PatchLinearEmbedding,
    PatchMLPEmbedding,
    PerTimepointEmbedding,
)

__all__ = [
    # Activations
    "get_activation",
    # Channel strategies
    "ChannelStrategy",
    "FixedChannelStrategy",
    "PerChannelStrategy",
    "SpatialProjectionStrategy",
    # Spatial projectors
    "LinearSpatialProjector",
    "PerceiverSpatialProjector",
    "SessionSpatialProjector",
    # Patch operations
    "patch_signal",
    "compute_patch_timestamps",
    # Temporal embeddings - new names
    "PatchLinearEmbedding",
    "PatchMLPEmbedding",
    "PatchCNNEmbedding",
    "PerTimepointEmbedding",
    "CWTEmbedding",
    "ContinuousCWTLayer",
]

from foundry.models.embeddings.base import (
    EmbeddingBase,
    FixedChannelWindowEmbedding,
    get_activation,
)
from foundry.models.embeddings.cnn import CNNEmbedding
from foundry.models.embeddings.cwt import ContinuousCWTLayer, CWTEmbedding
from foundry.models.embeddings.linear import LinearEmbedding
from foundry.models.embeddings.mlp import MLPEmbedding
from foundry.models.embeddings.spatial import SessionSpatialProjector

__all__ = [
    "EmbeddingBase",
    "FixedChannelWindowEmbedding",
    "LinearEmbedding",
    "MLPEmbedding",
    "CNNEmbedding",
    "CWTEmbedding",
    "ContinuousCWTLayer",
    "SessionSpatialProjector",
    "get_activation",
]

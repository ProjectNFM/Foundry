"""Foundry models package.

Re-exports the core model classes (:class:`POYOEEGModel`,
:class:`MaskedPOYOEEGModel`), the :class:`EEGTokenizer`, backbone components,
channel/temporal embedding strategies, session/channel encoders, and baseline
architectures.
"""

from foundry.models.backbones import (
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIOBackbone,
    PerceiverProcessor,
)
from foundry.models.embeddings import (
    ChannelStrategy,
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
    PatchCNNEmbedding,
    ContinuousCWTLayer,
    CWTEmbedding,
    CWTCNNEmbedding,
    PatchMLPEmbedding,
    PatchLinearEmbedding,
    PerTimepointLinearEmbedding,
    PerTimepointIdentityEmbedding,
    ResampleCNNEmbedding,
    LinearSpatialProjector,
    PerceiverSpatialProjector,
    SessionSpatialProjector,
    patch_signal,
    compute_patch_timestamps,
)
from foundry.models.signal_preparation import (
    PreparedSignal,
    compute_num_patches,
    normalize_encoder_inputs,
    normalize_reconstruction_targets,
    normalize_signal_length,
)
from foundry.models.tokenizer import EEGTokenizer

from foundry.models.poyo_eeg import POYOEEGModel
from foundry.models.masked_poyo_eeg import MaskedPOYOEEGModel
from foundry.models.relative_channel_encoder import RelativeChannelEncoder
from foundry.models.session_embedding import (
    DynamicSessionEncoder,
    SessionContextCache,
    SessionEmbeddingCache,
)
from foundry.models.baselines import (
    TemporalConvAvgPool,
    Linear,
    MLP,
    GRU,
    ShallowConvNet,
    EEGNetEncoder,
)

__all__ = [
    "POYOEEGModel",
    "MaskedPOYOEEGModel",
    "EEGTokenizer",
    "PreparedSignal",
    "compute_num_patches",
    "normalize_signal_length",
    "normalize_encoder_inputs",
    "normalize_reconstruction_targets",
    "TemporalConvAvgPool",
    "Linear",
    "MLP",
    "GRU",
    "ShallowConvNet",
    "EEGNetEncoder",
    "ChannelStrategy",
    "FixedChannelStrategy",
    "PerChannelStrategy",
    "SpatialProjectionStrategy",
    "PatchLinearEmbedding",
    "PatchMLPEmbedding",
    "PatchCNNEmbedding",
    "CWTEmbedding",
    "ContinuousCWTLayer",
    "PerTimepointLinearEmbedding",
    "PerTimepointIdentityEmbedding",
    "LinearSpatialProjector",
    "PerceiverSpatialProjector",
    "SessionSpatialProjector",
    "PerceiverDecoder",
    "PerceiverEncoder",
    "PerceiverIOBackbone",
    "PerceiverProcessor",
    "patch_signal",
    "compute_patch_timestamps",
    "RelativeChannelEncoder",
    "DynamicSessionEncoder",
    "SessionContextCache",
    "SessionEmbeddingCache",
    "ResampleCNNEmbedding",
    "CWTCNNEmbedding",
]

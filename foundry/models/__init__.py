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
    PatchMLPEmbedding,
    PatchLinearEmbedding,
    PerTimepointLinearEmbedding,
    PerTimepointIdentityEmbedding,
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
from foundry.models.baselines import (
    TemporalConvAvgPool,
    Linear,
    MLP,
    GRU,
    ShallowConvNet,
    EEGNetEncoder,
)
from foundry.models.labram import LaBraMEEGModel
from foundry.models.vqnsp import NormEMAVectorQuantizer, VQNSPModel
from foundry.models.masked_labram import (
    MaskedLaBram,
    apply_masking,
)
from foundry.models.patch_utils import (
    extract_labram_patches,
    prepare_labram_continuous_signal,
)
from braindecode.models.labram import LABRAM_CHANNEL_ORDER

__all__ = [
    "POYOEEGModel",
    "MaskedPOYOEEGModel",
    "LaBraMEEGModel",
    "VQNSPModel",
    "NormEMAVectorQuantizer",
    "MaskedLaBram",
    "apply_masking",
    "extract_labram_patches",
    "prepare_labram_continuous_signal",
    "EEGTokenizer",
    "PreparedSignal",
    "compute_num_patches",
    "normalize_signal_length",
    "infer_sampling_rate_from_timestamps",
    "resolve_signal_source",
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
    "LABRAM_CHANNEL_ORDER",
]

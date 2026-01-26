"""
Example usage of the modular EEG model architecture.

This demonstrates how to compose different building blocks to create custom models.
Each component is standalone and can be mixed and matched however you want.
"""

from foundry.models import (
    EEGModel,
    PatchEmbedding,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIOBackbone,
    PerceiverProcessor,
)


def create_default_model(readout_specs, embed_dim=256):
    """Create a model with PatchEmbedding and PerceiverIO backbone."""
    
    input_embedding = PatchEmbedding(embed_dim=embed_dim)
    
    backbone = PerceiverIOBackbone(
        embed_dim=embed_dim,
        depth=2,
        dim_head=64,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
    )
    
    model = EEGModel(
        input_embedding=input_embedding,
        backbone=backbone,
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=30.0,
        patch_size_seconds=0.5,
        patch_overlap_percentage=0.5,
        latent_step=0.1,
        num_latents_per_step=1,
    )
    
    return model


def create_model_with_custom_components(readout_specs, embed_dim=256):
    """
    Build a custom model using individual Perceiver components.
    
    This shows how you can use PerceiverEncoder, PerceiverProcessor, and
    PerceiverDecoder as separate building blocks instead of the full
    PerceiverIOBackbone wrapper.
    """
    
    input_embedding = PatchEmbedding(embed_dim=embed_dim)
    
    encoder = PerceiverEncoder(embed_dim=embed_dim, cross_heads=2)
    processor = PerceiverProcessor(embed_dim=embed_dim, depth=4, self_heads=12)
    decoder = PerceiverDecoder(embed_dim=embed_dim, cross_heads=2)
    
    model = EEGModel(
        input_embedding=input_embedding,
        backbone=PerceiverIOBackbone(
            embed_dim=embed_dim, depth=4, self_heads=12, cross_heads=2
        ),
        readout_specs=readout_specs,
        embed_dim=embed_dim,
        sequence_length=30.0,
    )
    
    return model

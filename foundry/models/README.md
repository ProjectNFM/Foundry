# Modular EEG Model Architecture

This directory contains a flexible, composable architecture for building EEG models. Components are standalone building blocks that can be mixed and matched without rigid interfaces.

## Philosophy

- **No enforced protocols**: Each component is just an `nn.Module` you can use however you want
- **Plug and play**: Swap embeddings, backbones, or entire architectures easily
- **Build your own**: Use components individually or compose them in new ways
- **Clear separation**: Embedding → Backbone → Readout

## Directory Structure

```
foundry/models/
├── embeddings/          # Input embedding components
│   └── patch.py        # PatchEmbedding: flatten & project patches
├── backbones/          # Model architectures
│   └── perceiver.py    # Perceiver IO components (encoder, processor, decoder)
├── eeg_model.py        # Reference implementation showing composition
├── poyo_eeg.py         # Original monolithic model (for backward compatibility)
└── README.md           # This file
```

## Components

### Embeddings (`embeddings/`)

**PatchEmbedding** - Converts variable-sized patches to fixed embeddings
- Dynamic projection layers based on input dimensions
- Simple interface: `embeddings = patch_emb(input_values)`

```python
from foundry.models import PatchEmbedding

patch_emb = PatchEmbedding(embed_dim=256)
embeddings = patch_emb(input_values)  # (batch, seq, embed_dim)
```

### Backbones (`backbones/`)

**Perceiver Components** - Three standalone modules:

1. **PerceiverEncoder** - Compress inputs into latents via cross-attention
2. **PerceiverProcessor** - Refine latents with self-attention layers
3. **PerceiverDecoder** - Generate outputs from latents via cross-attention

Use individually:

```python
from foundry.models import PerceiverEncoder, PerceiverProcessor, PerceiverDecoder

encoder = PerceiverEncoder(embed_dim=256)
processor = PerceiverProcessor(embed_dim=256, depth=4)
decoder = PerceiverDecoder(embed_dim=256)

# Use however you want
latents = encoder(latents, inputs, latent_ts_emb, input_ts_emb, mask)
latents = processor(latents, latent_ts_emb)
outputs = decoder(queries, latents, query_ts_emb, latent_ts_emb)
```

Or use the convenience wrapper:

```python
from foundry.models import PerceiverIOBackbone

backbone = PerceiverIOBackbone(embed_dim=256, depth=4)
outputs = backbone(inputs, input_ts_emb, latents, latent_ts_emb, 
                   queries, query_ts_emb, mask)
```

### Complete Model (`eeg_model.py`)

**EEGModel** - Reference implementation showing how to compose components

This is just one way to wire things together. You can build your own!

```python
from foundry.models import EEGModel, PatchEmbedding, PerceiverIOBackbone

embedding = PatchEmbedding(embed_dim=256)
backbone = PerceiverIOBackbone(embed_dim=256, depth=2)

model = EEGModel(
    input_embedding=embedding,
    backbone=backbone,
    readout_specs=readout_specs,
    embed_dim=256,
    sequence_length=30.0,
)
```

## Usage Examples

### Example 1: Use Default Components

```python
from foundry.models import EEGModel, PatchEmbedding, PerceiverIOBackbone

model = EEGModel(
    input_embedding=PatchEmbedding(embed_dim=256),
    backbone=PerceiverIOBackbone(embed_dim=256, depth=2),
    readout_specs=readout_specs,
    embed_dim=256,
    sequence_length=30.0,
)
```

### Example 2: Swap Just the Embedding

```python
# Create your own embedding
class ConvEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(...)
        
    def forward(self, input_values):
        return self.conv(input_values)

model = EEGModel(
    input_embedding=ConvEmbedding(embed_dim=256),
    backbone=PerceiverIOBackbone(embed_dim=256, depth=2),
    readout_specs=readout_specs,
    embed_dim=256,
    sequence_length=30.0,
)
```

### Example 3: Use Components Directly

```python
from foundry.models import PatchEmbedding, PerceiverEncoder, PerceiverProcessor

# Build your own architecture
class MyCustomModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = PatchEmbedding(embed_dim)
        self.encoder = PerceiverEncoder(embed_dim)
        self.processor = PerceiverProcessor(embed_dim, depth=6)
        # Add your own components
        self.my_custom_layer = MyLayer()
        
    def forward(self, inputs, ...):
        # Wire components however you want
        x = self.embedding(inputs)
        x = self.encoder(...)
        x = self.processor(x, ...)
        x = self.my_custom_layer(x)
        return x
```

### Example 4: Mix and Match Different Backbones

```python
# Use only the processor for a different architecture
from foundry.models import PatchEmbedding, PerceiverProcessor

class TransformerModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = PatchEmbedding(embed_dim)
        self.processor = PerceiverProcessor(embed_dim, depth=8)
        self.readout = nn.Linear(embed_dim, num_classes)
        
    def forward(self, inputs, timestamps, ...):
        x = self.embedding(inputs)
        x = self.processor(x, timestamps)
        return self.readout(x)
```
# Modular EEG Model Architecture

This directory contains composable EEG modeling components. Each piece is a
plain `nn.Module` so you can wire things together without extra framework
constraints.

## Philosophy

- **No enforced protocols**: components are standard `nn.Module`s
- **Plug and play**: swap embeddings, backbones, or entire assemblies
- **Clear separation**: Embedding → Backbone → Readout

## Directory Structure

```
foundry/models/
├── embeddings/          # Input embedding components
│   ├── linear.py        # LinearEmbedding
│   ├── mlp.py           # MLPEmbedding
│   └── cnn.py           # CNNEmbedding
├── backbones/           # Model architectures
│   └── perceiver.py     # Perceiver IO components (encoder, processor, decoder)
├── eeg_model.py         # Reference implementation showing composition
└── README.md            # This file
```

## Components

### Embeddings (`embeddings/`)

- **LinearEmbedding**: simple projection to `embed_dim`
- **MLPEmbedding**: MLP stack for richer projections
- **CNNEmbedding**: conv-based temporal embedding

```python
from foundry.models import LinearEmbedding

embedding = LinearEmbedding(embed_dim=256)
embeddings = embedding(input_values)  # (batch, seq, embed_dim)
```

### Backbones (`backbones/`)

**Perceiver Components** - three standalone modules:

1. **PerceiverEncoder** - compress inputs into latents via cross-attention
2. **PerceiverProcessor** - refine latents with self-attention layers
3. **PerceiverDecoder** - generate outputs from latents via cross-attention

```python
from foundry.models import PerceiverEncoder, PerceiverProcessor, PerceiverDecoder

encoder = PerceiverEncoder(embed_dim=256)
processor = PerceiverProcessor(embed_dim=256, depth=4)
decoder = PerceiverDecoder(embed_dim=256)

latents = encoder(latents, inputs, latent_ts_emb, input_ts_emb, mask)
latents = processor(latents, latent_ts_emb)
outputs = decoder(queries, latents, query_ts_emb, latent_ts_emb)
```

Or use the convenience wrapper:

```python
from foundry.models import PerceiverIOBackbone

backbone = PerceiverIOBackbone(embed_dim=256, depth=4)
outputs = backbone(
    inputs,
    input_ts_emb,
    latents,
    latent_ts_emb,
    queries,
    query_ts_emb,
    mask,
)
```

### Complete Model (`eeg_model.py`)

**EEGModel** is a reference implementation showing how to compose the pieces.

```python
from foundry.models import EEGModel, LinearEmbedding, PerceiverIOBackbone

embedding = LinearEmbedding(embed_dim=256)
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

### Example 1: Use Standard Components

```python
from foundry.models import EEGModel, LinearEmbedding, PerceiverIOBackbone

model = EEGModel(
    input_embedding=LinearEmbedding(embed_dim=256),
    backbone=PerceiverIOBackbone(embed_dim=256, depth=2),
    readout_specs=readout_specs,
    embed_dim=256,
    sequence_length=30.0,
)
```

### Example 2: Swap Just the Embedding

```python
class ConvEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
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
from foundry.models import LinearEmbedding, PerceiverEncoder, PerceiverProcessor

class MyCustomModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = LinearEmbedding(embed_dim)
        self.encoder = PerceiverEncoder(embed_dim)
        self.processor = PerceiverProcessor(embed_dim, depth=6)
        self.my_custom_layer = MyLayer()

    def forward(self, inputs, ...):
        x = self.embedding(inputs)
        x = self.encoder(...)
        x = self.processor(x, ...)
        return self.my_custom_layer(x)
```
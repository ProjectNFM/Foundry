# Foundry Models

`foundry/models` contains the EEG/iEEG model stack used in Foundry:

- a composable `EEGTokenizer` (`tokenizer.py`) for channel handling + temporal embedding,
- a Perceiver IO backbone (`backbones/perceiver.py`),
- the integrated POYO-style model (`poyo_eeg.py`),
- and baseline CNN models (`baselines.py`).

## Current Layout

```text
foundry/models/
├── __init__.py
├── tokenizer.py
├── poyo_eeg.py
├── baselines.py
├── utils.py
├── backbones/
│   ├── __init__.py
│   └── perceiver.py
└── embeddings/
    ├── __init__.py
    ├── activations.py
    ├── patching.py
    ├── channel/
    │   ├── __init__.py
    │   ├── processors.py
    │   └── spatial_projectors.py
    └── temporal/
        ├── __init__.py
        ├── per_timepoint.py
        ├── patch_linear.py
        ├── patch_mlp.py
        ├── patch_cnn.py
        └── cwt.py
```

## End-to-End Integration

```mermaid
flowchart LR
    A[TemporalData sample] --> B[POYOEEGModel.tokenize]
    B --> C[EEGTokenizer.pretokenize<br/>CPU dataloader stage]
    C --> D[Batch tensors]
    D --> E[EEGTokenizer.forward<br/>GPU stage]
    E --> F[Session embedding add]
    F --> G[PerceiverIOBackbone]
    G --> H[MultitaskReadout]
    H --> I[Task outputs]
```

### POYOEEGModel Composition

`POYOEEGModel` composes these modules in order:

1. `EEGTokenizer` -> produces `inputs` with shape `(B, num_tokens, embed_dim)`
2. session conditioning -> adds `session_emb(input_session_index)` to every input token
3. latent/query setup -> latent embeddings + rotary time embeddings
4. `PerceiverIOBackbone` -> encoder cross-attn, processor self-attn, decoder cross-attn
5. `MultitaskReadout` -> task-specific heads chosen by `output_decoder_index`

## EEGTokenizer Architecture

The tokenizer has two phases:

- `pretokenize(...)`: CPU-side per-sample prep in dataloading
- `forward(...)`: GPU-side token embedding in training/inference

```mermaid
flowchart TD
    A[input_values: B x C x T] --> B[channel_strategy.forward]
    B --> C{patch_duration?}
    C -->|yes| D[patch_signal -> B x P x C' x S]
    C -->|no| E[raw temporal path]
    D --> F[temporal_embedding]
    E --> F
    F --> G[LayerNorm]
    G --> H{PerChannelStrategy?}
    H -->|yes| I[reassemble C x N tokens<br/>+ optional channel embedding]
    H -->|no| J[return tokens]
    I --> J
```

### Channel Strategies (`embeddings/channel/processors.py`)

| Strategy | Output before temporal embedding | Typical use |
|---|---|---|
| `FixedChannelStrategy` | `(B, num_channels, T)` | Fixed-size channel layout |
| `PerChannelStrategy` | `(B * C_pad, 1, T)` | Independent per-channel tokenization |
| `SpatialProjectionStrategy` | `(B, num_sources, T)` | Learn a common source space from variable channels |

### Spatial Projectors (`embeddings/channel/spatial_projectors.py`)

Used only by `SpatialProjectionStrategy`:

- `LinearSpatialProjector`: shared linear projection from channels to sources
- `SessionSpatialProjector`: session-specific linear projection (optional shared MLP)
- `PerceiverSpatialProjector`: cross-attention projection into latent sources

### Temporal Embeddings (`embeddings/temporal/`)

| Embedding | Expected input | Output |
|---|---|---|
| `PatchLinearEmbedding` | `(B, P, C, S)` | `(B, P, D)` |
| `PatchMLPEmbedding` | `(B, P, C, S)` | `(B, P, D)` |
| `PatchCNNEmbedding` | `(B, P, C, S)` | `(B, P, D)` |
| `PerTimepointEmbedding` | `(B, T, input_dim)` | `(B, T, D)` |
| `CWTEmbedding` | `(B, num_sources, T)` + `input_sampling_rate`, `input_seq_len` | `(B, target_time_tokens, D)` |

## How Tokenizer Variants Relate

Tokenizer configs live in `configs/model/tokenizer/` and combine:

- one channel strategy family (`fixed`, `per_channel`, `spatial_linear`, `spatial_session`, `spatial_perceiver`)
- one temporal embedding family (`patch_linear`, `patch_mlp`, `patch_cnn`, `per_timepoint`, `cwt`)

```mermaid
flowchart LR
    subgraph Channel Strategy
      A[fixed]
      B[per_channel]
      C[spatial_linear]
      D[spatial_session]
      E[spatial_perceiver]
    end
    subgraph Temporal Embedding
      F[patch_linear]
      G[patch_mlp]
      H[patch_cnn]
      I[per_timepoint]
      J[cwt]
    end
    A --> F
    A --> G
    A --> H
    A --> I
    A --> J
    B --> F
    B --> G
    B --> H
    B --> I
    B --> J
    C --> F
    C --> G
    C --> H
    D --> F
    D --> G
    D --> H
    E --> F
    E --> G
    E --> H
```

### Config Coverage Matrix

| Channel strategy family | `patch_linear` | `patch_mlp` | `patch_cnn` | `per_timepoint` | `cwt` |
|---|---:|---:|---:|---:|---:|
| `fixed_*` | `fixed_patch_linear.yaml` | `fixed_patch_mlp.yaml` | `fixed_patch_cnn.yaml` | `fixed_per_timepoint.yaml` | `fixed_cwt.yaml` |
| `per_channel_*` | `per_channel_patch_linear.yaml` | `per_channel_patch_mlp.yaml` | `per_channel_patch_cnn.yaml` | `per_channel_per_timepoint.yaml` | `per_channel_cwt.yaml` |
| `spatial_linear_*` | `spatial_linear_patch_linear.yaml` | `spatial_linear_patch_mlp.yaml` | `spatial_linear_patch_cnn.yaml` | - | - |
| `spatial_session_*` | `spatial_session_patch_linear.yaml` | `spatial_session_patch_mlp.yaml` | `spatial_session_patch_cnn.yaml` | - | - |
| `spatial_perceiver_*` | `spatial_perceiver_patch_linear.yaml` | `spatial_perceiver_patch_mlp.yaml` | `spatial_perceiver_patch_cnn.yaml` | - | - |

## API Reference (Practical)

### Build a tokenizer + POYO model

```python
from foundry.models import (
    EEGTokenizer,
    POYOEEGModel,
    SpatialProjectionStrategy,
    PerceiverSpatialProjector,
    PatchLinearEmbedding,
)

embed_dim = 128
num_channels = 64
num_sources = 10
patch_samples = 25

tokenizer = EEGTokenizer(
    channel_strategy=SpatialProjectionStrategy(
        num_channels=num_channels,
        num_sources=num_sources,
        projector=PerceiverSpatialProjector(
            num_sources=num_sources,
            d_attn=64,
            num_heads=4,
        ),
    ),
    temporal_embedding=PatchLinearEmbedding(
        embed_dim=embed_dim,
        num_input_channels=num_sources,
        patch_samples=patch_samples,
    ),
    embed_dim=embed_dim,
    patch_duration=0.1,
)

model = POYOEEGModel(
    tokenizer=tokenizer,
    readout_specs=readout_specs,  # list[str], list[ModalitySpec], or dict
    embed_dim=embed_dim,
    sequence_length=30.0,
)
```

### Perceiver IO components directly

`backbones/perceiver.py` exposes:

- `PerceiverEncoder`
- `PerceiverProcessor`
- `PerceiverDecoder`
- `PerceiverIOBackbone` (wrapper around the three components)

### Baselines

`baselines.py` provides non-Perceiver references:

- `TemporalConvAvgPoolClassifier`
- `ShallowConvNet`
- `EEGNetEncoder`

These use the same multitask readout interface, but do not use `EEGTokenizer`.
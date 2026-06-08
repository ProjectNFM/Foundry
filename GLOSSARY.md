# Foundry

End-to-end training framework for EEG foundation models built on top of torch_brain. Foundry provides EEG-specific tokenization, Perceiver-based architectures, and a task-driven training system.

## Language

### Tokenization & Embedding

**Tokenization**:
The CPU-side process of converting a `Data` sample into a structured input dictionary for the model. Decomposes continuous recordings into discrete tokens: signal patches or segments, channel indices, timestamps, session indices, latent positions, and extracted targets.
_Avoid_: Preprocessing, featurization, feature extraction

**Token**:
A discrete unit produced by Tokenization. For input signals, a token is a segment of waveform data (a Patch or per-channel time slice) paired with its timestamp and channel identity. For outputs, a token is a query position defined by a timestamp and task index.
_Avoid_: Feature, sample, frame

**Patch**:
A fixed-duration sub-window of signal produced by unfolding along the time axis. Each Patch becomes one input Token.
_Avoid_: Frame, segment, chunk

**Channel Strategy**:
The method by which variable channel counts across sessions are handled during Tokenization. Determines whether channels are padded to a fixed count, processed independently, or projected into a shared space.
_Avoid_: Channel mode, input strategy

**Signal Embedding**:
The GPU-side learned transform that maps raw input Tokens (patch waveforms, signal segments) into dense vectors in the backbone's embedding space. Implementations include linear projections, CNNs, and continuous wavelet transforms.
_Avoid_: Tokenizer (when meaning the embedding layer), feature extractor, encoder (when meaning this layer)

**Temporal Embedding**:
The time-aware component of Signal Embedding that converts raw signal segments into dense vectors while preserving temporal structure. Encompasses both the learned signal transform (CWT, CNN, patched linear) and the continuous rotary position encoding derived from timestamps.
_Avoid_: Positional encoding (when meaning the full layer), time encoding

### Model Architecture

**Backbone**:
The Perceiver-IO architecture that transforms input tokens into output representations through a latent bottleneck. Composed of an Encoder, Processor, and Decoder in sequence.
_Avoid_: Transformer, network, architecture

**Encoder**:
Cross-attention module where latents attend to input tokens, compressing variable-length inputs into a fixed set of latent vectors.
_Avoid_: Input layer, cross-attention block (when referring to this specific component)

**Processor**:
Self-attention stack that refines latent representations over multiple layers. Its depth is the primary capacity knob of the model.
_Avoid_: Transformer layers, self-attention block (when referring to this specific component)

**Decoder**:
Cross-attention module where output queries attend to processed latents, producing per-query representations for task-specific readout. Not an autoregressive sequence decoder.
_Avoid_: Output layer, generator

**Latent**:
A learned query vector positioned at a regular time grid within the backbone. Latents form the fixed-size bottleneck between encoder and decoder — they are not derived from any input signal.
_Avoid_: Query token, memory token, hidden state

**Readout Head**:
A small `nn.Module` that projects backbone output embeddings to task-specific predictions. Pure forward pass — no loss, no data logic.
_Avoid_: Output head, decoder (when meaning projection), classifier

**Readout Router**:
The module that dispatches output embeddings to the correct Readout Head based on task index. Provides a single-task fast path that skips indexing entirely.
_Avoid_: MultitaskReadout, dispatcher

### Tasks & Training

**Task**:
Any training objective — supervised (classification, regression) or self-supervised (MAE reconstruction, contrastive). The unit of composition for multitask training.
_Avoid_: Modality (when meaning task), readout (when meaning task), objective

**Task Config**:
A Hydra-instantiable structured configuration that wires together a Readout Head, Target Extractor, Task Loss, and metrics for one Task. The single source of truth for everything a Task needs.
_Avoid_: ModalitySpec, readout spec, task definition

**Target Extractor**:
A frozen dataclass callable that pulls targets (timestamps and values) from a `Data` object during Tokenization. A pure data concern — no `nn.Module`, no GPU, no embed_dim dependency.
_Avoid_: Label loader, target transform, prepare_for_multitask_readout

**Task Index**:
An integer tensor that maps each output query to its corresponding Readout Head. Enables multiple tasks to share a single forward pass through the backbone.
_Avoid_: Decoder index, output_decoder_index, modality ID

**Task Loss**:
A composable loss function specific to one Task. Uniform signature: `(predictions, targets, sample_weights) → scalar`. Swappable from config without changing Python.
_Avoid_: Loss (as a class hierarchy), criterion

**SSL Strategy**:
A Lightning Callback that modifies how the model trains for self-supervised objectives. Hooks into the training loop to alter inputs, capture intermediates, and compute auxiliary losses. The model's forward pass stays unchanged.
_Avoid_: SSL readout, wrap_backbone, training wrapper

**FoundryModule**:
The unified LightningModule that orchestrates training for all task types. Delegates loss computation to per-task Task Losses, metric management to per-task MetricCollections, and SSL execution to SSL Strategy callbacks.
_Avoid_: ClassificationModule, RegressionModule, training module

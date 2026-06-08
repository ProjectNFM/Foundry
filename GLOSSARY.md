# Foundry

End-to-end training framework for EEG foundation models built on top of torch_brain. Foundry provides EEG-specific tokenization, Perceiver-based architectures, and a task-driven training system.

## Language

### Tokenization & Embedding

**Tokenization**:
The CPU-side process (running in the dataloader) that converts a raw `Data` sample into a structured input dictionary ready for the model. Tokenization crops the signal to the **context window**, decomposes the continuous recording into discrete **tokens** — signal **patches** or per-channel time slices — and assembles all supporting metadata: channel indices, timestamps, **session** indices, **latent** positions, and extracted **task** targets. This is where the boundary between data and model is drawn: everything upstream is I/O, everything downstream is computation.
*Avoid*: Preprocessing, featurization, feature extraction

**Token**:
A discrete element in the model's sequence interface, always paired with a timestamp. Foundry distinguishes three families of tokens. *Input tokens* are segments of waveform data (a **patch** or per-channel time slice) combined with a timestamp and channel identity. *Latent tokens* are learned query vectors on a fixed time grid that form the **backbone**'s bottleneck — they are not derived from any input signal. *Output tokens* (decoder queries) are positions where predictions are requested, defined by a timestamp, **session** index, and **task** index.
*Avoid*: Feature, sample, frame

**Embedding**:
A learned or fixed transform that maps a **token** (or an identifier such as a session ID or channel ID) into a dense vector in the model's embedding space. Examples include **signal embedding**, **time embedding**, session embedding, and channel embedding.

**Patch**:
A fixed-duration sub-window of signal produced by unfolding the recording along the time axis (`torch.unfold`). Each patch becomes one input **token**. Patches tile the **context window**; their duration and stride are set via configuration (`patch_duration`, `stride`). In non-per-channel mode, one patch produces one token; in per-channel mode, one patch produces one token per channel.
*Avoid*: Frame, segment, chunk

**Spatial Embedding Strategy**:
The strategy chosen to embed the spatial (channel) dimensions of a signal. Controls how multi-channel electrode layouts are transformed before temporal processing — options range from per-channel identity embeddings to learned spatial projections that compress variable electrode counts into a fixed number of latent sources.
*Avoid*: Channel mode, input strategy

**Temporal Embedding Strategy**:
The strategy chosen to embed the temporal dimensions of a signal. Determines how **patches** or time slices are projected into the model's embedding space — implementations include linear projections, CNNs, MLPs, and continuous wavelet transforms.
*Avoid*: Time encoding, positional encoding, time embedding

**Signal Embedding**:
The GPU-side learned transform that maps raw input **tokens** (patch waveforms or signal segments) into dense vectors in the **backbone**'s embedding space. This is composed of a **spatial embedding strategy** and a **temporal embedding strategy** orchestrated by the `EEGTokenizer` module. Runs on device during the forward pass, as distinct from **tokenization** which runs on CPU in the dataloader.
*Avoid*: Tokenizer (when meaning the embedding layer), feature extractor, encoder (when meaning this layer)

**Time Embedding**:
A fixed (non-learned) encoding that injects timestamp information into **token** representations, enabling the **backbone**'s attention layers to reason about temporal ordering and relative distances. Currently implemented via Rotary Position Embedding (RoPE). Applied to input tokens, **latent** tokens, and output tokens alike.

### Architecture

**Backbone**:
The Perceiver-IO architecture (`PerceiverIOBackbone`) that transforms input **tokens** into output representations through a **latent** bottleneck. Composed of three stages in sequence: an *Encoder* (cross-attention from latents to input tokens), a *Processor* (self-attention among latents), and a *Decoder* (cross-attention from output query tokens to latents). All attention layers use **time embeddings** (RoPE) to preserve temporal structure. The backbone is agnostic to **task** semantics — it produces generic embeddings that **readout heads** map to predictions.
*Avoid*: Transformer, network, architecture

**Latent**:
A learned query vector positioned at a regular time grid inside the **backbone**. Latents form the fixed-size bottleneck between encoder and decoder — they are not derived from any input signal. Their count is determined by the **context window** length, `latent_step`, and `num_latents_per_step`. Input **tokens** attend *into* latents during encoding; output **tokens** attend *from* latents during decoding. Note: the term "latent" also appears in the spatial projection layer (`PerceiverSpatialProjector`) to describe compressed channel sources — these are a separate concept from backbone latents.
*Avoid*: Query token, memory token, hidden state

**Context Window**:
The fixed temporal span of signal that the model sees for a single forward pass, set by `sequence_length` (in seconds). During training, the sampler (`RandomFixedWindowSampler`) crops random windows of this duration from longer recordings; during **tokenization**, the signal is padded or truncated to exactly this length so batches can collate. The context window length directly determines how many **patches** tile the input and how many **latent** positions span the **backbone**.

### Task & Readout

**Task**:
Any training objective — supervised (classification, regression) or self-supervised (MAE reconstruction, contrastive). A task is the unit of composition for multitask training: it bundles a target extraction rule, a loss function, and a **readout head**. Each output **token** carries a task index that tells the **readout router** which head to dispatch to. Multiple tasks can coexist in a single training run, each producing its own predictions and losses over the same **backbone** representations.
*Avoid*: Modality (when meaning task), readout (when meaning task), objective

**Readout Head**:
A small `nn.Module` (typically a linear projection) that maps **backbone** output embeddings to **task**-specific predictions. Pure forward pass — no loss computation, no data logic. Each task has its own readout head with an output dimensionality matching the task's label space.
*Avoid*: Output head, decoder (when meaning projection), classifier

**Readout Router**:
The module that dispatches output **token** embeddings to the correct **readout head** based on **task** index. When only a single task is active, the router provides a fast path that skips indexing entirely.
*Avoid*: MultitaskReadout, dispatcher

### Data Concepts

**Session**:
A distinct recording — one subject, one electrode montage, one experimental run. Each session carries a unique string ID (`data.session.id`) and its own channel layout. The model learns a per-session embedding (`session_emb`) that is added to both input and output **tokens**, allowing the **backbone** to condition on recording identity. When using session-specific spatial projectors, the session ID also selects the correct channel-to-source mapping. In the parent POYO framework, sessions decouple multi-subject training without assuming electrode correspondence across recordings.
*Avoid*: Recording (as a synonym — session is the canonical term)
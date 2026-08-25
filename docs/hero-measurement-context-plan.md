# HERO sample-derived relational context plan

**Status:** Proposed minimal follow-on to HERO v1  
**Date:** 2026-08-25  
**Scope:** Give HERO a per-channel relational context derived from the exact
unnormalized sample already supplied to the model. Channel type and absolute
electrode position are optional additional sources. No extra context windows,
calibration segments, recording caches, or sampling rules are introduced.

## Relationship to HERO v1

This is a controlled extension of the
[HERO v1 architecture](hierarchical-eeg-representation-plan.md), not the full
dual-stream content-context architecture deferred in the
[research roadmap](../experiments/ROADMAP.md).

HERO v1 applies the same local encoder to each normalized channel and then
fuses the unordered channel set with learned spatial slots. The fused output is
invariant to reassigning signals between channels because no stable channel
descriptor reaches the mixer. That is useful for montage-agnostic processing,
but it prevents the model from representing fixed spatial contrasts such as C3
versus C4.

The proposed extension branches the same sampled signal inside HERO:

1. a normalized content branch retains the existing shared channel encoder;
2. an unnormalized context branch summarizes every channel and performs
   cross-channel attention;
3. the resulting relational vector conditions which channel content each
   spatial slot reads;
4. coarse channel type and absolute position, when available, provide separate
   optional routing terms.

The temporal hierarchy, task decoder, output timestamps, and public fused
content interface remain unchanged.

## Central hypothesis

A low-bandwidth cross-channel context computed from the same unnormalized
sample can learn most of the useful channel relationships needed for spatial
fusion, including in iEEG recordings that have no trustworthy absolute
electrode coordinates.

On datasets with known positions, absolute coordinates may add an anatomical
anchor. The primary comparison is whether relational-only context performs as
well as relational context plus absolute position.

## Terminology

The context produced by this design is **sample-derived relational context**.
It is not guaranteed to be physical electrode position.

Relationships in unnormalized neural signals can reflect:

- physical proximity;
- common reference and volume conduction;
- functional coupling;
- anatomical region;
- spectral composition;
- amplitude scale and impedance;
- noise and channel quality;
- task-dependent activity in the current sample.

The model is allowed to use these relationships because they are part of the
same example given to the predictor. The context stream should not be described
as recording-stable anatomy unless separate evidence demonstrates that
property.

## Goals

1. Add no new dataset sampling, calibration, history, or caching behavior.
2. Derive one compact context vector per channel from the same sample used by
   the content branch.
3. Let channels communicate before the spatial slots select content.
4. Avoid a learned vocabulary of electrode names.
5. Make absolute position optional rather than required.
6. Support coordinate-poor EEG, ECoG, and SEEG through relational context.
7. Preserve channel-order equivariance before fusion and joint-permutation
   invariance after fusion.
8. Keep context low-bandwidth and restrict it to spatial-slot routing in the
   first implementation.
9. Preserve HERO's temporal hierarchy and post-fusion scaling.
10. Make local context, cross-channel relationships, type, and absolute
    position independently ablatable.

## Non-goals

- Do not sample a separate context segment.
- Do not add causal history, recording-level context, or test-time adaptation.
- Do not cache channel vectors across examples.
- Do not embed subject, session, dataset, task, or opaque channel IDs.
- Do not require electrode-name cleanup for coordinate-poor data.
- Do not claim that the relational context reconstructs physical geometry
  without a direct probe.
- Do not pass the context vector directly to the task decoder.
- Do not use context as the spatial-slot value in the first implementation.
- Do not run cross-channel attention over every channel-time token.

## Architecture

```text
                         one sampled signal [B,C,T]
                                   │
                    sanitize and resample once
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
      normalized content branch          unnormalized context branch
                │                                     │
     shared local channel encoder        shared temporal summarizer
                │                                     │
 channel content [B,C,T,D]            local summaries [B,C,Dc]
                │                                     │
                │                           cross-channel attention
                │                                     │
                │                         relational context [B,C,Dc]
                │                                     │
                │                          + optional channel type
                │                          + optional absolute position
                │                                     │
                └────────────── context-aware spatial slots
                                                      │
                                           fused content [B,T,D]
                                                      │
                                        existing HERO temporal hierarchy
```

There is one input example, one time interval, one set of channel masks, and one
resampling decision. The two branches are internal model computations over the
same example.

## Single-sample input contract

The public API remains signal-first:

```python
representation = model.encode(
    signal,                       # [B, C, T], unnormalized
    sampling_rate=...,            # or explicit timestamps
    channel_mask=None,            # [B, C]
    sample_mask=None,             # [B, T]
    channel_type=None,            # optional [B, C]
    channel_position=None,        # optional [B, C, 3]
    channel_position_valid=None,  # optional [B, C]
)
```

No `context_signal`, `context_duration`, calibration loader, or context cache is
part of this design.

The unnormalized input is the common source for both branches. HERO constructs
the normalized content view internally.

## Data preparation and normalization

HERO v1 currently normalizes each channel during tokenization and then
resamples it. The context branch needs the corresponding unnormalized sample.
To avoid duplicate resampling, use this order:

1. resolve and filter neural channels;
2. sanitize non-finite values and construct masks;
3. calculate per-channel mean and standard deviation from valid source samples;
4. resample the unnormalized signal once onto the canonical grid;
5. pass the resampled unnormalized signal to the context branch;
6. apply the stored per-channel statistics to construct the normalized content
   view;
7. pass the normalized view to the existing channel-content encoder.

Conceptually:

```python
raw = sanitize_and_resample_once(signal)
normalized = (raw - source_mean) / source_std

channel_content = content_encoder(normalized)
channel_context = relational_context_encoder(raw)
```

Resampling and affine normalization are linear operations away from finite
filter boundaries, but this reordering is not assumed to be numerically
identical to HERO v1. Deterministic comparison tests must quantify boundary and
representation differences. The signal-only control used in the experiment
must use the same refactored one-resample data path so that context is the only
trained difference.

If raw units create optimization problems across datasets, a later controlled
variant may present the context encoder with normalized waveform shape plus
explicit raw mean and log standard deviation. That is not the first condition
because the current hypothesis specifically concerns relationships available
in the unnormalized sample.

## Local channel-context summarizer

The context branch first compresses every channel independently with shared
weights:

```text
raw channel [T]
    -> small shared temporal convolution stack
    -> bounded temporal pooling
    -> local summary [D_context]
```

Recommended initial design:

- two or three Conv1d blocks shared across channels;
- small channel width, for example 32;
- bounded downsampling before global pooling;
- masked mean or learned single-query pooling;
- one output vector per channel;
- no channel-index embedding;
- no task-specific parameters.

This vector may contain amplitude, spectral, quality, and temporal-shape
information from the current sample. The independent-summary control exposes
these vectors to slot routing without cross-channel attention, allowing the
experiment to distinguish local raw-signal information from learned
relationships.

## Cross-channel relational encoder

The local summaries form a masked channel set:

```text
local summaries [B,C,D_context]
    -> cross-channel self-attention
    -> feed-forward block
    -> optional second block
    -> relational context [B,C,D_context]
```

Recommended initial design:

- two transformer-style blocks;
- four attention heads;
- context width 32 or 64;
- no channel-order positional embedding;
- no channel-name or channel-index embedding;
- standard channel mask in attention and output zeroing.

The encoder must be permutation-equivariant. For any channel permutation `P`:

```text
RelationalEncoder(Ps, Pmask) == P RelationalEncoder(s, mask)
```

Cross-channel attention is applied once to `C` summary vectors, not separately
at every time step. Its cost is `O(B * C^2 * D_context)` per example. This is
acceptable for the initial 2/16/64-channel targets and remains separate from
the duration-scaling cost of the temporal hierarchy. Larger channel-count
regimes can test sparse attention later if this mechanism is useful.

## Optional channel type

Use a small fixed ontology rather than free-form strings:

| Index | Type |
|---:|---|
| 0 | unknown |
| 1 | EEG |
| 2 | ECoG |
| 3 | SEEG |
| 4 | LFP |
| 5 | microelectrode |

Channel type is encoded separately and contributes its own spatial-routing
term. It helps distinguish sensor families but cannot identify two channels of
the same type. Type-only is therefore a useful added-parameter control.

## Optional absolute electrode position

When trustworthy positions exist, encode normalized 3-D coordinates with a
small Fourier-feature or radial-basis encoder followed by an MLP.

For standard scalp EEG, a data-layer resolver may map names such as `C3`, `C4`,
or `Cz` to a standard montage. The name is a deterministic lookup key only; it
is never embedded by the model.

For ECoG or SEEG, use numeric positions only when their coordinate frame and
units are explicit. Otherwise set `channel_position_valid=false` and rely on
relational context plus channel type.

Unknown positions receive an explicit learned missing-position representation.
An all-zero coordinate alone must not be interpreted as a real location.

Absolute position is an optional source and an experimental comparator. The
model must remain usable when every position is missing.

## Context-aware spatial slots

Keep context sources separate in the attention logits. For spatial slot `s`,
head `h`, channel `c`, and time `t`:

```text
content_logit = q_content[s,h] dot K_content(x[c,t]) / sqrt(d)
local_logit = q_local[s,h] dot K_local(local_context[c]) / sqrt(dc)
relation_logit = q_relation[s,h] dot K_relation(relation_context[c]) / sqrt(dc)
type_logit = q_type[s,h] dot K_type(type_context[c]) / sqrt(dc)
position_logit = q_position[s,h] dot K_position(position_context[c]) / sqrt(dc)
```

The active condition selects the applicable terms:

```text
logit = content_logit
      + gamma_local[h] * local_logit
      + gamma_relation[h] * relation_logit
      + gamma_type[h] * type_logit
      + position_valid[c] * gamma_position[h] * position_logit
```

Then:

```text
weight = softmax_channel(logit, channel_mask)
slot[s,h,t] = sum_channel(weight * V_content(x[c,t]))
```

Only normalized channel content enters `V_content`. Context changes selection
but is not itself a value supplied to the temporal hierarchy or decoder.

Each source has a separately logged learned gate. This provides exact ablations
and makes it possible to see whether the model uses relational, type, or
absolute-position context.

## Why same-sample context is still constrained

Because the context branch sees the target sample, it can contain task-relevant
information. This is intentional and not considered data leakage: it is part of
the input supplied to the predictor.

The branch is nevertheless constrained to test a particular mechanism:

1. one vector per channel rather than a time-resolved context sequence;
2. narrow context width;
3. shared per-channel summarization;
4. permutation-equivariant cross-channel processing;
5. routing-only access to the spatial slots;
6. no direct decoder or task-head access;
7. normalized signal remains the only source of slot values.

These restrictions do not prove that context represents physical geometry.
They test whether a compact description of the current channel set helps HERO
select and compare channel content.

## Invariance contract

Let `x` be the signal, `t` channel type, `p` optional absolute position, `u`
the channel mask, and `P` a channel permutation. The model must satisfy:

```text
f(Px, Pt, Pp, Pu) == f(x, t, p, u)
```

This is invariance to storage order. The local and relational context vectors
must permute with the input before fusion.

When position is supplied, the model should not generally satisfy:

```text
f(Px, t, p, u) == f(x, t, p, u)
```

because that operation moves signals between physical positions without moving
their descriptors.

With relational-only context and no absolute anchors, the model remains
invariant to jointly relabeling all channels. It learns relationships within
the observed set rather than a universal anatomical label.

## Configuration

```yaml
model:
  channel_context:
    mode: disabled
    # disabled | local | relational | position | relational_position

    context_dim: 32

    local_encoder:
      num_layers: 3
      kernel_size: 7
      pooling: learned_query

    relational_encoder:
      num_blocks: 2
      num_heads: 4

    channel_type:
      enabled: true

    absolute_position:
      enabled: false
      encoding: fourier
      num_fourier_bands: 6

    spatial_routing:
      separate_source_logits: true
      learned_gate: per_source_per_head
      use_context_in_values: false
```

The default remains `disabled` until the controlled experiment supports a
change.

## Phased implementation plan

### Phase 0: Refactor the single-sample data path

**Objective:** Produce unnormalized and normalized views of one resampled sample
without changing dataset sampling.

Implement:

1. Sanitize and resample the raw signal once.
2. Preserve source-sample per-channel normalization statistics.
3. Construct the normalized view after resampling.
4. Pass the raw canonical view and normalized content view through batching.
5. Keep one channel mask, sample mask, timestamp sequence, and target set.

Tests:

- the context and content views cover the exact same samples and timestamps;
- masks are identical and aligned;
- invalid values cannot affect either branch;
- padded channels are zero in both views;
- no second dataset sample or context window is requested;
- the signal-only model using the refactored path is compared numerically and
  through a short training smoke test with the current HERO v1 path.

**Exit criterion:** One-resample branching is correct and any difference from
the current normalization order is measured and documented.

#### Phase 0 implementation outcome

Implemented on 2026-08-25 with the following concrete semantics:

- `encode(signal=...)` now treats `signal` as unnormalized and creates both
  views internally.
- Non-finite values in a valid channel invalidate that source time for the
  shared sample mask. Non-finites in padded channels are ignored.
- Per-channel mean and population standard deviation are calculated from the
  valid source-rate samples. A scale of one is used for constant, empty, and
  padded channels so normalization remains finite.
- The sanitized raw signal is resampled once. The normalized content view is
  then constructed on the canonical grid using the stored source statistics.
- Tokenized batches carry raw `input_values`, normalized
  `input_content_values`, `input_normalization_mean`, and
  `input_normalization_std`. Both views share `input_timestamps`,
  `sample_mask`, and `channel_mask`; task extraction is unchanged.
- `forward()` consumes the prepared content view directly when it is present,
  avoiding a second resampling or normalization pass. Direct callers that
  supply only raw `input_values` use the same preparation path as `encode()`.

The deterministic 256 Hz to 128 Hz comparison uses a two-second, nonzero-mean
5 Hz plus 17 Hz analytic signal. Relative to HERO v1's normalize-then-resample
order, the refactored content view has mean/max absolute interior error
`0.01012/0.01012` normalized units (excluding 24 samples per boundary). Across
the complete signal, including filter boundaries, mean/max absolute error is
`0.05012/5.58427`. With the fixed small test model, the resulting complete
representation has mean/max absolute difference `0.05698/3.82955`. Both the
legacy prepared path and the refactored path complete a finite forward,
backward, and optimizer step in the deterministic smoke test.

### Phase 1: Local and relational context encoders

**Objective:** Produce one permutation-equivariant relational vector per channel
from the unnormalized current sample.

Implement:

1. Shared local temporal context encoder.
2. Mask-aware temporal pooling to one vector per channel.
3. Two-block cross-channel self-attention encoder.
4. Context width and attention diagnostics.

Tests:

- local summaries are independent across channels before relation attention;
- relational outputs respond to changing another valid channel;
- masked or padded channels cannot affect valid context;
- channel permutation produces the same permutation of context vectors;
- context output is time-constant;
- gradients reach both the local and relational encoders;
- full forward/backward remains finite under representative raw input scales.

**Exit criterion:** Relational context is mechanically correct before it is
connected to spatial fusion.

#### Phase 1 implementation outcome

Implemented on 2026-08-25 with the following concrete semantics:

- `channel_context_mode` is `disabled` by default and supports `local` and
  `relational` Phase 1 modes. The branch owns no parameters when disabled and
  remains disconnected from spatial fusion in every mode until Phase 3.
- `encode_channel_context(...)` uses the same signal-first preparation path as
  `encode(...)`, so the context encoder receives the exact sanitized,
  once-resampled raw canonical view from Phase 0.
- The shared local encoder applies configurable same-padded Conv1d blocks,
  per-time feature normalization and GELU, then mask-aware bounded average
  downsampling after each block. A learned single query pools the bounded grid
  to one `context_dim` vector per channel.
- The relational encoder uses two configurable pre-normalized transformer
  blocks by default. It has no channel index, name, or positional embedding;
  invalid keys and queries are zeroed explicitly, including all-channel-masked
  examples.
- `ChannelContext` exposes local vectors, optional relational vectors, temporal
  pooling weights, per-block/per-head relational attention weights, and the
  context width. Both vector outputs are time-constant with shape `[B,C,Dc]`.
- Contract tests cover local independence, valid-channel relational response,
  masking and all-masked batches, joint permutation equivariance of vectors
  and diagnostics, explicit local/relational modes, gradient reachability, and
  finite full-model backward passes across raw channel scales from `1e-4` to
  `1e4`.

### Phase 2: Optional static metadata sources

**Objective:** Add channel type and optional absolute position without making
either mandatory.

Implement:

1. Coarse channel-type mapping and embedding.
2. Optional standard-montage name-to-position resolver for scalp EEG.
3. Position-valid masks and missing-position embedding.
4. Small absolute-position encoder.

Tests:

- unresolved names remain valid signal channels;
- all-position-missing batches are supported;
- no electrode-name string enters the model;
- type and position follow channel permutation, padding, and filtering;
- coordinate-frame mismatches are rejected or treated as missing.

**Exit criterion:** Type and position can be supplied independently, and
coordinate-poor iEEG requires no special sampling or data path.

#### Phase 2 implementation outcome

Implemented on 2026-08-25 with the following concrete semantics:

- Channel type uses the fixed six-entry ontology from this plan. Tokenization
  converts data-layer strings to integer indices; generic or unrecognized
  values, including ambiguous `iEEG`, map to `unknown`. No electrode-name
  string crosses the model boundary.
- `ChannelTypeEncoder` embeds the ontology independently per channel and zeros
  padded channels. Type routing is independently enabled with
  `channel_type_enabled`.
- `AbsolutePositionEncoder` applies configurable Fourier features and a small
  MLP to normalized 3-D numeric coordinates. It owns an explicit learned
  missing-position vector, while the routing term is validity-masked so a
  missing position cannot behave like a shared real coordinate.
- Explicit data-layer coordinates are accepted only when all selected channels
  declare one recognized coordinate frame and unit. Mixed, absent, or unknown
  frame/unit metadata makes those coordinates missing without invalidating the
  signal channels. Meter, centimeter, millimeter, micrometer, and already
  normalized coordinates are converted to fixed 10-cm-head-radius units.
- Missing EEG coordinates can be resolved from bare channel names through MNE
  `standard_1020`/`standard_1005` lookup. Names are used only for this
  deterministic CPU-side lookup. Unresolved EEG names and coordinate-poor
  ECoG/SEEG channels remain valid with `channel_position_valid=false`.
- NeuralBench normalized head-frame positions are preserved by its adapter,
  including its documented unresolved-channel sentinel. Tokenized batches now
  carry `channel_type`, `channel_position`, and `channel_position_valid`.
- Tests cover all-position-missing inputs, unresolved names, mixed coordinate
  frames, padding/filtering, independent type/position permutation, missing
  versus real zero coordinates, and NeuralBench metadata preservation.

### Phase 3: Context-conditioned spatial-slot routing

**Objective:** Let spatial slots select normalized channel content using local,
relational, type, and optional position context.

Implement:

1. Separate key projections and attention-logit terms per context source.
2. Learned per-source, per-head gates.
3. Routing-only context access.
4. Context shuffling hooks used only by controlled experiments.
5. Logging of gate values, logit scales, and spatial attention summaries.

Tests:

- all context gates disabled reproduce the refactored signal-only mixer;
- jointly permuting all channel-aligned inputs leaves fused output unchanged;
- shuffling relational context across channels can change output;
- context never enters slot values or the task decoder directly;
- unknown positions do not behave as a shared real coordinate;
- temporal complexity after spatial fusion remains unchanged;
- the additional cross-channel cost is measured separately as a function of
  channel count.

**Exit criterion:** Context affects only the intended routing mechanism and the
controls are exact.

#### Phase 3 implementation outcome

Implemented on 2026-08-25 with the following concrete semantics:

- The spatial mixer retains its original normalized-content key and value
  projections. Local, relational, type, and position sources each own a
  separate context key projection, slot/head query, and learned per-head scalar
  gate. Context is never concatenated into or substituted for slot values.
- Gates initialize to exact zero. A source is ablated by setting its gate to
  zero; with every context gate zero, the contextual model reproduces the
  refactored signal-only representation exactly.
- `channel_context_mode` supports `disabled`, `local`, `relational`,
  `position`, and `relational_position`. Type and absolute position can also be
  toggled independently, allowing all planned Phase 4 conditions without a
  separate data path.
- `relational_context_permutation` is an explicit experimental-only hook that
  must contain each channel index exactly once. Normal tokenization never emits
  it. Reassignment changes routing when the relational gate is active.
- `Representation.spatial_routing` exposes detached per-source/per-head gates,
  per-source logit RMS values, and spatial attention averaged over valid time.
  `Representation.channel_context` retains the local/relational diagnostics.
- `estimate_relational_context_cost()` separately reports attention elements
  and approximate multiply-adds as a function of batch and channel count. It
  is quadratic in channels and independent of sample duration; the existing
  post-fusion temporal encoder and its output shapes are unchanged.
- Tests establish exact disabled-gate equivalence, joint permutation
  invariance with all sources active, relational-shuffle sensitivity,
  routing-only context access, absent-position neutrality, source-specific
  diagnostics, gradient reachability for every source and gate, unchanged
  temporal scaling, and explicit quadratic cross-channel cost.

### Phase 4: Motor Imagery sufficiency experiment

**Primary question:** Is same-sample relational channel context sufficient for
MI, or does known absolute electrode position add useful information?

Keep fixed:

- NeuralBench MI dataset, split, and target contract;
- flat temporal mode and eight spatial slots;
- content channel encoder, temporal depth, width, and local window;
- optimizer, scheduler, precision, batch size, and full epoch budget;
- seeds 33, 34, and 35;
- best-validation-checkpoint test evaluation.

Conditions:

| Condition | Local raw summary | Cross-channel relations | Type | Absolute position | Purpose |
|---|:---:|:---:|:---:|:---:|---|
| Signal-only | No | No | No | No | Refactored HERO control. |
| Type-only | No | No | Yes | No | Added-parameter control with no within-EEG identity. |
| Local-context | Yes | No | Yes | No | Tests unnormalized per-channel information without relationships. |
| Relational-only | Yes | Yes | Yes | No | Main coordinate-free intervention. |
| Position-only | No | No | Yes | Yes | Tests known anatomy without inferred relationships. |
| Relational + position | Yes | Yes | Yes | Yes | Tests the incremental value of absolute position. |
| Shuffled relational | Yes | Yes, misassigned | Yes | No | Tests correct channel-context binding. |

Primary metric: three-seed mean held-out test balanced accuracy.

Also report:

- per-class recall and the left-fist/right-fist confusion;
- train and validation curves;
- context-source gate values;
- relational attention maps;
- spatial-slot attention summarized by known scalp position;
- parameter count, peak memory, wall-clock time, and selected epoch.

Define relational sufficiency before launch. A suggested criterion is:

1. relational-only improves meaningfully over signal-only and local-context;
2. relational-only is within 2 percentage points of relational-plus-position;
3. shuffled relational context loses the relational-only improvement;
4. the result is consistent across seeds.

Interpretation:

- Relational beats local: cross-channel relationships add value.
- Local matches relational: raw per-channel summaries, not relations, explain
  the gain.
- Relational matches relational-plus-position: absolute position is not needed
  for this task under the tested montage.
- Relational-plus-position wins clearly: absolute anatomy remains useful.
- Position-only wins but relational does not: the signal-derived encoder fails
  to infer the needed stable spatial structure.
- Shuffled relational also improves: capacity or a global sample summary, not
  correct channel binding, is the likely mechanism.
- No context condition helps: investigate information loss in the normalized
  channel-content encoder and optimization before adding more context capacity.

**Exit criterion:** Select the simplest supported routing sources or retain
signal-only HERO.

### Phase 5: Coordinate-poor iEEG and cross-task validation

Run this phase only if relational context is useful in Phase 4.

**Objective:** Determine whether the coordinate-free relational stream helps
when absolute positions are unavailable and whether optional position remains
safe across tasks.

Evaluate:

1. at least one ECoG or SEEG task with all absolute positions withheld;
2. relational-only versus signal-only and local-context;
3. shuffled relational context;
4. P300 with relational-only, position-only, and their combination;
5. Sleep with relational-only and signal-only;
6. channel removal and variable-channel subsets;
7. mixed batches with present and missing positions if supported by the data
   contract.

Where ground-truth coordinates exist but are deliberately hidden, use them only
for evaluation probes:

- pairwise-distance prediction from relational vectors;
- nearest-neighbor recovery;
- correlation between learned attention and physical distance.

These probes test whether physical geometry emerges. They are not training
losses in the first experiment.

**Exit criterion:** Relational context improves at least one credible
coordinate-poor task, degrades gracefully with variable channels, and does not
require any extra sampling behavior.

## Evaluation requirements

| Category | Required evidence |
|---|---|
| Task utility | Three-seed mean and spread, held-out test metrics, per-class recall. |
| Relationship specificity | Local-only, relational, and shuffled-relational controls. |
| Position sufficiency | Position-only versus relational-only versus combined. |
| Invariance | Joint channel permutation equality and context-misassignment sensitivity. |
| Mechanism | Source gates, relational attention, and slot-routing summaries. |
| Geometry | Evaluation-only distance/neighborhood probes where positions exist. |
| Efficiency | Parameters, context-branch memory/time, and unchanged post-fusion duration scaling. |

## Main risks and mitigations

### Context becomes a parallel classifier

**Risk:** Same-sample unnormalized summaries carry task labels directly.

**Mitigation:** Restrict context to one vector per channel and spatial-routing
logits. Compare local-only with relational context, and do not expose context
directly to the decoder.

### Raw scale destabilizes training

**Risk:** Physical units and amplitude ranges differ greatly across datasets.

**Mitigation:** Audit raw distributions, use numerically stable convolutions and
clipping only if pre-registered, and retain a follow-up option that separates
normalized waveform shape from explicit raw mean/log-scale.

### Relational context captures reference rather than geometry

**Risk:** Cross-channel dependencies are dominated by the referencing scheme or
common noise.

**Mitigation:** Call the output relational context rather than inferred
position. Use geometry probes and test robustness across references when data
permits.

### Cross-channel attention adds capacity without useful binding

**Risk:** Gains arise from more parameters or a global sample statistic.

**Mitigation:** Include local-only and shuffled-relational controls, log
attention behavior, and parameter-match if the initial result is ambiguous.

### Normalization refactor changes the baseline

**Risk:** Resampling raw before applying stored normalization statistics changes
HERO independently of context.

**Mitigation:** Use the refactored path for every experimental condition and
quantify it against the current path before interpreting context gains.

### Quadratic channel cost becomes limiting

**Risk:** `O(C^2)` summary attention is expensive for very high channel counts.

**Mitigation:** Measure it explicitly. Do not introduce sparse or low-rank
attention until usefulness is established; that is a later efficiency
hypothesis.

### Absolute positions are inconsistently defined

**Risk:** Scalp and intracranial coordinates use incompatible frames.

**Mitigation:** Treat position as optional, carry validity/frame provenance,
and never guess a coordinate for unresolved iEEG contacts.

## Decisions before implementation

1. Whether local temporal pooling begins with masked mean or a learned query.
2. Context width 32 versus 64.
3. Whether raw input requires a fixed numerical clipping policy.
4. Initialization of the context-source gates.
5. The standard scalp coordinate encoding used by the optional position branch.
6. The pre-registered minimum MI improvement and 2-point sufficiency margin.

Recommended starting choices are width 32, learned single-query temporal
pooling, two four-head cross-channel blocks, routing-only context, and separate
per-head gates initialized so the signal-content term dominates initially.

## Handoff checklist

Before coding:

- [ ] Specify the one-resample raw/normalized branching semantics.
- [ ] Audit raw amplitude distributions for the initial datasets.
- [ ] Define the channel-type ontology.
- [ ] Define optional coordinate validity and frame handling.

Before training:

- [ ] Pass single-sample alignment and mask tests.
- [ ] Quantify the normalization/refactor difference from HERO v1.
- [ ] Pass relational permutation-equivariance tests.
- [ ] Verify routing-only access and exact disabled-source controls.
- [ ] Verify the shuffled-relational condition.
- [ ] Create a dedicated experiment hypothesis file.
- [ ] Pre-register the primary metric and relational-sufficiency criterion.
- [ ] Use matched seeds and the full production training schedule.
- [ ] Use the clean committed snapshot workflow, the `long` partition, and
      record Slurm job IDs and snapshot paths.

Before adopting relational context:

- [ ] Demonstrate a gain beyond independent local summaries.
- [ ] Demonstrate dependence on correct channel-context binding.
- [ ] Establish whether absolute position adds meaningful performance.
- [ ] Validate at least one coordinate-poor iEEG task.
- [ ] Report context cost separately from the temporal hierarchy.

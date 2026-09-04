# NeuroSoft convolution--BiGRU implementation handoff

**Status:** Core implementation complete (2026-08-28).
**Purpose:** Record the implemented standalone model used as the matched
from-scratch and supervised-pretraining architecture in Phase 2 onward of the
[NeuroSoft supervised-pretraining roadmap](neurosoft-supervised-pretraining-roadmap.md).

## Implementation completion record

The model and its transfer contract are implemented in
`foundry/models/neurosoft_conv_bigru.py`, exported from `foundry.models`, and
configured by `configs/model/neurosoft_conv_bigru.yaml`. The implementation:

- provides one fresh, session-specific `Linear(C_session, adapter_dim)` input
  adapter per configured recording, with padded channels excluded before the
  adapter;
- implements the shared depthwise-separable temporal frontend, bidirectional
  GRU, masked mean pooling, and shared `ReadoutRouter` described below;
- preserves zero time padding after learned normalization and after each
  temporal block, so a batch's padding cannot affect valid right-edge
  convolution windows or pooled predictions; and
- exposes full-finetuning and frozen-representation component selections.
  `load_pretrained_weights(..., components=...)` records the selected transfer
  boundary, and the standard training entry point accepts
  `run.pretrained_transfer_regime=frozen_representation` together with
  `run.freeze_pretrained=true` to load/freeze only `temporal_frontend` and
  `gru`. The target adapter and router remain newly initialized and trainable.

Focused model, transfer, and transfer-regime regression tests cover this
contract. The separate Phase-2 experiment matrix, FLOP validation, and
end-to-end scientific runs described below remain experiment work rather than
part of this core implementation.

## Decision summary

`NeurosoftConvBiGRU` is implemented as a new standalone model. It is not in
`foundry/models/baselines.py` and does not subclass `BaselineEEGModel`.

The initial, fixed recipe is:

```text
raw 0.5 s NeuroSoft signal at 2,000 Hz, shape (B, C_session, 1,000)
  -> fresh session-specific channel adapter, C_session -> 64
  -> one shared depthwise-separable temporal convolution block
       temporal kernel = 64 samples (32 ms), stride = 4
       64 adapter features -> 128 shared temporal features
  -> shared 2-layer bidirectional GRU, hidden size 128 per direction
  -> length-aware temporal mean pooling, embedding size 256
  -> shared 8-logit task readout
```

The model is offline: bidirectionality may use the whole 0.5 s window.  The
causal requirement in the roadmap applies to train/validation/test partitions,
not to online streaming inference.

The raw waveform is deliberately **not** pre-resampled.  The first convolution
operates on all 1,000 samples.  Its stride is learned internal temporal
compression so that the BiGRU receives about 250 time steps; it is not a claim
that high-frequency raw signal is irrelevant.  This avoids discarding possible
high-frequency ECoG content before the model can represent it.

CNN-to-(Bi)GRU is a conventional neural-signal baseline pattern: the CNN
extracts local/spatial-temporal features and the recurrent stack summarizes
their longer temporal structure.  The exact widths and kernels below are a
declared NeuroSoft recipe, not purported universal optima.  See the
CNN--BiGRU baseline in [Du-IN](https://proceedings.neurips.cc/paper_files/paper/2024/file/92559987ee79e42a2b01d534a54682ee-Paper-Conference.pdf),
the convolutional GRU encoder in [Universal EEG Encoder](https://arxiv.org/abs/1911.12152),
and the EEG CNN--RNN ablation showing the value of a convolutional frontend
and no general benefit from very deep recurrent stacks
[here](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2018.00085/full).

## Scope and non-goals

This is a new architecture for NeuroSoft 8-band acoustic decoding and its
later supervised-pretraining experiments.  It must support minipigs and
monkeys with differing recording channel counts.

It is not:

- a revision of the historical plain `GRU` baseline;
- an EEGNet variant;
- a generic variable-length EEG model for every Foundry dataset;
- a streaming/causal-inference model; or
- an immediate CNN-depth, GRU-depth, width, attention, or multi-scale sweep.

The existing historical GRU results only establish that a simple
single-session BiGRU can learn this task.  They do not validate this model's
transfer boundary or pretraining behavior.

## Module location and minimal integration

The model is implemented in:

```text
foundry/models/neurosoft_conv_bigru.py
```

`NeurosoftConvBiGRU` is exported from `foundry/models/__init__.py`, and its
dedicated Hydra model config is:

```text
configs/model/neurosoft_conv_bigru.yaml
```

Do not modify `foundry/models/baselines.py` except for no changes at all.  The
new class must be a standalone `nn.Module`, own its input adapters, and expose
the normal Foundry model contract:

- `task_configs` property;
- `tokenize(data)` for `NeuralDataModule` workers;
- `forward(input_values=..., task_index=..., ...)` returning the normal
  task-output dictionary;
- `transferable_components()` (defined below).

Reuse only stable existing utilities rather than duplicating task semantics:

- `TaskConfig.normalize_task_configs` and `extract_multitask_targets`;
- `build_readout_router` / `ReadoutRouter` for existing multitask head routing;
- `chain`, `pad8`, and `pad2d` for token collation; and
- the existing `FoundryModule`, metric callbacks, fraction-manifest machinery,
  and `ComputeTrackingCallback`.

It is acceptable for this standalone class to contain a small explicit
tokenization method mirroring the raw EEG/ECoG handling needed by NeuroSoft.
Do not inherit the generic baseline channel strategy solely to reuse that
method.  If duplicated tokenization becomes a maintenance concern, extract a
small neutral task-I/O helper in a separate module; do not refactor every
baseline as part of this implementation.

## Exact architecture

### 1. Input and tokenization

For every window, retain supported EEG/ECoG/iEEG channels in their recorded
order and produce:

```text
input_values:         (C_session, 1000), float32
input_session_ids:    session/recording identifier, string
input_channel_counts: scalar C_session, long
input_seq_len:        scalar 1000, long
task_index:            existing padded task-index tensor
target_values:         existing chained task targets
target_weights:        existing chained target weights
```

The collated forward input is `(B, C_pad, T_pad)`.  The first production
recipe uses fixed `T_pad = 1000`; nevertheless, preserve `input_seq_len` and
build the masking path correctly.  Any time padding must remain zero and must
not contribute to convolutional or pooled representations.

Use the actual session/recording ID as the adapter key.  The key must be
canonical, serializable, and identical in source, target, checkpoints, and
manifests.

### 2. Session-specific input adapter

Implement a model-owned `SessionInputAdapter` rather than reusing
`SessionSpatialProjector`.

For each configured session `s`, register exactly one:

```text
adapter.layers[s]: Linear(C_s, adapter_dim=64, bias=True)
```

At each time point it maps `(C_s)` to `(64)`, producing `(B, 64, T)`.
For a batch with padded channels, slice exactly the valid `C_s` channels before
the layer.  An unknown session ID must raise a clear error -- never silently
fall back to a padded/shared channel projection.

The session adapter is the **only** session-specific part of the model.  Do
not put a trainable `common_layer` inside the adapter: all shared mixing belongs
to the shared encoder below.  This makes the checkpoint policy unambiguous.

Implement an explicit construction path for a downstream target model with
only its target adapter.  It must not require source-session adapter IDs to
exist.  Newly added target adapters use the framework's declared deterministic
initialization under the run seed; record the initializer in the resolved Hydra
config/checkpoint hyperparameters.

### 3. Shared separable temporal block

Input: `(B, 64, 1000)`.

The initial block is:

```text
LayerNorm over the 64 feature channels at each time point
-> depthwise Conv1d(64 -> 64, groups=64, kernel_size=64, stride=4,
                    padding=30)
-> pointwise Conv1d(64 -> 128, kernel_size=1)
-> normalization independent of batch composition
-> GELU
-> dropout
```

Output: `(B, 128, L)`.  The fixed starting shape is exactly `L=250` because
`floor((1000 + 2*30 - 64) / 4 + 1) = 250`.  For future variable-length input,
calculate valid output lengths from this same Conv1d formula and create the
corresponding temporal mask; do not hard-code `250` outside the fixed-shape
test.

Use a per-sample/feature normalization such as `LayerNorm` (after transposing
to `(B, L, 128)`) rather than BatchNorm running statistics.  This keeps the
shared encoder's behavior independent of which sessions are mixed in a batch
and makes frozen-transfer behavior well defined.

The exact dropout probability may start at `0.3` to match the historical GRU
starting point.  Define it as a config field and keep it fixed across Phase-2
scientific comparisons.

`conv_depth` must be configurable but defaults to `1`.  The initial Phase-2
matrix uses only one block.  A later depth-and-transfer study may set
`conv_depth=2` by stacking an explicitly specified second separable block; it
must be a separate experiment with parameter/FLOP reporting, not an
unrecorded per-session adjustment.

### 4. Shared recurrent encoder and pooling

Transpose convolution features to `(B, L, 128)` and pass them to:

```text
nn.GRU(
    input_size=128,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    batch_first=True,
)
```

The default is fixed at two layers and hidden size 128 **per direction**.  Its
output is `(B, L, 256)`.

Use `pack_padded_sequence(..., enforce_sorted=False)` / `pad_packed_sequence`
or an equivalent correct mask-aware implementation.  The final session
embedding is the masked mean over valid output steps:

```text
embedding[b] = sum_t(mask[b, t] * gru_output[b, t]) / sum_t(mask[b, t])
```

Do not use the last recurrent state as the initial classifier representation:
masked mean pooling uses all evidence in the offline window and avoids an
arbitrary directional/end-of-window preference.  Do not add attention pooling
to the base recipe.

Expose `gru_num_layers`, but leave `2` as the only initial scientific setting.
Later use a fixed-width `hidden_size=128` family (for example 1/2/4 layers) if
studying depth.  That is intentionally a **depth-and-scale** study: parameter
count and FLOPs grow with layer count and must be reported rather than matched
away.  Keep hidden width constant as agreed.

PyTorch's native GRU applies its `dropout` only between recurrent layers.
Avoid silently changing regularization in a future 1-versus-2-layer comparison:
use fixed external feature/readout dropout across every depth, and either set
the native GRU dropout to zero or explicitly report inter-layer dropout as
part of the depth intervention.

### 5. Classification output

Use the normal `ReadoutRouter` on the 256-dimensional pooled embedding.  For
the NeuroSoft task it produces eight logits.  Keep all eight output logits even
when a particular target session lacks a label class; absent-class predictions
remain errors and existing supported-class metric aggregation controls only
metric denominators.

The 8-way classifier is shared because the label meanings are invariant across
sessions.  It is part of the transferable shared model during full fine-tuning.

## Checkpoint and transfer contract

The transfer boundary is mandatory.  The implementation must not use the
current generic baseline behavior, which has no GRU transfer policy.

### Transferable parameters

The following are shared and transferable:

- all shared temporal-block normalization, depthwise-convolution, and
  pointwise-convolution parameters;
- all BiGRU parameters;
- pooling-related parameters, if any are added later; and
- the shared 8-way `ReadoutRouter` parameters.

The following must never transfer from a source run to a new target session:

- every `session_adapter.layers.<source_session_id>.*` tensor;
- optimizer state; and
- any session-specific normalization/statistics should such a component ever
  be introduced (it is not part of this design).

Implement `transferable_components()` so the existing validated loader can
select exactly the shared components.  Prefer top-level module names such as:

```python
("temporal_frontend", "gru", "router")
```

Keep the adapter in a separate top-level `session_adapter` module so its
complete exclusion is structurally obvious.  Do not rely on permissive loading
to ignore source adapter keys; they must be deliberately excluded and listed
as such in the `TransferReport`.

### Required transfer modes

1. **Scratch control:** fresh target adapter, fresh frontend/GRU/router;
   all parameters trainable.
2. **Full fine-tuning:** fresh target adapter; load frontend, GRU, and
   classifier from the source checkpoint; all parameters trainable.  This is
   the primary supervised-pretraining comparison.
3. **Frozen shared representation with target adapter and readout:** fresh
   target adapter and fresh classifier; load and freeze only the temporal
   frontend and BiGRU.  Train the target adapter and classifier.  This is not
   a strict linear probe because the input adapter is learned; use this full
   name in reports.

The third mode is implemented through the model's
`transferable_components_for_mode("frozen_representation")`, which returns
`("temporal_frontend", "gru")`. The checkpoint loader accepts this explicit
component tuple, and `main.py` resolves it from
`run.pretrained_transfer_regime`. This avoids loading and then overwriting the
router, while keeping the selected transfer boundary in the transfer report.

In all modes, verify that the target adapter has a different state-dict prefix
and is not present in the loaded-key list.  Source adapter tensors should be
reported as intentionally excluded, not as silently unexpected tensors.

## Hydra configuration surface

Create a dedicated model config with explicit, resolved fields similar to:

```yaml
_target_: foundry.models.NeurosoftConvBiGRU

num_samples: 1000
adapter_dim: 64
session_configs: ???       # {canonical_session_id: physical_channel_count}
temporal_channels: 128
temporal_kernel_samples: 64
temporal_stride: 4
conv_depth: 1
dropout_rate: 0.3
gru_hidden_size: 128
gru_num_layers: 2
gru_bidirectional: true
gru_dropout: 0.0           # external dropout is used consistently
```

`session_configs` is required, never `null`.  Construct it from the exact
source or target recording manifest before model instantiation.  A
single-session scratch run supplies one entry; a source-pretraining run supplies
only allowed source sessions; a target adaptation run supplies the one target
session.  The source-subject exclusion rule is therefore represented both in
the data manifest and in adapter construction.

Create distinct Phase-2 experiment configs rather than adapting the historical
`gru_neurosoft_8band_intrasession_*.yaml` files.  They must mirror the Phase-1
EEGNet matrix:

- `intrasession-causal` split;
- Phase-0 eligibility and nested fraction manifests;
- fractions 5%, 10%, 25%, 50%, and 100%;
- seeds 42, 43, and 44;
- `val/neurosoft_acoustic_stim_8band_supported_f1` for early stopping and
  checkpoint selection;
- selected-checkpoint test evaluation;
- minipig and monkey configs; and
- `ComputeTrackingCallback` with a FLOP value validated for this exact model,
  input shape, precision, and training step.

Do not copy EEGNet's FLOP number.  The validation must cover both convolution
and recurrent operations, as the roadmap requires.

## Tests and acceptance criteria

Focused contract tests are implemented in
`tests/test_models/test_neurosoft_conv_bigru.py`, with CLI transfer-regime
coverage in `tests/test_pretrained_transfer_regimes.py`. The remaining
end-to-end and experiment-matrix requirements below still apply before
scientific jobs are launched.

### Architecture and data-path tests

- Construct with two sessions of different channel counts; tokenize, collate,
  forward, compute a task loss, and backpropagate.
- Verify exact feature shapes for `(B, C_s, 1000)` and the Conv1d length
  formula; verify default output is 256-dimensional before the router.
- Verify a 0.5 s raw 2 kHz signal reaches the convolution without preprocessing
  resampling.
- Verify padded channels are never consumed by a smaller session adapter.
- Verify variable time padding is masked after the frontend and does not alter
  pooled embeddings or predictions for valid samples.
- Verify unknown session IDs fail loudly.
- Verify `conv_depth=2` and `gru_num_layers` construction paths work, without
  treating them as part of the initial experiment matrix.

### Transfer tests

- Save a source Lightning checkpoint with source adapter IDs; load it into a
  target-only model with a different adapter ID.
- In full fine-tuning, assert the shared frontend, GRU, and router tensors load
  bitwise; assert the fresh target adapter remains unchanged by loading and is
  trainable.
- In frozen-representation mode, assert only frontend and GRU load/freeze;
  assert both target adapter and fresh router remain trainable and unmodified by
  loading.
- Assert source adapter tensors appear as intentionally excluded in the
  transfer report.
- Deliberately change a shared tensor shape and assert strict loading fails
  before mutating any target tensor.

### End-to-end smoke tests

- Run one short CPU/GPU train/validation/test job for a minipig and monkey
  session with the Phase-2 config semantics.
- Run a two-session source/pretraining smoke job followed by one independent
  target adaptation for every transfer mode.
- Confirm the supported-class validation metric selects the checkpoint and the
  selected checkpoint is the one evaluated on test data.
- Confirm parameter count, trainable count, steps, processed windows, signal
  seconds, FLOPs, precision, wall time, and best-checkpoint counters are logged.

## Explicit implementation boundaries

Do not submit a scientific Slurm array while this work is uncommitted or before
the repository is clean.  Once experiments are authorized, use the project's
normal immutable snapshot launcher workflow and the `long` partition, recording
the job ID and snapshot bundle path in the relevant experiment record.

Before any production Phase-2 run, add a new MS-authored experiment report and
link it to the completed Phase-1 EEGNet learning-curve report.  Model-depth or
transfer-regime comparisons that become scientific claims require their own
linked experiment records rather than being folded into the initial scratch
baseline report.

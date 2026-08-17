# Embedding Visualization Redesign: Implementation Plan

**Status:** Proposed  
**Date:** 2026-08-17  
**Primary entry point:** `foundry/training/callbacks/embedding_viz.py`

## 1. Objective

Replace the current embedding visualization callback with a deterministic,
model-aware system that supports meaningful comparisons across validation
events and compatible runs.

The redesign must analyze two distinct representation families:

1. **Channel representations:** the per-channel vector fused with temporal
   sample representations before they enter the Perceiver. This is `ch_emb`
   with shape `(B, C, D_channel)`, whether it is produced by a static lookup or
   by the dynamic relative-channel encoder, and whether fusion uses addition or
   concatenation.
2. **Backbone representations:** one mean-pooled vector per input window from
   the final processed Perceiver latents.

The implementation must make the scientific meaning of every point and metric
explicit. It must not treat dynamic channel representations as fixed channel
identities: the dynamic encoder does not receive channel identity, so its
output is a signal- and context-conditioned observation that may vary across
windows for the same channel.

## 2. Success criteria

The redesign is complete when:

- Repeated validation passes use exactly the same seeded validation windows.
- Compatible runs select the same visualization observations and log a sample
  fingerprint that verifies this.
- Static and dynamic channel representations are visualized through one clear
  representation contract.
- Dynamic plots expose variation across time for the same recording-specific
  channel.
- Channel consistency is analyzed at recording-specific, canonical-electrode,
  and anatomical levels when the required metadata exists.
- Backbone plots consistently support dataset, subject, session, and per-task
  class coloring.
- Cosine-based scores provide comparable numerical summaries alongside plots.
- Every scheduled event produces the same set of applicable outputs; frequency
  is controlled independently.
- Unscheduled events perform no representation capture or analysis.
- The default configuration adds no more than approximately 10% wall-clock
  overhead to a representative full validation pass. This target informs fixed
  defaults and must never cause hardware-dependent adaptive sampling.
- Distributed runs aggregate observations once and log only from global rank
  zero.
- Static, dynamic, disabled, Perceiver, and non-Perceiver cases are covered by
  tests and fail or skip explicitly rather than silently changing semantics.

## 3. Current behavior to replace

The existing callback has several incompatible behaviors under one name:

- It hooks `model.backbone.processor`, mean-pools its output, and calls the
  result an embedding.
- It separately hooks `relative_channel_encoder`, but only in dynamic mode.
- Static channel lookup embeddings are not visualized.
- Models without `backbone.processor` disable the entire callback, including
  otherwise applicable analysis.
- The first buffered observations win, which biases results toward early
  validation batches and recordings with many channels.
- Channel observations are colored by dataset, so repeated windows from the
  same channel cannot be inspected.
- The scalp plot averages dynamic observations by bare electrode name across
  recordings, discarding within-channel temporal variation.
- PCA is independently fitted to raw vectors at every event, while the logged
  silhouette score uses Euclidean geometry in a separate PCA space.
- Epoch-based scheduling does not map cleanly to step-based validation.
- Sanity validation produces misleading step-zero artifacts.
- Classification labels are selected using an implicit sleep-task/first-task
  heuristic.
- Validation samplers advance their random generators, so windows may differ
  between validation passes.
- There is no explicit distributed aggregation or test suite for callback
  semantics.

## 4. Scope and non-goals

### In scope

- Deterministic validation sampling for all validation, not only this callback.
- Explicit representation and sample-metadata contracts.
- Deterministic hierarchical observation selection.
- PCA plots based on L2-normalized representations.
- Norm diagnostics.
- Cosine-based clustering metrics.
- Conditional anatomical visualization and scores using reliably resolved
  canonical EEG positions.
- Stable W&B keys, category colors, availability counts, and sample
  fingerprints.
- Configuration migration and automated tests.

### Out of scope

- t-SNE or UMAP.
- A shared PCA coordinate frame across epochs or runs.
- Claiming that inferred canonical montage coordinates are recording-specific
  measured sensor locations.
- Inventing anatomical coordinates for unmatched, bipolar, ECoG, SEEG, or
  dataset-specific channels.
- A generic latent-token temporal visualization. The backbone representation is
  initially limited to one mean-pooled vector per input window.
- Channel plots for models that do not have an explicit pre-backbone channel
  representation.

## 5. Core contracts

### 5.1 Stable observation identity

Define a validation-window identity from explicit metadata:

```text
(dataset_id, subject_id, session_id, absolute_start, window_duration)
```

The identity must be serializable and stable across processes. Use a
cryptographic or otherwise process-stable hash with the configured seed; do not
use Python's randomized `hash()`.

Exact identity matching is guaranteed only for compatible runs with the same
validation data, split, window-length configuration, and seed. Runs with
different window durations can use deterministic anchors but do not contain
literally identical signal windows; this limitation must be visible in the
sample metadata.

Log a fingerprint computed from the sorted selected identities at every event.
The same fingerprint is required before treating two plots as paired samples.

### 5.2 Explicit sample metadata

Carry these fields through tokenization, collation, `FoundryModule`, and
`StepOutput`:

- Dataset ID
- Subject ID
- Session/recording ID
- Absolute window start
- Window duration
- Channel vocabulary indices or stable channel IDs
- Channel-validity mask
- Per-task targets and validity

Dataset, subject, and session identity must be resolved centrally in the data
layer. The callback must not contain dataset-specific string parsing. Where an
upstream dataset lacks an explicit dataset field, add a documented adapter at
the dataset/datamodule boundary. Preserve namespace components in the stable
IDs.

### 5.3 Typed representation payload

Replace forward hooks with an explicit, typed representation payload. Add an
optional payload to `ModelOutput` and propagate a detached validation form to
`StepOutput`. It should contain:

- `channel_representations`: `(B, C, D_channel)` when an explicit channel
  representation exists.
- `backbone_representations`: `(B, D_backbone)`, mean-pooled from the final
  processed Perceiver latents.
- Channel mode and masks needed to interpret the tensors.

Capture must be enabled only for a scheduled, non-sanity validation event. Do
not retain these tensors during ordinary training or unscheduled validation.

Static and dynamic POYO models must expose the same `channel_representations`
field. The channel identity supplied alongside a dynamic vector is analysis
metadata only; it is not an input to the dynamic encoder.

## 6. Deterministic validation

Validation determinism is a data-loader guarantee, independent of whether the
embedding callback is enabled.

### Required behavior

- Reset the validation sampler generator to `seed + validation_offset` before
  every validation pass.
- Handle both `dataloader.sampler.generator` and
  `dataloader.batch_sampler.generator`; the latter is required for
  `VariableLengthBatchSampler`.
- Preserve stochastic training sampling.
- Ensure repeated fixed-length validation iterations yield identical
  `(recording, start, end)` tuples in identical order.
- Ensure repeated variable-length validation iterations yield identical window
  lengths, offsets, batches, and ordering.
- Preserve correct, deterministic sharding under distributed validation.
- Sanity validation may consume the same deterministic window sequence, but
  embedding capture and logging must remain disabled while
  `trainer.sanity_checking` is true.

Prefer enforcing this in the datamodule/sampler lifecycle rather than relying
on callback ordering. Update or replace `DeterministicSamplerCallback` so there
is one authoritative mechanism rather than two partially overlapping ones.

## 7. Observation selection

Selection occurs only during scheduled events and is independent of batch
order and hardware speed.

### 7.1 Window budget

For `N` total validation windows, use:

```text
window_budget = min(N, max(256, ceil(0.10 * N)), 2048)
```

This selects all windows for validation sets smaller than 256, scales with
medium validation sets, and caps large runs.

### 7.2 Hierarchical allocation

Allocate the window budget deterministically:

1. Balance allocation across datasets before considering dataset size.
2. Within each dataset, maximize subject diversity.
3. Select at most 8 sessions per dataset.
4. Target at least 16 windows per selected session.
5. When the budget cannot support the requested depth, reduce the number of
   selected sessions before reducing windows per session.
6. When a small validation set has fewer observations than a quota, include all
   available observations and redistribute unused capacity deterministically.

Use stable hashes to choose datasets' subjects, sessions, and windows. Never use
arrival order as a tie-breaker.

### 7.3 Channel-observation budget

Cap dynamic channel observations at 16,384 `(window, channel)` vectors.

- Admit complete channel sets for a selected window so within-window geometry
  and anatomical scores remain valid.
- Stop before admitting a window that would exceed the cap.
- Balance selected channel windows across the already selected sessions.
- Static models resolve only the lookup vectors for channel identities present
  in the deterministic validation selection; do not plot train-only or unseen
  vocabulary entries and do not duplicate a static vector for every window.

### 7.4 Distributed selection

Each rank may capture its local deterministic candidates, but selection must be
equivalent to selection over the global validation population:

- Gather compact identities and selected tensors across ranks.
- Deduplicate by stable observation identity.
- Apply final ordering and quotas deterministically.
- Compute figures and metrics only on global rank zero.
- Broadcast nothing unless another callback needs the results.

## 8. Shared representation processing

### 8.1 Normalization

For both representation families:

- Record raw L2 norms before normalization.
- Exclude zero or non-finite vectors from cosine/PCA calculations and log their
  counts.
- L2-normalize valid vectors with a documented epsilon.
- Run PCA on the normalized vectors.
- Use a deterministic PCA solver and seed where the solver requires one.

All cosine metrics operate on the original normalized representation vectors,
not on PCA coordinates. PCA is an event-level visual summary; its axes may
rotate or flip between events and must not be interpreted as a shared temporal
coordinate system.

### 8.2 Stable presentation

- Derive categorical colors from stable names/IDs so colors do not change when
  group subsets change.
- Use task class order from `TaskConfig.class_mapping`.
- Reuse the same PCA coordinates for every alternate coloring of a given
  representation family at an event.
- Include sample counts, group counts, excluded counts, and explained variance
  in plot subtitles or accompanying metrics.
- Close every Matplotlib figure after conversion to `wandb.Image`.

## 9. Channel-representation outputs

### 9.1 Recording-specific temporal-consistency figure

Produce a small-multiple figure with up to 8 deterministically selected
recordings:

- One panel per recording.
- One point per dynamic `(window, channel)` observation.
- Color by recording-specific channel identity.
- Use identical normalized-PCA coordinates within the figure.
- Tight same-color clusters represent temporal consistency; overlap between
  colors represents poor within-recording channel separability.
- Static mode shows one point per validation-observed channel rather than
  duplicated points.

### 9.2 Canonical-electrode figure

Produce a global channel view colored by conservatively normalized bare
electrode name:

- Strip namespaces centrally.
- Normalize case and whitespace.
- Apply only explicitly vetted aliases; do not collapse bipolar or ambiguous
  names into standard electrodes.
- Preserve recording-specific points so variation across recordings remains
  visible.

### 9.3 Anatomical figure

Use canonical 3D positions from MNE's standard montages when a channel name can
be resolved reliably.

- Color resolved channels with the existing scalp-position color-wheel idea.
- Retain unresolved channels in gray as context.
- Generate the anatomical output only when at least one recording contains 9
  or more resolved channel positions.
- Make clear that these are canonical inferred positions, not measured
  recording-specific coordinates.
- Plot recording-specific channel centroids across sampled windows; repeated
  canonical electrodes from different recordings retain separate points.

### 9.4 Channel metrics

Use cosine geometry and deterministic weighting. Log counts with every score.

#### Temporal consistency

For each dynamic recording-specific channel with at least two windows, compute
the mean cosine similarity from each observation to its leave-one-out channel
centroid. Macro-average over channels, then recordings, so high-channel-count
recordings do not dominate.

Static channel representations have temporal consistency `1.0` by construction;
label this provenance in the availability/count metadata rather than implying
it was empirically estimated from changing vectors.

#### Within-recording separability

For each eligible dynamic observation, classify it by the nearest leave-one-out
channel centroid within its recording. Report macro accuracy over channels and
recordings, plus the mean cosine margin between the correct centroid and the
nearest incorrect centroid. Omit this score for static mode because a single
lookup vector per channel cannot support a non-leaking leave-one-out estimate.

#### Cross-recording canonical-electrode consistency

For canonical electrode labels present in at least two recordings, classify
each recording-specific channel centroid against canonical electrode centroids
built from other recordings only. Report macro accuracy and cosine margin.
This score applies to static and dynamic modes.

#### Anatomical organization

For each recording with at least 9 resolved positions:

- Normalize canonical 3D coordinates to the scalp sphere and compute pairwise
  angular/geodesic distance.
- Compute Spearman correlation between physical distances and pairwise cosine
  distances between channel representations.
- **Centroid score:** use recording-specific channel centroids across sampled
  windows.
- **Per-window score:** in dynamic mode, compute the score separately for each
  eligible window containing at least 9 resolved channels.
- Log the median, interquartile range, number of eligible recordings/windows,
  and resolved-channel counts.
- Omit undefined constant-input correlations and count them explicitly.

Avoid a full all-pairs observation matrix for channel metrics. Use group
centroids and bounded deterministic comparisons so runtime remains predictable.

## 10. Backbone-representation outputs

### 10.1 Representation

Mean-pool the final processed Perceiver latents over their latent-token axis to
produce one `(D_backbone,)` vector per selected input window. Do not add
individual latent-token or temporal plots in this implementation.

### 10.2 PCA views

Fit one PCA on all valid normalized selected backbone vectors at an event and
reuse its coordinates for:

- Dataset coloring
- Subject coloring
- Session coloring
- One class-colored view per configured classification task

For a task class view, include a window only when it has one valid class or
multiple targets that all agree. Exclude windows containing conflicting labels
from that task's plot and report the exclusion count. Never choose a task using
name heuristics or dictionary order.

### 10.3 Backbone scores

Compute cosine silhouette scores on the normalized original backbone vectors
for every available grouping:

- Dataset
- Subject
- Session
- Each classification task

Compute the cosine-distance matrix once per event and reuse it for all grouping
labels. Exclude singleton/invalid groups according to one documented rule and
log included/excluded observations and groups. Omit a score when fewer than two
eligible groups remain.

## 11. Availability behavior

Treat channel and backbone analysis independently.

| Model capability | Channel output | Backbone output |
|---|---|---|
| Dynamic channel + Perceiver | Full dynamic analysis | Full backbone analysis |
| Static channel + Perceiver | Validation-observed lookup analysis | Full backbone analysis |
| Disabled channel + Perceiver | Explicitly unavailable; no zero scatter | Full backbone analysis |
| No explicit channel representation + Perceiver | Explicitly unavailable | Full backbone analysis |
| Non-Perceiver baseline | Applicable model-specific channel output only, if an explicit contract exists | Explicitly unavailable |

Unavailability should not disable the other representation family. Log
availability and coverage counters under stable keys so missing figures can be
distinguished from callback failures.

## 12. Scheduling and configuration

Replace the current callback arguments:

- Remove `every_n_epochs`.
- Remove `compute_tsne`.
- Remove the single overloaded `max_samples`.
- Remove callback-level `class_names`; obtain names per task from task config.

Proposed configuration:

```yaml
embedding_visualization:
  _target_: foundry.training.callbacks.EmbeddingVisualizationCallback
  every_n_validation_runs: 5
  sample_seed: ${run.seed}
  window_fraction: 0.10
  min_windows: 256
  max_windows: 2048
  max_channel_observations: 16384
  max_sessions_per_dataset: 8
  min_windows_per_session: 16
  max_recording_panels: 8
  min_positioned_channels: 9
```

Scheduling rules:

- Count complete, non-sanity validation runs.
- `every_n_validation_runs: 1` means every full validation pass.
- Decide whether an event is scheduled in `on_validation_epoch_start` before
  enabling representation capture.
- The first complete validation pass is event 1.
- Skip all embedding output during sanity validation.
- Every scheduled event produces the same applicable figure and metric set.

Update all experiment overrides that currently set `every_n_epochs` or
`compute_tsne`. Prefer a clean configuration migration over indefinitely
supporting aliases whose semantics are misleading.

## 13. W&B output contract

Use stable, hierarchical names. Final spelling may be adjusted once to match
project conventions, then treated as API:

```text
val/embedding_viz/sample/window_count
val/embedding_viz/sample/channel_observation_count
val/embedding_viz/sample/fingerprint

val/embedding_viz/channel/pca_by_recording
val/embedding_viz/channel/pca_canonical_electrode
val/embedding_viz/channel/pca_anatomy
val/embedding_viz/channel/norm_distribution
val/embedding_viz/channel/temporal_consistency
val/embedding_viz/channel/within_recording_accuracy
val/embedding_viz/channel/within_recording_margin
val/embedding_viz/channel/canonical_accuracy
val/embedding_viz/channel/canonical_margin
val/embedding_viz/channel/anatomy_centroid_spearman
val/embedding_viz/channel/anatomy_window_spearman

val/embedding_viz/backbone/pca_dataset
val/embedding_viz/backbone/pca_subject
val/embedding_viz/backbone/pca_session
val/embedding_viz/backbone/pca_task/<task_name>
val/embedding_viz/backbone/norm_distribution
val/embedding_viz/backbone/silhouette/dataset
val/embedding_viz/backbone/silhouette/subject
val/embedding_viz/backbone/silhouette/session
val/embedding_viz/backbone/silhouette/task/<task_name>
```

Also log compact availability/coverage counters. Do not upload full raw
embedding tables by default; they would materially increase storage and upload
time. The deterministic identity fingerprint and scalar metrics are the
cross-run comparison contract.

Use the trainer's actual global step or validation-event counter in titles;
never label step-based runs only by `current_epoch`.

## 14. Implementation phases

### Phase 1: Deterministic validation foundation

Likely files:

- `foundry/data/samplers.py`
- `foundry/data/datamodules/base.py`
- `foundry/training/callbacks/lifecycle.py`
- Sampler/datamodule tests

Tasks:

1. Establish one generator-reset mechanism for validation.
2. Support ordinary and variable-length samplers.
3. Add repeated-iteration and distributed-sharding tests.
4. Verify training sampler state continues to advance normally.

Deliverable: validation indices and windows repeat exactly across passes.

### Phase 2: Metadata and representation contracts

Likely files:

- Dataset/datamodule metadata adapters
- `foundry/models/ssl_meta.py` or a new representation-contract module
- `foundry/models/poyo_eeg.py`
- `foundry/models/masked_poyo_eeg.py`
- `foundry/training/step_output.py`
- `foundry/training/module.py`

Tasks:

1. Carry explicit dataset, subject, session, start, duration, and channel IDs.
2. Add the typed optional representation payload.
3. Expose static and dynamic `ch_emb` through the same field.
4. Mean-pool processed latents in the model contract.
5. Gate capture before forward execution for scheduled validation only.
6. Remove dependence on forward hooks.

Deliverable: callbacks receive self-describing, detached tensors and metadata.

### Phase 3: Deterministic selector and distributed aggregation

Likely files:

- New helper module under `foundry/training/callbacks/` or
  `foundry/training/visualization/`
- `foundry/training/callbacks/embedding_viz.py`

Tasks:

1. Implement stable identities, hashes, quotas, and fingerprints.
2. Implement window and channel-observation budgets.
3. Gather/deduplicate observations across ranks.
4. Add selection unit tests that randomize batch order and rank partitioning.

Deliverable: the selected global observation set is invariant to batch order,
worker count, device speed, and distributed partitioning.

### Phase 4: Metrics

Prefer pure functions in a dedicated module so they can be tested without
Lightning or W&B.

Tasks:

1. Implement normalization and invalid-vector accounting.
2. Implement channel temporal, separability, canonical, and anatomy metrics.
3. Implement reusable cosine-distance silhouette calculations for backbone
   groupings.
4. Define macro-averaging and exclusion behavior exactly in docstrings.
5. Validate metrics against small hand-computable synthetic examples.

Deliverable: deterministic metric dictionaries with stable keys and coverage
metadata.

### Phase 5: Figures and callback orchestration

Tasks:

1. Implement shared deterministic PCA and stable color utilities.
2. Implement channel small multiples, canonical view, anatomical view, and norm
   distribution.
3. Implement backbone dataset/subject/session/per-task views and norm
   distribution.
4. Implement scheduling, sanity exclusion, availability logging, and W&B
   emission.
5. Remove t-SNE and the old plotting paths.

Deliverable: every scheduled event emits the complete applicable output set.

### Phase 6: Configuration migration and validation

Likely files:

- `configs/trainer/default.yaml`
- Pretraining and downstream experiment overrides
- Callback tests and selected integration tests

Tasks:

1. Replace old callback arguments everywhere.
2. Add configuration-composition tests to catch stale overrides.
3. Run static, dynamic, disabled, pretraining, downstream classification, and
   multi-task smoke tests.
4. Test an event with insufficient anatomy and one with 9+ resolved channels.
5. Confirm W&B keys and image lifecycle using a fake experiment/logger.

Deliverable: no configuration references the removed API and all capability
combinations behave explicitly.

### Phase 7: Performance and artifact review

Tasks:

1. Add a benchmark script or documented profiling command that separately
   times capture, CPU transfer, selection, metrics, plotting, and W&B image
   conversion.
2. Profile representative 2-channel, standard 10-20, 64-channel, and
   129-channel validation runs.
3. Compare callback-enabled and callback-disabled full validation wall time.
4. Adjust only the fixed default budgets if representative overhead exceeds
   approximately 10%; never adapt budgets based on measured hardware speed.
5. Inspect early and later artifacts from a known dynamic pretraining run and a
   static downstream run.

Deliverable: recorded performance evidence and reviewed example artifacts.

## 15. Testing matrix

### Unit tests

- Window-budget boundary cases.
- Stable hash and fingerprint behavior across process orderings.
- Hierarchical quota allocation, redistribution, and session-depth guarantees.
- Channel cap with complete-window admission.
- L2 normalization, zero/non-finite exclusion, and norm statistics.
- Stable categorical colors.
- Unambiguous and conflicting task-label extraction.
- Temporal consistency and leave-one-out centroid metrics.
- Cross-recording canonical leave-one-recording-out behavior.
- Anatomical eligibility at 8 versus 9 positioned channels.
- Canonical 3D distance and Spearman calculations.
- Backbone cosine silhouette exclusions and reuse.

### Integration tests

- Fixed-length validation repeats the same windows over multiple passes.
- Variable-length validation repeats lengths, windows, and batch order.
- Training iterations remain stochastic.
- Scheduled versus skipped event capture.
- Sanity validation produces no embedding logs.
- Dynamic POYO emits both representation families.
- Static POYO emits validation-observed channel entries without duplicates.
- Disabled-channel POYO emits only backbone outputs.
- Non-Perceiver baseline skips Perceiver outputs cleanly.
- Multi-task model emits one valid class view and score per task.
- Multi-device aggregation logs once and matches single-device selection.
- No Matplotlib figures remain open after logging.
- Callback works with no W&B logger and clears all temporary state.

### Artifact review

For a known run, verify manually that:

- The sample fingerprint remains constant across events.
- Dataset, subject, session, and task colors remain stable.
- Same-channel dynamic observations are visible within each recording panel.
- Static plots contain one point per selected channel identity.
- Gray anatomical context is retained.
- Anatomical outputs are absent below 9 positioned channels and present at or
  above 9.
- Titles use meaningful step/event information.

## 16. Risks and mitigations

### Metadata is unavailable or inconsistent

Mitigation: resolve metadata at the dataset boundary, validate it before
training, namespace IDs, and log availability. Do not add callback-specific
parsers.

### Deterministic validation changes existing metric values

This is expected: current validation can sample different offsets on different
passes. Mitigation: document the behavior change, verify split coverage, and
compare variance before and after on a representative run.

### High-cardinality plots remain crowded

Mitigation: enforce session quotas, use small multiples for recording-specific
channel plots, use stable colors, and keep numerical scores as the primary
cross-run evidence.

### PCA plots appear to rotate between events

Mitigation: state clearly that PCA is independently optimized per event and do
not use coordinates as a trajectory. Use cosine metrics for longitudinal
claims.

### Silhouette computation dominates runtime

Mitigation: cap windows at 2,048, compute one cosine-distance matrix, and reuse
it for every backbone grouping. Profile before merge.

### Large-channel recordings dominate memory

Mitigation: cap channel observations at 16,384, admit complete windows, balance
windows across sessions, and transfer only scheduled selected candidates.

### Static metrics look artificially better than dynamic metrics

Mitigation: identify temporal consistency as `1.0 by construction` for static
mode, omit non-estimable leave-one-out metrics, and emphasize cross-recording
canonical and anatomical metrics for comparisons.

## 17. Completion checklist

- [ ] Validation sampling is repeatable for fixed and variable lengths.
- [ ] Stable metadata and representation payloads replace forward hooks.
- [ ] Deterministic hierarchical sampling and fingerprints are implemented.
- [ ] Static, dynamic, disabled, and unavailable modes are explicit.
- [ ] Channel figures and cosine metrics match the agreed semantics.
- [ ] Conditional anatomical figures and both anatomical scores are present.
- [ ] Backbone figures and per-group cosine silhouette scores are present.
- [ ] t-SNE and epoch-based scheduling are removed.
- [ ] Sanity and unscheduled validation perform no capture/logging.
- [ ] Distributed aggregation logs once from global rank zero.
- [ ] All configs use the new callback API.
- [ ] Unit and integration tests pass.
- [ ] Representative artifact review is complete.
- [ ] Representative measured overhead is approximately 10% or less, or fixed
      defaults are reduced and re-profiled.


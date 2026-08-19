# Plan: NeuralBench-compatible protocols on Foundry H5 data

**Status:** Agreed design; not yet implemented.

## Objective

Add a reusable protocol layer that lets a Foundry run use the exact split and
task contract of a version-pinned NeuralBench task while retaining Foundry's
existing H5 brainsets, datamodules, models, Hydra workflow, launch snapshots,
and result storage.  The brainset pipeline materializes versioned protocol
metadata into those H5 files; the runtime consumes that metadata directly.

The first implementation is Brain Invaders P300 / `Korczowski2014A`
(`BI2014a`).  It must support a canonical NeuralBench-compatible benchmark and
leave Foundry's LOSO protocol available as a separately labelled robustness
benchmark.

This is **not** a migration to NeuralBench as the training framework and does
not change where Foundry stores raw or processed data.  NeuralBench remains the
external specification and reference implementation; Foundry remains the
runtime.

## Agreed decisions

- Preserve each H5's EEG signal, canonical raw event tables, and recording
  identity.  The pipeline adds versioned protocol metadata groups to the same
  H5 files through an explicit schema migration/reprocessing step.
- The canonical configuration must implement the full NeuralBench **task
  contract**, not merely its train/validation/test subject assignment.
- Store the externally generated split in a checked-in, versioned manifest.
  Foundry jobs must not import NeuralBench or calculate the split live.
- Build a reusable external-protocol abstraction, but implement and validate
  only P300 first.
- Disable Foundry's recording-wide normalization in the compatibility protocol.
  It remains available only for explicitly non-compatible legacy/ablation runs.
- Retain LOSO as a second, clearly distinct protocol.  Do not pool it with the
  canonical fixed-partition result.

## What “NeuralBench-compatible” means

A run may use a Foundry model and still be NeuralBench-compatible at the task
level.  The compatibility claim covers all of the following, as pinned in the
protocol lock:

1. source study/dataset identity;
2. subject, session, and recording membership in `train`, `valid`, and `test`;
3. event inclusion, label mapping, and event-to-window alignment;
4. epoch start, duration, boundary handling, and baseline semantics;
5. signal preprocessing and normalization policy;
6. class weighting, loss, monitored validation metric, checkpoint-selection
   rule, and final test evaluation;
7. task-level random seeds and reported aggregation rule.

It does **not** by itself claim numerical equality with a NeuralBench model.
Numerical parity additionally requires the same model implementation,
initialization, optimizer, scheduler, augmentation, and hardware-level
determinism.  Foundry model results must therefore be described as “evaluated
under the NeuralBench P3/BI2014a task protocol,” not as reproductions of a
NeuralBench model result, until a model-specific parity test has passed.

## Current P300 fit and gaps

The H5 inventory already contains one recording for each of the 64 BI2014a
subjects (`sub001_0_0.h5` through `sub064_0_0.h5`).  This aligns naturally with
NeuralBench P3's subject-level split.  The pipeline will add the corresponding
assignment and P3 sampling-unit table to each existing H5 rather than creating
a second data representation.

The current Foundry implementation is not task-compatible:

- `MOABBPipeline._create_splits()` writes Foundry-specific three-fold
  assignments into each H5 file.  The pipeline must add a separate, versioned
  NeuralBench assignment rather than overwrite those legacy fields.
- `BrainInvadersP300` currently expands a window from stimulus onset to
  `onset + 1.0 s` and then keeps the earliest event in the slice.  NeuralBench
  P3 declares `start: -0.2`, `duration: 1.0`, and
  `neuro.baseline: [0.0, 0.2]`.  With rapid stimuli, that pre-stimulus window
  may include an earlier event, so the compatibility path must retain an
  explicit anchor ID.  The exact baseline coordinate convention must be
  captured from the reference run; it must not be guessed from the YAML fields.
- `BrainInvadersP300._ensure_normalized()` performs recording-wide z-scoring.
  This both differs from the external task and leaks held-out recording data
  into its own normalization statistics.  It must be disabled for the
  compatibility protocol.
- Foundry trains with `trainer.fit()` only.  It has a test dataloader but does
  not invoke it after selecting the best validation checkpoint.  The canonical
  protocol must run and report test metrics exactly once per completed run.
- Existing P300 experiments monitor positive-class F1.  NeuralBench P3
  declares validation balanced accuracy as the monitored quantity.  The
  compatibility configuration must use the pinned NeuralBench selection metric
  and preserve any other metrics as auxiliary outputs only.

These facts make a split-only patch insufficient.

## Architecture

### 1. Protocol specifications are first-class data artifacts

Add a small, dependency-light package:

```text
foundry/data/protocols/
  __init__.py
  schema.py                 # typed protocol and manifest validation
  registry.py               # protocol ID -> packaged manifest/contract
  manifest.py               # read-only lookup and integrity checks
  sampling.py               # convert events into protocol windows
  transforms.py             # protocol-scoped baseline/normalization transforms
```

Use a stable identifier, for example:

```text
neuralbench/p3/korczowski2014a/0.2.3
```

The identifier refers to the NeuralBench release used to construct the
reference artifact, not to a mutable “latest” specification.  A later upgrade
is a new protocol ID, never an in-place edit of an old one.

Each protocol must declare:

```yaml
id: neuralbench/p3/korczowski2014a/0.2.3
format_version: 1
reference:
  framework: neuralbench
  version: 0.2.3
  task: p3
  dataset: Korczowski2014A
  config_sha256: <hash>
  exporter_commit: <hash>
source:
  foundry_brainset: korczowski_brain_invaders_2014a
  required_subjects: [sub001, ...]
split:
  unit: subject
  manifest: subjects.json
window:
  trigger: p300_trials
  start: -0.2
  duration: 1.0
  boundary_policy: <reference-verified value>
preprocessing:
  baseline: [0.0, 0.2]
  normalization: <reference-verified value>
training:
  class_weighting: <reference-verified value>
  loss: <reference-verified value>
  monitor: val/bal_acc
  monitor_mode: max
evaluation:
  checkpoint: best
  test_after_fit: true
  task_seeds: [33, 34, 35]
```

Values marked reference-verified are populated only after an exporter has
inspected the installed NeuralBench implementation and a reference execution
has validated them.

### 2. Split manifests are frozen, transparent, and H5-independent

Store the P300 artifact under version control, for example:

```text
configs/protocols/neuralbench/p3/korczowski2014a/0.2.3/
  protocol.yaml
  subjects.json
  event_inventory.json
  protocol.lock.json
```

`subjects.json` maps the canonical Foundry subject IDs to one split.  It also
records the exact identifier emitted by NeuralBench before conversion.  It is
an input to the brainset pipeline, which writes the assignment into every
matching H5.  It must cover every expected subject exactly once and must not
contain a subject absent from the H5 brainset.

`event_inventory.json` is a compact, reviewable reference fingerprint per
recording: number of eligible events by label, sampling rate, recording
duration, and hashes of canonical `(onset, label)` tables.  It detects a source
or staging mismatch without committing raw data.

`protocol.lock.json` pins the NeuralBench, NeuralFetch, MOABB, scikit-learn,
and Python versions used by the exporter, the source/revision hashes, and the
hashes of every accompanying artifact.  The lock must state whether the
reference data were downloaded through MOABB or another NeuralBench backend.

The pipeline writes each protocol under a namespaced H5 group, for example:

```text
neuralbench_protocols/
  p3_korczowski2014a_0_2_3/
    metadata/              # protocol ID and lock hash
    split_assignment        # one value: train, valid, or test
    sampling_units/         # explicit, labelled anchor/window table
```

The signal, channels, canonical `p300_trials`, and existing Foundry split
fields are not altered.  Multiple protocol groups may coexist in the same H5.
The H5 derived-version metadata must be incremented and the migration recorded
so old results remain attributable to the pre-protocol files.

### 3. Pipeline materializes sampling units; runtime chooses the matching sampler

Extend the relevant brainset pipeline with a protocol-materialization stage.
For every eligible reference event it writes one `sampling_units` row with at
least:

- stable `unit_id` and canonical source-event/anchor ID;
- exact absolute `window_start` and `window_end`;
- anchor timestamp and target label; and
- any task-specific baseline or grouping metadata needed at runtime.

This table is deliberately not represented as a regular `Interval`: P300
windows overlap, while `torch_brain.Interval` requires disjoint intervals for
sorting and slicing.  The pipeline retains the existing short, disjoint
`p300_trials` table as the canonical event table.

Add a `ProtocolWindowSampler` that reads the materialized unit table and emits
a `ProtocolDatasetIndex(recording_id, start, end, unit_id)`.  For P300 it emits
every declared window exactly once; it may deterministically shuffle only the
training order.  It must not apply the current random tiling/jitter rule.

`BrainInvadersP300.__getitem__` receives the protocol index, slices the
existing H5 signal, and selects the target using `unit_id`/anchor ID rather
than the current "earliest event" heuristic.  It then returns the regular
`torch_brain`/Foundry sample, so tokenizers, models, collation, losses, and
Lightning remain unchanged.

The generic protocol schema must declare a `sampling_mode` and a matching
sampler implementation:

| Sampling mode | Pipeline artifact | Runtime sampler |
|---|---|---|
| `explicit_windows` | labelled window/anchor table | `ProtocolWindowSampler` |
| `interval_tiling` | selected disjoint intervals | existing `FastRandomFixedWindowSampler` |
| `fixed_stride` | domains plus fixed stride/window rule | deterministic stride sampler |
| `predefined_partition` | session/recording assignment plus units | selected protocol sampler |

This is intentionally not a universal emulation of NeuralBench inside
`FastRandomFixedWindowSampler`: the pipeline declares the sampling units, and
Foundry selects the sampler that consumes them faithfully.

Do not overload `split_type="intersubject"` for this behaviour.  Add a
separate configuration field such as `data.protocol_id`; this prevents a
NeuralBench fixed partition from being mistaken for Foundry's current CV mode.

The protocol interface should be generic over split unit (`subject`,
`session`, `recording`, or event index).  P300 implements only `subject`
initially.  This is sufficient groundwork for future NeuralBench tasks without
preemptively supporting every split form.

### 4. Compatibility preprocessing is scoped and non-leaking

Implement baseline correction as a window-level transform with the exact
reference coordinate convention.  Apply it after extracting the intended
epoch, before the model sees the signal.

The compatibility configuration must set the current recording-wide
normalization path to off.  Any remaining model-local normalization must be
made explicit in the resolved config and compared to the reference task before
using the phrase “full task contract.”

Epochs that cannot provide the reference window or baseline must follow the
reference boundary rule.  The rule is a hard acceptance test because silent
padding and truncation can change both the event count and the signal content.

### 5. Training and test execution are a task-policy concern

Add an explicit evaluation policy to Foundry's run path.  For protocols that
declare `test_after_fit: true`, `main.py` must:

1. fit using only the train and validation partitions;
2. restore the checkpoint selected by the protocol's validation monitor; and
3. invoke `trainer.test(..., ckpt_path="best")` against the untouched test
   dataloader.

Persist test metrics with a `test/` prefix, the selected checkpoint path, and
the protocol ID/lock hash.  Do not use a maximum validation metric as the
reported benchmark result.

This policy should be reusable but opt-in.  It must not silently alter existing
experiments that were configured as validation-only studies.

### 6. NeuralBench is an offline reference dependency, not a Foundry runtime dependency

Create a small exporter environment under `tools/neuralbench_reference/` (or a
similarly isolated location) pinned to Python 3.12 and the selected NeuralBench
release.  It is used only to:

- resolve the reference task and dataset configuration;
- materialize the official split from the same study metadata;
- export subject IDs and the event inventory;
- record dependency versions and source hashes; and
- run a reference smoke experiment when validating an upgrade.

Foundry's Python 3.11 environment must be able to read the committed artifacts
without importing NeuralBench.  This keeps production jobs simple, avoids
cross-framework dependency conflicts, and means a historical protocol remains
usable even if NeuralBench later changes.

## P300 implementation phases

### Phase 0 — Reference capture (no Foundry behaviour change)

1. Build the isolated, pinned NeuralBench reference environment.
2. Resolve `p3` with `Korczowski2014A` and save the effective task config.
3. Export the exact subject assignments and a recording/event inventory.
4. Compare the reference study identifiers to Foundry's 64 H5 subject IDs.
5. Add the locked manifest and a human-readable provenance note.

**Gate:** the manifest contains all and only Foundry's 64 BI2014a subjects;
each appears in exactly one split; regenerated output matches byte-for-byte.

If the reference and H5 event inventories disagree, stop here.  Resolve the
source/staging discrepancy before implementing Foundry-side sampling.

### Phase 1 — Pipeline split and sampling-unit materialization

1. Add protocol schema, registry, and manifest validation.
2. Add a versioned H5 schema migration path and protocol-materialization stage
   to the BI2014a pipeline.
3. Write the external split assignment and explicit P300 sampling-unit table
   into the namespaced protocol group of every H5.
4. Add `protocol_id` plumbing through Hydra data configuration and
   `NeuralDataModule` into `BrainInvadersP300`.
5. Implement `ProtocolWindowSampler` and `ProtocolDatasetIndex` for the P300
   `explicit_windows` sampling mode.
6. Add a P300 data config that selects this sampler and protocol group.

**Gate:** Foundry’s train/valid/test recording sets exactly equal the exported
sets; their union is all 64 subjects; their intersection is empty; every
eligible anchor yields one declared sampling unit; and checksums of EEG,
channels, canonical event tables, and legacy split groups are unchanged by the
migration.

### Phase 2 — Full data task contract

1. Implement anchor-based target selection and reference-aligned event
   filtering, epoch windows, boundary rule, baseline correction, and
   normalization policy.
2. Add a compatibility-specific task config: target mapping, class weights,
   loss, metric set, and validation monitor.
3. Add a deterministic event-level comparison tool that emits the Foundry
   inventory in the same canonical form as the reference exporter.
4. Compare all sampled event IDs/times and a selected set of post-transform
   window arrays to the reference output with explicit tolerances.

**Gate:** event membership and labels match exactly; window timing and baseline
behaviour match the reference; no sample from a test subject participates in
any fitted data transform.

### Phase 3 — Train/select/test contract

1. Add opt-in post-fit best-checkpoint test evaluation.
2. Write the selected checkpoint and `test/` metrics to the logger and run
artifacts.
3. Add a canonical P300 NeuralBench-compatible experiment config and a
three-seed sweep consistent with the protocol lock.
4. Preserve a distinct Foundry LOSO config and labels such as
`protocol=foundry/loso/bi2014a/v1`.

**Gate:** a smoke run shows exactly one final test evaluation, never uses test
metrics for model selection, and records the protocol lock hash in every run.

### Phase 4 — Reference baseline and publication gate

1. Run the external NeuralBench task and the Foundry compatibility task on a
   documented reference model/config.
2. Reconcile any difference: event inventory, preprocessing, model details,
   numerical nondeterminism, or implementation divergence.
3. Publish a compact parity report stating which compatibility level has
   passed.
4. Only then retire the current P300 headline result and replace it with the
   canonical test result.

**Gate:** the report distinguishes exact data/task compatibility from optional
model-result parity and documents all unresolved deviations.

## Testing strategy

Tests must not require downloading BI2014a or installing NeuralBench in the
normal Foundry CI environment.

- **Schema tests:** reject missing subjects, duplicate assignments, unknown H5
  subjects, invalid split names, or altered manifest hashes.
- **Pipeline-migration tests:** use a small synthetic H5 inventory and the
  committed P300 fixture to verify partition membership, protocol-group schema,
  explicit sampling units, and preservation of all non-protocol H5 content.
- **Sampling tests:** use synthetic events to verify negative-offset windows,
  boundary handling, anchor-ID target selection in overlapping windows, label
  retention, and baseline correction.
- **Leakage tests:** prove all fitted preprocessing statistics are derived from
  the correct partition at the appropriate scope, or that no fitted statistic
  exists.
- **Run-policy tests:** ensure a compatibility run calls `fit` and then one
  best-checkpoint `test`, while a legacy run retains its original behaviour.
- **Reference-artifact tests:** run only in an optional Python 3.12 reference
  environment; regenerate the manifest and inventory and compare with the
  committed lock.

## Rollout and future tasks

Treat each NeuralBench task/dataset pair as a new locked protocol.  Reuse the
protocol interface, exporter, manifest validation, sampling machinery, and
test policy; do not assume P300's subject split or epoch semantics apply to
other tasks.

For every future integration, require:

1. a pinned external version and exported task artifact;
2. an H5/source identity audit;
3. a split and preprocessing parity test;
4. an explicit distinction between canonical compatibility and Foundry-native
   robustness protocols; and
5. a recorded pretraining-overlap status for every evaluated foundation model.

## Non-goals for the first implementation

- replacing `torch_brain`, Foundry datamodules, Hydra, or Foundry model code;
- downloading data through NeuralBench in production Foundry jobs;
- changing EEG signals, canonical event tables, or legacy H5 split fields;
- claiming equality with NeuralBench model scores before model-specific parity
  is demonstrated; or
- generalizing to every NeuralBench task before the P300 gates pass.

## References

- [NeuralBench P3 task specification](https://facebookresearch.github.io/neuroai/neuralbench/tasks/eeg/p3.html)
- [NeuralBench split transform](https://facebookresearch.github.io/neuroai/neuralbench/generated/neuralbench.transforms.SklearnSplit.html)
- [NeuralBench execution and evaluation workflow](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/quickstart/01_run_first_task.html)

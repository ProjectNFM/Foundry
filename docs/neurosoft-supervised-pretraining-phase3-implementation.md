# NeuroSoft supervised pretraining Phase 3 implementation plan

**Status:** Approved handoff plan (2026-09-03)

**Scope:** Implement, review, and execute Phase 3 of the
[NeuroSoft supervised pretraining roadmap](neurosoft-supervised-pretraining-roadmap.md).

**Prerequisites:** The Phase 0 audit and Phase 2 normalized Conv--BiGRU scratch
baselines are complete.

**Scientific boundary:** Phase 3 validates infrastructure. Phase 4 is the first
phase allowed to make a supervised-pretraining performance claim.

## Outcome

At the end of Phase 3, the repository must be able to:

1. generate immutable, leakage-free source-data manifests for all planned
   Phase 4 volume, Phase 5 subject-diversity, and Phase 6 species-composition
   conditions;
2. train `NeurosoftConvBiGRU` on multiple source sessions using each source
   session's `intrasession-causal` train and validation partitions;
3. select the best source checkpoint using the unweighted mean of
   source-session supported-class validation macro-F1;
4. retain fixed optimizer-step milestones and the best source checkpoint;
5. hand checkpoints to independent target-session jobs through a verified,
   human-readable checkpoint manifest;
6. strictly load the declared shared components into a target-only model with
   a fresh target adapter;
7. run both full finetuning and frozen-representation transfer; and
8. pass a ten-job, two-stage minipig/monkey smoke matrix without using source
   test data or making a scientific transfer claim.

This document is normative for Phase 3. If implementation reveals that a
decision below is infeasible, stop, document the issue, and amend this plan
before changing the experiment.

Two earlier roadmap phrases are explicitly superseded by the interview
decisions captured here:

- replace "permissive loading" with strict, component-scoped transfer plus a
  fresh target adapter; and
- replace "target finetuning jobs for every fraction and seed" in the Phase 3
  smoke itself with the ten canonical jobs below. Phase 3 still implements and
  composition-tests the general fraction/seed interface; Phase 4 performs the
  scaled scientific evaluation.

Update those roadmap lines in the same implementation change so future agents
do not encounter two active specifications.

## Agreed decisions

The following decisions were made during the Phase 3 design interview.

| Area | Decision |
|---|---|
| Runtime stages | Run a two-source-session smoke stage, then a complete leave-one-subject-out source-pool canary. |
| Transfer modes | Validate both full finetuning and frozen representation. |
| Smoke targets | Minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw`; monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`. |
| Source training balance | Keep normal example-proportional training. Explicit diversity manifests are the exception and balance subjects/classes by design. |
| Source recipe | Start from the exact normalized Phase 2 Conv--BiGRU recipe. Do not tune it inside Phase 3. |
| Compute budget | Use optimizer steps, not epochs, as the primary pretraining budget. |
| Transfer strictness | Strictly validate selected shared components. Source adapters are intentionally excluded; permissive loading is not the normal path. |
| Checkpoint handoff | A hash-verified shared-filesystem manifest is authoritative. W&B is a secondary index/copy. |
| Data/test policy | Source pretraining may use source train and source validation only. Source test access is forbidden. Canonical downstream jobs evaluate target test once. |
| Phase 3 performance gate | Require correct, finite, non-collapsed execution, not improvement over scratch. |
| Scaling axes | Study source-example volume separately from source-subject diversity. Do not add a session-count axis. |
| Volume selection | Preserve every eligible source recording; select nested fractions within each recording and represented class. |
| Diversity selection | Use nested `1 ⊂ 2 ⊂ 4 ⊂ all` subject chains for each selection seed at matched example and class budgets. |
| Diversity label control | Use the common subject-level class intersection as the primary comparison. Generate an anchored eight-class sensitivity family separately. |
| Species composition | Generate and validate Phase 6 minipig-only, monkey-only, and 50/50 manifests during Phase 3; do not add mixed-species training to the Phase 3 smoke matrix. |
| Seeds | Store the source-data selection seed in the data manifest and the model initialization seed in Hydra. Pair values `42`, `43`, and `44` in planned runs while logging the roles separately. |

## Existing implementation to preserve

Phase 3 must extend, not replace, the following working pieces.

| Existing location | Reusable behavior |
|---|---|
| `docs/neurosoft-phase0-audit.json` | Content-hashed session eligibility, causal split hashes, target-specific source pools, volume caps, diversity bins, and species-composition budgets. |
| `foundry/data/fraction_manifest.py` | Stable interval identities, canonical JSON hashing, class-aware deterministic permutations, and nesting semantics. |
| `foundry/data/datamodules/base.py` | Dataset construction, interval filtering, deterministic fixed-window loaders, fraction manifests, and per-recording train-only normalization. |
| `foundry/models/neurosoft_conv_bigru.py` | Session adapters, shared temporal frontend/BiGRU/router, and model-declared transfer components. |
| `foundry/training/pretrained.py` | Atomic validate-before-apply checkpoint transfer and structured `TransferReport`. |
| `foundry/training/callbacks/metrics.py` | Collection of validation predictions by session. |
| `foundry/training/callbacks/compute.py` | Steps, windows, signal seconds, wall time, parameter counts, and best-checkpoint compute counters. |
| `main.py` | Hydra construction, normalization artifact handling, transfer loading, best-validation target test evaluation, and W&B/snapshot provenance. |
| `configs/experiment/auditory_decoding/neurosoft_conv_bigru_global_zscore_{minipigs,monkeys}.yaml` | The fixed Phase 2 downstream architecture, normalization, optimizer, and checkpoint recipe. |

Do not change old Phase 1/2 manifest hashes or reinterpret completed runs.
Backward-compatible configs without a source manifest must retain their
current behavior.

## Gaps in the current repository

The implementation agent must not mistake existing nearby features for a
complete Phase 3 pipeline.

1. The current `split_type=loso` puts non-target subjects' complete trials in
   train and the held-out subject in validation. It is a historical
   validation-only baseline and is not the Phase 3 source protocol.
2. The Phase 0 JSON describes source pools and caps, but it does not contain
   the concrete selected train interval IDs required for volume/diversity
   runs.
3. `SessionMetricsCallback` logs per-session metrics directly to the logger;
   it does not expose their unweighted mean as a checkpoint monitor.
4. The standard Lightning checkpoint callback retains `best` and `last`, not
   checkpoints at fixed fractions of a step budget;
5. `main.py` currently discards the returned `TransferReport` after logging;
6. `run.pretrained_checkpoint` accepts an unchecked checkpoint path rather
   than a checkpoint-manifest contract;
7. `_prepare_fraction_provenance` assumes a single target recording and is not
   a source-pretraining provenance record;
8. the compute callback assumes one constant `flops_per_window`, which must be
   revalidated for multisession/species-qualified input heads; and
9. raw recording IDs can overlap between minipigs and monkeys.

## Terminology and invariants

### Canonical identities

Use species-qualified session identifiers everywhere in new source manifests
and checkpoint provenance:

```text
minipigs:sub-03_ses-01_task-AcousStim_acq-RH_desc-raw
monkeys:sub-03_ses-01_task-AcousStim_acq-RH_desc-raw
```

Every record must carry both:

- `species` and the raw dataset `recording_id`; and
- `canonical_recording_id = "<species>:<recording_id>"`.

Interval identities in the new manifest family must hash the canonical
recording ID, positional index, hexadecimal start/end timestamps, and raw
label. Do not modify the old target fraction identity algorithm. Source models
must use canonical IDs as adapter keys. For backward compatibility, a model
constructed without explicit ID aliases may continue using raw IDs.

### Accounting units

Keep the scientific selection unit separate from the runtime processing unit:

- one **selected example** is one audited labeled causal interval/trial;
- one **input window** is the fixed 0.5-second tensor emitted by the sampler;
- one input window contains 1,000 raw time samples; and
- the base frontend produces approximately 250 recurrent time steps per
  window.

The existing sampler emits `floor(interval_duration / 0.5 seconds)` windows
from an interval and drops an incomplete final batch. It may also jitter window
locations between epochs. Therefore, the generator must derive and record, for
every interval and manifest:

- selected interval/example count;
- available 0.5-second windows before batch dropping;
- realized windows per nominal epoch after batch dropping;
- raw input samples and signal seconds per nominal epoch; and
- the sampler/window-accounting implementation version.

Before freezing the generated manifests, verify the expected NeuroSoft
invariant that every eligible acoustic interval produces exactly one
0.5-second window. If it does, record that proof in `README.md`, and example
and pre-drop window counts may be used interchangeably in the scaling tables.
If it does not, stop and amend the volume/diversity allocation rules before
launch: equal example counts and equal processed-input counts would be
different experimental controls, and this plan does not authorize choosing
between them silently.

**Approved NeuroSoft amendment (2026-09-03).** Live source recordings contain
both genuine 0.1-second alternating stimulus/rest annotations and 0.75-second
annotations, in addition to the intended 0.5-second trials. For NeuroSoft
supervised pretraining, an interval is eligible when its duration is at least
0.5 seconds within a 1 ns timestamp tolerance. Shorter intervals are excluded
from both source-train and source-validation selections. Every eligible
interval yields exactly one onset-anchored window `[start, start + 0.5 s)`;
thus a 0.75-second trial deliberately contributes its first 0.5 seconds. The
manifest records per-class selection counts and hashes of the reconstructed
train/validation interval-ID sequences—not repeated raw index arrays. Its
selection and sampler implementation versions identify this policy.

Runs must record cumulative processed windows, raw samples, signal seconds,
recurrent steps, optimizer steps, and estimated FLOPs. The human-readable
`effective_epochs` value is based on what the training loader can actually
emit in one complete pass:

```text
effective_epochs = cumulative_processed_windows / realized_train_windows_per_epoch
```

Also retain Lightning's epoch and batch counters for debugging. The step
budget remains authoritative even when it ends mid-epoch.

### Seeds

The following fields have different meanings and must never be collapsed in
provenance:

```text
source_selection_seed   # subjects/examples in a source manifest
model_seed              # parameter initialization and loader RNG
target_fraction_seed    # downstream nested target subset
```

Planned scientific cells pair the numeric values `42`, `43`, and `44`, but
the schema must permit rerunning another initialization on the same data
manifest.

### Split and test policy

For every selected source recording:

- optimization uses only its audited `intrasession-causal` train intervals;
- source checkpoint selection uses its complete causal validation partition;
- volume fractions never reduce source validation;
- source test interval IDs and metrics are absent from source manifests; and
- calling a test loader or `trainer.test()` in source-pretraining mode raises
  a clear error.

Target adaptation retains the existing causal train/validation/test protocol.

## Artifact layout

Commit source-data manifests with the code. They must therefore be present in
every immutable Git snapshot.

```text
manifests/neurosoft_supervised/v1/
├── index.json
├── README.md
├── source_pools/
│   ├── minipigs/target-sub-01.json
│   └── monkeys/target-sub-01.json
├── phase3_smoke/
│   ├── minipigs/target-sub-06.json
│   └── monkeys/target-sub-01.json
├── source_volume/
│   └── <species>/target-<subject>/fraction-<value>/selection-<seed>.json
├── subject_diversity/
│   ├── common_classes/<species>/target-<subject>/subjects-<n>/selection-<seed>.json
│   └── eight_class_anchor/<species>/target-<subject>/subjects-<n>/selection-<seed>.json
└── species_composition/
    └── <target-species>/target-<subject>/<composition>/selection-<seed>.json
```

Expected logical selection-manifest counts are:

| Family | Count | Basis |
|---|---:|---|
| Phase 3 two-session smoke | 2 | one per selected target species |
| Phase 4 source volume | 144 | 12 target subjects x 4 fractions x 3 selection seeds |
| Phase 5 common-class diversity | 129 | audit-supported subject bins x 3 seeds |
| Phase 5 anchored sensitivity | 129 | same bins, separately labeled sensitivity family |
| Phase 6 composition | 108 | 12 targets x 3 compositions x 3 seeds |

Generate all 144 volume manifests even though the selected interval set at
100% is seed-invariant. Keeping the selection seed in each logical manifest
makes the intended run pairing and audit table explicit.

Checkpoint artifacts are produced at runtime and live on compute-node-visible
shared storage, outside the Git snapshot:

```text
<checkpoint-root>/<run-id>/
├── checkpoints/<checkpoint-name>.ckpt
├── manifests/<checkpoint-name>.json
├── manifests/<checkpoint-name>.md
├── transfer-report.json             # downstream only
└── transfer-report.md               # downstream only
```

Introduce a narrowly named environment/config setting such as
`FOUNDRY_CHECKPOINT_ROOT`; do not overload `FOUNDRY_SNAPSHOT_ROOT`. The
resolved root must be shared and visible to later downstream jobs. On the
legacy cluster use an explicit `/network/scratch/...` location. On Clariden
use a mounted `/capstor` location.

## Source manifest schemas

Implement typed, versioned schema objects in a new module such as
`foundry/data/source_manifest.py`. JSON examples below are abbreviated; the
implementation must not omit the described validation fields.

### Source-pool manifest

Create one source-pool file per target subject. A file may contain the three
composition pools already represented in the Phase 0 audit.

```json
{
  "schema": "neurosoft-source-pool",
  "version": 1,
  "phase0_audit_sha256": "...",
  "target": {
    "species": "minipigs",
    "subject": "sub-06",
    "eligible_recordings": ["..."]
  },
  "pools": {
    "same_species": {
      "source_subjects": ["minipigs:sub-01", "..."],
      "source_recordings": ["minipigs:sub-01_ses-01_...", "..."],
      "source_subject_count": 6,
      "source_recording_count": 38,
      "class_counts": {"low_bass": 509, "...": 0},
      "target_leakage": []
    }
  },
  "manifest_hash": "..."
}
```

The pool hash must cover the target, source subjects/recordings, source train
split hashes, composition, and parent audit hash.

### Source-selection manifest

One selection manifest represents the complete source-data input to one
pretraining run.

```json
{
  "schema": "neurosoft-source-selection",
  "version": 1,
  "selection_id": "volume_minipigs_target-sub-06_f0.25_sel42",
  "family": "source_volume",
  "phase0_audit_sha256": "...",
  "source_pool_manifest": "../../../../source_pools/minipigs/target-sub-06.json",
  "source_pool_hash": "...",
  "target": {"species": "minipigs", "subject": "sub-06"},
  "condition": {
    "source_composition": "minipigs_only",
    "requested_fraction": 0.25,
    "subject_count_bin": null,
    "source_selection_seed": 42,
    "class_coverage_policy": "all_available"
  },
  "summary": {
    "source_subject_count": 6,
    "source_recording_count": 38,
    "selected_train_examples": 7239,
    "available_train_windows": 7239,
    "realized_train_windows_per_epoch": 7232,
    "selected_signal_seconds": 3619.5,
    "validation_examples": 804,
    "available_validation_windows": 804,
    "represented_class_union": ["..."],
    "represented_class_intersection": ["..."],
    "requested_fraction": 0.25,
    "realized_fraction": 0.2501
  },
  "subjects": ["minipigs:sub-01", "..."],
  "recordings": [
    {
      "species": "minipigs",
      "subject": "sub-01",
      "recording_id": "sub-01_ses-01_...",
      "canonical_recording_id": "minipigs:sub-01_ses-01_...",
      "raw_channel_count": 32,
      "supported_channel_count": 18,
      "train_source_intervals_hash": "...",
      "train_selected_indices": [0, 4],
      "train_selected_interval_ids": ["...", "..."],
      "train_counts_by_class": {"low_bass": 54, "...": 0},
      "available_train_windows": 2,
      "valid_source_intervals_hash": "...",
      "valid_interval_ids": ["...", "..."],
      "available_validation_windows": 2
    }
  ],
  "source_test_policy": "forbidden",
  "target_leakage": [],
  "manifest_hash": "..."
}
```

Store positional indices for efficient selection and stable IDs/hashes for
verification. Counts in `summary` must be generated from the actual lists, not
copied from requested values. JSON hashes use sorted keys and compact ASCII
encoding, excluding only the top-level `manifest_hash` field.

The numbers above illustrate field shape, not authoritative counts. Generated
values come only from the audited data and the configured sampler. In
particular, distinguish raw channels from the supported channels that reach a
session adapter.

### Manifest index

`index.json` is a generated catalog, not a source of scientific truth. Each
entry contains:

- selection ID and relative file path;
- manifest hash and parent pool hash;
- target species/subject;
- family, condition, selection seed, and composition;
- subject/recording/example/window/class counts; and
- eligibility plus any explicit failure reason.

Hydra sweep resolvers query this index and then load the selected manifest.
Every selected manifest must still be independently hash-checked.

`README.md` is generated from the same index and provides human-readable
tables by target and family. Never maintain its counts manually.

## Manifest generation algorithms

Create a deterministic, data-backed tool such as:

```text
tools/generate_neurosoft_source_manifests.py
```

It must take the processed-data root, Phase 0 audit, output directory, task
config, fractions, and selection seeds as explicit inputs. The normal command
is expected to resemble:

```bash
uv run python tools/generate_neurosoft_source_manifests.py \
  --data-root "$SCRATCH/brainsets/processed" \
  --audit docs/neurosoft-phase0-audit.json \
  --task configs/tasks/neurosoft_acoustic_stim_8band.yaml \
  --output manifests/neurosoft_supervised/v1 \
  --fractions 0.10 0.25 0.50 1.00 \
  --selection-seeds 42 43 44
```

Generation is read-only with respect to processed neural data. Write output
atomically. Regenerating twice with the same inputs must produce byte-identical
JSON and Markdown.

### Common preparation

For each of the 12 eligible target subjects:

1. verify the Phase 0 artifact hash;
2. reload both species' configured recordings and audited causal splits;
3. reject any runtime/audit split-hash mismatch;
4. select eligible source recordings only;
5. exclude every recording from the target subject in the target species;
6. build canonical subject, recording, and interval identities;
7. calculate per-recording, per-subject, and aggregate class availability; and
8. write the target-specific source-pool manifest before any derived
   selection.

Never select a source condition from validation metrics or model outcomes.

### Phase 3 two-session manifests

Use the following fixed sources.

| Target | Source 1 | Source 2 |
|---|---|---|
| Minipig `sub-06` | `minipigs:sub-02_ses-01_task-AcousStim_acq-LH_desc-raw` (6 classes, 753 train examples) | `minipigs:sub-03_ses-06_task-AcousStim_acq-LH_desc-raw` (8 classes, 861 train examples) |
| Monkey `sub-01` | `monkeys:sub-02_ses-02_task-AcousStim_acq-RH_desc-raw` (8 classes, 3,193 train examples) | `monkeys:sub-05_ses-01_task-AcousStim_acq-RH_desc-raw` (7 classes, 1,735 train examples) |

Use complete causal train and validation partitions for these smoke manifests.
The sources belong to two distinct non-target subjects and have an eight-class
aggregate union.

### Phase 4 volume manifests

For each target, selection seed, source recording, and represented class:

1. create one deterministic permutation using the canonical recording ID,
   class ID, and source selection seed;
2. select prefix lengths `ceil(fraction * available_count)` for 10%, 25%, and
   50%, and all labeled interval/examples for 100%;
3. prove `10% ⊂ 25% ⊂ 50% ⊂ 100%` for every recording/class and for
   the complete manifest;
4. preserve every eligible source recording, with at least one selected train
   example from each represented recording/class when the source count is
   positive; and
5. attach every selected recording's complete causal validation partition.

The source-subject and source-recording sets must be identical across volume
fractions. Record both requested and realized fractions.

### Phase 5 primary subject-diversity manifests

For each target and selection seed:

1. deterministically rank candidate same-species subjects;
2. form one nested chain with audit-supported bins:
   - minipigs: `1 ⊂ 2 ⊂ 4 ⊂ 6` subjects;
   - monkeys: `1 ⊂ 2 ⊂ 4` subjects;
3. use the Phase 0 `common_present_classes` and
   `common_per_class_cap` for that target;
4. hold the total selected count for every common class exactly constant
   across bins;
5. divide each class allocation as evenly as integer constraints permit across
   selected subjects;
6. distribute a subject/class allocation across that subject's eligible
   recordings in proportion to available causal train examples, using a
   deterministic largest-remainder rule and stable tie-breaking;
7. ensure every included validation recording has at least one selected train
   example, or fail the manifest rather than validating an untrained adapter;
8. select interval prefixes using deterministic recording/class
   permutations; and
9. include complete causal validation intervals for the selected recordings.

Log the realized recording/session count. Session count is not a scientific
axis and may differ between subject selections; it must remain visible as a
design attribute.

### Phase 5 eight-class anchor sensitivity manifests

Generate this as a separate family and mark it `sensitivity_only=true`.

1. Restrict the one-subject anchor to a source subject whose aggregate source
   recordings contain all eight classes.
2. Keep that anchor in every larger nested bin.
3. Add other subjects by the seed-specific deterministic ranking.
4. Derive one conservative eight-class count vector that is feasible for
   every bin in that chain, then hold it fixed across bins.
5. Record the number of eligible anchors and whether independent one-subject
   selection replication is possible.

For monkey pools where only one all-eight-class anchor exists, state loudly
that seeds replicate interval/model randomness but not one-subject identity.
These manifests must not be pooled with the primary common-class comparison.

### Phase 6 species-composition manifests

For every target and seed, generate `minipigs_only`, `monkeys_only`, and
`mixed_50_50` conditions. These rules use labeled example counts once the
one-example/one-window invariant has passed; otherwise generation must stop at
the accounting-unit gate above.

- Exclude the target subject from any condition containing its species.
- Match total selected examples and their derived window counts across the
  three conditions.
- Allocate exactly half of mixed examples/windows to each species, allowing at
  most a one-example/window rounding difference.
- Match the per-class count vector across compositions when feasible.
- If the Phase 0 equal-total budget cannot also support a common per-class
  vector, reduce to the largest feasible common vector and report both the
  original and realized budgets. Never hide a class-distribution mismatch
  behind equal total volume.
- Allocate within species across subjects/classes first, then recordings,
  using deterministic largest-remainder rules.
- Record expected per-condition FLOPs. Equal window count is primary; small
  adapter/channel-count FLOP differences are reported rather than silently
  compensated with unequal labeled volume.

No Phase 6 manifest is executed during Phase 3.

## Data loading and normalization

### DataModule surface

Add optional source-manifest configuration without breaking target fraction
runs. A suitable interface is:

```yaml
data:
  role: source_pretraining
  selection_manifest: ${source_manifest}
  split_type: intrasession-causal
  source_test_policy: forbidden
```

Rules:

- `selection_manifest` and `training_fraction` are mutually exclusive;
- source-manifest selection happens before normalization is fitted;
- train and validation loaders use only the selected/declared interval IDs;
- the test loader raises in `source_pretraining` mode;
- model session configs are derived only from manifest recordings; and
- any mismatch between the Hydra species/data config and manifest is fatal.

The source manifest loader should reuse neutral hashing/interval-selection
helpers but should not embed NeuroSoft subject lists in the generic
DataModule.

### Species-qualified adapter keys

Do not mutate cached `Data.session.id`. A raw recording ID alone is not a safe
alias key because the two species may use the same value. Instead, make the
data path supply a dataset namespace alongside the raw ID and resolve the pair
through a nested alias map in the data-driven model metadata:

```python
{
    "minipigs": {
        "sub-03_ses-01_...": "minipigs:sub-03_ses-01_..."
    },
    "monkeys": {
        "sub-03_ses-01_...": "monkeys:sub-03_ses-01_..."
    }
}
```

Add the namespace as non-destructive sample metadata before tokenization; do
not rewrite the underlying session object. Extend `NeurosoftConvBiGRU` to
resolve `(namespace, raw_id)` during tokenization and use canonical IDs in
`input_session_ids` and `session_adapter.layers`. A missing namespace, alias,
or ambiguous raw ID in source-pretraining mode is fatal. When no alias mapping
is supplied, preserve the existing raw-ID behavior for old single-dataset
configs. Source adapter parameters remain outside every transferable
component.

### Normalization

Use `recording_train_global_zscore` exactly as in Phase 2:

- fit one frozen mean/scale from each selected recording's selected causal
  train intervals;
- apply it to that recording's source train and source validation windows;
- persist the stats and manifest hashes;
- never pool normalization statistics across recordings; and
- never transfer source normalization statistics to a target session.

Target adaptation independently fits target normalization from the selected
target causal-training partition.

The existing normalization implementation currently fits from
`_effective_sampling_intervals("train")`; tests must prove the new manifest
selection is already active at that point.

### Training sampler

Normal source pretraining remains example-proportional: feed the selected
intervals to the NeuroSoft onset-anchored fixed-window sampler without a
session-balanced sampler. Recording/session contribution therefore follows the
number of eligible windows. Each eligible interval contributes exactly one
non-jittered window from its onset, and incomplete batches follow the existing
`drop_last` behavior; both facts must be included in the accounting metadata.

The Phase 5 generator itself balances its fixed budget across subjects and
classes. Once that manifest is built, the ordinary example-proportional loader
faithfully executes the designed allocation.

Log selected examples, available windows, and actually processed windows by
canonical recording and subject at run start and fit end.

## Source validation and checkpoint selection

### Required aggregate

At every source validation event:

1. accumulate predictions and targets by full canonical source session;
2. compute supported-class macro-F1 independently for each session;
3. record each session's represented classes, aggregation mask, support, and
   supported macro-F1;
4. compute the unweighted arithmetic mean across sessions with a defined
   supported F1; and
5. log the scalar under a stable key, for example:

```text
val/source_session_mean_supported_f1
```

The scalar must be available in Lightning callback metrics before early
stopping and `ModelCheckpoint` make their decisions. Direct logger-only output
is insufficient.

Extend `SessionMetricsCallback` or add a narrowly focused
`SourceSessionMetricsCallback`. Avoid implementing a second definition of
supported-class F1. In distributed validation, gather complete per-session
predictions/targets correctly or explicitly reject unsupported distributed
execution; never average per-rank session metrics.

### Best checkpoint

Source early stopping and the standard best checkpoint monitor:

```text
val/source_session_mean_supported_f1
```

with mode `max` and patience 40 validation events. All Phase 3 target
adaptations use the corresponding best source checkpoint.

Do not use aggregate window-level F1, mean source loss, target validation, or
any source/target test metric to choose a source checkpoint.

## Compute milestones and accounting

### Fixed milestones

Implement a callback such as `ComputeMilestoneCheckpointCallback` that saves
at the following fractions of configured `trainer.max_steps`:

```text
1%, 3%, 10%, 30%, 100%
```

Convert percentages to monotonically increasing integer optimizer steps with
a documented rounding rule. Deduplicate collisions for very small debug
budgets and record the realized percentages. Phase 3 budgets do not collide:

| Stage | `max_steps` | Validation cadence | Nominal milestone steps |
|---|---:|---:|---|
| Two-session smoke | 500 | every 100 optimizer steps | 5, 15, 50, 150, 500 |
| Full-pool canary | 5,000 | every 500 optimizer steps | 50, 150, 500, 1,500, 5,000 |

Save milestones after completed optimizer steps, not microbatches. Use atomic
temporary-file-to-final-name replacement. Preserve callback state across
Slurm requeues so milestones are neither lost nor silently overwritten.

If early stopping ends a later scientific run before a nominal milestone,
record it as `not_reached`; never relabel the last checkpoint as 100%.

### FLOPs

Do not blindly copy the Phase 2 `768098304` FLOPs/window constant. Validate
profiler coverage for:

- every distinct supported source channel count;
- the session adapter;
- depthwise and pointwise convolutions;
- both directions and all layers of the GRU;
- the router and loss; and
- forward plus backward/optimizer work represented by the callback's chosen
  definition.

Support either a validated per-canonical-session FLOPs table or a documented
analytic correction to the shared base cost. The compute callback must sum
the realized batch composition. Store raw per-session window counts so FLOPs
can be recomputed.

At every validation event and checkpoint record:

- optimizer and scheduler steps;
- processed windows/examples, raw samples, signal seconds, and effective
  epochs;
- cumulative estimated FLOPs and method/version;
- total/trainable parameters and effective batch size;
- precision, GPU model, peak memory, and wall time; and
- the monitored aggregate validation score.

## Checkpoint manifest and transfer contract

### Authoritative checkpoint manifest

For every best and milestone checkpoint, write a canonical JSON manifest and
a generated Markdown companion. The JSON includes at least:

```json
{
  "schema": "neurosoft-pretraining-checkpoint",
  "version": 1,
  "checkpoint": {
    "kind": "best",
    "path": "checkpoints/best.ckpt",
    "sha256": "...",
    "size_bytes": 0
  },
  "trained_on": {
    "source_selection_id": "...",
    "source_manifest_path": "...",
    "source_manifest_hash": "...",
    "excluded_target": {"species": "minipigs", "subject": "sub-06"},
    "subjects": ["..."],
    "recordings": ["..."],
    "selected_train_examples": 0,
    "available_train_windows": 0,
    "realized_train_windows_per_epoch": 0,
    "processed_windows": 0,
    "completed_effective_epochs": 0.0,
    "optimizer_steps": 0,
    "class_union": ["..."],
    "class_intersection": ["..."]
  },
  "selection": {
    "monitor": "val/source_session_mean_supported_f1",
    "monitor_value": 0.0,
    "source_session_scores": {"canonical-id": 0.0}
  },
  "compute": {
    "cumulative_flops": 0,
    "flop_method": "...",
    "signal_seconds": 0.0,
    "wall_time_seconds": 0.0,
    "gpu": "...",
    "precision": "bf16-mixed"
  },
  "recipe": {},
  "normalization_artifact_hashes": {},
  "git_sha": "...",
  "snapshot_bundle": "...",
  "slurm_job_id": "...",
  "wandb": {"project": "...", "group": "...", "run_id": "..."},
  "manifest_hash": "..."
}
```

The Markdown companion must make the following visible without reading JSON:

- checkpoint kind, score, path, and SHA-256;
- target subject excluded from pretraining;
- source species, subjects, complete recording list, and class coverage;
- selected examples, available/processed windows, and signal duration;
- completed effective epochs, including fractional progress;
- optimizer steps, FLOPs, wall time, and hardware;
- architecture/optimizer/normalization recipe; and
- Git, snapshot, Slurm, and W&B provenance.

Generate Markdown from the finalized JSON. Do not maintain two independent
sources of truth.

### Downstream input

Add a new config field:

```yaml
run:
  pretrained_checkpoint_manifest: ???
```

When set, the loader must:

1. resolve the manifest and checkpoint on shared storage;
2. verify the manifest hash and checkpoint SHA-256;
3. verify source-manifest, architecture, and task identities;
4. verify that the current target matches the manifest's excluded target;
5. reject simultaneous use of `pretrained_checkpoint` and
   `pretrained_checkpoint_manifest`;
6. choose components from the requested model-declared transfer regime;
7. invoke the existing atomic strict loader; and
8. persist the complete transfer report in JSON/Markdown and W&B metadata.

Keep direct `run.pretrained_checkpoint` support for old experiments, but all
new NeuroSoft Phase 3+ jobs use the manifest field.

### Transfer regimes

Use strict component-scoped loading.

| Regime | Load | Keep fresh | Train |
|---|---|---|---|
| `full_finetuning` | `temporal_frontend`, `gru`, `router` | target `session_adapter` | every target parameter |
| `frozen_representation` | `temporal_frontend`, `gru` | target `session_adapter`, `router` | target adapter and router only |

The transfer report must show every source adapter tensor as intentionally
excluded. Missing, shape-mismatched, or dtype-mismatched selected shared
tensors are fatal. `permissive` remains available only for an explicitly
documented future architecture-change experiment.

Given the same model seed and target config, assert that loading does not
change the freshly initialized target adapter. For frozen representation,
also assert that the fresh router is unmodified.

## Hydra configuration

### Proposed files

Add reusable configs rather than one YAML per data condition:

```text
configs/data/neurosoft_minipigs/source_pretraining.yaml
configs/data/neurosoft_monkeys/source_pretraining.yaml
configs/experiment/pretraining/neurosoft_conv_bigru_supervised_minipigs.yaml
configs/experiment/pretraining/neurosoft_conv_bigru_supervised_monkeys.yaml
configs/experiment/auditory_decoding/neurosoft_conv_bigru_transfer_minipigs.yaml
configs/experiment/auditory_decoding/neurosoft_conv_bigru_transfer_monkeys.yaml
```

Expose these stable Hydra dimensions:

```yaml
source_manifest: ???

run:
  seed: ???                         # model seed
  pretrained_checkpoint_manifest: null
  pretrained_transfer_regime: null # full_finetuning/frozen_representation

data:
  role: source_pretraining
  selection_manifest: ${source_manifest}
```

Add resolvers that query `manifests/neurosoft_supervised/v1/index.json` by
family, species, target, fraction/bin, composition, and selection seed. A
resolver returns paths only; it must not reconstruct or modify selections.

Run names include target species/subject, manifest family/condition,
selection seed, model seed, and checkpoint kind. These values must also be
logged as structured config; names are not provenance.

### Fixed source recipe

Use the exact normalized Phase 2 starting recipe:

```yaml
model:
  adapter_dim: 64
  temporal_channels: 128
  temporal_kernel_samples: 64
  temporal_stride: 4
  conv_depth: 1
  dropout_rate: 0.3
  gru_hidden_size: 128
  gru_num_layers: 2
  gru_bidirectional: true
  gru_dropout: 0.0

data:
  sequence_length: 0.5
  input_normalization:
    mode: recording_train_global_zscore

class_weights:
  mode: none

hyperparameters:
  batch_size: 16
  learning_rate: 0.0015
  weight_decay: 0.018

trainer:
  precision: bf16-mixed
  gradient_clip_val: 1.0
```

Do not add a pretraining learning-rate, regularization, class-weight, or
scheduler sweep. If a source smoke run is unstable or collapsed, pause and
create a separate one-hypothesis recipe experiment.

### Downstream recipe

Target transfer runs must reuse the matching Phase 2 target data manifest,
normalization, optimizer, fraction, seed, early stopping, and checkpoint
selection. For Phase 3 canonical runs:

```text
target training fraction = 100%
model seed = 42
target fraction seed = 42
evaluate target test = true
```

Development attempts set `evaluate_test=false`. Do not add an actual reduced-
data Phase 3 job; config composition and integration tests cover other target
fractions/seeds.

## Implementation work packages

Implement and review in the following order. Do not launch training before all
earlier gates pass.

### WP1 -- Schemas, generator, and static validation

- Add typed source-pool/source-selection schema and hash validation.
- Add canonical species-qualified identities without changing old hashes.
- Implement volume, diversity-primary, diversity-sensitivity, composition,
  smoke, index, and README generation.
- Add a standalone validator, or a `--validate-only` mode, that checks every
  generated manifest against processed data and the Phase 0 audit.
- Generate all committed `v1` manifests and review aggregate counts/diffs.

Gate: two independent generations are byte-identical; every expected manifest
exists; every leakage list is empty; all interval and nesting checks pass.

### WP2 -- Source DataModule and canonical session routing

- Load and verify one source-selection manifest.
- Restrict train/validation intervals exactly as declared.
- Forbid source test access;
- fit normalization after selection;
- auto-populate canonical session configs, namespaces, and aliases; and
- update the model tokenizer/routing path compatibly.

Gate: a synthetic two-session, unequal-channel dataset trains, validates, and
backpropagates while an unknown/cross-species/target-leaking manifest fails
before training.

### WP3 -- Source-session metric aggregate

- Compute per-session supported metrics from complete predictions/targets.
- Expose the unweighted source-session mean as a callback metric;
- log aggregation masks and session support; and
- make early stopping and best checkpoint consume that exact scalar.

Gate: a hand-computed unequal-session-size example matches exactly and differs
from the intentionally unequal pooled/window-weighted reference.

### WP4 -- Compute milestones and checkpoint manifests

- Implement resumable optimizer-step milestone saving.
- Upgrade FLOP accounting for realized session composition;
- write atomic JSON/Markdown checkpoint manifests; and
- validate best/milestone checkpoint provenance.

Gate: a short interrupted/resumed run saves each reached milestone once,
reports exact counters, and verifies every checkpoint hash.

### WP5 -- Strict manifest-based transfer

- Add checkpoint-manifest input to `main.py`.
- Validate all provenance before loading;
- persist transfer reports; and
- cover full-finetuning and frozen-representation parameter states.

Gate: source adapters never load, target adapters remain bitwise fresh, and a
tampered checkpoint/manifest or mismatched target fails atomically.

### WP6 -- Hydra configs and analysis/reporting

- Add generic pretraining/transfer configs and index resolvers.
- Add config composition tests for both species, both transfer modes, every
  target fraction, and seeds 42/43/44;
- create the Phase 3 experiment record; and
- create its W&B API-backed analysis script before launch.

Gate: all planned Phase 3 commands compose without hardcoded recording lists
and resolve to the intended manifest/checkpoint policies.

### WP7 -- Execute the ten-job matrix

Run the stages sequentially and stop at each gate. Never submit all downstream
jobs before their source checkpoint manifests exist and pass validation.

## Required tests

### Manifest unit tests

- canonical hash determinism and sensitivity to every scientific field;
- species-qualified recording/interval collision avoidance;
- target-subject exclusion across same/cross/mixed pools;
- eligible-recording-only source selection;
- exact interval-ID reconstruction and failure on split drift;
- per-recording/per-class volume nesting;
- volume recording-set invariance;
- nested subject chains for each seed;
- exact common-class count equality across diversity bins;
- subject/class/recording allocation and deterministic remainder ties;
- eight-class anchor invariants and replication-limit metadata;
- Phase 6 equal volume, 50/50 allocation, and class-vector checks;
- malformed version, missing parent, bad hash, duplicate ID, and path errors;
- byte-identical regeneration; and
- expected index counts and no duplicate selection IDs.

### Data/normalization integration tests

- source train/validation selections match their manifest exactly;
- derived available and per-epoch window counts match the real sampler;
- every eligible NeuroSoft interval satisfies the one-example/one-window gate;
- source validation is invariant across volume fractions;
- source test loader and `trainer.test()` fail;
- normalization sees selected source train intervals only;
- one recording's normalization never affects another;
- canonical aliases select the correct session adapter;
- the same raw ID in two species resolves through its namespace without a
  collision;
- padded channels/time remain excluded; and
- old target fraction configs behave unchanged.

### Metric tests

- supported-class F1 is computed separately per session;
- absent-class predictions remain errors while absent positive classes are
  excluded from the macro denominator;
- unequal session sizes receive equal aggregate weight;
- aggregation masks and class counts are logged;
- undefined/empty sessions fail loudly rather than disappearing; and
- the aggregate is visible to early stopping, checkpointing, and compute
  tracking at the same validation event.

### Checkpoint/compute tests

- milestone rounding, deduplication, naming, and exact global-step trigger;
- gradient accumulation counts optimizer steps correctly;
- requeue/resume preserves counters and avoids duplicate milestone writes;
- best checkpoint score/counters match the monitored aggregate;
- effective epochs use processed windows divided by the realized train windows
  per nominal epoch;
- per-session FLOP sums match a hand calculation and counters are monotonic;
- checkpoint and source-manifest hashes detect tampering;
- Markdown is generated from JSON and shows source data plus epochs; and
- an unreached milestone is explicit.

### Transfer/end-to-end tests

- full finetuning loads frontend/GRU/router and trains all target parameters;
- frozen representation loads/freezes frontend/GRU, leaves router fresh, and
  trains only router plus target adapter;
- source adapters are intentionally excluded in both modes;
- target adapter is bitwise unchanged by loading;
- shared mismatch fails before any target tensor changes;
- manifest target must equal the downstream target;
- direct checkpoint and manifest checkpoint inputs are mutually exclusive;
- target validation selects the downstream checkpoint; and
- target test runs exactly once from that checkpoint.

### Suggested verification commands

Adjust filenames to the implemented modules, then run at least:

```bash
uv run ruff check foundry tests tools main.py
uv run pytest -q \
  tests/test_fraction_manifest.py \
  tests/test_source_manifest.py \
  tests/test_models/test_neurosoft_conv_bigru.py \
  tests/test_source_session_metrics.py \
  tests/test_compute_milestone_checkpoint.py \
  tests/test_pretrained_loading.py \
  tests/test_pretrained_transfer_regimes.py \
  tests/test_neurosoft_phase3_configs.py

uv run python tools/generate_neurosoft_source_manifests.py <arguments-above>
uv run python tools/validate_neurosoft_source_manifests.py \
  --data-root "$SCRATCH/brainsets/processed" \
  --manifest-root manifests/neurosoft_supervised/v1
```

Also run the broader existing test suite if practical. A pre-existing
unrelated failure must be recorded with evidence; do not weaken tests to make
Phase 3 pass.

## Phase 3 experiment record

Before launching, create one MS-reviewed infrastructure experiment file in
`experiments/inbox/`, following the repository experiment template. Suggested
stem:

```text
20260903-MS-neurosoft-supervised-pretraining-pipeline
```

Its single hypothesis is that the manifest-to-pretraining-to-transfer pipeline
passes the declared correctness, leakage, provenance, checkpoint, and compute
gates for both species. It must not hypothesize a performance gain.

Create the matching analysis script before launch:

```text
analysis/20260903-MS-neurosoft-supervised-pretraining-pipeline_analysis.py
```

The script uses `wandb.Api()`, checks the exact canonical run set, exports
tables to `analysis/csv/`, and reports:

- run completeness/status;
- source aggregate and per-session validation metrics;
- source/target manifest and checkpoint hashes;
- milestone/best checkpoint counters;
- transfer loaded/excluded/frozen/trainable counts;
- downstream validation/test checkpoint identity;
- compute monotonicity and profiler method; and
- descriptive target F1 versus the existing matched scratch control.

Create separate future experiment records for Phase 4 volume, Phase 5 subject
diversity, the eight-class sensitivity analysis, and Phase 6 composition. One
Phase 3 infrastructure report must not become an omnibus scientific report.

## Canonical ten-job execution matrix

All new Phase 3 runs use model seed 42. Target jobs use 100% causal target
training data and target fraction seed 42. Reuse the completed Phase 2 scratch
runs; do not rerun scratch.

### Stage A -- two-session smoke (six jobs)

| # | Species | Run | Input | Budget/test policy |
|---:|---|---|---|---|
| 1 | Minipigs | source pretraining | fixed two-session minipig manifest | 500 steps; source test forbidden |
| 2 | Monkeys | source pretraining | fixed two-session monkey manifest | 500 steps; source test forbidden |
| 3 | Minipigs | full finetuning | job 1 best checkpoint manifest | target test once |
| 4 | Minipigs | frozen representation | job 1 best checkpoint manifest | target test once |
| 5 | Monkeys | full finetuning | job 2 best checkpoint manifest | target test once |
| 6 | Monkeys | frozen representation | job 2 best checkpoint manifest | target test once |

Stage A gate:

- both source jobs are finite and predict more than one validation class;
- both hand-checked source-session means match logged values;
- all milestones and best manifests validate;
- downstream strict transfer reports contain the exact expected components;
- frozen/trainable parameter sets are correct; and
- target jobs are finite, select on validation, and evaluate test once.

Target performance may be lower than scratch without failing this gate. A
clear collapse or invariant violation requires diagnosis before Stage B.

### Stage B -- full-pool canary (four jobs)

| # | Species | Run | Input | Budget/test policy |
|---:|---|---|---|---|
| 7 | Minipigs | source pretraining | full same-species pool excluding minipig `sub-06` (38 eligible recordings) | 5,000 steps; source test forbidden |
| 8 | Monkeys | source pretraining | full same-species pool excluding monkey `sub-01` (4 eligible recordings) | 5,000 steps; source test forbidden |
| 9 | Minipigs | full finetuning | job 7 best checkpoint manifest | target test once |
| 10 | Monkeys | full finetuning | job 8 best checkpoint manifest | target test once |

Stage B repeats every Stage A infrastructure gate and additionally proves that
normalization, adapter construction, per-session validation, compute
accounting, and checkpoint manifests scale to complete target-specific source
pools.

Frozen representation is not repeated in Stage B because Stage A validates
its transfer contract. Fixed milestones are load/hash-validated, but only the
best source checkpoint receives an actual Phase 3 target adaptation.

## Performance and test discipline

Phase 3 acceptance is performance-neutral. Require:

- finite source and downstream losses/metrics;
- non-identical representations and more than one predicted validation class;
- a validation trajectory that is not an obvious data/adapter failure; and
- all engineering/scientific invariants in this plan.

Do not require pretrained target F1 to equal or exceed scratch. Report the
comparison descriptively and do not use it to alter sources, recipe, budget,
or checkpoint choice.

Source test data is never loaded. During development, downstream
`evaluate_test=false`. Only the final canonical downstream jobs above use
`evaluate_test=true`, and they test exactly once from their target-validation
selected checkpoint. Phase 4 retains the formal three-seed, all-session
100%-data transfer gate.

## Review checklist before execution

### Scientific review

- [ ] Every target-specific source pool excludes the complete target subject.
- [ ] Source train/validation are audited causal partitions; source test is
      inaccessible.
- [ ] Volume changes examples without changing source recordings.
- [ ] Primary diversity changes subjects at fixed class/example counts.
- [ ] Anchored diversity is labeled sensitivity-only.
- [ ] Composition conditions match volume and class distribution as closely as
      feasible, with differences explicit.
- [ ] Selection seeds and model seeds are separate.
- [ ] Best source selection is an unweighted session mean.
- [ ] Phase 3 contains no performance hypothesis.

### Engineering review

- [ ] Old configs, target fraction manifests, and transfer APIs remain
      backward compatible.
- [ ] All file and content hashes are verified before data access/loading.
- [ ] Source manifest selection precedes normalization fitting.
- [ ] Species-qualified adapters cannot collide.
- [ ] Source adapters never transfer.
- [ ] Checkpoint writes are atomic and resumable.
- [ ] Human-readable checkpoint summaries are generated from canonical JSON.
- [ ] FLOP coverage includes convolution and recurrent operations and realized
      session composition.
- [ ] W&B-offline execution still leaves complete shared-filesystem artifacts.
- [ ] Test coverage exercises failure paths, not only successful runs.

### Execution review

- [ ] All relevant tests and data-backed manifest validations pass.
- [ ] The experiment record and analysis script are committed.
- [ ] The repository is clean: `git status --short` prints nothing.
- [ ] `FOUNDRY_SNAPSHOT_ROOT` is a compute-node-visible shared directory.
- [ ] `FOUNDRY_CHECKPOINT_ROOT` is a distinct compute-node-visible shared
      directory.
- [ ] The chosen partition and timeout fit the cluster and canary duration.
- [ ] Every command uses the normal `python main.py ... -m` snapshot workflow.

## Launch discipline and recording

Do not launch from an uncommitted or dirty repository. Do not disable snapshot
creation or bypass the clean-Git check.

On the legacy cluster:

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
# Set an explicit shared checkpoint root under /network/scratch.
# Submit production work to long unless the investigator chooses otherwise.
python main.py <phase3-overrides> -m
```

On Clariden:

```bash
export FOUNDRY_SNAPSHOT_ROOT=<mounted-/capstor-path>/foundry-launches
# Set a distinct mounted /capstor checkpoint root.
# Use normal for production; debug only for a genuinely short canary.
python main.py <phase3-overrides> -m
```

Immediately after every submission, record in the Phase 3 experiment file:

- exact command and resolved config/manifest IDs;
- Git SHA and branch;
- Slurm job ID;
- immutable snapshot bundle path;
- shared checkpoint root/output path;
- W&B project/group/run name and run ID; and
- any dependency between source and downstream submissions.

If a job is requeued, resumed, timed out, or replaced, retain the complete
history and mark which attempt is canonical. Never overwrite an old
checkpoint manifest in place.

## Final Phase 3 acceptance criteria

Phase 3 is complete only when all of the following are true:

1. all committed manifest families regenerate byte-identically and pass the
   data-backed validator;
2. every source manifest has empty target leakage and no source-test payload;
3. volume manifests are nested and recording-invariant;
4. primary diversity manifests have nested subjects plus identical per-class
   and total example budgets across bins;
5. Phase 6 manifests have reviewed equal-volume/50-50/class-coverage reports;
6. the source DataModule executes exact manifest train/validation selections
   and forbids test;
7. unweighted source-session validation F1 matches a hand reference and drives
   best checkpoint selection;
8. all reached fixed milestones and best checkpoints have verified JSON and
   human-readable Markdown manifests;
9. compute counters are monotonic, resumable, and based on documented
   convolution/GRU-aware FLOP coverage;
10. strict full and frozen transfers load/freeze/train exactly the declared
    components while keeping the target adapter fresh;
11. all ten canonical jobs finish or have a documented, resolved replacement;
12. canonical downstream test evaluation occurs once per target run from the
    validation-selected downstream checkpoint;
13. the W&B API-backed analysis reproduces the run/provenance/compute tables;
14. the Phase 3 experiment record contains every job ID and snapshot path; and
15. no scientific performance claim has been made from the Phase 3 smoke
    results.

After acceptance, update the roadmap's overall status and Phase 3 status,
link the completed Phase 3 experiment record, and recalibrate Phase 4 compute
assumptions from the measured full-pool canaries. Do not launch Phase 4 until
that update is reviewed and committed.

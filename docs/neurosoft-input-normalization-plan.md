# NeuroSoft train-split input-normalization implementation plan

**Status:** Proposed implementation plan (2026-08-31).  
**Scope:** NeuroSoft raw EEG/ECoG/iEEG inputs, beginning with the
`NeurosoftConvBiGRU` acoustic-stimulus experiments.  
**Motivating evidence:** [Conv--BiGRU compact capacity screen](../experiments/inbox/20260831-MS-neurosoft-conv-bigru-compact-capacity.md).

## Decision

Implement an opt-in, recording-specific, per-channel standardization transform.
For recording `r` and supported neural channel `c`, fit statistics from the
training partition only. Apply the same frozen values to every train,
validation, and test window from that recording:

```text
x_normalized[r, c, t] = (x[r, c, t] - mean_train[r, c])
                        / max(std_train[r, c], scale_floor)
```

Apply this transform in the data pipeline immediately before model tokenization,
while the signal is still in recording-native time-by-channel layout. It is
therefore independent of batch size, padding, and the Conv--BiGRU adapter.

This is the production remedy for the minipig failure. It makes the
input-dependent session-adapter term comparable to its learned bias and
handles gain differences between recordings. The diagnostic already shows
that it restores learnability; the controlled pilot rerun remains necessary to
establish validation and test performance.

## Rationale and non-decisions

The affected minipig recording has median channel standard deviation
approximately `7e-5 V`. In the Conv--BiGRU, its
`Linear(C, D, bias=True)` session adapter becomes bias-dominated, and the
following LayerNorm produces nearly identical representations for distinct
windows. Per-channel standardization fixes the cause without changing the
adapter's transfer boundary or model capacity.

| Alternative | Decision | Reason |
|---|---|---|
| Fixed multiplication such as `x * 1e4` | Diagnostic control only | It is arbitrary across recordings and does not handle channel-level gain differences. |
| Volts-to-microvolts conversion | Optional explicit unit convention | It does not by itself standardize different gains; never hide it as a magic multiplier. |
| `bias=False` session adapter | Architectural ablation only | It addresses this interaction, but needlessly restricts the model and does not condition inputs generally. |
| BatchNorm before the adapter | Do not implement | It depends on batch composition and is awkward for variable channel counts. |
| Per-window z-scoring | Do not implement | It lets each test window set its own scale and removes potentially useful trial-level amplitude variation. |
| Whole-recording statistics | Do not implement in causal protocols | It includes validation/test/future waveform data. |

The first normalized rerun must not also change filtering, rereferencing,
labels, loss balancing, model capacity, optimizer, or regularization.

## Statistical and leakage contract

### Fit population

1. Resolve the final audited train split and fraction manifest.
2. Call `dataset.get_sampling_intervals("train")` and form the union of
   included intervals for each configured recording.
3. Convert intervals to raw sample indices using the recording domain origin
   and sampling rate. Merge overlaps before computing statistics so a sample is
   never weighted by how often a window overlaps it.
4. Select EEG, ECoG, sEEG, and iEEG channels in the exact existing tokenizer
   order.
5. Compute each channel's mean and population standard deviation in chunks.
   Freeze the result before validation/test loaders are built.

Statistics use no targets, but must still not read validation/test waveforms.
For `intrasession-causal`, they are based solely on the earlier train portion
of the recording.

### Numerical policy

- Accumulate sum, squared sum, and sample count in `float64`; persist
  `float32` mean/scale arrays for runtime.
- Use population standard deviation (`ddof=0`) and a configurable positive
  `scale_floor`.
- Abort before training for non-finite values, an empty train population, or a
  channel at/below the scale floor. Include recording and channel identifiers
  in the error; never silently create `NaN`/infinite values.
- Do not clip or robust-scale in the first implementation. Median/MAD scaling
  is a separate preprocessing experiment if artifacts prove mean/std unstable.

### Session policy

The first supported mode is within-recording normalization: every recording
must possess its own train-derived statistics. This supports single-session
scratch and intrasession protocols.

For an intersession or LOSO held-out recording with no samples in its own train
partition, fitting on that recording leaks. The initial implementation must
reject that configuration. A future policy may specify an external calibration
segment or independently learned source-session normalizer, but must never be
an implicit fallback.

## Proposed implementation

### Components

| Location | Responsibility |
|---|---|
| `foundry/data/normalization.py` (new) | Immutable `RecordingChannelStats`; interval union; chunked train-stat fitting; validation and provenance hash. |
| `foundry/data/transforms/recording_standardize.py` (new) | Non-mutating `RecordingChannelStandardize` callable using frozen stats. |
| `foundry/data/transforms/__init__.py` | Export the transform. |
| `foundry/data/datamodules/base.py` | Add the fit/load lifecycle, transform insertion, and normalization provenance. |
| NeuroSoft data/experiment YAML | Explicitly declare the normalization mode and numerical parameters. |
| `foundry/models/neurosoft_conv_bigru.py` | No normalizer or new learned parameters; only share a neutral modality/channel-selection helper if needed. |
| `tests/` | Unit, integration, leakage, and checkpoint-provenance regression tests. |

Do not extend `RescaleSignal`: it mutates arrays in place and its arbitrary
default factor is not suitable for fitted standardization.

### Transform contract

The new transform should conceptually be:

```python
RecordingChannelStandardize(
    stats_by_recording: Mapping[str, RecordingChannelStats],
    supported_modalities={"eeg", "ecog", "seeg", "ieeg"},
    scale_floor: float,
)
```

For each fetched `Data` window it must:

1. select the same first-present neural signal field and supported channels as
   `NeurosoftConvBiGRU.tokenize`;
2. look up frozen statistics by canonical `data.session.id`;
3. require the exact expected channel count and order; and
4. produce a normalized window without mutating the cached raw recording,
   unsupported channels, time domain, sampling rate, or metadata.

Extract shared signal-field/channel-selection logic if necessary. The model and
normalizer must not maintain subtly different definitions of supported inputs.

### DataModule lifecycle

The current DataModule appends the tokenizer before instantiating the dataset.
When normalization is enabled, use this lifecycle:

```text
instantiate/access raw dataset
  -> resolve and validate train split / fraction manifest / audit
  -> fit or load frozen train-only statistics
  -> create RecordingChannelStandardize(stats)
  -> compose: required transforms -> user transforms -> standardizer -> tokenizer
  -> construct loaders
```

Dataset-required transforms remain first; the tokenizer remains last. The
disabled path must preserve current behavior.

Fit once per DataModule setup, never per sampled window, worker, or rank. In
distributed jobs, rank zero writes the immutable run-local stats artifact and
hash; synchronize, then have all ranks load exactly that artifact.

### Configuration and provenance

Add a disabled-by-default data configuration:

```yaml
data:
  input_normalization:
    mode: disabled  # or recording_train_channel_zscore
    supported_modalities: [eeg, ecog, seeg, ieeg]
    scale_floor: 1.0e-8
    accumulator_dtype: float64
```

The normalized Conv--BiGRU follow-up selects
`recording_train_channel_zscore`. Means and scales are runtime artifacts, not
committed YAML literals.

At run start, save `input_normalization_stats.npz` and a JSON manifest
containing:

- mode and numerical settings;
- recording IDs, signal fields, selected channel names/order, sample rates,
  sample counts, means, scales, and floored-channel flags;
- train-interval hash, fraction-manifest hash when present, Phase-0 audit
  artifact hash, source Git SHA, and SHA-256 of the stats artifact.

Log both artifacts to W&B and record their path/hash in checkpoint metadata.
Test-only evaluation must load the checkpoint's exact stats artifact, never
refit from a potentially changed dataset split.

## Required tests

### Unit coverage

- Fit synthetic two-channel signals over disjoint and overlapping intervals;
  verify unique-sample weighting.
- Verify near-zero mean and unit population standard deviation over fitting
  samples, including channel-specific means/scales.
- Verify supported-modality selection and unchanged unsupported
  channels/metadata.
- Verify non-mutation and idempotence of raw source data across repeated
  fetches.
- Reject unknown recording IDs, channel reorder/count changes, non-finite
  values, missing train support, and near-zero scales.
- Reject missing, corrupt, or hash-mismatched stats artifacts.

### Integration coverage

- Build a synthetic train/valid/test split with distinct held-out amplitudes
  and prove only train samples affect fitted stats.
- Verify transform order: required transforms, standardizer, tokenizer.
- Verify padded channels and time stay zero after collation and never enter the
  session adapter.
- Verify disabled normalization gives the existing tokenized inputs.
- Verify normalized minipig-like inputs have non-identical Conv--BiGRU
  embeddings, while a raw tiny-scale control reproduces the known collapse.
- Verify checkpoint/test-only restore reloads recorded statistics rather than
  fitting anew.

## Staged validation

1. **Data audit:** Fit the representative minipig and monkey recordings without
   training. Confirm train-only fitting and approximately zero-mean/unit-scale
   normalized train channels; validation/test use the frozen train values.
2. **Learnability gate:** Repeat the 16-example overfit test using the
   production transform. Require finite loss, near-100% accuracy,
   non-identical embeddings, and adapter-weight gradients no longer orders of
   magnitude below bias gradients.
3. **Controlled pilot:** Rerun the original full Conv--BiGRU minipig pilot
   with only `recording_train_channel_zscore` added. Keep the audited
   recording, causal split, seeds 42/43/44, unweighted loss, optimizer,
   regularization, precision, early stopping, and checkpoint policy fixed.
   Run the matched monkey control as a regression check.
4. **Report:** Create a new linked experiment record; do not rewrite the
   abandoned capacity screen. Record the stats-manifest hash for every run and
   compare raw versus normalized results with the same splits and seeds.
5. **Capacity follow-up:** Only after both architectures use this same
   normalization protocol, compare compact and full Conv--BiGRU capacity.

Predeclare pilot gates: finite metrics, at least two predicted validation
classes, minipig train supported F1 above the raw collapse regime, and
three-seed mean minipig test supported F1 above the matched raw full-model
pilot.

## Delivery and launch checklist

1. Implement fitter, transform, tests, and DataModule/config/provenance
   support.
2. Run the relevant unit/integration suite and the no-training data audit.
3. Commit the implementation and verify `git status --short` is empty.
4. Create the follow-up experiment record.
5. Set
   `FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches` and
   submit using the normal `python main.py ... -m` workflow. Use Slurm's
   `long` partition unless explicitly changed.
6. Record the returned Slurm job ID and immutable snapshot-bundle path in the
   experiment record.

## Acceptance criteria

The feature is ready for scientific use only when train-only immutability and
checkpoint provenance are tested; raw disabled behavior is preserved; padding
and channel contracts remain correct; the learnability gate succeeds; and the
controlled three-seed rerun is reported against the raw baseline.

# Hierarchical EEG representation: agreed design and validation handoff

**Status:** No-training validation passed; no training runs yet  
**Date:** 2026-08-24  
**Scope:** First architecture family for a scalable, signal-first EEG content representation. This document records decisions made before implementation so future work can distinguish agreed requirements from deferred research questions.

## Why this work exists

The flat POYO/Perceiver-IO structure is not viable across the three NeuralBench target geometries:

| Target | Channels | Duration | Failure exposed by the flat model |
|---|---:|---:|---|
| P300 | 16 | 1 s | Short-sequence regime is feasible, but transfer/generalization remains difficult. |
| Motor imagery | 64 | 4 s | Independent per-channel tokenization produces 25,600 input tokens at 100 Hz and a 40:1 input-to-latent compression ratio. |
| Sleep staging | 2 | 30 s | A `0.1 s × 16` latent schedule creates 4,800 latents and impractical quadratic self-attention cost. |

The objective is **not** to tune a separate best model for P300, MI, and Sleep. They are common targets for testing whether one representation architecture scales across different channel counts and durations. The long-term goal remains a pretrained neurofoundation model that accepts signal, physical time, and masks with minimal required metadata.

Relevant internal evidence is recorded in:

- [Research roadmap](../experiments/ROADMAP.md)
- [NeuralBench POYO baseline synthesis](../experiments/04-neuralbench-from-scratch-baselines/README.md)
- [MI performance-gap analysis](neuralbench-poyo-mi-performance-gap.md)
- [Sleep profiling analysis](neuralbench-poyo-sleep-profiling.md)

## Design principles

1. **Expose one reusable temporal content stream.** Channel-local states are important internal scaffolding but are not part of the first public API.
2. **Fuse channels once, carefully, after local temporal processing.** The expensive representation must scale over time after this point, not over channels × time.
3. **Use physical time, not sample counts.** Sampling rate/timestamps define resampling, output timestamps, receptive fields, and masks.
4. **Use local, growing-context computation.** No full-sequence self-attention and no fixed number of latents for a whole recording.
5. **Make every aggregation explicit and testable.** Do not average channels, slots, or long epochs by default.
6. **Keep the first family content-only.** The roadmap's later `measurement_context` factorization is a separate hypothesis, not a hidden side effect of this model.
7. **Use one shared encoder across the targets.** Only readout class count and generic mask-aware temporal pooling may be task-specific.

## Terminology: content, memory, and measurement context

These concepts must not be conflated.

| Term | Version-one status | Meaning |
|---|---|---|
| `content[t]` | Implement | Fused, time-resolved, normalized-signal representation intended for downstream transfer. |
| Channel-local states | Internal only | Shared local temporal features before channel aggregation. |
| Spatial and temporal slots | Internal only | Fixed-number learned factors used to avoid premature pooling. |
| Long-range content memory | Deferred | A bounded summary carried across arbitrary-length/chunked recordings. It is not required for 1–30 s target windows. |
| `measurement_context` | Deferred | A distinct roadmap pathway for pre-normalization scale, offset, quality, montage/recording information, and optional metadata. It must not be mixed silently into content. |
| `coverage` | Implement | Explicit accounting of observed samples, time intervals, and channels. |

There is intentionally no public `context` output in version one. A later measurement-context stream needs its own controlled routing, bandwidth limits, and nuisance-information probes.

## Agreed reference architecture

### Input contract

```python
representation = model.encode(
    signal,              # [batch, channels, samples]
    sampling_rate=...,   # or explicit timestamps
    channel_mask=None,
    sample_mask=None,
)
```

Version one is strictly montage-agnostic:

- no required channel names, coordinates, device, subject, session, or task;
- a channel permutation must leave fused output unchanged (with masks permuted consistently);
- only signal, physical time, and masks are required;
- the content path receives normalized signal only.

### Reference capacity

| Component | Decision |
|---|---|
| Content width | 256 |
| Canonical input/fine rate | 128 Hz, band-limited resampling |
| Mid/coarse rates | 32 Hz / 8 Hz (4× downsampling at each transition) |
| Spatial slots per fine bin | 8 |
| Temporal slots per downsampling transition | 4 |
| Local attention window | 32 tokens at every level |
| Local attention blocks | 2 per level |
| Parameters across levels | Separate fine, mid, and coarse weights |
| Top-down fusion | Aligned gated residual |

The reference model is scoped honestly to the relevant scalp-EEG band for the three targets. It is not yet a claim to preserve high-frequency iEEG/ECoG content. A broader-band variant is future work.

### Encoder topology

```text
normalized signal + sampling rate + masks
  │
  ├─ band-limited resampling to a 128 Hz physical-time grid
  ├─ shared per-channel local convolutional encoder
  │    (the same weights for every channel)
  │
  ├─ fine-scale, signal-conditioned spatial-slot mixer
  │    8 learned slots read the set of channel-local features at each fine bin
  │    slots are concatenated, then gated/projected to one 256-d fine token
  │
  ├─ fine stream: 128 Hz ─ local-window attention ───────────────┐
  │          │                                                    │
  │          └─ local anti-aliased reduction + 4 temporal slots  │
  ├─ mid stream:  32 Hz ─ local-window attention ────────────┐   │
  │          │                                                │   │
  │          └─ local anti-aliased reduction + 4 temporal slots
  ├─ coarse stream: 8 Hz ─ local-window attention             │   │
  │                                                            │   │
  └─ aligned gated residual coarse → mid, then mid → fine ────┘───┘
                                                               │
                                                     content[t], timestamps,
                                                     coverage
```

#### Shared local channel encoder

The first front end is intentionally simple: band-limited resampling followed by a shared learned convolutional encoder. It does not include a CWT branch, channel-ID embedding, coordinates, or task-specific token rate. This prevents the initial result from being confounded by another tokenizer comparison.

Each causal convolution is followed by LayerNorm across features independently at each time step. Temporal GroupNorm is deliberately excluded because it would let masked padding and arbitrarily distant samples change every valid token, violating the bounded-locality contract.

#### Spatial-slot aggregation

At each fine time bin, channel-local features form an unordered masked set. Eight learned queries read this set and form eight spatial factors. The factors are concatenated and mapped through a gated projection to the single fused fine token.

This must be permutation-invariant in the fused output. It is deliberately more expressive than mean pooling while avoiding full channel-by-channel self-attention at every timestep. The slots are internal factors, not assumed to correspond to anatomical regions.

#### Temporal hierarchy and aggregation safeguards

Each rate reduction is local. Before decimation, a bounded depthwise/local encoder provides anti-aliasing behavior; a fixed set of four temporal slots then reads a short, overlapping local neighborhood. Their concatenation and a gated projection form the next-level token.

This is not whole-window pooling. The number of tokens grows with recording duration at every level. The fine path remains available after coarse features are created.

The 32-token local-attention window has different physical spans by design:

| Level | Rate | Window span |
|---|---:|---:|
| Fine | 128 Hz | 250 ms |
| Mid | 32 Hz | 1 s |
| Coarse | 8 Hz | 4 s |

Multiple local blocks increase effective receptive field further. This achieves longer physical context at coarser levels with linear-in-duration attention cost.

#### Top-down fusion

Coarse tokens are aligned only to finer timestamps within their receptive-field coverage, projected, and added through learned gates. A residual fine/mid path is always preserved. Do not use indiscriminate broadcast addition, full fine-to-coarse cross-attention, or coarse replacement of local features in version one.

The implementation finds overlapping intervals with ordered boundary searches and gathers only the bounded candidate set. It does not construct a dense fine-by-coarse alignment matrix, so alignment memory grows linearly with duration for this fixed architecture.

### Public output contract

The first external representation should be:

```python
representation.content             # [batch, fine_time, 256]
representation.content_timestamps  # physical timestamps
representation.coverage            # sample/channel/time observation metadata
```

Task heads use only `content` and mask-aware temporal pooling. They must not reach into channel-local features, spatial slots, or a selected hidden scale.

### Implemented coverage and mask semantics

- Input sample intervals are derived from physical timestamp midpoints. The causal channel stack expands each fine token's left boundary by its actual convolutional dependency chain.
- Every local-attention block unions the receptive fields of exactly the keys visible in its bounded window. Each temporal reduction then unions the fixed low-pass support and eight-token slot neighborhood. Top-down fusion unions only valid overlapping coarse intervals.
- Receptive-field intervals are structural dependency bounds clipped to real input samples. A masked hole inside an interval does not contribute signal, while the validity masks separately report whether the required observed support is present.
- During rate conversion, invalid source samples are conservatively mapped into output bins before the filter-support dilation is applied. Isolated invalid samples therefore cannot disappear during downsampling.
- Uniform explicit timestamps within the near-canonical tolerance are retained exactly when resampling is skipped; they are not rewritten onto a nominal 128 Hz clock.

## What is deliberately excluded from version one

- Full-sequence Perceiver/Transformer self-attention.
- A fixed global latent count for an arbitrary-duration recording.
- Long-range recurrent or bounded content memory.
- CWT/wavelet branches and a simultaneous tokenizer sweep.
- Channel names, coordinates, session/subject IDs, and required acquisition metadata.
- A pre-normalization measurement-context path.
- Pretraining objectives or checkpoint transfer.
- Task-specific temporal ladders, slot counts, or internal-scale readouts.

These are potential follow-on questions, not omissions to fill opportunistically while implementing the reference model.

## Validation phase: no training runs required

Implement this phase first using deterministic synthetic/fake data and unit or integration tests. It establishes that the model contract and claimed scaling properties are true before any supervised experiment is interpreted.

### A. Shape, timestamp, and coverage tests

Use fake batches with 1 s, 4 s, and 30 s durations; channel counts 2, 16, and 64; mixed valid lengths; padded channels; missing sample regions; and at least two input sampling rates.

Assert:

1. Fine/mid/coarse output lengths follow the documented 128/32/8 Hz rates.
2. Fine `content` timestamps are monotonically increasing and correctly tied to input physical time.
3. Token receptive-field/coverage metadata agrees with masks and padding.
4. Padded channels and masked samples cannot contribute nonzero information.
5. Output shape is independent of the original channel count except through valid coverage, and token count scales with duration rather than channel count after spatial fusion.

### B. Permutation and masking contract tests

For a fixed fake example:

1. Permute channels and permute `channel_mask` identically; verify fused `content`, timestamps, and coverage are unchanged within numerical tolerance.
2. Add masked/padded channels containing arbitrary large values; verify valid output is unchanged.
3. Remove a real channel through the mask; verify output remains finite and coverage reports the reduction.
4. Verify batch order and padding order do not affect per-example output.

Do not require channel-local intermediate states to be invariant; they should be equivariant to the input permutation. Only the fused public output has the invariance contract.

### C. Physical-time and resampling tests

Generate analytic signals (sine waves, impulses, chirps, and combinations) at multiple sampling rates representing the same continuous-time signal.

Assert:

1. After resampling, output timestamps and level rates agree.
2. Corresponding valid-time `content` values are close under a defined tolerance; document boundary differences caused by finite filters.
3. An impulse affects only the documented bounded receptive-field interval at each level.
4. Frequencies above the version-one canonical-band limit do not alias into low-frequency coarse tokens beyond the accepted filter tolerance.

### D. Aggregation anti-collapse tests

These tests validate mechanisms, not learned semantic quality:

1. Construct distinct channel patterns and verify spatial slots/gated projection are shape-correct, finite, and sensitive to each unmasked input channel under a gradient/Jacobian sanity check.
2. Construct temporally distinct local patterns with identical means; verify the temporal-slot path is not algebraically equivalent to mean pooling.
3. Verify the fine residual path remains nonzero when top-down gates are set to zero or are initialized near zero.
4. Verify top-down alignment never reads coarse tokens outside the fine token's documented local coverage.

These are not claims that slots have learned useful factors. Semantic value is tested only in later controlled training experiments.

### E. Complexity and memory profiling

Run forward/backward microbenchmarks on synthetic inputs with durations that increase geometrically and with channel counts 2/16/64.

Record:

- elapsed time and peak allocated GPU memory;
- token counts at all levels;
- behavior as duration grows at fixed channels;
- behavior as channel count grows at fixed duration;
- comparison with the existing flat POYO token/latent accounting.

Acceptance criterion: no quadratic duration curve attributable to a full-sequence attention matrix; after spatial fusion, temporal token count and dominant temporal cost must be independent of channel count.

### F. Test organization and documentation

Add focused tests near the model modules rather than one opaque end-to-end test. Keep fixtures deterministic and CPU-compatible where possible. GPU profiling may be a separate script/test target because timing assertions should not be part of normal unit tests.

Before any training launch, document:

- exact rate/receptive-field formulae;
- mask semantics and numerical tolerances;
- parameter count and FLOP/token accounting;
- known limitations of the 128 Hz scalp-EEG scope.

## Validation evidence (2026-08-24)

The no-training validation phase passes. The focused CPU suite is
`tests/test_models/test_hero.py` (31 deterministic tests), and the standalone
GPU target is `scripts/profile_hero.py`. The measured JSON Lines output is
preserved in
[`hierarchical-eeg-validation-profile.jsonl`](hierarchical-eeg-validation-profile.jsonl).

| Gate | Evidence | Result |
|---|---|---|
| A. Shape, timestamp, coverage | 1/4/30 s and 2/16/64-channel geometry; mixed-length batch; channel and time padding; fine/mid/coarse timestamp hooks | Pass |
| B. Permutation and masking | Full public representation comparison after channel permutation; arbitrary `1e6` masked values; channel removal; batch reversal | Pass (`atol=rtol=2e-5` for content) |
| C. Physical time and resampling | Sine, chirp, combination, and impulse cases at 64/128/256 Hz; per-level impulse locality; one- and two-stage filter attenuation | Pass |
| D. Aggregation anti-collapse | Per-channel Jacobian support; equal-mean temporal patterns; zero top-down gate; bounded overlap alignment | Pass |
| E. Complexity and memory | Reference-width synthetic CUDA forward/backward sweeps on a Quadro RTX 8000 | Pass; duration timing/memory slopes 0.56/0.84, versus 2 for quadratic scaling |
| F. Organization and documentation | Focused module tests, separate non-asserting GPU profiler, formulae and limitations below | Pass |

The validation work exposed and fixed one material issue: the Kaiser-sinc
low-pass implementation had applied the `2 * cutoff` factor only at its center
tap. The corrected formula is `h[n] = 2 fc sinc(2 fc (n - M))`, windowed and
renormalized. After the fix, a 48 Hz signal carried through both fixed
anti-alias/decimation stages has less than 1% of the RMS of a 2 Hz signal in
the valid interior.

### Exact rate, timestamp, and receptive-field formulae

For a canonical-grid input of length `T`:

- `T_fine = T`, `T_mid = floor(T_fine / 4)`, and
  `T_coarse = floor(T_mid / 4)`. Partial reduction tails are dropped.
- Fine timestamps are `domain_start + (i + 0.5) / 128` after resampling.
  Uniform explicit timestamps within 0.5 Hz of the canonical rate are retained
  exactly. Mid reduction token `j` takes fine timestamp `4j + 2`; coarse token
  `j` takes mid timestamp `4j + 2`. Their spacings are therefore exactly
  1/32 s and 1/8 s on the canonical grid.
- Input intervals use adjacent timestamp midpoints. At the edges, the nearest
  half-step is reflected. The three causal kernel-7 channel convolutions expand
  the nominal fine dependency 18 samples to the left and zero to the right.
- A 32-token local-attention block updates interval `R[j]` to the union of
  `R[k]` for `k in [j - 15, j + 16]`, clipped to existing valid tokens. This
  recurrence is applied once per block and independently at each level.
- A reduction updates output interval `R_out[j]` to the union of the input
  intervals reached by 33-tap filtering (radius 16) around the eight-token
  neighborhood `k in [4j - 2, 4j + 5]`. Away from boundaries this is the union
  over input indices `[4j - 18, 4j + 21]`; invalid filter support is excluded.
- Top-down fusion unions a fine interval only with valid coarse intervals that
  overlap it. Ordered boundary searches select this bounded candidate set; no
  dense fine-by-coarse matrix is constructed.

All receptive fields are clipped to real input sample intervals. They are
structural dependency bounds rather than a claim that every sample within the
interval was observed.

### Mask semantics and numerical tolerances

`True` means observed for every mask. Invalid channel/sample values are zeroed
before learned operations, invalid outputs are zeroed after each local block,
and downsampling validity requires at least 0.999 of the normalized absolute
33-tap support. Resampling first maps every invalid source sample
conservatively to output bins, then dilates invalidity by the 33-tap support.
`coverage.sample_support`, per-level validity, channel count/fraction, and
receptive-field intervals report these decisions separately.

The deterministic acceptance tolerances are:

- channel permutation, padding, and masking: `atol=rtol=2e-5`;
- 64-to-128 Hz sine equivalence: `atol=rtol=0.08`, excluding 24 fine tokens at
  each boundary;
- 64/256-to-128 Hz chirp and combination equivalence: `atol=rtol=0.12`,
  excluding 32 fine tokens at each boundary;
- impulse-effect detection: maximum feature delta greater than `1e-6`, with
  every affected fine/mid/coarse token required to contain the impulse time in
  its reported interval;
- fixed-filter rejection: a high-frequency RMS below 25% of a low-frequency
  reference after one stage and below 1% after two stages.

### Parameter, token, FLOP, and memory accounting

The reference configuration has **8,613,680 parameters**. For duration `d` in
seconds it creates `128d + 32d + 8d = 168d` temporal tokens. Local-attention
pair count is `sum_l B_l T_l min(32, T_l)` with two blocks `B_l=2`; temporal
slot pair count is `sum_reductions T_out * 4 * 8`. Both are linear in duration
once a level is longer than its fixed window. Channel count appears only in
the shared channel encoder and spatial fusion before the single fine stream.

The profiling script reports an analytical dense multiply-add estimate. It
counts two FLOPs per multiply-add and excludes normalization, activations,
mask/index operations, and overlap-search bookkeeping. This scope is recorded
in every output file rather than presented as hardware-counter FLOPs.

Duration sweep at 16 channels (batch size 1, FP32, median of three
forward/backward measurements after one warm-up):

| Seconds | Fine/mid/coarse tokens | Temporal GFLOPs | Elapsed (s) | Peak allocated MiB |
|---:|---:|---:|---:|---:|
| 1 | 128 / 32 / 8 | 0.711 | 0.0479 | 120.9 |
| 2 | 256 / 64 / 16 | 1.422 | 0.0511 | 172.3 |
| 4 | 512 / 128 / 32 | 2.845 | 0.0590 | 294.0 |
| 8 | 1,024 / 256 / 64 | 5.690 | 0.0600 | 543.3 |
| 16 | 2,048 / 512 / 128 | 11.380 | 0.1325 | 1,052.5 |
| 30 | 3,840 / 960 / 240 | 21.338 | 0.4031 | 1,983.5 |

At fixed 4 s, 2/16/64 channels all retain 512/128/32 temporal tokens and
2.845 temporal GFLOPs. Their measured peak allocations are 199.1/294.0/619.2
MiB; the increase is confined to the channel-local pre-fusion path. The
corresponding pre-fusion estimates are 3.23/18.35/70.18 GFLOPs.

For comparison, the flat POYO accounting uses 100 input tokens/channel/s,
`latent_step=0.1`, 16 latents/step, and depth 4. Its 4 s, 64-channel case has
25,600 input tokens, 640 latents, 16.384 million cross-attention pairs, and
1.638 million processor self-attention pairs. At 30 s and 16 channels it has
4,800 latents and 92.16 million processor self-attention pairs. HERO's
documented attention and slot pair counts remain linear in the same duration.

### Known limitations of this validation

- Synthetic checks establish mechanics and scaling, not semantic utility,
  downstream accuracy, learned slot specialization, or transfer quality.
- The canonical 128 Hz grid and successive anti-alias filters target the
  scalp-EEG bandwidth needed by the named tasks. They are not validation for
  high-frequency iEEG/ECoG content.
- HERO v1 accepts only uniform sampling and integral source rates when actual
  resampling is needed. It has no arbitrary-length recurrent memory.
- GPU timings are hardware/software-specific and intentionally have no unit
  test threshold; structural bounded-window tests and analytical accounting
  carry the complexity contract.

## Controlled supervised experiment ladder

After the validation phase passes, test from scratch under the common NeuralBench P300, MI, and Sleep contracts. Use a single shared encoder and task-specific readout only.

| Stage | Question | Main comparison |
|---|---|---|
| 1 | Do multiple spatial factors help before temporal hierarchy? | One-factor pooling vs 8 spatial slots in an otherwise flat temporal control. |
| 2 | Does hierarchy repair flat-model geometry? | Parameter-/compute-matched flat control vs reference hierarchy. |
| 3 | Do anti-collapse mechanisms matter? | Reference hierarchy vs no temporal slots; gated top-down fusion vs simple pooling/addition. |
| 4 | Is a win robust rather than a capacity accident? | Vary one axis only: slot count, width, or local window. |
| 5 | Does the selected architecture hold? | Three seeds × P300/MI/Sleep, against historical POYO and matched EEGNet. |

The indispensable causal baseline is a **flat control that reuses the new normalized local encoder and spatial-slot mixer**. It changes only the temporal organization. Existing POYO is a practical historical comparator but cannot by itself establish why a new model wins; EEGNet is the absolute task-specific reference.

For every trained condition report held-out metrics, parameter count, FLOPs, peak memory, wall-clock time, level token counts, and duration scaling. Do not start pretraining until the winning scratch architecture is both competitive and demonstrably efficient. Pretraining and later label-efficiency/nuisance probes are follow-on experiments.

## Literature motivation and limits

The chosen structure is motivated, not copied, from relevant work:

- [EEGPT](https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf) separates short-term spatial aggregation from longer-term temporal modeling, uses learned summary tokens, and evaluates MI, sleep, and ERP tasks.
- [CBraMod](https://arxiv.org/abs/2412.07236) argues that EEG spatial and temporal dependencies are heterogeneous and models them separately rather than as one undifferentiated patch-attention problem.
- [EEG-DINO](https://papers.miccai.org/miccai-2025/0278-Paper3347.html) supports multilevel EEG representations and decoupled channel/temporal structure, though its hierarchy is primarily a pretraining design.
- [CLEF](https://arxiv.org/abs/2605.10817) supports the importance of session-scale EEG modeling, but does not validate bounded memory slots; long-range memory is therefore deliberately deferred.

None of these papers proves that this exact architecture will solve Foundry's three targets. The staged controls above are required to establish that.

## Handoff checklist

Before implementation:

- [ ] Read this document, the roadmap, and the three NeuralBench issue/synthesis documents linked above.
- [ ] Preserve the agreed exclusions; do not add context streams, metadata, CWT, pretraining, or global attention as incidental implementation changes.
- [ ] Specify exact local convolution kernels, attention window boundary rules, and timestamp/coverage data structures before coding.
- [ ] Define the flat compute-/parameter-matched control alongside the hierarchy.

Before training:

- [x] Complete and pass all fake-data contract, invariance, resampling, anti-collapse, and scaling checks.
- [x] Review profiling evidence that the claimed scaling behavior holds.
- [ ] Create one hypothesis file per decision experiment following the experiment-tracking workflow.
- [ ] For any production Slurm launch, require a clean committed repository, set `FOUNDRY_SNAPSHOT_ROOT`, submit to `long`, and record job ID plus snapshot bundle path.

# POYO EEG Tokenization Optimization Report

## Scope

This report analyzes the tokenization path used by `configs/model/poyo_eeg.yaml`, with implementation details from:

- `foundry/models/poyo_eeg.py` (`POYOEEGModel.tokenize`)
- `foundry/data/transforms/patching.py` (`patch_time_series`)
- `foundry/models/embeddings.py` (`FixedChannelWindowEmbedding.pretokenize`)
- `temporaldata` `RegularTimeSeries` behavior (`timestamps` property)

The goal is to identify practical speedups and separate optimizations that belong in this repository vs `temporaldata`.

## Current Tokenization Scheme (from config)

From `configs/model/poyo_eeg.yaml`:

- `sequence_length = 1.0s`
- `patch_duration = 0.1s`
- `stride = 0.1s` (non-overlapping)
- base embedding shape uses `num_channels = 64`, `patch_samples = 25`

Derived tokenization geometry:

- `num_patches = ceil((1.0 - 0.1) / 0.1) + 1 = 10`
- per-example patch tensor size: `10 x 64 x 25 = 16,000` values

Notes:

- AJILE experiment configs override to `num_channels=94`, `patch_samples=50`, still with 10 patches per 1s window.
- Stride equals duration, so patch overlap is currently not a source of growth.

## Where Time Is Spent

Microbenchmarks were run with `uv run python` on synthetic 1s windows and current code paths. Timings are per sample.

### Baseline timings

| Operation | Base-like (250Hz, 64ch, 25 samples/patch) | AJILE-like (500Hz, 94ch, 50 samples/patch) |
|---|---:|---:|
| `RegularTimeSeries.timestamps` allocation | ~0.0036 ms | ~0.0036 ms |
| `patch_time_series` | ~0.0540 ms | ~0.0613 ms |
| `FixedChannelWindowEmbedding.pretokenize` | n/a | ~0.0142 ms (94ch exact) |
| channel tokenizer (`InfiniteVocabEmbedding.tokenizer` over all channels) | ~0.0084 ms | ~0.0084 ms |
| session tokenizer | ~0.0009 ms | ~0.0009 ms |

### Main slowdown sources

1. `patch_time_series` does repeated regularity checks (`np.diff`, `np.allclose`) on every sample window.
2. `RegularTimeSeries.timestamps` allocates a fresh dense timestamp array each call.
3. `pretokenize` always allocates and copies a padded signal buffer, even when channel count already matches target.
4. Channel IDs are converted to string and tokenized with Python list-comprehension per sample.

## Measured Improvement Opportunities

### 1) Skip repeated regularity checks for regular signals

A benchmark variant that removes per-call regularity checks (while keeping the same patch extraction behavior) showed:

- base-like case: **~61% faster** `patch_time_series`
- AJILE-like case: **~55% faster**
- overlap-heavy case (46 patches): **~39% faster**

Interpretation: checks are a large fraction of current patching cost for short windows.

### 2) Add no-copy fast path in `pretokenize`

When `num_channels_actual == num_channels` (common in AJILE overrides), a fast path that avoids zero-padding/copying showed:

- `pretokenize`: **~73.5% faster** (0.0142 ms -> 0.0038 ms)

Interpretation: current copying dominates `pretokenize` even for small tensors.

## Prioritized Optimizations (this repo)

## P0 (high impact, low risk)

- Add a `patch_time_series_regular(...)` path (or extend current API) that accepts:
  - `sampling_rate`
  - `start_time`
  - `assume_regular=True`
- In `POYOEEGModel.tokenize`, call this fast path for `RegularTimeSeries`.
- Avoid constructing full `timestamps` arrays when only start time + sampling rate are needed.

Expected impact:

- large relative speedup in patching (roughly 1.4x to 2.5x on that stage in tested cases)

## P1 (high impact, very low risk)

- In `FixedChannelWindowEmbedding.pretokenize`, add fast path for exact channel count:
  - return `torch.from_numpy(patches_array).float()` directly
  - return `channel_tokens` directly
  - build `input_mask` as all-ones without padding copy

Expected impact:

- significant speedup for exact-channel datasets (up to ~3.7x for pretokenize stage in benchmark)

## P1 (moderate impact, low risk)

- Cache patch index matrices by `(patch_samples, stride_samples, num_patches)` to avoid rebuilding:
  - `np.arange(patch_samples)[None, :] + stride_samples * np.arange(num_patches)[:, None]`
- Use a small LRU cache because these shapes are typically stable within one run.

Expected impact:

- modest but free savings in high-throughput training loops

## P2 (moderate impact, medium risk)

- Reduce repeated string conversions:
  - avoid `astype(str)` each sample if channel IDs are already strings
  - pre-store normalized channel IDs during dataset load/hook
- If profiling confirms significance, replace list-based tokenization with vectorized/cached lookup for common channel sets.

Expected impact:

- depends on dataset/channel cardinality; usually smaller than patching gains

## P3 (situational)

- Optional tokenization cache keyed by `(session_id, window_start, window_end, channel_mask_version, model_tokenizer_signature)`.
- Most useful when windows repeat (evaluation, hyperparameter sweeps, debugging).

Expected impact:

- very high if cache hit rate is high; negative if low (extra memory and bookkeeping)

## Optimizations to Propose in `temporaldata`

1. `RegularTimeSeries` API to expose `(start_time, sampling_rate, length)` directly for downstream regular-grid ops without allocating full timestamp arrays.
2. Optional cached `timestamps` array (or lazy memoization) with safe invalidation rules.
3. Built-in regular-series patch/window method returning patch views or efficient copies from C/NumPy kernels.
4. Metadata/contract for "known regular" after slicing so downstream callers can safely skip re-validating regularity.

Why these help:

- current `timestamps` property always builds a new `np.arange(...) / sampling_rate` array, which is avoidable overhead in repeated tokenization.
- downstream libraries currently must infer regularity repeatedly.

## Additional Observations / Risks

- `poyo_eeg` base config uses fixed `patch_samples` in embedding config, while runtime patch length is computed from `patch_duration * sampling_rate`.
  - If sampling rate differs from assumed value, shape mismatch or silent inefficiency can happen.
  - Consider validating `computed_patch_samples == input_embedding.patch_samples` at startup.
- Overlap (`stride < patch_duration`) increases `num_patches` quickly and amplifies all tokenization overhead.

## Recommended Execution Plan

1. Implement P0 + P1 first (regular fast patching + no-copy pretokenize path).
2. Add benchmark test(s) in-repo:
   - one base-like case
   - one AJILE-like case
   - one overlap-heavy case
3. Re-profile dataloader worker throughput (samples/s) with and without changes.
4. If worker CPU still dominates, implement P1 cache for patch indices and reevaluate.
5. Open upstream issue/PR to `temporaldata` with concrete API proposal and benchmark evidence.

## Expected End Result

With non-overlapping 1s windows, tokenization is already lightweight in absolute terms, but there is clear avoidable overhead. The best return-on-effort is to remove repeated regularity/timestamp work and avoid unnecessary copies in pretokenization. These are straightforward changes with low algorithmic risk and immediate throughput upside.

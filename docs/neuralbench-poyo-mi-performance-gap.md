# NeuralBench POYO Motor Imagery — Performance Gap Analysis

**Date:** 2026-08-24  
**Related experiment:**
[NeuralBench POYO-EEG Tokenizer Baselines](../experiments/04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-poyo-tokenizer-baselines.md)  
**Parent EEGNet experiment:**
[NeuralBench Matched EEGNet — Three-Task Test Parity](../experiments/04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-matched-test-parity.md)

## Problem

The POYO tokenizer baseline experiment shows a large test balanced-accuracy
gap between EEGNet and POYO on Motor Imagery (Schalk2004Bci2000, 64 channels,
4.0 s epochs), while the same experimental framework produces more comparable
results on P300 and Sleep Stage. This document summarises the root causes
identified by comparing configs, model code, and the tokenizer pipeline.

## Systematic config differences (POYO vs EEGNet, all tasks)

| Setting | EEGNet Matched | POYO Baseline |
|---|---|---|
| `trainer.precision` | `32-true` (FP32) | `16-mixed` (AMP) |
| `run.compile` | not set | `default` |
| `num_workers` | 10 | 6 |

The `num_workers` difference does not affect model performance. The other two
are potentially significant and interact with the token-count characteristics
of each task (see below).

## Root cause: token count explosion on Motor Imagery

The per-channel CWT-CNN tokenizer (`target_token_rate=100`, `channel_fusion=concat`)
combined with `PerChannelStrategy` produces the following total input token
counts per sample:

| Task | Channels | Duration | Tokens / ch | Total input tokens |
|---|---|---|---|---|
| P300 | 16 | 1.0 s | 100 | 1,600 |
| **Motor Imagery** | **64** | **4.0 s** | **400** | **25,600** |
| Sleep Stage | 2 | 30.0 s | 3,000 | 6,000 |

MI produces 16× more tokens than P300 and 4× more than Sleep Stage. This
creates three compounding problems:

### 1. Extreme Perceiver compression ratio

The latent bottleneck uses the defaults from `configs/model/poyo_eeg.yaml`
(`latent_step=0.1`, `num_latents_per_step=16`):

| Task | Latent tokens | Input tokens | Compression ratio |
|---|---|---|---|
| P300 | 160 | 1,600 | 10 : 1 |
| **Motor Imagery** | **640** | **25,600** | **40 : 1** |
| Sleep Stage | 4,800 | 6,000 | 1.25 : 1 |

MI's 640 latents must compress 25,600 input tokens — a 40 : 1 ratio, far
more aggressive than either other task. The encoder cross-attention matrix is
640 × 25,600 ≈ 16.4 M elements. Spatial information from 64 independently
tokenized channels is likely lost in this bottleneck.

### 2. Mixed precision validated only on a small-token regime

The `16-mixed` precision choice follows the
[P300 profiling results](neuralbench-poyo-p300-profiling.md), which showed a
4× wall-clock speedup with no early metric degradation. However, P300 has
only 1,600 input tokens. At 25,600 tokens the FP16 attention softmax over
the key dimension is far more numerically fragile: small logits underflow to
zero, producing effectively dead attention patterns. This is a well-known
failure mode of FP16 attention at long sequence lengths.

### 3. `torch.compile` interaction

POYO uses `torch.compile(mode="default")` (applied in `main.py`), while
EEGNet runs uncompiled. Compiled kernels may fuse attention operations
differently for very large tensors, and these fusions can interact poorly
with AMP autocast boundaries. This combination was never tested at the MI
token scale.

## Secondary factors

- **POYO dropout is aggressive for this regime.** The base `poyo_eeg.yaml`
  config has `ffn_dropout=0.2`, `lin_dropout=0.4`, `atn_dropout=0.2`.
  Combined with highly diluted attention, these further reduce the effective
  training signal per step.

- **EEGNet has an architectural advantage on spatial integration.** Its
  depthwise spatial convolution (`Conv2d(F1, F1*D, (64, 1))`) directly mixes
  all 64 channels in a single learned operation — its core inductive bias for
  motor-imagery BCI. POYO tokenizes each channel independently and relies on
  cross-attention to learn spatial relationships, a much harder optimisation
  problem with 64 channels.

## Recommended next steps

1. **Isolate precision:** re-run MI POYO with `precision: 32-true` (matching
   EEGNet). If the gap closes substantially, mixed precision at 25,600 tokens
   is confirmed as the primary cause.

2. **Reduce input token count:** lower `target_token_rate` (e.g. 50 or 25 Hz)
   for MI, reducing total tokens from 25,600 to 12,800 or 6,400.

3. **Increase latent capacity:** decrease `latent_step` or increase
   `num_latents_per_step` to narrow the compression ratio towards the
   P300/Sleep regime.

None of these changes affect NeuralBench evaluation fidelity — they are
POYO-specific architectural decisions inherited from defaults tuned on other
datasets.

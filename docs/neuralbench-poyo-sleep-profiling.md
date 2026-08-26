# NeuralBench POYO sleep staging profiling

**Date:** 2026-08-24  
**Configuration:** Sleep Stage / Kemp2000Analysis, POYO `embed_dim=256`, depth 4,
dynamic channel embeddings, session embeddings disabled, CWT-CNN tokenizer,
batch size 64, six DataLoader workers, FP16-mixed, one Quadro RTX 8000.

## Problem

The POYO sleep staging runs did not complete within the 12-hour Slurm timeout.
EEGNet on the same task and dataset finishes in ~25 minutes. The target is
<6 hours on RTX 8000 nodes.

## Root cause: latent sequence length

The default `poyo_eeg.yaml` sets `latent_step=0.1` and
`num_latents_per_step=16`. These values are tuned for short windows (P300 at
1 s, MI at 4 s) but produce a catastrophically long latent sequence for 30 s
sleep epochs:

| Task | Window | Latent tokens | Self-attn cost (O(n²)) | vs P300 |
|------|--------|---------------|------------------------|---------|
| P300 | 1.0 s | 160 | 25,600 | 1× |
| MI | 4.0 s | 640 | 409,600 | 16× |
| **Sleep** | **30.0 s** | **4,800** | **23,040,000** | **900×** |

The Perceiver processor runs quadratic self-attention over all latent tokens
at each of its 4 depth layers. Cross-attention (4,800 latents × 6,000 input
tokens) adds a comparable cost.

## Measured timings (RTX 8000)

Isolated `scaled_dot_product_attention` benchmarks on this node:

| Operation | Sequence dims | Time (fwd) |
|-----------|---------------|------------|
| Self-attn (processor) | 4,800 × 4,800, 8 heads | 174 ms |
| Cross-attn (encoder) | 4,800 × 6,000, 8 heads | 222 ms |
| Self-attn (reduced, 240 tok) | 240 × 240, 8 heads | 0.6 ms |
| Self-attn (reduced, 60 tok) | 60 × 60, 8 heads | <0.1 ms |

Estimated full training-step time (fwd + bwd ≈ 3× fwd, attention only):

- Current (4,800 latents): **~2,760 ms per batch**
- Reduced (240 latents): **~674 ms per batch**
- Reduced (60 latents): **~668 ms per batch**

## Dataset size

Kemp2000Analysis produces 116,493 training samples (1,820 batches at bs=64),
38,071 validation samples, and 41,656 test samples. This is an order of
magnitude larger than P300 or MI.

## Training-time estimates

| Latent config | Latents | Epoch (attn-only) | 40 epochs | 10 ep (early stop) |
|---------------|---------|--------------------|-----------|--------------------|
| `0.1 s, 16/step` (current) | 4,800 | ~84 min | ~56 h | ~14 h |
| `0.5 s, 4/step` | 240 | ~20 min | ~13.5 h | ~3.4 h |
| `1.0 s, 2/step` | 60 | ~20 min | ~13.4 h | ~3.4 h |

These are attention-only lower bounds. Real wall time includes CWT
tokenization, FFN layers, data loading, validation, and logging, adding an
estimated 30–50 % overhead. The true runtime for the current config likely
exceeds 70 hours for 40 epochs.

## Why EEGNet finishes in ~25 minutes

EEGNet is a lightweight 2D CNN that processes the raw (batch, 1, 2, 3600)
tensor directly. Its cost scales linearly with sequence length — a few Conv2d
layers with ~10 K parameters. Each batch takes ~50–100 ms, giving ~2.3 min per
epoch. Early stopping around epoch 10–15 yields ~25 min total.

The POYO bottleneck is fundamentally **O(n²) self-attention** (with n = 4,800)
versus **O(n) convolution** (with n = 3,600 raw samples).

## Why excessive latents are unnecessary here

Sleep staging produces **one label per 30 s epoch**. The decoder only needs to
map the latent representation to a single classification output. Having 4,800
latent tokens — one every 6.25 ms — provides temporal granularity far beyond
what 5-class sleep staging requires. Even 30–60 latents (one per second) is
generous for capturing the spectral macro-features (spindles, K-complexes,
slow waves) that distinguish sleep stages.

## Recommended fix

Override `latent_step` and `num_latents_per_step` in the sleep experiment YAML:

```yaml
model:
  embed_dim: 256
  depth: 4
  channel_emb_mode: dynamic
  latent_step: 1.0
  num_latents_per_step: 2
```

This gives 60 latent tokens (one pair per second). Self-attention becomes
negligible and cross-attention shrinks ~80× (query dimension drops from 4,800
to 60). Estimated total training time drops to **1–2 hours** for 40 full
epochs — well within the 6-hour budget.

### Conservative alternative

`latent_step=0.5, num_latents_per_step=4` → 240 latents. Self-attention is
still negligible (0.6 ms vs 174 ms). Estimated training time: ~5 hours for
40 epochs.

## Additional optimizations (if margins are tight)

1. **Reduce `target_token_rate`** from 100 to 30–50 tokens/s: cuts input
   tokens from 6,000 to 1,800–3,000, reducing cross-attention proportionally.
2. **Reduce depth** from 4 to 2: halves processor self-attention layers.
3. **Increase batch size** to 128–256: the 48 GB RTX 8000 has headroom once
   latents are reduced; larger batches amortize fixed overhead.
4. **Cap `max_epochs`** at 20: with patience=5, convergence is unlikely to
   extend beyond epoch 15 for a from-scratch baseline.

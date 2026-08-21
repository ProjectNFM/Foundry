# NeuralBench POYO P300 profiling

**Date:** 2026-08-21  
**Configuration:** P300 / Korczowski2014A, POYO `embed_dim=256`, depth 4,
dynamic channel embeddings, session embeddings disabled, CWT-CNN or
ResampleCNN tokenizer, batch size 64, six DataLoader workers, FP32, one Quadro
RTX 8000.

## Scope

These are short, isolated profiling runs. They are not production baseline
runs and use the WandB group `NB_P300_POYO_TOKENIZER_PROFILE`.

## Runtime measurement

Both tokenizer conditions completed two full training-and-validation epochs
(552 training batches per epoch):

| Tokenizer | First epoch | Steady-state epoch | 40-epoch estimate |
|---|---:|---:|---:|
| CWT-CNN | ~4.7 min | ~3.2 min | ~2h10m–2h20m |
| ResampleCNN | ~5.1 min | ~3.3 min | ~2h15m–2h25m |

The GPU stayed at roughly 90–98% utilization during steady-state training,
so the dominant cost is GPU computation rather than the six-worker input
pipeline.

## CUDA trace

The 20-batch CWT profiling run used five warm-up and five traced training
batches. Its artifacts are:

- Operator table: `/network/scratch/s/sobralm/runs/NB_P300_POYO_TOKENIZER_PROFILE/profiler/fit-p300_poyo.txt`
- Chrome/TensorBoard trace: `/network/scratch/s/sobralm/runs/NB_P300_POYO_TOKENIZER_PROFILE/profiler/cn-a007.server.mila.quebec_703952.1787342646624916518.pt.trace.json`

The active trace averaged approximately 473 ms CUDA time per training step.
The important attribution is not ordinary POYO attention: the four
self-attention layers plus encoder/decoder attention account for substantially
less than the largest single operation. The trace links that operation to the
backward pass of `RelativeChannelEncoder`'s temporal pooling:

```python
torch.einsum("bcn,bcnd->bcd", weights, tokens)
```

This creates many small batched matrix multiplies for P300's `(B=64, C=16,
N=100, D=192)` tensor. Its `BmmBackward` is about 294 ms CUDA time per traced
step, whereas the aggregate rotary-attention calls are about 62 ms per step.

## Implemented optimization

Commit `ba70346` replaces that einsum with its algebraically equivalent
elementwise weighted reduction:

```python
(weights.unsqueeze(-1) * tokens).sum(dim=2)
```

The dynamic-channel test suite passes (25 tests). A GPU microbenchmark with
gradients through both operands at the exact P300 tensor shape measured:

| Pooling implementation | Forward + backward |
|---|---:|
| einsum / BmmBackward | 52.276 ms |
| elementwise multiply + sum | 2.115 ms |

This is a ~25x speedup for the isolated pooling operation. The follow-up
20-batch end-to-end profiler completed, but a full epoch re-timing is still
needed to quantify the net training-run speedup; profiler instrumentation and
startup overhead obscure it at 20 batches.

## Next steps

1. Run one uninstrumented P300 epoch with `ba70346` and compare it with the
   ~3.2 minute CWT baseline.
2. If the parity contract permits it, benchmark `run.compile=default` and
   `precision=16-mixed` separately. Both can help an RTX 8000, but they change
   the matched FP32 protocol and require a metric check.
3. Profile MI and Sleep only after selecting the production precision and
   compilation policy; their larger token counts make that choice material.

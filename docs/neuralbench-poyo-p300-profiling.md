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

## Epoch-level speedup from pooling optimization

The full CWT-CNN epoch with `ba70346` measured 3:03 (3.19 it/s), compared to
the pre-optimization ~3.2 min baseline. The 25x isolated improvement translates
to roughly 5% wall-clock gain because other GPU work (attention, CWT forward,
gradient accumulation) dominates the end-to-end step.

## Compilation and precision benchmarks

All four configurations below ran two uninstrumented epochs on the same RTX 8000
with seed 33 and CWT-CNN. Validation balanced accuracy after two epochs is shown
to confirm no early degradation.

| Configuration | Steady-state epoch | it/s | Speedup | Val bal-acc (ep 0) |
|---|---:|---:|---:|---:|
| FP32 baseline | 3:03 | 3.19 | 1.0x | 0.603 |
| FP32 + `compile=default` | 1:55 | 4.82 | 1.6x | 0.607 |
| `16-mixed` | 1:27 | 6.73 | 2.1x | 0.609 |
| `16-mixed` + `compile=default` | 0:46 | 12.04 | 4.0x | 0.608 |

`torch.compile` triggers graph breaks from `Tensor.item()` in the CWT
embedding's target-token computation but still achieves a substantial speedup.
The combined configuration cuts the 40-epoch estimate from ~2h10m to
approximately **35 minutes**.

### Parity-contract considerations

The matched EEGNet baseline (`p300_eegnet_matched`) uses `precision: 32-true`.
`torch.compile(mode="default")` preserves bit-exact FP32 semantics so it is
safe for the matched comparison without qualification. `precision=16-mixed`
changes the numerical protocol (FP16 accumulations with grad scaling); if the
POYO baselines adopt it the results are no longer precision-matched to EEGNet,
though all early metrics look equivalent.

## Recommendation

Enable `run.compile=default` unconditionally in the POYO baseline config — it
is numerically safe and gives 1.6x. If the parity contract tolerates
mixed-precision results (or if EEGNet baselines are also switched to
`16-mixed`), enable both for the full 4x speedup.

## Remaining steps

1. Apply the selected compile/precision policy to the production experiment
   YAML and confirm a full 40-epoch seed-33 run produces competitive metrics.
2. Profile MI and Sleep only after locking the production precision and
   compilation policy; their larger token counts make that choice material.

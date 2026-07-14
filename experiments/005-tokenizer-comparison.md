# Tokenizer Architecture Comparison for Masked Pretraining

**Status:** Draft
**Date started:** 2026-07-09
**Parent experiment:** None (root)
**Follow-up experiments:** [Kemp Sleep EDF — Tokenizer Baseline Comparison](../experiments/006-kemp-sleep-tokenizer-baseline.md)

## Background

The EEG tokenizer converts raw per-channel time-series into token embeddings
that feed into the Perceiver backbone. The current default tokenizer
(`per_channel_resample_cnn`) was chosen without a systematic comparison against
alternatives. Three tokenizer architectures are available with comparable
parameter counts:

1. **ResampleCNN** — a 1-D CNN that resamples the raw signal to a fixed token
   rate (100 Hz). Uses 2 conv layers with 12 filters, kernel size 9, and GELU
   activation. Channel identity is concatenated via a 64-dim embedding.

2. **CWT-CNN** — applies a Continuous Wavelet Transform (9 log-spaced
   frequencies, 0.5–30 Hz) before the CNN, giving the network explicit
   time-frequency features. Same CNN backbone (2 layers, 64 filters, kernel 9)
   on top of the CWT coefficients.

3. **PerTimestepLinear** — a simple linear projection of each individual
   timepoint (input_dim=1 → embed_dim), with no temporal context baked into the
   tokenizer. All temporal modelling is left to the Perceiver.

Comparing these on the same masked-reconstruction pretraining task will reveal
whether injecting inductive bias at the tokenizer level (resampling, wavelet
decomposition) helps or whether the Perceiver can learn equivalent
representations from raw per-timepoint features.

## Question

Which tokenizer architecture yields the lowest reconstruction loss when
pretraining with masked reconstruction on multi-session EEG data?

## Hypothesis

The CWT-CNN tokenizer will achieve the lowest reconstruction loss because the
explicit time-frequency decomposition provides richer input features to the
Perceiver. ResampleCNN will be a close second, while PerTimestepLinear will
underperform both because it pushes all temporal modelling onto the Perceiver
without any local signal processing.

## Experiment

### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, TemporalBlockMasking (block_size=10, mask_ratio=0.5)
- **Data:** klinzing_sleep_ds005555 via OpenNeuroMultiBrainset, intrasession
  split, sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=100, lr=1e-4, weight_decay=0.01, max_epochs=200,
  bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining, group=PRETRAIN_TOKENIZER_SWEEP
  - `pretrain_tokenizer_per_channel_resample_cnn`: TBD
  - `pretrain_tokenizer_per_channel_cwt_cnn`: TBD
  - `pretrain_tokenizer_per_channel_per_timestep`: TBD

### Launch command

```bash
# SLURM sweep (3 tokenizers in parallel):
uv run python main.py experiment=pretraining/poyo_pretrain_tokenizer_sweep -m
```

### Key config overrides

All overrides are captured in the sweep config
`configs/experiment/pretraining/poyo_pretrain_tokenizer_sweep.yaml`. The Hydra
sweeper varies `model/tokenizer` over:

- `per_channel_resample_cnn`
- `per_channel_cwt_cnn`
- `per_channel_per_timestep`

Notable non-default settings vs the base model config:
- `model.masking` overridden from `RandomTokenMasking` to `TemporalBlockMasking`
  (block_size=10, mask_ratio=0.5)
- `model.zero_output_timestamps: false`
- `model.normalize_inputs: true`
- `module.warmup_epochs: 0`
- `data.split_type: intrasession`, `data.task_type: null`

## Results

TBD

### Summary

TBD

### Metrics

| Metric | ResampleCNN | CWT-CNN | PerTimestepLinear |
|--------|-------------|---------|-------------------|
| Best val loss | TBD | TBD | TBD |
| Final train loss | TBD | TBD | TBD |
| Epoch of best val loss | TBD | TBD | TBD |

### Analysis

TBD

**Analysis script:** `analysis/005_tokenizer_comparison.py`

```bash
uv run python analysis/005_tokenizer_comparison.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If one tokenizer clearly dominates, adopt it as the default for all future
  pretraining and investigate why (parameter efficiency, inductive bias, etc.).
- If CWT-CNN wins, experiment with CWT hyperparameters (number of frequencies,
  frequency range, n_cycles) as a follow-up.
- If PerTimestepLinear is competitive despite its simplicity, this suggests the
  Perceiver's cross-attention is sufficient for temporal modelling and tokenizer
  complexity may be unnecessary overhead.
- Consider extending the comparison to spatial tokenizer strategies (e.g.,
  `spatial_session_*`) once the best per-channel temporal embedding is
  identified.
- Note: the sweep config references `per_channel_per_timestep` but the closest
  existing tokenizer config file is `per_channel_per_timepoint_linear.yaml` —
  verify the Hydra name resolves correctly before launch.

# Information Leak Fixes: Channel Encoder Masking + Signal Zeroing + Tokenizer Comparison

**Status:** In Progress
**Date started:** 2026-08-12
**Parent experiment:** [Masking Parameter Sweep](20260811-MS-masking-parameter-sweep.md)
**Follow-up experiments:** [Masking Parameter Sweep](20260811-MS-masking-parameter-sweep.md) (on hold), [Multi-Length Pretraining](20260811-MS-multi-length-pretraining.md) (on hold) — both paused pending these results
**Tags:** pretraining, mae, masked, dynamic_ch, channel_encoder, information_leak, bugfix, ablation, signal_zeroing, tokenizer_comparison, cwt_cnn, resample_cnn

## Background

The [masking parameter sweep](20260811-MS-masking-parameter-sweep.md) tests how
mask ratio and block size affect downstream transfer, using the standard B2 data
configuration with CWT-CNN + dynamic channel embeddings. All prior pretraining
runs with `channel_emb_mode="dynamic"` — including
[experiment 018](../_legacy/018-dynamic-channel-embeddings.md) (which showed a 72%
reconstruction loss reduction vs disabled) and the
[data scaling group](../02-data-scaling/README.md) — contained **two sources of
information leakage** that made reconstruction artificially easy:

### Leak 1: Channel encoder pooling over masked tokens

When `MaskedPOYOEEGModel` computes channel embeddings via the
`RelativeChannelEncoder`, the encoder's temporal pooling stage attends over ALL
token embeddings — including those that will be masked and used as reconstruction
targets. This means:

1. The channel embeddings encode information about the signal at masked positions.
2. Reconstruction queries use these leaked channel embeddings to reconstruct
   masked tokens, making the task artificially easier.
3. The encoder backbone doesn't need to learn as rich representations because
   the decoder already has partial answers via the channel embeddings.

**Fix:** Thread a `token_mask` through the pipeline so the
`RelativeChannelEncoder` only pools over visible tokens when computing channel
embeddings (`disable_channel_encoder_token_mask=false`, the default post-fix).

### Leak 2: Temporal embedding receptive field

Both CWT-CNN and ResampleCNN temporal embeddings apply convolutions/wavelets
over the raw signal before masking splits tokens into visible/masked sets. The
receptive field of these operations at visible token positions extends into
adjacent masked token positions, encoding information about masked signal into
visible token embeddings. This is a subtler leak than Leak 1 but potentially
significant, especially with large kernel sizes or wide wavelet supports.

**Fix:** Zero the raw signal at masked time positions *before* the temporal
embedding (`zero_masked_signal=true`, the default). The mask is upsampled from
token resolution to raw sample resolution so the CWT/CNN sees zeros where masked
tokens will be, preventing any receptive-field bleed.

All previous reconstruction loss numbers for dynamic channel embedding models
(e.g. 0.11 in exp 018) benefited from both leaks. Post-fix, the reconstruction
task becomes strictly harder — the decoder loses both its unfair channel embedding
signal and the receptive-field bleed. The key question is whether this harder task
produces better representations.

### Tokenizer comparison on B2 data

The original [tokenizer sweep](../_legacy/003-tokenizer-comparison.md) compared
ResampleCNN and CWTCNN only on the smaller sleep brainset (1 dataset). The data
scaling experiments used CWT-CNN exclusively. With both leak fixes in place, we
now have a clean setup to compare these tokenizers at the B2 data scale
(3 datasets, 37k ch·h).

## Question

1. Does fixing both information leaks — channel encoder masking and signal
   zeroing — change pretraining dynamics and improve downstream transfer quality?
2. Does signal zeroing provide additional benefit beyond the channel encoder fix?
3. With all leaks fixed, does ResampleCNN match or exceed CWTCNN at the B2 data
   scale?

## Hypothesis

1. **Reconstruction loss will increase** (post-fix val loss > pre-fix val loss)
   because the decoder can no longer exploit either leak. Expected: 20–50%
   relative increase from the channel encoder fix, plus an additional 5–15%
   from signal zeroing.

2. **Downstream transfer will improve** because the encoder backbone must
   compensate for both removed shortcuts by learning more informative
   representations. Signal zeroing should provide incremental improvement
   beyond the channel encoder fix alone.

3. **Training will be slower to converge** but ultimately reach a better
   representation quality, as the model optimizes a genuinely harder objective.

4. **ResampleCNN and CWTCNN will be closer in performance** at B2 scale than
   in the original small-scale tokenizer sweep. With proper leak fixes, the
   tokenizer choice may matter less than data scale and masking strategy.

## Experiment

### Setup

- **Model:** POYO masked pretraining + dynamic channel embeddings, session_emb disabled
- **Data:** B2 = Klinzing + Shirazi + Pavlov (`three_dataset_pretrain.yaml`)
- **Task:** MAE pretraining (masked reconstruction), TemporalBlockMasking
  (mask_ratio=0.5, block_size=10)
- **Training:** 400k max steps, batch_size=64, lr=1e-4, warmup 2k + cosine
  decay, bf16-mixed, intersubject validation, early stopping patience=10
- **WandB:** `foundry_pretraining`, group `CHANNEL_LEAK_FIX`

### Pretraining runs

| Run | Ch encoder fix | Signal zeroing | Tokenizer | Notes |
|-----|:-:|:-:|:-:|-------|
| pretrain_leak_baseline | No | No | CWT-CNN | Pre-fix behavior (already running) |
| pretrain_leak_fixed | Yes | No | CWT-CNN | Channel encoder fix only (already running) |
| pretrain_all_fixed_cwt | Yes | Yes | CWT-CNN | Both fixes, CWT tokenizer |
| pretrain_all_fixed_resample | Yes | Yes | ResampleCNN | Both fixes, ResampleCNN tokenizer |

All runs use identical config, data, seeds, and compute budget. The only
differences are the leak fix flags and (for the last run) the tokenizer.

### Launch commands — Pretraining

```bash
# leak-baseline: Both leaks present (pre-fix behavior) — ALREADY RUNNING
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  +model.disable_channel_encoder_token_mask=true \
  +model.zero_masked_signal=false \
  run.name=pretrain_leak_baseline run.group=CHANNEL_LEAK_FIX -m

# ch-fix-only: Channel encoder fix, no signal zeroing — ALREADY RUNNING
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  +model.zero_masked_signal=false \
  run.name=pretrain_leak_fixed run.group=CHANNEL_LEAK_FIX -m

# all-fixed-cwt: Both fixes, CWT-CNN (default post-commit behavior)
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  run.name=pretrain_all_fixed_cwt run.group=CHANNEL_LEAK_FIX -m

# all-fixed-resample: Both fixes, ResampleCNN
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  model/tokenizer=per_channel_resample_cnn \
  run.name=pretrain_all_fixed_resample run.group=CHANNEL_LEAK_FIX -m
```

### Launch commands — Downstream evaluation

```bash
for NAME in pretrain_leak_baseline pretrain_leak_fixed pretrain_all_fixed_cwt pretrain_all_fixed_resample; do
  # For ResampleCNN pretrained model, override tokenizer in downstream too
  if [ "$NAME" = "pretrain_all_fixed_resample" ]; then
    TOK_OVERRIDE="model/tokenizer=per_channel_resample_cnn"
  else
    TOK_OVERRIDE=""
  fi

  # Kemp Sleep
  uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
  uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
  # PhysioNet MI
  uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
  uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
  # Brain Invaders P300
  uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
  uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX $TOK_OVERRIDE -m
done
```

### Key config overrides

| Config | Purpose |
|--------|---------|
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Base config (intersubject val, patience=10, 400k steps) |
| `configs/data/openneuro/three_dataset_pretrain.yaml` | B2 data (3 brainsets) |
| `+model.disable_channel_encoder_token_mask=true` | Disable channel encoder fix (for baseline) |
| `+model.zero_masked_signal=false` | Disable signal zeroing (for baseline + ch-fix-only) |
| `model/tokenizer=per_channel_resample_cnn` | Switch to ResampleCNN tokenizer |

### Key comparisons

- **leak-baseline vs leak-fixed (pretraining):** Impact of channel encoder fix alone.
  Expect higher val loss (harder task) for the fixed run.
- **leak-fixed vs all-fixed-cwt (pretraining):** Incremental impact of signal
  zeroing beyond the channel encoder fix.
- **leak-baseline vs all-fixed-cwt (pretraining):** Total impact of both fixes combined.
- **all-fixed-cwt vs all-fixed-resample (pretraining):** Tokenizer comparison at
  B2 scale with clean leak-free setup.
- **All 4 runs (downstream):** F1 on 3 tasks × 2 modes (finetune + linear probe).
  Expect the fully-fixed models to produce better linear probes (representation
  quality) and comparable or better finetuning.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

- If the fix significantly improves downstream transfer, all prior dynamic
  channel embedding results (exp 018, 020, 021, data scaling) should be
  considered as lower bounds on the method's true potential.
- The magnitude of the leak's impact may interact with mask_ratio: at higher
  ratios (M1–M3 from the masking sweep), the leak provides more information
  (more masked tokens → more signal leaking into channel embeddings). Re-running
  the full masking sweep post-fix could reveal a different optimal mask ratio.
- Consider whether the fix changes the relative value of dynamic vs disabled
  channel embeddings. If the fix narrows the gap (because part of the previous
  advantage was the leak), this would need to be accounted for in future
  architecture comparisons.
- If ResampleCNN matches CWTCNN at B2 scale, it is the preferred tokenizer due
  to lower computational cost (no wavelet transform).
- Signal zeroing interaction with block_size: larger blocks zero out larger
  contiguous regions, which may reduce the amount of useful signal the tokenizer
  can extract. This could interact differently with CWT (wider receptive field)
  vs ResampleCNN (narrower receptive field).

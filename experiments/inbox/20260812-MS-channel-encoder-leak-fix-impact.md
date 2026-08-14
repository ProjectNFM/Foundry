# Information Leak Fixes: Channel Encoder Masking + Signal Zeroing + Tokenizer Comparison

**Status:** Completed
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


| Run                         | Ch encoder fix | Signal zeroing | Tokenizer   | Notes                                      |
| --------------------------- | -------------- | -------------- | ----------- | ------------------------------------------ |
| pretrain_leak_baseline      | No             | No             | CWT-CNN     | Pre-fix behavior (already running)         |
| pretrain_leak_fixed         | Yes            | No             | CWT-CNN     | Channel encoder fix only (already running) |
| pretrain_all_fixed_cwt      | Yes            | Yes            | CWT-CNN     | Both fixes, CWT tokenizer                  |
| pretrain_all_fixed_resample | Yes            | Yes            | ResampleCNN | Both fixes, ResampleCNN tokenizer          |


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


| Config                                                          | Purpose                                                 |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Base config (intersubject val, patience=10, 400k steps) |
| `configs/data/openneuro/three_dataset_pretrain.yaml`            | B2 data (3 brainsets)                                   |
| `+model.disable_channel_encoder_token_mask=true`                | Disable channel encoder fix (for baseline)              |
| `+model.zero_masked_signal=false`                               | Disable signal zeroing (for baseline + ch-fix-only)     |
| `model/tokenizer=per_channel_resample_cnn`                      | Switch to ResampleCNN tokenizer                         |




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



### Summary

Both information leaks were confirmed as real and massive shortcuts for the
reconstruction objective, but fixing them has negligible impact on downstream
transfer quality. The encoder backbone was already learning useful
representations despite the decoder exploiting the leaked information.

### Pretraining losses

The leak fixes cause dramatic pretraining loss increases, confirming the leaks
were providing enormous reconstruction shortcuts:


| Run                      | Tokenizer   | Best Val Loss | Δ vs Baseline |
| ------------------------ | ----------- | ------------- | ------------- |
| Baseline (no fixes)      | CWT-CNN     | 0.0576        | —             |
| Ch-encoder fix only      | CWT-CNN     | 0.0775        | +34.4%        |
| Both fixes (CWT-CNN)     | CWT-CNN     | 0.2838        | +392.5%       |
| Both fixes (ResampleCNN) | ResampleCNN | 0.3028        | +425.3%       |


The channel encoder masking fix alone increases loss by 34%. Adding signal
zeroing raises it by an additional 266 pp (from +34% to +393%), indicating
that the temporal embedding receptive-field bleed was the larger of the two
leaks. All pretraining runs terminated via early stopping (`state=failed`
reflects SLURM timeout, not training failure — all runs completed early
stopping).

### Downstream finetuning (mean F1 ± std, 3-fold CV)


| Run                      | Kemp Sleep    | PhysioNet MI  | Brain Invaders P300 |
| ------------------------ | ------------- | ------------- | ------------------- |
| Baseline (no fixes)      | 0.735 ± 0.006 | 0.882 ± 0.042 | 0.325 ± 0.021       |
| Ch-encoder fix only      | 0.736 ± 0.004 | 0.887 ± 0.042 | 0.334 ± 0.017       |
| Both fixes (CWT-CNN)     | 0.738 ± 0.000 | 0.888 ± 0.038 | 0.332 ± 0.006       |
| Both fixes (ResampleCNN) | 0.723 ± 0.008 | 0.882 ± 0.038 | 0.310 ± 0.023       |


CWT-CNN variants show small positive deltas (+0.001 to +0.009) across all
tasks, but these are within noise given the standard deviations.
ResampleCNN underperforms CWT-CNN on all three finetuning tasks.

### Downstream linear probe (mean F1 ± std, 3-fold CV)


| Run                      | Kemp Sleep    | PhysioNet MI  | Brain Invaders P300 |
| ------------------------ | ------------- | ------------- | ------------------- |
| Baseline (no fixes)      | 0.635 ± 0.012 | 0.674 ± 0.016 | 0.299 ± 0.014       |
| Ch-encoder fix only      | 0.635 ± 0.011 | 0.673 ± 0.018 | 0.302 ± 0.018       |
| Both fixes (CWT-CNN)     | 0.632 ± 0.010 | 0.649 ± 0.029 | 0.301 ± 0.014       |
| Both fixes (ResampleCNN) | 0.601 ± 0.015 | 0.661 ± 0.004 | 0.293 ± 0.008       |


Linear probe results are flat to slightly negative. The channel encoder fix
alone has essentially zero effect. Signal zeroing adds a small negative delta
on PhysioNet MI linear probe (−0.024), suggesting the much harder pretraining
objective may slightly hurt frozen-backbone representation quality at this
training budget.

### Analysis

Script: `analysis/037_channel_encoder_leak_fix_impact.py`

```bash
uv run python analysis/037_channel_encoder_leak_fix_impact.py
```



### Figures

Pretraining loss curvesBest pretraining val lossDownstream comparison gridAblation deltas vs baselineTokenizer comparison

## Conclusions

**Hypothesis 1 (reconstruction loss increases): CONFIRMED.** The channel encoder
fix alone raised val loss by +34%, and adding signal zeroing raised it by +393%
total. This exceeds the hypothesized 20–50% + 5–15% range — the signal zeroing
leak was far larger than expected.

**Hypothesis 2 (downstream transfer improves): PARTIALLY CONFIRMED / NEGLIGIBLE.**
Finetuning shows tiny positive deltas for CWT-CNN leak-fixed models (+0.001 to
+0.009), but these are within noise. Linear probe is flat to slightly negative.
The encoder backbone was already learning useful representations despite the
decoder exploiting the leaked shortcuts — the leaks primarily made the decoder's
job easier without degrading the encoder's learned features.

**Hypothesis 3 (slower convergence, better quality): REFUTED.** The fixed models
did not reach better representation quality despite the harder objective. The
much harder pretraining task (5x higher loss) did not translate to improved
downstream transfer within the same training budget.

**Hypothesis 4 (tokenizer convergence at B2 scale): REFUTED.** CWT-CNN
outperforms ResampleCNN across nearly all tasks and modes. The gap is
particularly large on Kemp Sleep linear probe (0.632 vs 0.601, Δ = −0.032)
and Brain Invaders P300 finetuning (0.332 vs 0.310, Δ = −0.022). CWT-CNN
remains the preferred tokenizer.

## Notes for future experiments

- **Keep both fixes as default.** They make the pretraining objective
honest and don't hurt downstream performance. All future pretraining
runs should use `disable_channel_encoder_token_mask=false` (default)
and `zero_masked_signal=true` (default).
- **Resume the masking parameter sweep with fixes applied.** The leak
magnitude interacts with mask_ratio (more masked tokens = more leaked
information). Post-fix, the optimal mask_ratio and block_size may differ
from the pre-fix sweep results. This is the natural next experiment.
- **Try longer pretraining.** The dramatically harder objective (5x higher
loss) may need more training steps to show downstream benefits. The
current 400k-step budget with early stopping patience=10 may be
insufficient for the fixed models to fully converge to better
representations. Consider increasing patience or max steps.
- Prior dynamic channel embedding results (exp 018, 020, 021, data scaling)
had inflated reconstruction losses but apparently comparable representation
quality — the leaks were decoder-side shortcuts that didn't meaningfully
change encoder learning.
- The fix changes the *interpretation* of reconstruction loss as a proxy for
representation quality. Post-fix, the loss is a more honest signal, which
may make it a better metric for comparing pretraining configurations.
- Signal zeroing interaction with block_size: larger blocks zero out larger
contiguous regions, which may reduce the amount of useful signal the tokenizer
can extract. This could interact differently with CWT (wider receptive field)
vs ResampleCNN (narrower receptive field).


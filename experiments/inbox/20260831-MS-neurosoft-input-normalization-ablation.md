# NeuroSoft Input Normalization Recovery Ablation

**Status:** Draft
**Date started:** 2026-08-31
**Parent experiment:** [Phase 2 -- Convolution--BiGRU Recipe Recovery](20260831-MS-neurosoft-conv-bigru-recipe-recovery.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, input-normalization, eegnet, convolution-bigru, recovery, ablation, intrasession-causal

## Background

The [Conv--BiGRU recipe-recovery experiment](20260831-MS-neurosoft-conv-bigru-recipe-recovery.md) showed that changing the optimizer recipe did not recover the representative minipig session: every full-sized Conv--BiGRU cell remained at the class-prior validation F1. The subsequent [compact-capacity screen](20260831-MS-neurosoft-conv-bigru-compact-capacity.md) isolated the cause as recording-scale mismatch: tiny raw minipig ECoG values allow the GRU session-adapter bias to dominate before LayerNorm, yielding indistinguishable embeddings. A train-split-only, recording-level per-channel z-score transform was added in commit `9c4d4a4` to address that mechanism.

This recovery screen tests that new data-pipeline transform in the same causal, full-data setting on one representative minipig session and one matched monkey session. It includes EEGNet as a control: its first convolution is followed by BatchNorm, so it is expected to be materially less sensitive to absolute input scale than the GRU. The comparison deliberately uses a single shared training recipe and one fixed seed; it is a diagnostic validation screen, not a multi-seed performance estimate.

## Question

On one representative causal session per species, does train-split-only per-channel input normalization improve best validation supported macro-F1 for Conv--BiGRU and EEGNet under the same training recipe?

## Hypothesis

Enabling `recording_train_channel_zscore` will lift the minipig GRU above its raw class-prior validation regime (about 0.043 supported macro-F1) by restoring signal-dependent gradients and multi-class predictions. It will not reduce validation macro-F1 for either model or species. EEGNet will show a smaller gain, if any, because its BatchNorm makes it comparatively insensitive to global recording scale. Thus, the normalization gain will be larger for the GRU than for EEGNet.

## Experiment

### Setup

- **Models:** `NeurosoftConvBiGRU` (called GRU below) and `EEGNetEncoder`, each trained from scratch.
- **Data:** One 8-band, full-data, audited `intrasession-causal` session per species: minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw` and monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`.
- **Independent variable:** `data.input_normalization.mode`: `disabled` versus `recording_train_channel_zscore`. In the enabled condition, one frozen mean and scale are fitted per supported channel from causal-training waveform samples only, then reused for train and validation.
- **Controls:** Model, normalization mode, and no other setting vary. Both architectures use batch size 16, learning rate 0.0015, weight decay 0.018, 0.5 s windows, 200 maximum epochs, patience 40, unweighted cross-entropy, seed 42, and `bf16-mixed` precision.
- **Primary metric:** Best `val/neurosoft_acoustic_stim_8band_supported_f1` (validation supported macro-F1), used for both early stopping and the final 2 x 2 comparison.
- **Scope:** Eight local runs total: 2 species x 2 models x normalization disabled/enabled. This is one session per species, not a multi-recording or multi-seed sweep.
- **Test discipline:** `run.evaluate_test=false`; this experiment answers the requested validation-only question and does not consume the held-out test set.
- **WandB:** project `neurosoft_supervised_pretraining`, group `PHASE2_INPUT_NORMALIZATION_ABLATION`.

### Launch command

Commit this experiment record and configurations, confirm the repository is clean, then run one four-cell multirun for each species locally. The local-GPU launcher creates a snapshot for every cell and, with `max_batch_size=1`, runs them sequentially.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_minipigs \
  normalization_ablation_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  data.input_normalization.mode=disabled,recording_train_channel_zscore \
  hydra/launcher=local_gpu \
  -m

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_monkeys \
  normalization_ablation_recording_id=sub-01_ses-04_task-AcousStim_acq-RH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  data.input_normalization.mode=disabled,recording_train_channel_zscore \
  hydra/launcher=local_gpu \
  -m
```

Record the four snapshot paths and WandB run IDs here after the launcher returns.

### Key config overrides

- New experiment configs: `configs/experiment/auditory_decoding/neurosoft_input_normalization_ablation_{minipigs,monkeys}.yaml`.
- CLI sweep: `model=eegnet,neurosoft_conv_bigru` x `data.input_normalization.mode=disabled,recording_train_channel_zscore`.
- Common recovery recipe: `learning_rate=0.0015`, `weight_decay=0.018`, `batch_size=16`, `max_epochs=200`, `patience=40`, `run.seed=42`.
- Validation-only: `run.evaluate_test=false`.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD -- run `uv run python analysis/20260831-MS-neurosoft-input-normalization-ablation_analysis.py` after the four WandB runs are available.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Treat this one-seed screen as mechanistic evidence only. If the GRU normalization cell clears the learnability gate, repeat the paired comparison across seeds 42--44 before making a performance claim.
- Preserve the generated normalization stats manifest SHA-256 for every enabled run; it establishes that validation data did not influence fitted statistics.
- If EEGNet changes materially, inspect the fitted per-channel statistics and optimization curves before attributing the effect to architecture, since BatchNorm predicts relative scale robustness.

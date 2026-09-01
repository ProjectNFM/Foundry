# NeuroSoft Input Normalization Recovery Ablation

**Status:** In Progress
**Date started:** 2026-08-31
**Parent experiment:** [Phase 2 -- Convolution--BiGRU Recipe Recovery](20260831-MS-neurosoft-conv-bigru-recipe-recovery.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, input-normalization, eegnet, convolution-bigru, recovery, ablation, intrasession-causal

## Background

The [Conv--BiGRU recipe-recovery experiment](20260831-MS-neurosoft-conv-bigru-recipe-recovery.md) showed that changing the optimizer recipe did not recover the representative minipig session: every full-sized Conv--BiGRU cell remained at the class-prior validation F1. The subsequent [compact-capacity screen](20260831-MS-neurosoft-conv-bigru-compact-capacity.md) isolated the cause as recording-scale mismatch: tiny raw minipig ECoG values allow the GRU session-adapter bias to dominate before LayerNorm, yielding indistinguishable embeddings. A train-split-only, recording-level per-channel z-score transform was added in commit `9c4d4a4` to address that mechanism.

The initial one-seed screen confirmed the GRU scale-recovery mechanism: per-channel z-scoring raised best validation supported macro-F1 from 0.0427 to 0.2538 for the minipig GRU and from 0.1413 to 0.7403 for the monkey GRU. However, it reduced the peak EEGNet metric, particularly for monkey (0.4559 to 0.2379). EEGNet's BatchNorm makes it relatively insensitive to uniform scale, but per-channel z-scoring additionally reweights and recenters individual electrodes before its spatial convolution.

Phase 2 separates global recording scale correction from per-channel equalization. It adds exactly the four missing train-split-only global-z-score cells—one for each species-model pair at seed 42—to the completed eight-cell comparison. It is validation-only and designed to determine whether amplitude rescue without channel equalization preserves EEGNet while retaining the GRU recovery.

## Question

Across representative causal sessions from both species, does train-split-only global recording z-scoring preserve EEGNet validation performance better than per-channel z-scoring while retaining the GRU recovery?

## Hypothesis

`recording_train_global_zscore` will lift the minipig GRU above its raw class-prior validation regime (about 0.043 supported macro-F1) and attain validation F1 close to the existing per-channel seed-42 condition. Because it applies one scalar mean and scale to the whole recording, it will preserve relative channel amplitudes and therefore attain EEGNet validation F1 above the existing per-channel mode, especially for monkey. If global z-scoring does not preserve EEGNet, the adverse effect is more likely due to the changed optimization/regularization regime than to loss of relative channel scale.

## Experiment

### Setup

- **Models:** `NeurosoftConvBiGRU` (called GRU below) and `EEGNetEncoder`, each trained from scratch.
- **Data:** One 8-band, full-data, audited `intrasession-causal` session per species: minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw` and monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`.
- **Independent variable:** `data.input_normalization.mode`: `disabled`, `recording_train_global_zscore`, or `recording_train_channel_zscore`. The global mode fits one frozen mean and standard deviation across every supported channel-time sample in a recording's causal-training partition, then broadcasts those scalars to every supported channel. The per-channel mode instead fits one mean and scale per supported channel. Both reuse their train-only statistics unchanged for validation.
- **Controls:** Model, normalization mode, and no other setting vary. Both architectures use batch size 16, learning rate 0.0015, weight decay 0.018, 0.5 s windows, 200 maximum epochs, patience 40, unweighted cross-entropy, seed 42, and `bf16-mixed` precision.
- **Primary metric:** Best `val/neurosoft_acoustic_stim_8band_supported_f1` (validation supported macro-F1), used for early stopping and the 3 x 2 single-seed comparison. Validation curves remain supporting evidence because a best epoch can be noisy.
- **Scope:** Phase 1 completed 8 runs: 2 species x 2 models x raw/per-channel z-score. Phase 2 adds 4 runs: 2 species x 2 models x global z-score. This remains one session per species and one seed, not a performance estimate.
- **Test discipline:** `run.evaluate_test=false`; this experiment answers the requested validation-only question and does not consume the held-out test set.
- **WandB:** project `neurosoft_supervised_pretraining`, group `PHASE2_INPUT_NORMALIZATION_ABLATION`. The completed Phase-1 runs are `1qwq1x7r`, `lx7f8e1t`, `xfo8t5yt`, `l7gek5ps`, `gvpixafp`, `fu7dnhkn`, `cra6xbbd`, and `677tch9g`; Phase 2 adds the four global-z-score runs to that comparison.

### Launch command

Commit this experiment record and configurations, confirm the repository is clean, then run one two-cell multirun for each species locally. The local-GPU launcher creates a snapshot for every cell and, with `max_batch_size=1`, runs them sequentially.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_minipigs \
  normalization_ablation_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  hydra/launcher=local_gpu \
  -m

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_monkeys \
  normalization_ablation_recording_id=sub-01_ses-04_task-AcousStim_acq-RH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  hydra/launcher=local_gpu \
  -m
```

Record the four snapshot paths and WandB run IDs here after the launchers return.

### Key config overrides

- New experiment configs: `configs/experiment/auditory_decoding/neurosoft_input_normalization_ablation_{minipigs,monkeys}.yaml`.
- Added input-normalization mode: `recording_train_global_zscore`, implemented with one train-only recording-wide mean and standard deviation broadcast across supported channels.
- The Phase-2 config fixes `data.input_normalization.mode=recording_train_global_zscore`; the CLI sweeps only `model=eegnet,neurosoft_conv_bigru` at the existing seed 42.
- Common recovery recipe: `learning_rate=0.0015`, `weight_decay=0.018`, `batch_size=16`, `max_epochs=200`, `patience=40`, `run.seed=42`.
- Validation-only: `run.evaluate_test=false`.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

The completed seed-42 diagnostic is reproducible with:

```bash
uv run python analysis/20260831-MS-neurosoft-input-normalization-ablation_analysis.py
```

It dynamically collects the original WandB group, so rerun it after the four Phase-2 cells finish to produce the 12-cell comparison.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Treat the resulting 12-cell screen as mechanistic, single-seed evidence only; repeat across seeds before making a performance claim.
- Preserve the generated normalization stats manifest SHA-256 for every enabled run; it establishes that validation data did not influence fitted statistics.
- Compare EEGNet's trajectories as well as peak F1; its monkey raw peak was much higher than its typical trajectory.
- If global z-scoring preserves EEGNet and rescues GRU, evaluate a later global-scale-only mode (division by train-only recording standard deviation without centering) to isolate whether centering contributes.

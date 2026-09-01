# Phase 3 -- NeuroSoft Input-Normalization Seed Replication

**Status:** Draft
**Date started:** 2026-09-01
**Parent experiment:** [NeuroSoft Input Normalization Recovery Ablation](20260831-MS-neurosoft-input-normalization-ablation.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, input-normalization, eegnet, convolution-bigru, replication, seeds, intrasession-causal

## Background

The parent [single-seed normalization ablation](20260831-MS-neurosoft-input-normalization-ablation.md) completed the 12-cell comparison at seed 42: two species, EEGNet and Conv--BiGRU (GRU), and raw, train-channel-z-score, and train-global-z-score inputs. Global z-scoring recovered the GRU in both species and improved EEGNet relative to channel-wise normalization, but the evidence is a single seed and cannot establish that these rankings are stable.

This phase fills the two missing seeds in the predeclared three-seed protocol. It repeats every seed-42 cell without changing data, split, optimizer, model capacity, precision, or early-stopping criterion. Together with the completed seed-42 cells, it will yield a balanced 2 species x 2 models x 3 normalization modes x 3 seeds validation-only comparison.

## Question

Across seeds 42, 43, and 44, do the relative effects of train-global and train-channel z-scoring on validation supported macro-F1 replicate for EEGNet and GRU in both representative causal sessions?

## Hypothesis

The three-seed mean will preserve the seed-42 ordering: both z-score modes will materially improve GRU over raw input, global z-scoring will match or exceed channel-wise z-scoring for monkey GRU, and global z-scoring will exceed channel-wise z-scoring for EEGNet while remaining below raw EEGNet. Replication failure will be indicated by a reversed mean ordering or by effects that are inconsistent across the two added seeds.

## Experiment

### Setup

- **Models:** `EEGNetEncoder` and `NeurosoftConvBiGRU` (GRU), trained from scratch.
- **Data:** The parent experiment's audited, 8-band, full-data `intrasession-causal` sessions: minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw` and monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`.
- **Independent variables:** `run.seed` in `{43, 44}` and `data.input_normalization.mode` in `{disabled, recording_train_channel_zscore, recording_train_global_zscore}`. Model and species are retained as balanced comparison factors.
- **Controls:** Parent Hydra configs, batch size 16, learning rate 0.0015, weight decay 0.018, 0.5 s windows, 200 maximum epochs, patience 40, unweighted cross-entropy, and `bf16-mixed` precision are unchanged. Normalization statistics remain fit only on each recording's causal-training partition.
- **Scope:** 24 new cells: 2 species x 2 models x 3 normalization modes x 2 missing seeds. Combined with the completed 12 seed-42 cells, this produces 36 validation-only runs.
- **Primary metric:** Per-run best `val/neurosoft_acoustic_stim_8band_supported_f1`, summarized by species, model, and normalization mode as the three-seed mean and sample standard deviation. Curves are supporting evidence only.
- **Test discipline:** `run.evaluate_test=false`; no held-out test evaluation is consumed.
- **WandB:** project `neurosoft_supervised_pretraining`, group `PHASE3_INPUT_NORMALIZATION_REPLICATION`. Seed-42 parent runs remain in `PHASE2_INPUT_NORMALIZATION_ABLATION` and are included only by the Phase-3 analysis script.

### Launch command

Commit the Phase-3 record, confirm the repository is clean, set a shared snapshot root, and launch one 12-cell multirun per species. The existing configs are reused; no Phase-3 config duplication is required.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_minipigs \
  normalization_ablation_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  data.input_normalization.mode=disabled,recording_train_channel_zscore,recording_train_global_zscore \
  run.seed=43,44 \
  run.group=PHASE3_INPUT_NORMALIZATION_REPLICATION \
  hydra/launcher=local_gpu \
  -m

python main.py \
  experiment=auditory_decoding/neurosoft_input_normalization_ablation_monkeys \
  normalization_ablation_recording_id=sub-01_ses-04_task-AcousStim_acq-RH_desc-raw \
  model=eegnet,neurosoft_conv_bigru \
  data.input_normalization.mode=disabled,recording_train_channel_zscore,recording_train_global_zscore \
  run.seed=43,44 \
  run.group=PHASE3_INPUT_NORMALIZATION_REPLICATION \
  hydra/launcher=local_gpu \
  -m
```

Record the two snapshot bundle paths and the 24 WandB run names and IDs here after the launchers return.

### Key config overrides

- Reuses `configs/experiment/auditory_decoding/neurosoft_input_normalization_ablation_{minipigs,monkeys}.yaml`.
- Sweeps the three existing input-normalization modes and only the missing seeds, `43,44`.
- Overrides `run.group=PHASE3_INPUT_NORMALIZATION_REPLICATION` to keep the replication runs distinct while preserving the existing run-name interpolation.
- Keeps `run.evaluate_test=false` and all recovery-recipe hyperparameters unchanged.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

After all 24 Phase-3 cells complete, reproduce the combined 36-run table and figure with:

```bash
uv run python analysis/20260901-MS-neurosoft-input-normalization-replication_analysis.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Treat the three-seed comparison as evidence for the two selected representative sessions, not as a population-level species estimate.
- Inspect both the mean and per-seed ordering before choosing a normalization mode for a later test-set evaluation.
- Preserve each run's normalization-statistics SHA-256 to document train-only fitting across all seeds.

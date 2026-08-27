# Phase 1 -- EEGNet Single-Session Learning Curves

**Status:** Running
**Date started:** 2026-08-26
**Parent experiment:** [NeuroSoft Supervised-Pretraining Protocol and Data Audit](20260826-neurosoft-supervised-pretraining-protocol.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, phase1, eegnet, learning-curves, 8band, intrasession-causal, supervised-pretraining

## Background

The [Phase 0 protocol audit](20260826-neurosoft-supervised-pretraining-protocol.md)
frozen eligibility rules, nested fraction manifests, and staged run counts for
NeuroSoft supervised pretraining. Phase 1 measures how much nested causal
training data each eligible single session needs before EEGNet reaches stable
supported-class performance on the fixed `intrasession-causal` evaluation
protocol.

## Question

Does increasing the nested causal training fraction improve supported-class test
macro-F1 and reduce failure to reach the fixed performance target, with
diminishing returns at larger fractions?

## Hypothesis

Increasing nested causal training data improves supported-class test macro-F1
and reduces failure to reach the fixed performance target, with diminishing
returns at larger fractions.

## Experiment

### Setup

- **Eligible recordings:** 53 (40 minipig, 13 monkey) from the Phase 0 audit.
- **Supported cells:** 255 recording/fraction cells (193 minipig, 62 monkey).
- **Jobs:** 765 = 255 cells × 3 seeds (42, 43, 44).
- **Model:** EEGNet (fixed recipe from baseline experiments).
- **Task:** `neurosoft_acoustic_stim_8band` (25 stimuli → 8 frequency bands).
- **Split:** `intrasession-causal`.
- **Fractions:** 5%, 10%, 25%, 50%, 100%.
- **Monitoring:** `val/neurosoft_acoustic_stim_8band_supported_f1`.
- **Test evaluation:** best validation checkpoint.

Hydra experiment configs:

| Species | Config |
|---------|--------|
| Minipigs | `configs/experiment/auditory_decoding/eegnet_neurosoft_8band_learning_curves_minipigs.yaml` |
| Monkeys | `configs/experiment/auditory_decoding/eegnet_neurosoft_8band_learning_curves_monkeys.yaml` |

### Launch command

Production submissions require a clean committed repository, a compute-node-visible
`FOUNDRY_SNAPSHOT_ROOT`, and the normal `python main.py ... -m` workflow on the
`long` partition.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/eegnet_neurosoft_8band_learning_curves_minipigs \
  hydra/launcher=slurm_default \
  -m

python main.py \
  experiment=auditory_decoding/eegnet_neurosoft_8band_learning_curves_monkeys \
  hydra/launcher=slurm_default \
  -m
```

### Key config overrides

See the species-specific Hydra YAML files above. Non-default choices relative to
the singlesession EEGNet baseline:

- `split_type=intrasession-causal`
- `training_fraction_task=neurosoft_acoustic_stim_8band`
- `training_fraction` swept via `phase1_cell` resolver from `docs/neurosoft-phase0-audit.json`
- `training_fraction_seed=${run.seed}`
- `run.project=neurosoft_supervised_pretraining`
- early stopping / checkpoint monitor: `val/neurosoft_acoustic_stim_8band_supported_f1`
- sweep seeds: 42, 43, 44
- validated `flops_per_window` and `flop_method` are required before production
  launch; the configs deliberately refuse to run until those profiling metadata
  are supplied for the realized model/input shape.

### Slurm submission record

| Launch | Slurm job ID | Snapshot bundle path |
|--------|--------------|----------------------|
| Minipigs (`eegnet_neurosoft_8band_learning_curves_minipigs`) | 10521500 (73 array elements, 579 tasks) | `/network/scratch/s/sobralm/foundry-launches/20260827T152641_PHASE1_EEGNET_MINIPIGS_b6c65640_6c083754` |
| Monkeys (`eegnet_neurosoft_8band_learning_curves_monkeys`) | 10521501 (24 array elements, 186 tasks) | `/network/scratch/s/sobralm/foundry-launches/20260827T152736_PHASE1_EEGNET_MONKEYS_b6c65640_6262e1ed` |

## Results

TBD

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

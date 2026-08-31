# Phase 2 -- Convolution--BiGRU Compact Capacity Screen

**Status:** Draft
**Date started:** 2026-08-31
**Parent experiment:** [Phase 2 -- Convolution--BiGRU Recipe Recovery](20260831-MS-neurosoft-conv-bigru-recipe-recovery.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, phase2, convolution-bigru, compact, capacity, scratch, intrasession-causal

## Background

The full Conv--BiGRU recipe-recovery screen retained the 510k-parameter,
2-layer, 128-hidden-unit architecture while testing lower learning rates and
lighter regularization. Its completed minipig cells remained at the class-prior
validation F1 despite zero dropout and zero weight decay. That result makes
capacity/conditioning a plausible next cause. It also mirrors the historical
NeuroSoft capacity finding that smaller models can outperform the default
large model on limited data, although that result used POYO under a different
protocol.

This is a capacity-only replay of the original pre-recipe-screen pilot, not a
second recipe search. It holds the pilot data, unweighted loss, optimizer,
precision, split, test protocol, and three full-data seeds fixed. It
deliberately preserves stride 4, so the compact model has the same 250
recurrent time steps as the full model. The only training-relevant intervention
is the compact encoder: adapter 32, temporal channels 64, one bidirectional
GRU layer, and 64 hidden units per direction.

## Question

Does reducing Conv--BiGRU width and recurrent depth restore learnability and
improve supported test macro-F1 on the representative causal minipig and
monkey sessions under the original pre-recipe-screen pilot recipe?

## Hypothesis

The compact Conv--BiGRU will avoid the minipig class-prior collapse, attain
train supported F1 materially above 0.12, and improve the three-seed mean test
macro-F1 over the matched full-model pilot (0.041 minipig; 0.134 monkey). A
successful capacity control will at least match the EEGNet references (0.135
minipig; 0.208 monkey) under this fixed recipe.

## Experiment

### Setup

- **Model:** `NeurosoftConvBiGRU` compact configuration: 32-dimensional
  session adapter, one 64-channel separable temporal block, 1-layer
  bidirectional GRU with 64 hidden units per direction, and the unchanged
  8-logit router. The raw window length, temporal kernel (64), stride (4),
  pooling, transfer boundary, and target-only adapter policy are unchanged.
- **Capacity comparison:** full pilot architecture = adapter 64, temporal
  channels 128, 2-layer 128-hidden BiGRU; compact architecture = 32, 64,
  1-layer 64-hidden. This is expected to reduce the model from approximately
  511k parameters to tens of thousands without changing the input protocol.
- **Data and task:** Full-data, `intrasession-causal` 8-band acoustic-stimulus
  classification on the same audited sessions as the recovery screen:
  minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw` (18 supported model
  channels) and monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw` (29).
- **Training control:** Exactly the original pilot recipe: full data, batch
  size 16, learning rate 0.0015, weight decay 0.018, model dropout 0.3,
  maximum 200 epochs, patience 40, and gradient clipping 1.0. Run seeds 42,
  43, and 44 for each species.
- **Loss:** Unweighted cross-entropy (`class_weights.mode=none`), matching the
  original pilot and Phase-1 EEGNet training; no balancing is introduced.
- **Precision:** `bf16-mixed`, matching the successfully completed pilot.
- **FLOPs:** The compact model must receive a new species-specific
  forward-plus-backward FLOP profile before production accounting. The screen
  deliberately does not copy the full-model FLOP value, so its compute
  callback records windows/time but no FLOP total until that profile is added.
- **Evaluation:** No recipe selection is performed. Each run restores its
  validation-selected checkpoint and evaluates its test split once, exactly as
  the original pilot did.

### Launch command

Commit these files first, confirm a clean repository, set the shared snapshot
root, and run the three fixed seeds for each species separately.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_8band_compact_minipigs \
  phase2_compact_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  hydra/launcher=local_gpu \
  -m

python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_8band_compact_monkeys \
  phase2_compact_recording_id=sub-01_ses-04_task-AcousStim_acq-RH_desc-raw \
  hydra/launcher=local_gpu \
  -m
```

### Key config overrides

- `model=neurosoft_conv_bigru_compact` with adapter 32, temporal channels 64,
  GRU hidden size 64, and one GRU layer;
- the original pilot's fixed LR 0.0015, weight decay 0.018, and model dropout
  0.3;
- unweighted loss, the causal manifests, the checkpoint monitor, and one test
  evaluation from the restored best-validation checkpoint; and
- `bf16-mixed` precision.

### Gate criteria

Consider capacity the likely driver only if the compact control for each
species:

1. has finite metrics and predicts at least two validation classes;
2. raises minipig train supported F1 above 0.12;
3. improves the full-model pilot's corresponding three-seed mean test F1; and
4. matches the stated EEGNet test-F1 reference in both species.

If it still fails to learn the minipig training set, next isolate the
convolutional frontend with a same-split historical-plain-GRU control rather
than decreasing capacity again.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

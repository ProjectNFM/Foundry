# Phase 2 -- Convolution--BiGRU Recipe Recovery

**Status:** Draft
**Date started:** 2026-08-31
**Parent experiment:** [Phase 2 -- Convolution--BiGRU Scratch Pilot](20260828-MS-neurosoft-conv-bigru-pilot.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, phase2, convolution-bigru, scratch, recipe-recovery, hyperparameters, intrasession-causal

## Background

The Phase-2 scratch pilot completed its protocol-semantic runs but failed the
minipig learnability gate: all full-data seeds converged to the class-prior
prediction (`high_treble`) with supported test macro-F1 0.041. The matched
EEGNet runs on the same causal manifests achieve 0.135 mean test macro-F1 for
that session, while the new Conv--BiGRU has 510,728 parameters versus EEGNet's
5,368 and uses two 0.3 dropout sites. The pilot also used unweighted
cross-entropy despite selecting on macro-F1.

This experiment is a targeted **recipe** screen, not a new architecture claim.
It retains the transfer-boundary architecture and causal data protocol, but
tests whether lower learning rates plus lighter regularization and
train-split-only inverse-frequency class weights restore learnability. The
requested 16-example overfit diagnostic is deliberately omitted: learnability
is assessed on the full causal training set instead.

## Question

Can a lower-learning-rate, lightly regularized, class-weighted Conv--BiGRU
recipe achieve non-collapsed validation learning and match the Phase-1 EEGNet
test macro-F1 on the representative minipig and monkey sessions?

## Hypothesis

At least one screened recipe with learning rate at or below 0.0015, dropout at
or below 0.1, weight decay at or below 0.003, and train-derived inverse-
frequency class weights will learn more than one class and improve supported
validation macro-F1 beyond the class-prior solution in both species. Its
three-seed confirmation mean will meet or exceed the matched EEGNet reference:
0.135 for minipig `sub-06_ses-02...LH` and 0.208 for monkey
`sub-01_ses-04...RH`.

## Experiment

### Setup

- **Model:** The unchanged `NeurosoftConvBiGRU` transfer-boundary architecture:
  64-dimensional session adapter, one 128-channel separable temporal block,
  2-layer bidirectional GRU with 128 hidden units per direction, and an
  8-logit router. A capacity reduction is explicitly out of scope unless this
  fixed-architecture screen fails its learnability gate.
- **Data protocol:** The same full-data (`training_fraction=1.0`),
  `intrasession-causal`, Phase-0-audited target manifests as the parent pilot.
  The recording audit's raw 32-channel count is reported separately from the
  model's supported-channel inputs: 18 for the minipig and 29 for the monkey.
- **Pilot sessions:** minipig
  `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw` and monkey
  `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`.
- **Screen:** For each species, seed 42 evaluates the 12-cell Cartesian grid:
  learning rate `{0.00015, 0.0005, 0.0015}` × dropout `{0.0, 0.1}` × AdamW
  weight decay `{0.0, 0.003}`. The learning-rate range follows the request to
  test smaller rates for the large recurrent model; it includes the original
  0.0015 as an anchor with lighter regularization.
- **Loss:** `class_weights.mode=auto`, smoothing 0.5. Weights are computed
  from each run's causal train split only. The primary selection metric remains
  supported validation macro-F1.
- **Training:** batch size 16, 200 epochs maximum, patience 40, and the
  original gradient-clip value of 1.0. The screen logs gradients and Adam
  updates for the session adapter, convolutional frontend, GRU, and router.
- **Precision:** `16-mixed`, which is supported by the Quadro RTX 8000.
  Reconfirm the existing FLOP count under this precision before treating the
  resulting compute accounting as final.
- **Test discipline:** Screening runs set `run.evaluate_test=false`. Select one
  recipe per species using validation only, then rerun it with seeds 42/43/44
  and `run.evaluate_test=true` to obtain exactly one test result per seed.
- **Tracking:** The screen and confirmation use distinct WandB groups
  (`PHASE2_CONV_BIGRU_RECIPE_screen` and
  `PHASE2_CONV_BIGRU_RECIPE_confirm`) and run names encode the stage, recipe
  hyperparameters, and seed.

### Launch command

Commit the configurations and experiment record, ensure the repository is
clean, and launch one species-specific 12-cell screen at a time. The local
launcher produces immutable snapshots, so retain its snapshot path and every
WandB ID in Results.

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_8band_recipe_recovery_minipigs \
  phase2_recovery_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  hydra/launcher=local_gpu \
  -m

python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_8band_recipe_recovery_monkeys \
  phase2_recovery_recording_id=sub-01_ses-04_task-AcousStim_acq-RH_desc-raw \
  hydra/launcher=local_gpu \
  -m
```

For each species' selected recipe, override the three recipe values explicitly
and run seeds 42, 43, and 44 with `run.evaluate_test=true`; do not rerun the
test during the screen. For example:

```bash
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_8band_recipe_recovery_minipigs \
  phase2_recovery_recording_id=sub-06_ses-02_task-AcousStim_acq-LH_desc-raw \
  recovery_stage=confirm \
  hyperparameters.learning_rate=<selected_lr> \
  model.dropout_rate=<selected_dropout> \
  hyperparameters.weight_decay=<selected_weight_decay> \
  run.seed=42,43,44 \
  run.evaluate_test=true \
  hydra/launcher=local_gpu \
  -m
```

### Key config overrides

- `class_weights.mode=auto`, `class_weights.smoothing=0.5`;
- lower learning-rate grid, `dropout_rate` grid, and lower weight-decay grid;
- `trainer.precision=16-mixed` on RTX 8000;
- gradient/update watcher expanded to all Conv--BiGRU components;
- validation-only screening (`run.evaluate_test=false`); and
- same causal split, fraction manifests, checkpoint monitor, batch size,
  maximum epochs, patience, and compute-tracking method as the parent pilot.

### Gate criteria

Proceed to three-seed confirmation only if a recipe for each species:

1. has finite losses and learns at least two validation prediction classes;
2. exceeds the parent pilot's class-prior validation F1 (0.043 minipig,
   0.064 monkey) by at least 0.02; and
3. has a train supported F1 materially above its validation F1 floor, showing
   that the optimization path learned conditional signal rather than only the
   label prior.

Accept the revised Phase-2 scratch recipe only if its confirmation mean test
supported macro-F1 meets the matched EEGNet references above. If no screened
fixed-architecture recipe passes the learnability gate, create a separate
compact-capacity experiment (`adapter_dim=32`, `temporal_channels=64`,
`gru_hidden_size=64`, `gru_num_layers=1`) rather than blending capacity and
optimization conclusions.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

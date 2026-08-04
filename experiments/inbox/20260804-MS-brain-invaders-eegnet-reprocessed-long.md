# Brain Invaders EEGNet Reprocessed — Long Training (No Early Stopping)

**Status:** Draft
**Date started:** 2026-08-04
**Parent experiment:** [Brain Invaders EEGNet HP Search (Reprocessed Data)](20260804-MS-brain-invaders-eegnet-reprocessed-hp.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, eegnet, reprocessed, long_training

## Background

The [parent HP search](20260804-MS-brain-invaders-eegnet-reprocessed-hp.md) ran
EEGNet with patience=50 on the reprocessed Brain Invaders P300 data. Since that
run was launched, two critical fixes were committed to the dataset class
(`foundry/data/datasets/brain_invaders_p300.py`):

1. **Z-score normalization** (`_ensure_normalized`) — EEG signals are now
   globally normalized per recording before any sampling, which was missing in
   the parent run.
2. **Anchor-trial filtering** (`_keep_anchor_trial`) — When `epoch_duration`
   causes windows to overlap multiple stimuli, only the anchor trial is
   retained. Without this, the model received contradictory labels per window
   (e.g. one Target + two NonTargets), which prevented learning entirely.

An overfitting debug run on a single session (sub001, intrasession) confirmed
that EEGNet *can* learn P300 with these fixes, but showed a notable plateau in
loss before the model eventually improves. With patience=50, the parent
experiment likely terminates during this plateau before the model has a chance
to descend.

## Question

With the normalization and anchor-trial fixes in place, can EEGNet achieve
good baseline P300 classification (F1 ≥ 0.5) if given sufficient training
time (no early stopping, 1000 epochs)?

## Hypothesis

The combination of proper normalization and single-trial-per-window filtering
removes the two main blockers from the parent experiment. Given enough training
time to push through the initial plateau, EEGNet should reach F1 ≥ 0.5 on at
least one learning rate configuration — entering the literature-reported range
of 0.5–0.7 F1 for P300 on this dataset.

## Experiment

### Setup

- **Model:** EEGNet (F1=8, D=2, F2=16, kernel_length=128, dropout=0.5)
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), reprocessed, intersubject split
- **Task:** Binary P300 classification (Target vs NonTarget)
- **Fold:** 0 only (HP search phase)
- **Class weights:** auto (smoothing=1.0)
- **Early stopping:** Disabled
- **Max epochs:** 1000
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_EEGNET_REPROC_LONG

**Hyperparameter grid (5 jobs):**

| Parameter | Values |
|-----------|--------|
| learning_rate | 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 |
| class_weights.mode | auto (smoothing=1.0) |
| trainer.max_epochs | 1000 |
| early_stopping | disabled |

### Launch command

```bash
uv run python main.py experiment=p300/brain_invaders_eegnet_reprocessed_long -m
```

### Key config overrides

- Config: `configs/experiment/p300/brain_invaders_eegnet_reprocessed_long.yaml`
- Based on parent `brain_invaders_hp_eegnet_reprocessed.yaml`
- Early stopping removed entirely (model trains for full 1000 epochs)
- max_epochs increased from 500 → 1000
- Same LR sweep: 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
- All other settings identical to parent

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

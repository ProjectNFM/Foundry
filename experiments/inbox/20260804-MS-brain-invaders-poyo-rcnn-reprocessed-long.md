# Brain Invaders POYO ResampleCNN HP Search (Reprocessed, Long Training)

**Status:** Draft
**Date started:** 2026-08-04
**Parent experiment:** [Brain Invaders P300 HP Search](20260731-MS-brain-invaders-p300-hp-search.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, hp_search, poyo, resample_cnn, reprocessed, long_training

## Background

The [parent HP search](20260731-MS-brain-invaders-p300-hp-search.md) swept
hyperparameters across EEGNet and POYO CWT-CNN on Brain Invaders P300. The
best POYO result was F1=0.402 (CWT-CNN) and ResampleCNN scored 0.308 in the
[baselines](20260731-MS-brain-invaders-p300-baselines.md) — both far below
literature targets (~0.5–0.7 F1).

The root cause was a **data pipeline issue**: with `sequence_length=1.0s` and
`drop_short=True`, 90% of trials were dropped. The data has since been
**reprocessed** to fix this, and two additional critical fixes were applied to
the dataset class (`foundry/data/datasets/brain_invaders_p300.py`):

1. **Z-score normalization** (`_ensure_normalized`) — EEG signals are globally
   normalized per recording before sampling.
2. **Anchor-trial filtering** (`_keep_anchor_trial`) — When `epoch_duration`
   causes windows to overlap multiple stimuli, only the anchor trial is
   retained, preventing contradictory labels per window.

The [EEGNet reprocessed HP search](20260804-MS-brain-invaders-eegnet-reprocessed-hp.md)
and its [long training follow-up](20260804-MS-brain-invaders-eegnet-reprocessed-long.md)
are testing EEGNet on this corrected pipeline. This experiment extends the
investigation to POYO with the ResampleCNN tokenizer, using long training
(1000 epochs, no early stopping) to avoid the plateau issue observed in the
EEGNet overfitting debug runs.

## Question

With the reprocessed Brain Invaders data and dataset fixes (normalization +
anchor-trial filtering), can POYO ResampleCNN achieve reasonable P300
classification (F1 > 0.5) when given sufficient training time?

## Hypothesis

With access to the full trial set (instead of ~10%) and proper normalization,
POYO ResampleCNN should substantially exceed the original baseline of F1=0.308.
Long training (1000 epochs) will prevent premature stopping during loss
plateaus. The optimal learning rate is expected to be 1e-4 (consistent with
POYO's behavior on the original data and PhysioNet MI), potentially reaching
the literature range of 0.5–0.7 F1.

## Experiment

### Setup

- **Model:** POYO EEG (embed_dim=256, depth=4, ResampleCNN tokenizer, dynamic channel embeddings)
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), reprocessed, intersubject split
- **Task:** Binary P300 classification (Target vs NonTarget)
- **Fold:** 0 only (HP search phase; best config re-run on all 3 folds later)
- **Class weights:** auto (smoothing=1.0)
- **Early stopping:** Disabled (patience=1000 = effectively off)
- **Max epochs:** 1000
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_POYO_RCNN_REPROC_LONG

**Hyperparameter grid (5 jobs):**

| Parameter | Values |
|-----------|--------|
| learning_rate | 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 |
| class_weights.mode | auto (smoothing=1.0) |
| trainer.max_epochs | 1000 |
| early_stopping | disabled (patience=1000) |

### Launch command

```bash
uv run python main.py experiment=p300/brain_invaders_hp_poyo_rcnn_reprocessed_long -m
```

### Key config overrides

- Config: `configs/experiment/p300/brain_invaders_hp_poyo_rcnn_reprocessed_long.yaml`
- Tokenizer: `per_channel_resample_cnn` (instead of `per_channel_cwt_cnn`)
- Channel embedding: `dynamic` (fixed, not swept)
- class_weights.mode fixed to `auto` (POYO originally preferred smoothing=0.1, but aligning with EEGNet reprocessed for comparability)
- LR range extended downward (5e-5, 1e-5) to test if larger dataset benefits from slower learning
- No early stopping — trains for full 1000 epochs
- timeout_min increased to 1440 (24h) to accommodate long training
- All other settings match the original POYO HP search (embed_dim=256, depth=4, batch_size=64)

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

# Brain Invaders P300 Hyperparameter Search

**Status:** Draft
**Date started:** 2026-07-31
**Parent experiment:** [Brain Invaders P300 From-Scratch Baselines](20260731-MS-brain-invaders-p300-baselines.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, hp_search, eegnet, poyo, cwt_cnn, resample_cnn

## Background

The [baseline experiment](20260731-MS-brain-invaders-p300-baselines.md) showed
that all models perform poorly on Brain Invaders P300 with default
hyperparameters from sleep staging:

- **EEGNet collapsed entirely** (0.046 F1) — predicted all NonTarget,
  early stopped at epoch 2. The lr=1e-4 with patience=20 is insufficient.
- **POYO CWT-CNN was best** at 0.347 F1 but far below literature (~0.5–0.7).
- **POYO ResampleCNN** slightly worse at 0.308 F1.
- **Channel embeddings** had negligible effect.

The 83/17 class imbalance and short 1s window create a very different
optimization landscape from 30s sleep staging. The hyperparameters need
task-specific tuning.

## Question

Can task-specific hyperparameter tuning bring Brain Invaders P300
classification to reasonable performance levels (>0.5 F1) for each model
architecture?

## Hypothesis

1. **Higher learning rates** (1e-3 to 5e-4) will prevent the early collapse
   seen with EEGNet and improve POYO convergence speed.
2. **Stronger class weighting** (smoothing < 1.0 or focal loss) will force
   models to attend to the minority Target class.
3. **Longer patience** (50–100 epochs) will allow models to escape early
   plateaus instead of stopping prematurely.
4. **Architecture-specific tuning** (EEGNet kernel_length for 512Hz,
   POYO depth/heads for short windows) will provide additional gains.

## Experiment

### Setup

- **Models:** EEGNet, POYO CWT-CNN (dynamic ch. emb only)
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), intersubject split
- **Task:** Binary P300 classification (Target vs NonTarget)
- **Fold:** 0 only (HP search phase; best configs re-run on all 3 folds)
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_HP_SEARCH

**Hyperparameter grid:**

| Parameter | Values |
|-----------|--------|
| learning_rate | 1e-3, 5e-4, 1e-4 |
| class_weights.mode | none, auto (smoothing=1.0 when auto) |
| trainer.callbacks.early_stopping.patience | 50 |
| trainer.max_epochs | 500 |

**EEGNet-specific:**

| Parameter | Values |
|-----------|--------|
| model.F1 | 8, 16 |
| model.kernel_length | 64, 128, 256 |
| model.dropout | 0.25, 0.5 |

**POYO-specific:**

| Parameter | Values |
|-----------|--------|
| model.depth | 2, 4 |
| model.num_heads | 4, 8 |
| model.embed_dim | 128, 256 |

### Launch command

```bash
# POYO CWT-CNN dynamic (12 jobs: 3 lr × 2 class_weights.mode × 2 embed_dim)
uv run python main.py experiment=p300/brain_invaders_hp_search_poyo -m

# EEGNet (12 jobs: 3 lr × 2 class_weights.mode × 2 F1)
uv run python main.py experiment=p300/brain_invaders_hp_search_eegnet -m
```

### Key config overrides

- POYO config: `configs/experiment/p300/brain_invaders_hp_search_poyo.yaml`
- EEGNet config: `configs/experiment/p300/brain_invaders_hp_search_eegnet.yaml`
- Patience increased to 50 (from 20 in baselines) to prevent premature stopping
- `max_epochs: 500` (reduced from 1000 since HP search is single-fold)
- EEGNet `kernel_length: 128` (doubled from 64 to better match 512 Hz)
- POYO `model.channel_emb_mode: dynamic` fixed across sweep
- All runs use fold 0 only

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

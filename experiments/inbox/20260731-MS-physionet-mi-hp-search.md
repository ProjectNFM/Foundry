# PhysioNet Motor Imagery Hyperparameter Search

**Status:** Draft
**Date started:** 2026-07-31
**Parent experiment:** [PhysioNet Motor Imagery From-Scratch Baselines](20260731-MS-physionet-mi-baselines.md)
**Follow-up experiments:** TBD
**Tags:** motor_imagery, physionet, hp_search, eegnet, poyo, cwt_cnn

## Background

The [baseline experiment](20260731-MS-physionet-mi-baselines.md) had mixed
results:

- **EEGNet achieved 0.873 F1** — a strong baseline for intersubject MI,
  but potentially improvable with tuned hyperparameters.
- **All POYO runs crashed** before training (OOM or data-loading issue with
  64 channels × 640 samples at batch_size=32). Need to reduce batch size or
  model size to fit in memory, then tune HPs.

The 64-channel, 4s-window MI task is the most memory-intensive configuration
tested so far. POYO needs batch size reduction and possibly architecture
downsizing to run at all on this dataset.

## Question

Can hyperparameter tuning improve EEGNet beyond 0.873 F1 on PhysioNet MI,
and can POYO CWT-CNN (with reduced batch size) achieve competitive or
superior performance once it runs successfully?

## Hypothesis

1. **EEGNet can reach ~0.90 F1** with tuned lr/weight_decay and larger
   spatial filters (F1=16 or D=4) to exploit the 64-channel spatial info.
2. **POYO CWT-CNN will run** with batch_size=8–16 and can match or exceed
   EEGNet once HP-tuned, given the richer architecture.
3. **Higher learning rates** (5e-4 to 1e-3) may speed convergence for both
   models on this relatively straightforward binary task.

## Experiment

### Setup

- **Models:** EEGNet, POYO CWT-CNN (dynamic ch. emb only)
- **Data:** PhysionetMI (`physionet_mi/allsess`), intersubject split
- **Task:** Binary motor imagery classification (Left Hand vs Right Hand)
- **Fold:** 0 only (HP search phase; best configs re-run on all 3 folds)
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=PHYSIONET_MI_HP_SEARCH

**Hyperparameter grid:**

| Parameter | Values |
|-----------|--------|
| learning_rate | 1e-3, 5e-4, 1e-4 |
| weight_decay | 0.0, 0.01, 0.1 (EEGNet only) |
| class_weights.mode | none, auto (smoothing=1.0 when auto) |
| trainer.callbacks.early_stopping.patience | 50 |
| trainer.max_epochs | 500 |

**EEGNet-specific:**

| Parameter | Values |
|-----------|--------|
| model.F1 | 8, 16 |
| model.D | 2, 4 |
| model.kernel_length | 64, 128 |
| model.dropout | 0.25, 0.5 |

**POYO-specific:**

| Parameter | Values |
|-----------|--------|
| hyperparameters.batch_size | 8, 16 |
| model.depth | 2, 4 |
| model.num_heads | 4, 8 |
| model.embed_dim | 128, 256 |

### Launch command

```bash
# POYO CWT-CNN dynamic (24 jobs: 3 lr × 2 batch_size × 2 class_weights.mode × 2 embed_dim)
uv run python main.py experiment=motor_imagery/physionet_hp_search_poyo -m

# EEGNet (36 jobs: 3 lr × 3 weight_decay × 2 class_weights.mode × 2 F1)
uv run python main.py experiment=motor_imagery/physionet_hp_search_eegnet -m
```

### Key config overrides

- POYO config: `configs/experiment/motor_imagery/physionet_hp_search_poyo.yaml`
- EEGNet config: `configs/experiment/motor_imagery/physionet_hp_search_eegnet.yaml`
- POYO `batch_size: 8, 16` (reduced from 32 to prevent OOM with 64 channels)
- POYO `model.channel_emb_mode: dynamic` fixed across sweep
- Patience increased to 50 (from 20 in baselines)
- `max_epochs: 500`
- All runs use fold 0 only

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

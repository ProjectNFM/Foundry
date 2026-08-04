# Brain Invaders P300 Reprocessed — 3-Fold Baselines (All Architectures, Inter + Intra)

**Status:** Draft
**Date started:** 2026-08-04
**Parent experiment:** [EEGNet Reprocessed Long Training](20260804-MS-brain-invaders-eegnet-reprocessed-long.md), [POYO ResampleCNN Reprocessed Long Training](20260804-MS-brain-invaders-poyo-rcnn-reprocessed-long.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, baseline, reprocessed, 3fold, eegnet, poyo, cwt_cnn, resample_cnn, intersubject, intrasession

## Background

Two HP search experiments on the reprocessed Brain Invaders P300 data
established the best learning rates for each architecture:

- **EEGNet** ([long training](20260804-MS-brain-invaders-eegnet-reprocessed-long.md)):
  best lr=1e-3, val F1=0.337 at ~94 epochs. Mild overfitting (train-val
  gap=+0.05).
- **POYO ResampleCNN** ([long training](20260804-MS-brain-invaders-poyo-rcnn-reprocessed-long.md)):
  best lr=1e-4, val F1=0.327 at ~44 epochs. Extreme overfitting (train
  F1=0.950, train-val gap=+0.62).

Both remain far below literature targets (0.5–0.7 F1). This experiment
runs a systematic cross-architecture comparison on all 3 folds to produce
the definitive baseline table for the reprocessed data. CWT-CNN (not yet
tested on reprocessed data) is included as it outperformed ResampleCNN by
+3.5pp F1 in the [original baselines](20260731-MS-brain-invaders-p300-baselines.md).

Critically, **intrasession splits** are included alongside intersubject to
diagnose whether the poor performance is due to cross-subject variability
(intersubject fails but intrasession works) or a deeper issue with the data
or models (both fail).

## Question

How do the 5 model conditions (EEGNet, POYO CWT-CNN ×2, POYO ResampleCNN ×2)
compare on reprocessed Brain Invaders P300 with tuned hyperparameters, and
does switching from intersubject to intrasession splits reveal that models
can learn P300 when train/val share the same subjects?

## Hypothesis

1. **CWT-CNN will outperform ResampleCNN** on reprocessed data, consistent
   with the +3.5pp advantage seen in the original baselines and across other
   tasks (KempSleep, PhysioNet MI).
2. **All models will remain below 0.5 F1** at the intersubject level, given
   that fold-0 HP search results peaked at 0.337 (EEGNet) and 0.327 (RCNN).
3. **Dynamic channel embeddings will have negligible effect**, consistent
   with prior findings on Brain Invaders P300 (<1pp difference) and
   PhysioNet MI (−0.7 to −0.8pp).
4. **EEGNet and POYO will perform similarly** (~0.30–0.34 F1), with no
   clear architectural winner — matching the near-equivalence seen on
   PhysioNet MI.
5. **Intrasession will substantially outperform intersubject**, potentially
   reaching 0.5+ F1, confirming that models can learn P300 features when
   train/val share the same subject and that cross-subject variability is
   the primary bottleneck.

## Experiment

### Setup

- **Models:**
  - EEGNet (F1=8, D=2, F2=16, kernel_length=128, dropout=0.5, lr=1e-3)
  - POYO CWT-CNN (embed_dim=256, depth=4, lr=1e-4)
  - POYO ResampleCNN (embed_dim=256, depth=4, lr=1e-4)
- **Channel embeddings (POYO only):** disabled / dynamic
- **Split types:** intersubject, intrasession
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), reprocessed
- **Task:** Binary P300 classification (Target vs NonTarget)
- **Folds:** 0, 1, 2
- **Class weights:** auto (smoothing=1.0) for all conditions
- **Early stopping:** patience=50
- **Max epochs:** 500
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_REPROC_3FOLD

**Conditions (30 total = 5 models × 3 folds × 2 split_types):**

| Condition         | Model  | Tokenizer      | channel_emb | LR   | Split        | Folds   |
|-------------------|--------|----------------|-------------|------|--------------|---------|
| eegnet            | EEGNet | N/A            | N/A         | 1e-3 | inter, intra | 0, 1, 2 |
| cwt-disabled      | POYO   | CWT-CNN        | disabled    | 1e-4 | inter, intra | 0, 1, 2 |
| cwt-dynamic       | POYO   | CWT-CNN        | dynamic     | 1e-4 | inter, intra | 0, 1, 2 |
| rcnn-disabled     | POYO   | ResampleCNN    | disabled    | 1e-4 | inter, intra | 0, 1, 2 |
| rcnn-dynamic      | POYO   | ResampleCNN    | dynamic     | 1e-4 | inter, intra | 0, 1, 2 |

### Launch command

```bash
# POYO models (24 jobs: 2 tokenizers × 2 channel_emb × 3 folds × 2 split_types)
uv run python main.py experiment=p300/brain_invaders_reprocessed_3fold_poyo -m

# EEGNet (6 jobs: 3 folds × 2 split_types)
uv run python main.py experiment=p300/brain_invaders_reprocessed_3fold_eegnet -m
```

### Key config overrides

- POYO config: `configs/experiment/p300/brain_invaders_reprocessed_3fold_poyo.yaml`
- EEGNet config: `configs/experiment/p300/brain_invaders_reprocessed_3fold_eegnet.yaml`
- EEGNet lr=1e-3 (best from long training HP search)
- POYO lr=1e-4 (best from ResampleCNN long training HP search)
- Patience=50, max_epochs=500
- class_weights=auto (smoothing=1.0) for all conditions
- Hydra sweep includes `data.split_type: intersubject, intrasession`
- Same WandB group `BI_P300_REPROC_3FOLD` for both configs

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

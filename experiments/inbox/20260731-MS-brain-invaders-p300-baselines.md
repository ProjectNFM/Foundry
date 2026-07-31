# Brain Invaders P300 From-Scratch Baselines

**Status:** Draft
**Date started:** 2026-07-31
**Parent experiment:** [KempSleep 30s-Epoch From-Scratch Baselines](../_legacy/023-kemp-30s-baselines.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, baseline, from_scratch, eegnet, poyo, cwt_cnn, resample_cnn

## Background

Experiment 023 established strong from-scratch baselines for KempSleep 5-class
sleep staging, showing that POYO CWT-CNN is the best architecture (0.730 F1),
followed by POYO ResampleCNN (0.699) and EEGNet (0.692). Dynamic channel
embeddings provided negligible benefit at 30s epochs.

This experiment extends the same baseline methodology to the Brain Invaders
2014a P300 dataset — a binary classification task (Target vs NonTarget) with
16 EEG channels at 512 Hz. P300 differs fundamentally from sleep staging:
trials are short (~1s), event-locked, and the discriminative signal is a
transient ERP component rather than sustained oscillatory patterns. This will
test whether the architectural rankings from sleep staging transfer to an
event-related potential paradigm.

## Question

How do the different model architectures (EEGNet, POYO CWT-CNN, POYO
ResampleCNN) and channel embedding modes (disabled, dynamic) compare on
Brain Invaders P300 binary classification when trained from scratch with
3-fold intersubject cross-validation?

## Hypothesis

1. **EEGNet may be more competitive here** than on sleep staging, since it was
   designed for short-window BCI tasks and P300 trials are ~1s.
2. **CWT-CNN will still outperform ResampleCNN** since wavelet decomposition
   should capture the P300 component's time-frequency signature well.
3. **Dynamic channel embeddings may matter more** for P300 than for 30s sleep
   staging, since the spatial distribution of the P300 is informative and
   the 16-channel montage provides meaningful topographic structure.
4. **Class imbalance will be a factor** — P300 datasets are typically ~80/20
   NonTarget/Target, making F1 a more informative metric than accuracy.

## Experiment

### Setup

- **Models:**
  - **EEGNet:** F1=8, D=2, F2=16, kernel_length=64, dropout=0.5,
    17 channels (auto-detected), 512 samples (1s × 512 Hz)
  - **POYO (CWT-CNN):** POYOEEGModel, embed_dim=256, depth=4, 8 heads,
    dim_head=128, `per_channel_cwt_cnn` tokenizer
  - **POYO (ResampleCNN):** Same backbone, `per_channel_resample_cnn` tokenizer
- **Channel embeddings (POYO only):** `disabled` / `dynamic`
- **Session embeddings:** `disabled` for all POYO conditions
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), intersubject
  split, all 3 folds (0, 1, 2)
- **Task:** Binary P300 classification (Target vs NonTarget), class-weighted
  cross-entropy
- **Training:** sequence_length=1.0s, batch_size=64, lr=1e-4,
  weight_decay=0.01, max_epochs=1000, early stopping on val F1 (patience=20),
  bf16-mixed precision
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_BASELINES

**Conditions (15 total = 5 models × 3 folds):**

| Condition                           | Model      | Tokenizer      | channel_emb | Folds |
| ----------------------------------- | ---------- | -------------- | ----------- | ----- |
| eegnet                              | EEGNet     | N/A            | N/A         | 0,1,2 |
| poyo-cwt-ch-disabled                | POYO       | CWT-CNN        | `disabled`  | 0,1,2 |
| poyo-cwt-ch-dynamic                 | POYO       | CWT-CNN        | `dynamic`   | 0,1,2 |
| poyo-rcnn-ch-disabled               | POYO       | ResampleCNN    | `disabled`  | 0,1,2 |
| poyo-rcnn-ch-dynamic                | POYO       | ResampleCNN    | `dynamic`   | 0,1,2 |

### Launch command

```bash
# POYO models (12 jobs: 2 tokenizers × 2 channel_emb × 3 folds)
uv run python main.py experiment=p300/brain_invaders_baselines -m

# EEGNet (3 jobs: 3 folds)
uv run python main.py experiment=p300/eegnet_brain_invaders_baselines -m
```

### Key config overrides

- POYO experiment config:
  `configs/experiment/p300/brain_invaders_baselines.yaml`
- EEGNet experiment config:
  `configs/experiment/p300/eegnet_brain_invaders_baselines.yaml`
- `data.split_type: intersubject` for all conditions
- `hyperparameters.sequence_length: 1.0` (P300 trial windows)
- `hyperparameters.sampling_rate: 512`
- `model/session_emb: disabled` for all POYO conditions
- Hydra multirun sweeps `hyperparameters.fold_number` over 0, 1, 2

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

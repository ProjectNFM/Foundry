# POYO Overfitting Diagnosis — Regularization & Frozen Tokenizer Ablation

**Status:** Draft
**Date started:** 2026-08-05
**Parent experiment:** [Brain Invaders P300 Reprocessed — 3-Fold Baselines](20260804-MS-brain-invaders-p300-reprocessed-3fold.md)
**Follow-up experiments:** TBD
**Tags:** p300, brain_invaders, poyo, overfitting, regularization, weight_decay, dropout, frozen_tokenizer, ablation

## Background

The [3-fold baselines experiment](20260804-MS-brain-invaders-p300-reprocessed-3fold.md)
revealed a striking divergence between POYO and EEGNet on Brain Invaders P300:
both achieve similar validation F1 (~0.32–0.40), but POYO memorises training
data completely (train F1 0.95–0.98, overfit gap +0.60–0.64) while EEGNet shows
zero overfitting (train ≈ val F1). This happens consistently across all POYO
variants (CWT-CNN / ResampleCNN × disabled / dynamic channel embeddings),
all 3 folds, and both intersubject and intrasession splits.

EEGNet uses aggressive built-in regularization (depthwise separable convolutions,
dropout=0.5 on all layers) and has far fewer parameters. POYO (embed_dim=256,
depth=4) uses relatively mild regularization (weight_decay=0.01, ffn/atn
dropout=0.2, lin_dropout=0.4) and has substantially more capacity distributed
across both its transformer backbone and CWT-CNN tokenizer.

This experiment systematically tests whether the overfitting is due to:
1. **Insufficient weight regularization** — the 0.01 weight decay is too low
   for POYO's parameter count
2. **Insufficient dropout** — the 0.2/0.4 dropout rates leave too much
   memorization capacity
3. **Tokenizer memorization** — the CWT-CNN tokenizer itself learns to
   memorise training patterns rather than extracting generalisable features

## Question

Which source of excess capacity drives POYO's extreme overfitting on Brain
Invaders P300: insufficient weight decay, insufficient dropout, or tokenizer
co-adaptation with training data?

## Hypothesis

1. **Weight decay alone will be insufficient** — increasing from 0.01 to 0.1
   will modestly reduce the overfit gap (by ~0.1–0.2) but not eliminate it,
   because L2 regularization does not prevent feature co-adaptation.
2. **Heavy dropout will have the largest single effect** — setting all dropouts
   to 0.5 (matching EEGNet's level) will substantially reduce the gap (by
   ~0.2–0.4) because it directly prevents neuron co-adaptation, which is the
   mechanism most likely responsible for memorization in transformers.
3. **Frozen tokenizer will reveal partial memorization** — freezing the CWT-CNN
   will reduce the gap somewhat (by ~0.1–0.2) because the tokenizer contributes
   memorization capacity, but the transformer backbone alone still has enough
   capacity to overfit.
4. **The combined condition (all three) will approach EEGNet's dynamics** —
   near-zero overfit gap — confirming that POYO's overfitting is a capacity/
   regularization problem rather than a fundamental architectural deficiency.

## Experiment

### Setup

- **Model:** POYO (embed_dim=256, depth=4, CWT-CNN tokenizer, dynamic channel embeddings)
- **Data:** BrainInvadersP300 (`brain_invaders_p300/allsess`), reprocessed
- **Task:** Binary P300 classification (Target vs NonTarget)
- **Split:** Intersubject, fold 0 only
- **Class weights:** auto (smoothing=1.0)
- **Early stopping:** patience=50, monitor=val/p300_binary_f1
- **Max epochs:** 500
- **Hardware:** 1× L40S, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=BI_P300_OVERFIT_REGULARIZATION

**Conditions (7 total):**

| # | Condition        | weight_decay | ffn_dropout | lin_dropout | atn_dropout | cwt_lr_mult |
|---|------------------|--------------|-------------|-------------|-------------|-------------|
| 1 | Baseline         | 0.01         | 0.2         | 0.4         | 0.2         | 1.0         |
| 2 | WD 0.05          | 0.05         | 0.2         | 0.4         | 0.2         | 1.0         |
| 3 | WD 0.1           | 0.1          | 0.2         | 0.4         | 0.2         | 1.0         |
| 4 | Dropout 0.3      | 0.01         | 0.3         | 0.3         | 0.3         | 1.0         |
| 5 | Dropout 0.5      | 0.01         | 0.5         | 0.5         | 0.5         | 1.0         |
| 6 | Frozen tokenizer | 0.01         | 0.2         | 0.4         | 0.2         | 0.0         |
| 7 | Combined         | 0.1          | 0.5         | 0.5         | 0.5         | 0.0         |

### Launch command

```bash
# Submit all 7 conditions to SLURM
bash scripts/launch_overfit_regularization.sh

# Or run locally for debugging (single condition runs in-process)
bash scripts/launch_overfit_regularization.sh local
```

### Key config overrides

- Config: `configs/experiment/p300/brain_invaders_poyo_overfit_regularization.yaml`
- Launch script: `scripts/launch_overfit_regularization.sh` (7 explicit CLI calls)
- Base: POYO CWT-CNN Dynamic from 3-fold experiment, restricted to fold 0 intersubject
- Each condition passes explicit overrides for weight_decay, dropout, and cwt_lr_multiplier
- `cwt_lr_multiplier: 0` freezes the CWT-CNN tokenizer (lr=0 for all `.cwt.` parameters)
- All other settings match the parent experiment exactly

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

# Discriminative LR Finetuning

**Status:** Draft
**Date started:** 2026-07-23
**Parent experiment:** [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md)
**Follow-up experiments:** TBD

## Background

Experiment 009 systematically swept learning rate and warmup for both pretrained
and from-scratch CWT-CNN finetuning on Kemp Sleep fold 0. The result was
decisive: **no HP configuration enabled pretrained finetuning to beat scratch**.
The best configs for both conditions were identical (lr=1e-4, warmup=0), and
the −2 pp gap persisted.

Two findings from 009 motivate a discriminative learning rate approach:

1. **Pretrained val F1 peaks at epoch 0** and monotonically declines — continued
   full-model finetuning destroys useful representations faster than it adapts
   the task head.
2. **Exp 008 showed the frozen pretrained backbone contains real signal**
   (linear probe F1 0.418 vs 0.267 random init, +15 pp). Full finetuning at
   uniform lr improves over the frozen backbone (0.543) but still trails scratch
   (0.563).

The failure mode is therefore not "pretrained features are useless" but
"uniform finetuning cannot preserve backbone features while adapting the head."
Standard transfer learning practice addresses this with **discriminative
(layerwise) learning rates**: a low LR for pretrained backbone components and a
higher LR for randomly-initialized task-specific components (embeddings, readout
heads).

This requires extending `FoundryModule._build_param_groups()` to split
parameters by `POYOEEGModel.transferable_components()` — implemented as
`hyperparameters.backbone_learning_rate` (backbone) vs
`hyperparameters.learning_rate` (head).

## Question

Can discriminative learning rates (low backbone LR, higher head LR) preserve
pretrained CWT-CNN representations during finetuning and achieve higher val F1
than both (a) uniform-lr finetuning from exp 009 (0.5425) and (b) the
from-scratch baseline (0.5629)?

## Hypothesis

Yes — with backbone LR in the 1e-6 to 1e-5 range and head LR at 1e-4 or 1e-3,
the model will adapt the classification head quickly while making only small,
non-destructive updates to the pretrained backbone. This should:

1. Beat exp 009 uniform-lr finetuning (0.5425) by reducing representation drift.
2. Beat or match the from-scratch baseline (0.5629) by combining pretrained
   feature quality with task-specific head adaptation.
3. Show val F1 that improves or holds steady beyond epoch 0 (unlike exp 009's
   immediate decline).

## Experiment

### Setup

- **Model:** POYOEEGModel with CWT-CNN tokenizer (per_channel_cwt_cnn),
  embed_dim=256, depth=4, same architecture as experiments 005–009
- **Data:** KempSleepEDF2013, inter-subject split, fold 0 for search,
  all 3 folds for final validation
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
  (smoothing=1.0)
- **Pretrained checkpoint:** CWT-CNN from exp 005 (wandb: `wlmobz7y`,
  val_loss=0.0364)
- **Param groups:** backbone = {tokenizer, backbone, rotary_emb, latent_emb};
  head = {channel_emb, session_emb, task_emb, readout heads}
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM, 6h wall time (SLURM)
- **WandB:** project=foundry_finetuning

**Sweep axes (Phase 1 grid):**

| Hyperparameter | Values | Rationale |
| --- | --- | --- |
| Backbone LR | 1e-6, 1e-5, 3e-5 | Conservative range below exp 009 best uniform lr (1e-4) |
| Head LR | 1e-4, 1e-3 | 1e-4 matches exp 009 scratch best; 1e-3 matches exp 008 linear probe |
| Warmup | 0 (fixed) | Exp 009 showed warmup hurt; hold constant to isolate discriminative LR |
| ES patience | 50 (fixed) | Same as exp 009 |

**Conditions:**

| Phase | Condition | Group | Runs | Purpose |
| --- | --- | --- | --- | --- |
| 1a | Pretrained discriminative LR grid | KEMP_DISCRIMINATIVE_LR_SEARCH | 6 (3 bb_lr × 2 head_lr) | Find best backbone/head LR pair |
| 1b | Frozen backbone (head only) | KEMP_DISCRIMINATIVE_LR_CONTROLS | 1 | Sanity check vs exp 008 linear probe |
| 1b | Scratch uniform LR | KEMP_DISCRIMINATIVE_LR_CONTROLS | 1 | Fair baseline under exp 010 config |
| 2 | 3-fold validation | KEMP_DISCRIMINATIVE_LR_VALIDATION | 3 | Final comparison with error bars |

### Launch command

```bash
# Phase 1a — Discriminative LR grid (6 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_discriminative_lr -m

# Phase 1b — Controls (fold 0):

# Frozen backbone, head lr=1e-3 (linear-probe-style under exp 010 training setup):
uv run python main.py experiment=sleep_staging/poyo_kemp_discriminative_lr \
    run.freeze_pretrained=true \
    hyperparameters.learning_rate=1e-3 \
    hyperparameters.backbone_learning_rate=null \
    'run.name=kemp_discriminative_lr_frozen_head' \
    run.group=KEMP_DISCRIMINATIVE_LR_CONTROLS \
    'run.tags=[sleep_staging,poyo,kemp,finetuning,discriminative_lr,control,exp010]'

# From-scratch uniform LR baseline:
uv run python main.py experiment=sleep_staging/poyo_kemp_discriminative_lr \
    run.pretrained_checkpoint=null \
    hyperparameters.backbone_learning_rate=null \
    hyperparameters.learning_rate=1e-4 \
    'run.name=kemp_discriminative_lr_scratch_uniform' \
    run.group=KEMP_DISCRIMINATIVE_LR_CONTROLS \
    'run.tags=[sleep_staging,poyo,kemp,scratch,discriminative_lr,control,exp010]'

# Phase 2 — 3-fold validation (fill in best backbone_lr and head_lr from Phase 1):
uv run python main.py experiment=sleep_staging/poyo_kemp_discriminative_lr \
    hyperparameters.backbone_learning_rate=<best_bb_lr> \
    hyperparameters.learning_rate=<best_head_lr> \
    run.group=KEMP_DISCRIMINATIVE_LR_VALIDATION \
    'run.name=kemp_discriminative_lr_val_fold${hyperparameters.fold_number}' \
    'run.tags=[sleep_staging,poyo,kemp,finetuning,discriminative_lr,validation,exp010]' \
    'hyperparameters.fold_number=0,1,2' -m
```

### Key config overrides

Uses new config
`configs/experiment/sleep_staging/poyo_kemp_discriminative_lr.yaml`.

Key differences from exp 009 config (`poyo_kemp_finetune_hp_search.yaml`):

- **Discriminative LR** via `hyperparameters.backbone_learning_rate` (new) +
  `hyperparameters.learning_rate` (head), implemented in
  `FoundryModule._build_backbone_head_param_groups()`
- **No LR/warmup sweep** — warmup fixed at 0; only backbone_lr × head_lr swept
- **Tokenizer fixed** to `per_channel_cwt_cnn`
- **Patience** 50, same as exp 009
- Controls use `run.freeze_pretrained=true` (frozen backbone) or
  `run.pretrained_checkpoint=null` (scratch) with uniform LR

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/010_discriminative_lr.py`

```bash
uv run python analysis/010_discriminative_lr.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If discriminative LR beats uniform finetuning but not scratch, try
  **two-stage finetuning**: freeze backbone and train head to convergence (exp
  008 config), then unfreeze backbone at backbone_lr=1e-6 for a short second
  stage. This directly addresses the epoch-0 peak from exp 009.
- If the best backbone_lr is at the floor (1e-6), try **fully frozen backbone
  with longer head training** (increase max_epochs / patience for head-only
  runs) before declaring finetuning exhausted.
- Monitor **backbone representation drift** (e.g., CKA or cosine distance from
  pretrained checkpoint weights) as an early-stopping signal — val F1 alone may
  miss the point where backbone features degrade.
- If discriminative LR also fails, pivot to **alternative pretraining
  objectives** (contrastive, temporal prediction) rather than further
  finetuning HP tuning.
- **Gradual unfreezing** (unfreeze backbone layers one at a time, deepest
  first) is a natural extension if partial backbone adaptation helps.

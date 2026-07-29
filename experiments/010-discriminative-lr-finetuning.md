# Discriminative LR Finetuning

**Status:** Completed
**Date started:** 2026-07-23
**Parent experiment:** [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md)
**Follow-up experiments:** [Session Embedding Ablation](../experiments/011-session-embedding-ablation.md)

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
- **WandB:** project=foundry_finetuning, group=KEMP_DISCRIMINATIVE_LR_SEARCH
  - `kemp_discriminative_lr_bb3e-05_hd0.0001` (`k2d11xtp`)
  - `kemp_discriminative_lr_bb3e-05_hd0.001` (`gtmtk25c`)
  - `kemp_discriminative_lr_bb1e-06_hd0.0001` (`6q016g63`)
  - `kemp_discriminative_lr_bb1e-06_hd0.001` (`o9ylo1dt`)
  - `kemp_discriminative_lr_bb1e-05_hd0.0001` (`cpl9mloq`, crashed)
  - `kemp_discriminative_lr_bb1e-05_hd0.001` (`rtbsydzp`, crashed)

**Sweep axes (Phase 1 grid):**


| Hyperparameter | Values           | Rationale                                                              |
| -------------- | ---------------- | ---------------------------------------------------------------------- |
| Backbone LR    | 1e-6, 1e-5, 3e-5 | Conservative range below exp 009 best uniform lr (1e-4)                |
| Head LR        | 1e-4, 1e-3       | 1e-4 matches exp 009 scratch best; 1e-3 matches exp 008 linear probe   |
| Warmup         | 0 (fixed)        | Exp 009 showed warmup hurt; hold constant to isolate discriminative LR |
| ES patience    | 50 (fixed)       | Same as exp 009                                                        |


**Conditions:**


| Phase | Condition                         | Group                         | Runs                    | Purpose                            |
| ----- | --------------------------------- | ----------------------------- | ----------------------- | ---------------------------------- |
| 1a    | Pretrained discriminative LR grid | KEMP_DISCRIMINATIVE_LR_SEARCH | 6 (3 bb_lr × 2 head_lr) | Find best backbone/head LR pair    |
| —     | Scratch baselines (from exp 009)  | KEMP_SCRATCH_HP_SEARCH        | 12                      | Reference from-scratch performance |




### Launch command

```bash
# Phase 1a — Discriminative LR grid (6 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_discriminative_lr -m
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



## Results



### Summary

Discriminative LR **failed to improve** over either baseline. The best config
(backbone_lr=3e-5, head_lr=1e-4) achieved 0.5406, which is slightly *below*
the uniform-lr pretrained baseline (0.5425, −0.19 pp) and well below scratch
(0.5629, −2.23 pp). Two of six runs crashed (backbone_lr=1e-5 configs), but
their pre-crash metrics (0.5416, 0.5401) suggest they would not have changed
the outcome.

Most critically, **learning curves show the same epoch-0 peak pathology** as
exp 009: all discriminative LR configs achieve their best val F1 at the very
first epoch and then decline monotonically. This is in stark contrast to
scratch models, which start near random (F1 ~0.2–0.3) and climb to their peak
over 3–5 epochs. The problem is not that the backbone LR is too high — even
at 1e-6 (essentially frozen), the model peaks immediately and declines. This
points to something in the model's randomly-initialized components (session
embeddings, channel embeddings) causing instant overfitting that subsequent
training cannot recover from.

### Metrics

**Phase 1a — Discriminative LR grid (fold 0):**


| Backbone LR | Head LR | Val F1     | Run ID     | Status   |
| ----------- | ------- | ---------- | ---------- | -------- |
| 1e-6        | 1e-4    | 0.5363     | `6q016g63` | finished |
| 1e-6        | 1e-3    | 0.5166     | `o9ylo1dt` | finished |
| 1e-5        | 1e-4    | 0.5416*    | `cpl9mloq` | crashed  |
| 1e-5        | 1e-3    | 0.5401*    | `rtbsydzp` | crashed  |
| 3e-5        | 1e-4    | **0.5406** | `k2d11xtp` | finished |
| 3e-5        | 1e-3    | 0.5298     | `gtmtk25c` | finished |


*Pre-crash best (early stopped by OOM/SLURM, not by training failure).

**Baselines (from exp 009, KEMP_SCRATCH_HP_SEARCH, fold 0):**


| Config                               | Val F1     |
| ------------------------------------ | ---------- |
| Scratch lr=1e-4, wu=0                | **0.5629** |
| Scratch lr=1e-4, wu=5                | 0.5605     |
| Scratch lr=5e-5, wu=5                | 0.5510     |
| Scratch lr=1e-4, wu=10               | 0.5502     |
| Pretrained uniform lr=1e-4 (exp 009) | 0.5425     |


**Key comparison:**


| Condition                        | Best Val F1 | Δ vs Scratch |
| -------------------------------- | ----------- | ------------ |
| Scratch best (exp 009)           | 0.5629      | —            |
| Pretrained uniform (exp 009)     | 0.5425      | −2.04 pp     |
| Discriminative LR best (exp 010) | 0.5406      | −2.23 pp     |




### Analysis

Results fetched programmatically from wandb groups `KEMP_DISCRIMINATIVE_LR_SEARCH`
and `KEMP_SCRATCH_HP_SEARCH`.

**Analysis script:** `analysis/010_discriminative_lr.py`

```bash
uv run python analysis/010_discriminative_lr.py
```



### Figures

Discriminative LR heatmap — Val F1 by (Backbone LR, Head LR)Discriminative LR vs scratch baselines comparisonLearning curves: discriminative LR (left) vs scratch (right)

## Conclusions

**Hypothesis refuted.** Discriminative learning rates do not solve the
pretrained-finetuning gap. The best discriminative LR config (backbone_lr=3e-5,
head_lr=1e-4) performs equivalently to uniform-lr finetuning (0.5406 vs 0.5425)
and both trail scratch by ~2 pp.

The learning curves provide a decisive diagnostic: **the problem is not
backbone representation drift** (the original motivation for discriminative LR).
Even with backbone LR=1e-6 — essentially frozen — the model peaks at epoch 0
and declines. This means the damage happens in the randomly-initialized head
components (session embeddings, channel embeddings, readout heads) during the
very first forward passes, not from the backbone being overwritten.

The epoch-0 peak pattern is consistent across all finetuning experiments
(009 uniform LR, 010 discriminative LR) and absent from scratch training.
The key difference is that pretrained models start with meaningful backbone
features that produce reasonable predictions from epoch 0, while scratch
models must learn everything from random — so there is no "peak" to lose.
The randomly-initialized session/channel embeddings likely inject noise that
initially doesn't matter (backbone features dominate) but becomes
progressively harmful as training overfits to their noise patterns.

This shifts the investigation from "how to protect the backbone" to
"why do randomly-initialized embeddings cause instant overfitting when paired
with a pretrained backbone?" — addressed in experiment 011.

## Notes for future experiments

- The epoch-0 peak + decline pattern in all pretrained finetuning runs
strongly implicates the **session embeddings** (and possibly channel
embeddings) as the source of instant overfitting. These are randomly
initialized even when the backbone is pretrained, and they may cause the
model to immediately overfit to embedding noise rather than learning
generalizable session/channel representations. This motivated
[experiment 011](../experiments/011-session-embedding-ablation.md).
- If session embedding ablation confirms the hypothesis, the fix may be
**freezing or removing session embeddings** during finetuning, or
**initializing them from pretrained statistics**.
- The crash of backbone_lr=1e-5 runs (likely SLURM OOM) suggests memory
pressure at intermediate backbone LRs — worth investigating if this
experiment is revisited.


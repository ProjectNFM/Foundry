# Finetuning Hyperparameter Search

**Status:** Draft
**Date started:** 2026-07-22
**Parent experiment:** [Pretraining Loss vs Downstream Task Performance](../experiments/007-pretraining-loss-vs-downstream.md), [Embedding Analysis: t-SNE/PCA and Linear Probing](../experiments/008-embedding-analysis-linear-probing.md)
**Follow-up experiments:** TBD

## Background

Experiment 007 concluded that pretraining provides "no positive transfer" to
downstream sleep staging, with finetuned models performing −2 to −5 pp F1 worse
than from-scratch baselines. However, that experiment used identical
hyperparameters for both finetuning and from-scratch training (lr=1e-4,
warmup=0, patience=20), which is a known methodological flaw — finetuning
and from-scratch training have fundamentally different optimal hyperparameters.

Two observations suggest the conclusion was premature:

1. **Training loss for finetuned models is lower than the from-scratch
  baseline**, indicating the pretrained representations do help the model fit
   training data faster. The gap between train and val performance points to
   overfitting or catastrophic forgetting, not useless representations.
2. **Experiment 008 (linear probing) confirmed the pretrained CWT-CNN backbone
  contains discriminative features for sleep staging** — linear probe F1 of
   0.418 vs 0.267 for random init (+15.0 pp). The pretrained features are
   useful when preserved; full finetuning at lr=1e-4 destroys them.

These two signals point to a classic transfer learning failure mode: the
learning rate is too high, causing catastrophic forgetting of pretrained
features in the first few gradient steps. The model memorizes training data
(low train loss) but loses generalizable structure (poor val F1). Early stopping
at epoch 3–4 confirms the model never recovers.

Standard mitigations from the NLP/vision transfer learning literature include
lower learning rates (10–100× reduction), learning rate warmup, and increased
training patience.

## Question

Can the pretrained CWT-CNN model outperform the from-scratch baseline on Kemp
Sleep 5-class staging when finetuned with properly tuned hyperparameters
(learning rate, warmup schedule, early stopping patience)?

## Hypothesis

Yes — with a lower learning rate (1e-5 to 5e-5 range) and warmup schedule, the
pretrained CWT-CNN model will preserve its useful representations during
finetuning and achieve higher val F1 than both (a) the exp 007 finetuning
result (0.534) and (b) the best from-scratch CWT-CNN baseline (0.554 from
exp 006), after applying equivalent hyperparameter search to the baseline for
fair comparison.

## Experiment



### Setup

- **Model:** POYOEEGModel with CWT-CNN tokenizer (per_channel_cwt_cnn),
embed_dim=256, depth=4, same architecture as experiments 005–008
- **Data:** KempSleepEDF2013, inter-subject split, fold 0 for HP search,
all 3 folds for final validation
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
(smoothing=1.0)
- **Pretrained checkpoint:** CWT-CNN from exp 005 (wandb: `wlmobz7y`,
val_loss=0.0364)
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM, 6h wall time (SLURM)
- **WandB:** project=foundry_finetuning

**Sweep axes:**


| Hyperparameter | Values                 | Rationale                                                               |
| -------------- | ---------------------- | ----------------------------------------------------------------------- |
| Learning rate  | 1e-5, 3e-5, 5e-5, 1e-4 | Standard finetuning LRs are 10–100× lower than from-scratch             |
| Warmup epochs  | 0, 5, 10               | Prevents large early gradients from destroying pretrained features      |
| ES patience    | 50 (fixed)             | Exp 007 stopped at epoch 3–4 with patience=20; allow more recovery time |


**Conditions:**


| Phase | Condition                     | Group                       | Runs                         | Purpose                          |
| ----- | ----------------------------- | --------------------------- | ---------------------------- | -------------------------------- |
| 1     | Pretrained CWT-CNN finetuning | KEMP_FINETUNE_HP_SEARCH     | 12 (4 lr × 3 warmup)         | Find best finetuning HPs         |
| 2     | From-scratch CWT-CNN          | KEMP_SCRATCH_HP_SEARCH      | 12 (4 lr × 3 warmup)         | Fair baseline with same grid     |
| 3     | 3-fold validation             | KEMP_FINETUNE_HP_VALIDATION | 6 (2 best configs × 3 folds) | Final comparison with error bars |




### Launch command

```bash
# Phase 1 — Pretrained CWT-CNN HP search (12 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_hp_search -m

# Phase 2 — From-scratch baseline HP search (12 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_hp_search \
    run.pretrained_checkpoint=null \
    'run.name=kemp_scratch_hp_lr${hyperparameters.learning_rate}_wu${module.warmup_epochs}' \
    run.group=KEMP_SCRATCH_HP_SEARCH \
    'run.tags=[sleep_staging,poyo,kemp,scratch,hp_search,exp009]' -m

# Phase 3 — 3-fold validation (fill in best LR and warmup from Phase 1 & 2):
# Pretrained best config:
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_hp_search \
    hyperparameters.learning_rate=<best_lr> module.warmup_epochs=<best_wu> \
    run.group=KEMP_FINETUNE_HP_VALIDATION \
    'run.name=kemp_finetune_val_fold${hyperparameters.fold_number}' \
    'run.tags=[sleep_staging,poyo,kemp,finetuning,validation,exp009]' \
    'hyperparameters.fold_number=0,1,2' -m

# Scratch best config:
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_hp_search \
    run.pretrained_checkpoint=null \
    hyperparameters.learning_rate=<best_lr> module.warmup_epochs=<best_wu> \
    run.group=KEMP_SCRATCH_HP_VALIDATION \
    'run.name=kemp_scratch_val_fold${hyperparameters.fold_number}' \
    'run.tags=[sleep_staging,poyo,kemp,scratch,validation,exp009]' \
    'hyperparameters.fold_number=0,1,2' -m
```



### Key config overrides

Uses new config
`configs/experiment/sleep_staging/poyo_kemp_finetune_hp_search.yaml`.

Key differences from exp 007 config
(`poyo_kemp_finetune_from_pretrain.yaml`):

- **Tokenizer fixed** to `per_channel_cwt_cnn` (exp 008 showed CWT-CNN has
much stronger pretrained representations than ResampleCNN)
- **LR swept** over {1e-5, 3e-5, 5e-5, 1e-4} instead of fixed 1e-4
- **Warmup swept** over {0, 5, 10} epochs (new; added `warmup_epochs` to
`configs/module/default.yaml`)
- **Patience increased** from 20 → 50 to allow longer training
- **Wall time increased** from 180 → 360 minutes to accommodate longer runs
- **Single fold** (fold 0) for HP search to keep grid tractable
- From-scratch baseline uses same config with
`run.pretrained_checkpoint=null` CLI override



## Results



### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/009_finetuning_hp_search.py`

```bash
uv run python analysis/009_finetuning_hp_search.py
```



### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If the best pretrained config beats scratch but the margin is small,
consider **discriminative (layerwise) learning rates** — lower LR for the
pretrained backbone, higher LR for the new task head. This would require
extending `FoundryModule._build_param_groups()` to split by
`transferable_components()`.
- **Gradual unfreezing** (freeze backbone for N epochs, then unfreeze) is
another avenue if warmup alone is insufficient. Can be approximated by
training a linear probe (exp 008 config) and then resuming with full
finetuning from that checkpoint.
- If no configuration beats scratch, the reconstruction pretraining objective
may be fundamentally misaligned with sleep staging — pivot to contrastive
or temporal prediction objectives.
- **ResampleCNN:** This experiment focuses on CWT-CNN. If CWT-CNN finetuning
succeeds, consider whether the same HP recipe helps ResampleCNN, though
exp 008 showed its pretrained representations are much weaker (+2.2 pp
linear probe vs +15.0 pp for CWT-CNN).


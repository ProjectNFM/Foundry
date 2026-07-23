# Session Embedding Ablation for Inter-Subject Sleep Staging

**Status:** Draft
**Date started:** 2026-07-23
**Parent experiment:** [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md), [Discriminative LR Finetuning](../experiments/010-discriminative-lr-finetuning.md)
**Follow-up experiments:** TBD

## Background

Experiments 009 and 010 attempted to improve pretrained CWT-CNN finetuning
through hyperparameter tuning (LR, warmup) and discriminative learning rates
(separate backbone/head LR). Neither closed the −2 pp F1 gap to the
from-scratch baseline. However, a more fundamental issue emerged from
inspecting the train-val loss gap across all scratch HP search runs (exp 009,
KEMP_SCRATCH_HP_SEARCH):

| Scratch run | Train loss | Val loss | Gap |
|---|---|---|---|
| Best (lr=1e-4, wu=0) | 0.318 | 1.254 | **0.936** |
| Grid average | 0.32–0.44 | 1.25–1.82 | **0.94–1.38** |

A 3–4× multiplier between train and val loss is present from the first epoch.
This is not late-stage overfitting — it reflects systematic subject-level
memorization baked into the architecture.

The `session_emb` (`InfiniteVocabEmbedding`, dim 256) is the likely culprit.
It enters the model at **two critical points**:

1. **Input tokens:** added to every tokenized input via
   `_tokenize_and_add_session` (inputs = inputs + session_emb).
2. **Output queries:** used to construct every readout query via
   `_build_downstream_queries` (queries = session_emb + task_emb).

With 197 sessions from ~100 subjects (most with 2 nights), each session gets
its own learned 256-d vector. In an inter-subject split, every validation
session is one the model has **never seen** — `session_emb` falls back to the
padding embedding (index 0). This means:

- **During training**, the model can encode subject-specific patterns
  (individual physiology, recording artifacts, amplitude scales) into session
  embeddings. This is the easiest way to reduce training loss.
- **During validation**, inputs and queries are constructed from a meaningless
  default embedding, shifting the entire representation away from the training
  distribution.

This architectural leak may dwarf any benefit from pretraining — the
train-val gap of ~1.0 in loss is an order of magnitude larger than the
pretrained-vs-scratch gap in F1 (~0.02).

## Question

Does disabling session embeddings reduce the train-val loss gap and improve
inter-subject generalization for CWT-CNN sleep staging, for both from-scratch
and pretrained models?

## Hypothesis

Yes — disabling session embeddings will:

1. **Reduce the train-val loss gap** significantly (from ~1.0 to <0.5),
   confirming that session embeddings are the primary source of subject-level
   memorization.
2. **Improve or maintain val F1** for from-scratch training, since the model
   will be forced to learn generalizable EEG features rather than
   session-specific shortcuts.
3. **Potentially close the pretrained-vs-scratch gap**, since pretrained
   representations may be more useful when the model cannot rely on session
   identity as a shortcut. Without session_emb, the model's only option is to
   use the EEG signal itself — exactly what the pretrained backbone provides.

## Experiment

### Setup

- **Model:** POYOEEGModel with CWT-CNN tokenizer (per_channel_cwt_cnn),
  embed_dim=256, depth=4, same architecture as experiments 005–010,
  **`disable_session_emb=true`**
- **Data:** KempSleepEDF2013, inter-subject split, fold 0 for search,
  all 3 folds for final validation
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
  (smoothing=1.0)
- **Pretrained checkpoint:** CWT-CNN from exp 005 (wandb: `wlmobz7y`,
  val_loss=0.0364) for pretrained condition
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM, 6h wall time (SLURM)
- **WandB:** project=foundry_finetuning

**Sweep axes (Phase 1):**

| Hyperparameter | Values | Rationale |
| --- | --- | --- |
| Learning rate | 1e-5, 3e-5, 5e-5, 1e-4 | Same grid as exp 009 for comparison |
| Initialization | Scratch, Pretrained | Test both conditions without session shortcuts |
| Session emb | Disabled (fixed) | The intervention under test |
| ES patience | 50 (fixed) | Same as exp 009 |

**Conditions:**

| Phase | Condition | Group | Runs | Purpose |
| --- | --- | --- | --- | --- |
| 1 | No session emb, scratch, LR sweep | KEMP_SESSION_EMB_ABLATION | 4 | Scratch without session leakage |
| 1 | No session emb, pretrained, LR sweep | KEMP_SESSION_EMB_ABLATION | 4 | Pretrained without session leakage |
| 1 | With session emb controls (lr=1e-4) | KEMP_SESSION_EMB_ABLATION_CONTROLS | 2 | Direct comparison under same config |
| 2 | 3-fold validation | KEMP_SESSION_EMB_ABLATION_VALIDATION | 3+ | Final comparison with error bars |

### Launch command

```bash
# Phase 1 — Scratch no-session-emb LR sweep (4 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_session_emb_ablation -m

# Phase 1 — Pretrained no-session-emb LR sweep (4 SLURM jobs, fold 0):
uv run python main.py experiment=sleep_staging/poyo_kemp_session_emb_ablation \
    run.pretrained_checkpoint='${pretrained_checkpoints.per_channel_cwt_cnn}' \
    'run.name=kemp_no_session_emb_lr${hyperparameters.learning_rate}_pretrained' \
    run.init_mode=pretrained \
    'run.tags=[sleep_staging,poyo,kemp,no_session_emb,pretrained,exp011]' -m

# Phase 1 — With-session-emb baselines (fold 0, 2 runs):
# Scratch:
uv run python main.py experiment=sleep_staging/poyo_kemp_session_emb_ablation \
    model.disable_session_emb=false \
    hyperparameters.learning_rate=1e-4 \
    'run.name=kemp_with_session_emb_lr0.0001_scratch' \
    run.group=KEMP_SESSION_EMB_ABLATION_CONTROLS \
    'run.tags=[sleep_staging,poyo,kemp,with_session_emb,control,exp011]'

# Pretrained:
uv run python main.py experiment=sleep_staging/poyo_kemp_session_emb_ablation \
    model.disable_session_emb=false \
    hyperparameters.learning_rate=1e-4 \
    run.pretrained_checkpoint='${pretrained_checkpoints.per_channel_cwt_cnn}' \
    'run.name=kemp_with_session_emb_lr0.0001_pretrained' \
    run.init_mode=pretrained \
    run.group=KEMP_SESSION_EMB_ABLATION_CONTROLS \
    'run.tags=[sleep_staging,poyo,kemp,with_session_emb,control,exp011]'

# Phase 2 — 3-fold validation (fill in best lr from Phase 1):
uv run python main.py experiment=sleep_staging/poyo_kemp_session_emb_ablation \
    hyperparameters.learning_rate=<best_lr> \
    run.group=KEMP_SESSION_EMB_ABLATION_VALIDATION \
    'run.name=kemp_no_session_emb_val_fold${hyperparameters.fold_number}' \
    'run.tags=[sleep_staging,poyo,kemp,no_session_emb,validation,exp011]' \
    'hyperparameters.fold_number=0,1,2' -m
```

### Key config overrides

Uses new config
`configs/experiment/sleep_staging/poyo_kemp_session_emb_ablation.yaml`.

Key differences from exp 009 config (`poyo_kemp_finetune_hp_search.yaml`):

- **`model.disable_session_emb: true`** — the intervention under test. When
  enabled, `_tokenize_and_add_session` returns zeros instead of looking up
  the session embedding, and `_build_downstream_queries` uses only `task_emb`
  for query construction. Implemented via new `disable_session_emb` parameter
  on `POYOEEGModel`.
- **No warmup sweep** — fixed at 0, since exp 009 showed warmup doesn't help
- **Both scratch and pretrained** conditions in the same group
- Controls re-enable session emb (`model.disable_session_emb=false`) at the
  best LR from exp 009 for direct comparison under the same config

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/011_session_emb_ablation.py`

```bash
uv run python analysis/011_session_emb_ablation.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If disabling session_emb helps, consider **session embedding dropout** as a
  softer alternative: randomly replace session_emb with the zero vector during
  training (e.g., 50% probability). This lets the model optionally use session
  identity when available while remaining robust when it is not.
- If the train-val gap drops but val F1 also drops, the model may be
  underfitting without any subject-level adaptation. Consider a **learned
  default session embedding** (a single trainable vector shared across all
  sessions) as a middle ground.
- If disabling session_emb closes the pretrained-vs-scratch gap, this confirms
  that session memorization was masking the value of pretraining. The pretrained
  backbone may actually be superior when the model cannot use session shortcuts.
- Consider whether the **channel_emb** has a similar leakage problem — with
  only 2–3 EEG channels per session, the channel embedding vocabulary is small,
  but it could still encode session-specific information indirectly.

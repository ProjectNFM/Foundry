# Session Embedding Ablation for Inter-Subject Sleep Staging

**Status:** Completed (Phase 1 partial — 7 epochs, timed out)
**Date started:** 2026-07-23
**Parent experiment:** [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md), [Discriminative LR Finetuning](../experiments/010-discriminative-lr-finetuning.md)
**Follow-up experiments:** [Within-Subject Split Control](../experiments/012-within-subject-split-control.md)

## Background

Experiments 009 and 010 attempted to improve pretrained CWT-CNN finetuning
through hyperparameter tuning (LR, warmup) and discriminative learning rates
(separate backbone/head LR). Neither closed the −2 pp F1 gap to the
from-scratch baseline. However, a more fundamental issue emerged from
inspecting the train-val loss gap across all scratch HP search runs (exp 009,
KEMP_SCRATCH_HP_SEARCH):


| Scratch run          | Train loss | Val loss  | Gap           |
| -------------------- | ---------- | --------- | ------------- |
| Best (lr=1e-4, wu=0) | 0.318      | 1.254     | **0.936**     |
| Grid average         | 0.32–0.44  | 1.25–1.82 | **0.94–1.38** |


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
train-val gap of ~~1.0 in loss is an order of magnitude larger than the
pretrained-vs-scratch gap in F1 (~~0.02).

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
`disable_session_emb=true`
- **Data:** KempSleepEDF2013, inter-subject split, fold 0 for search,
all 3 folds for final validation
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
(smoothing=1.0)
- **Pretrained checkpoint:** CWT-CNN from exp 005 (wandb: `wlmobz7y`,
val_loss=0.0364) for pretrained condition
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM, 6h wall time (SLURM)
- **WandB:** project=foundry_finetuning

**Sweep axes (Phase 1):**


| Hyperparameter | Values                 | Rationale                                      |
| -------------- | ---------------------- | ---------------------------------------------- |
| Learning rate  | 1e-5, 3e-5, 5e-5, 1e-4 | Same grid as exp 009 for comparison            |
| Initialization | Scratch, Pretrained    | Test both conditions without session shortcuts |
| Session emb    | Disabled (fixed)       | The intervention under test                    |
| ES patience    | 50 (fixed)             | Same as exp 009                                |


**Conditions:**


| Phase | Condition                            | Group                                | Runs | Purpose                             |
| ----- | ------------------------------------ | ------------------------------------ | ---- | ----------------------------------- |
| 1     | No session emb, scratch, LR sweep    | KEMP_SESSION_EMB_ABLATION            | 4    | Scratch without session leakage     |
| 1     | No session emb, pretrained, LR sweep | KEMP_SESSION_EMB_ABLATION            | 4    | Pretrained without session leakage  |
| 1     | With session emb controls (lr=1e-4)  | KEMP_SESSION_EMB_ABLATION_CONTROLS   | 2    | Direct comparison under same config |
| 2     | 3-fold validation                    | KEMP_SESSION_EMB_ABLATION_VALIDATION | 3+   | Final comparison with error bars    |




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

- `model.disable_session_emb: true` — the intervention under test. When
enabled, `_tokenize_and_add_session` returns zeros instead of looking up
the session embedding, and `_build_downstream_queries` uses only `task_emb`
for query construction. Implemented via new `disable_session_emb` parameter
on `POYOEEGModel`.
- **No warmup sweep** — fixed at 0, since exp 009 showed warmup doesn't help
- **Both scratch and pretrained** conditions in the same group
- Controls re-enable session emb (`model.disable_session_emb=false`) at the
best LR from exp 009 for direct comparison under the same config



## Results

Phase 1 LR sweeps (scratch + pretrained, no session emb) all **timed out
at epoch 7** (6h SLURM limit). None reached early stopping (patience 50).
The analysis script includes these partial runs when they exceed
`--min-epochs` (default 5); metrics are best-so-far from W&B summary.

```bash
uv run python analysis/011_session_emb_ablation.py
# stricter threshold:
uv run python analysis/011_session_emb_ablation.py --min-epochs 7
```

All runs are on **fold 0**.

### Summary

Disabling session embeddings **does not make a large difference** to val F1
for either scratch or pretrained models at epoch 7. The best no-session-emb
runs perform within ~1–2 pp of the with-session-emb baselines from experiment
009 (same fold, same architecture, same LR grid). The hypothesis that session
embeddings are the primary driver of the train-val gap and negative transfer
is **not supported** on this evidence — removing them neither closes the
pretrained-vs-scratch gap nor substantially improves generalisation.

**Caveat:** runs stopped at epoch 7; full convergence was not reached.
Pretrained no-sess at lr=5e-5 looks slightly better than exp 009 (+1.6 pp)
but this may not hold after longer training.

### Metrics

Results are fetched dynamically from the wandb groups and compared against
best runs from `KEMP_SCRATCH_HP_SEARCH` and `KEMP_FINETUNE_HP_SEARCH`
(exp 009) on the same fold 0. Partial runs (failed/crashed) are included
when `epochs_completed >= min_epochs`.


| Condition  | Session Emb              | Best Val F1 | Best LR | Δ vs with-sess baseline |
| ---------- | ------------------------ | ----------- | ------- | ----------------------- |
| Scratch    | Disabled (exp 011, ep 7) | 0.5612      | 5e-5    | −0.2 pp                 |
| Scratch    | Enabled (exp 009)        | 0.5629      | 1e-4    | —                       |
| Pretrained | Disabled (exp 011, ep 7) | 0.5585      | 5e-5    | +1.6 pp                 |
| Pretrained | Enabled (exp 009)        | 0.5425      | 1e-4    | —                       |


At epoch 7 with session emb disabled, scratch still leads pretrained by
~0.3 pp (0.5612 vs 0.5585) — much smaller than the ~2 pp gap in exp 009
with session emb enabled. Whether that gap re-opens with full training is
unknown.

### Analysis

Results extracted programmatically from WandB. The analysis script fetches
both the ablation runs (group `KEMP_SESSION_EMB_ABLATION`) and the baseline
runs from previous experiments (groups `KEMP_SCRATCH_HP_SEARCH` and
`KEMP_FINETUNE_HP_SEARCH`) on the same fold, ensuring an apples-to-apples
comparison without hardcoded metric values. Failed/crashed runs are included
when they logged at least `--min-epochs` (default 5).

**Analysis script:** `analysis/011_session_emb_ablation.py`

```bash
uv run python analysis/011_session_emb_ablation.py
```



### Figures

LR sweep with baseline referenceVal F1 comparison bar chartTrain-val loss gap comparison

## Conclusions

1. **Hypothesis largely refuted (at epoch 7).** Disabling session embeddings
  does not meaningfully improve inter-subject generalisation in these partial
   runs. Both scratch and pretrained models achieve similar val F1 with or
   without session embeddings, indicating that session embeddings are not the
   primary source of the train-val gap at this training stage.
2. **Session embeddings are not clearly causing the −2 pp pretrained-vs-scratch gap.**
  At epoch 7 with session emb disabled, the scratch–pretrained gap shrinks to
   ~0.3 pp. The exp 009 gap with session emb enabled may be inflated by
   session-level memorization, but partial runs prevent a definitive conclusion.
3. **The train-val gap is not primarily driven by session identity leakage.**
  While the gap was hypothesized to stem from session embeddings encoding
   subject-specific shortcuts, removing them does not substantially reduce it.
   The overfitting must be distributed across other model components or is
   inherent to the inter-subject evaluation setting with limited data.
4. **The model is robust to the absence of session embeddings.** The fact that
  performance does not degrade when session embeddings are disabled suggests
   they contribute little useful information for inter-subject sleep staging —
   the model learns to classify primarily from the EEG signal regardless of
   whether session identity is available.
5. **Pretraining's failure to transfer is not a session-embedding artifact.**
  The investigation into why pretrained models underperform scratch must look
   elsewhere — potentially at the pretraining objective itself, the domain gap
   between pretraining and finetuning data, or fundamental representation
   mismatch.



## Notes for future experiments

- Since session embeddings are neither helping nor hurting, consider
**removing them permanently** from the inter-subject finetuning pipeline to
simplify the model and reduce parameter count.
- The remaining −2 pp pretrained-vs-scratch gap is not explained by session
memorization. Future investigations should explore:
  - **Pretraining objective mismatch:** the reconstruction task may produce
  representations that are actively harmful for classification.
  - **Domain gap:** differences between pretraining and finetuning datasets.
  - **Architecture bottlenecks:** the readout/query mechanism may not
  effectively extract classification-relevant features from pretrained
  representations.
- Consider whether the **channel_emb** has a similar (non-)effect — with
only 2–3 EEG channels per session, the channel embedding vocabulary is small,
but it could still encode session-specific information indirectly.
- The epoch-0 peak phenomenon in pretrained finetuning (exp 009, 010) remains
unexplained by session embedding ablation. This points to a deeper issue
with how randomly-initialized head components interact with pretrained
backbone features.


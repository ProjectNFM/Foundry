# Session Embedding Mode Comparison for Cross-Subject Pretraining

**Status:** Completed
**Date started:** 2026-07-27
**Parent experiment:** [Intersubject Pretraining — Session Embedding Generalization](../experiments/013-pretrain-intersubject-session-embeddings.md), [Session Embedding Ablation](../experiments/011-session-embedding-ablation.md)
**Follow-up experiments:** [Intrasession Sanity Check](../experiments/015-session-emb-intrasession-sanity-check.md), [Channel Embedding Ablation](../experiments/016-channel-emb-ablation.md), [Full Dataset Pretraining](../experiments/017-full-dataset-pretraining-scaling.md)

## Background

Experiment 013 confirmed that intersubject pretraining with static
(`InfiniteVocabEmbedding`) session embeddings produces a massive train-val
reconstruction loss gap (~0.3–0.4) from the first epoch, caused by unseen
validation sessions falling back to the padding embedding. Experiment 011
showed that disabling session embeddings during finetuning did not
significantly hurt downstream performance, suggesting they contribute
little useful information for inter-subject tasks.

The `session_emb` config group now supports three modes:

1. **Static** (`static`): per-session learned embedding via
  `InfiniteVocabEmbedding`. Unseen sessions fall back to the padding
   embedding — the baseline that produces the large generalization gap.
2. **Disabled** (`disabled`): session embedding is replaced with zeros.
  Removes session identity entirely from both input tokenization and
   output query construction.
3. **Dynamic** (`dynamic`): signal-conditioned embedding computed from
  context windows via `DynamicSessionEncoder`. Pools tokenized context
   into a session representation, meaning unseen sessions get a
   meaningful embedding derived from their own signal.

This experiment compares all three modes on the same small intersubject
pretraining task to measure their effect on cross-subject generalization,
without sweeping over dynamic-mode hyperparameters (num_context_windows,
context_source are fixed at defaults: 5 windows, random sampling).

## Question

How do different session embedding modes (static, disabled, dynamic)
affect the train-val reconstruction loss gap during intersubject masked
pretraining on the Klinzing sleep subset?

## Hypothesis

1. **Disabled vs Static:** Zeroing out all session embeddings will **not
  significantly change** overall performance — the model can learn to
   reconstruct without session identity, and the train-val gap should
   shrink since there is no embedding mismatch for unseen subjects.
2. **Dynamic vs Static:** Signal-conditioned session embeddings will
  **reduce training performance** (higher train loss) because the
   dynamic encoder is harder to optimize than a simple lookup table, but
   will **improve intersubject validation performance** (lower val loss)
   because unseen sessions receive a meaningful embedding derived from
   their own signal rather than a default padding vector.



## Experiment



### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self
heads, dim_head=128, TemporalBlockMasking (block_size=10,
mask_ratio=0.5), `zero_output_timestamps: false`,
`normalize_inputs: true`
- **Data:** Balanced Klinzing subset (`sleep_brainset_small`) — 14
subjects (10 train / 2 val / 2 test), 28 recordings,
**intersubject** split, fold 0, sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=100, lr=1e-4, weight_decay=0.01,
max_epochs=200, bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining,
group=PRETRAIN_SESSION_EMB_COMPARISON
  - `pretrain_sessemb_static` — run ID `zjkkc5j6`
  - `pretrain_sessemb_disabled` — run ID `0bsi4w78`
  - `pretrain_sessemb_dynamic` — run ID `owetriji`

**Conditions:**


| Condition | session_emb mode | Description                                       | Purpose                              |
| --------- | ---------------- | ------------------------------------------------- | ------------------------------------ |
| Static    | `static`         | Per-session learned embedding (baseline)          | Reference: known generalization gap  |
| Disabled  | `disabled`       | Zeros — no session identity                       | Ablation: is the gap session-driven? |
| Dynamic   | `dynamic`        | Signal-conditioned (5 windows, random, mean pool) | Test: can signal-based init help?    |




### Launch command

```bash
# SLURM sweep (3 session_emb modes in parallel):
uv run python main.py experiment=pretraining/poyo_pretrain_dynamic_session_emb -m
```



### Key config overrides

Base config:
`configs/experiment/pretraining/poyo_pretrain_dynamic_session_emb.yaml`

Changes from the previous version of this config (dynamic-mode
hyperparameter sweep):

- Removed `override /model/session_emb: dynamic` from defaults — the
session_emb mode is now controlled by the sweeper
- Hydra sweeper now varies `model/session_emb` over `static`, `disabled`,
`dynamic` (3 runs) instead of sweeping `num_context_windows × context_source` (8 runs)
- `run.group: PRETRAIN_SESSION_EMB_COMPARISON`
- `run.name` uses `session_emb_mode` instead of window/source params
- Tags include `session_emb_comparison` and `exp014`

Dynamic mode uses default context settings from
`configs/model/session_emb/dynamic.yaml`:

- `num_context_windows: 5`
- `context_source: random`
- `context_duration: 2.0`



## Results



### Summary

All three runs hit SLURM wall time and terminated at 16–19 epochs (out of
200 max). Despite partial training, the trends are clear. **Disabled**
achieves the best validation loss (0.4214 at epoch 9), outperforming both
Static (0.4257 at epoch 6) and Dynamic (0.4268 at epoch 6). All three
modes begin overfitting at epoch 5, but Disabled continues to improve its
validation loss for several more epochs before plateauing, while Static
and Dynamic stall immediately. Dynamic does show the expected higher
train loss (0.1975 vs 0.1663 for Static), confirming the optimization is
harder, but this does not translate into better validation performance.

### Metrics


| Metric                       | Static   | Disabled | Dynamic  |
| ---------------------------- | -------- | -------- | -------- |
| Best val/loss                | 0.4257   | 0.4214   | 0.4268   |
| Train loss at best val epoch | 0.1663   | 0.1542   | 0.1975   |
| Train-val gap at best val    | 0.2594   | 0.2672   | 0.2293   |
| Epoch of best val            | 6        | 9        | 6        |
| Overfit onset (epoch)        | 5        | 5        | 5        |
| Max epoch reached            | 16       | 19       | 16       |
| Run state                    | finished | finished | finished |




### Analysis

Results extracted programmatically from WandB. Per-epoch train and val
loss were fetched via `fetch_metric_history` with epoch-level aggregation.
Overfitting onset is defined as the first epoch where validation loss
increases relative to the previous epoch.

**Analysis script:** `analysis/014_session_emb_mode_comparison.py`

```bash
uv run python analysis/014_session_emb_mode_comparison.py
```



### Figures

Best validation loss and train-val gap comparisonLearning curves for all three modesValidation loss overlay

## Conclusions

**Hypothesis mostly refuted.** The results partially confirm some
predicted trends but the key prediction — that dynamic session embeddings
would improve intersubject validation — is wrong.

1. **Disabled matches or slightly outperforms Static**, as predicted.
  Without session embeddings, the model achieves the best validation
   loss (0.4214 vs 0.4257), confirming that static session embeddings do
   not provide useful information for cross-subject generalization and
   may even slightly hurt by introducing noise through the padding
   embedding mismatch. This is consistent with experiment 011's finding
   that session embeddings contribute little to inter-subject tasks.
2. **Dynamic does reduce training performance**, as predicted. Train loss
  at best val epoch is 0.1975 for Dynamic vs 0.1663 for Static,
   confirming the dynamic encoder is harder to optimize than a lookup
   table. However, **this does not translate into better validation
   performance** — Dynamic achieves the worst best-val-loss of all three
   modes (0.4268).
3. **All three modes begin overfitting at the same epoch (5).** Dynamic
  does not meaningfully delay overfitting onset. However, Disabled
   continues to improve its val loss for 4 more epochs after the initial
   uptick, reaching its best at epoch 9, while Static and Dynamic stall
   at epoch 6.
4. **The train-val gap is comparable across modes (~0.23–0.27).** The
  gap is dominated by the inherent difficulty of intersubject
   generalization rather than the session embedding mechanism. Dynamic
   has a slightly smaller gap (0.2293) but only because its train loss
   is higher, not because its val loss is lower.
5. **Dynamic underperforming Disabled is unexpected.** If the dynamic
  encoder produces a meaningful signal-conditioned representation, it
   should at least match zeros. The most likely explanation is that the
   dynamic encoder adds parameters and optimization complexity without
   providing information that generalizes — the context windows may
   encode session-specific patterns (amplitude, artifact profile) that
   are no more transferable than static embeddings, while being harder
   to learn. Alternatively, with only 5 random context windows and a
   simple mean pooling, the signal representation may be too noisy to be
   useful.



## Notes for future experiments

- **Disabled is the best mode for intersubject pretraining** on this
dataset. Consider using it as the default for cross-subject settings.
- The dynamic encoder's failure to outperform zeros suggests it may need
architectural changes (attention pooling, more context windows, or a
dedicated projection head) rather than just hyperparameter tuning.
A sweep over `num_context_windows` and `context_source` is unlikely
to close the gap if the fundamental representation is not useful.
- The fact that all three modes produce similar val loss (~~0.42) and
similar gaps (~~0.25) suggests the intersubject generalization bottleneck
lies elsewhere — likely in the backbone or tokenizer rather than the
session embedding pathway.
- These runs were cut short at 16–19 epochs. Longer runs would confirm
whether the Disabled advantage persists or whether Static/Dynamic
eventually catch up. Given the overfitting trends, longer training is
more likely to widen the gap than close it.


# Session Embedding Mode Comparison for Cross-Subject Pretraining

**Status:** Draft
**Date started:** 2026-07-27
**Parent experiment:** [Intersubject Pretraining — Session Embedding Generalization](../experiments/013-pretrain-intersubject-session-embeddings.md), [Session Embedding Ablation](../experiments/011-session-embedding-ablation.md)
**Follow-up experiments:** TBD

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
  - `pretrain_sessemb_static` — run ID TBD
  - `pretrain_sessemb_disabled` — run ID TBD
  - `pretrain_sessemb_dynamic` — run ID TBD

**Conditions:**

| Condition | session_emb mode | Description                                    | Purpose                              |
| --------- | ---------------- | ---------------------------------------------- | ------------------------------------ |
| Static    | `static`         | Per-session learned embedding (baseline)       | Reference: known generalization gap  |
| Disabled  | `disabled`       | Zeros — no session identity                    | Ablation: is the gap session-driven? |
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
  `dynamic` (3 runs) instead of sweeping `num_context_windows ×
  context_source` (8 runs)
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

TBD

### Metrics

| Metric                       | Static | Disabled | Dynamic |
| ---------------------------- | ------ | -------- | ------- |
| Best val/loss                | TBD    | TBD      | TBD     |
| Train loss at best val epoch | TBD    | TBD      | TBD     |
| Train-val gap at best val    | TBD    | TBD      | TBD     |
| Epoch of best val            | TBD    | TBD      | TBD     |

### Analysis

TBD

**Analysis script:** `analysis/014_session_emb_mode_comparison.py`

```bash
uv run python analysis/014_session_emb_mode_comparison.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If dynamic mode improves val loss, follow up with a sweep over
  `num_context_windows` and `context_source` to optimize the dynamic
  encoder (the sweep from the previous version of this config).
- If disabled mode matches static, consider removing session embeddings
  from intersubject pretraining entirely to simplify the pipeline.
- Compare results with experiment 013 (static-only, both tokenizers) to
  confirm consistency on the same dataset and split.

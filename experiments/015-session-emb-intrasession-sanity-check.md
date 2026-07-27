# Session Embedding Mode Comparison — Intrasession Sanity Check

**Status:** Draft
**Date started:** 2026-07-27
**Parent experiment:** [Session Embedding Mode Comparison](../experiments/014-session-emb-mode-comparison.md)
**Follow-up experiments:** TBD

## Background

Experiment 014 compared static, disabled, and dynamic session embedding
modes under **intersubject** pretraining. Disabled mode achieved the best
validation loss, consistent with the expectation that static per-session
embeddings provide no useful signal for unseen subjects. However, the
differences were small (0.42–0.43 range), raising the question of whether
session embeddings matter at all or whether the intersubject regime simply
masks their contribution.

This experiment repeats the same comparison under **intrasession**
splitting, where train and validation windows come from the same
recordings. In this regime, static session embeddings should have a clear
advantage: the lookup table sees every session during training, so the
embedding can encode session-specific calibration information (amplitude
scale, electrode impedance, artifact profile) that directly helps
reconstruction.

## Question

Does the relative ranking of session embedding modes change when
switching from intersubject to intrasession splitting, confirming that
static embeddings are primarily useful for seen-session reconstruction?

## Hypothesis

1. **Static will clearly outperform Disabled and Dynamic** because every
  validation session was seen during training — the embedding can
   specialise to each session without encountering the padding-embedding
   mismatch that dominated exp 014.
2. **Dynamic may slightly outperform Disabled** because its
  signal-conditioned representation provides useful session-level
   calibration even when the session is seen, while Disabled throws away
   all session identity.
3. The **train-val gap will be smaller** across all modes compared to
  exp 014, since intrasession splitting removes the cross-subject
   generalization bottleneck.



## Experiment



### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self
heads, dim_head=128, TemporalBlockMasking (block_size=10,
mask_ratio=0.5), `zero_output_timestamps: false`,
`normalize_inputs: true`
- **Data:** Balanced Klinzing subset (`sleep_brainset_small`) — 14
subjects, 28 recordings, **intrasession** split, fold 0,
sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=512, lr=1e-4, weight_decay=0.01,
max_epochs=200, bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining,
group=PRETRAIN_SESSION_EMB_INTRASESSION
  - `pretrain_sessemb_intra_static` — run ID `TBD`
  - `pretrain_sessemb_intra_disabled` — run ID `TBD`
  - `pretrain_sessemb_intra_dynamic` — run ID `TBD`

**Conditions:**


| Condition | session_emb mode | split_type   | Purpose                               |
| --------- | ---------------- | ------------ | ------------------------------------- |
| Static    | `static`         | intrasession | Expected best: no embedding mismatch  |
| Disabled  | `disabled`       | intrasession | Ablation: can model do without?       |
| Dynamic   | `dynamic`        | intrasession | Test: does signal-based help on seen? |




### Launch command

```bash
# SLURM sweep (3 session_emb modes, intrasession split):
uv run python main.py experiment=pretraining/poyo_pretrain_dynamic_session_emb \
    data.split_type=intrasession \
    run.group=PRETRAIN_SESSION_EMB_INTRASESSION \
    'run.name=pretrain_sessemb_intra_${model.session_emb.session_emb_mode}' \
    'run.tags=[pretraining,mae,masked,session_emb_comparison,intrasession,exp015]' \
    -m
```



### Key config overrides

Base config:
`configs/experiment/pretraining/poyo_pretrain_dynamic_session_emb.yaml`
(same as exp 014)

Overrides from exp 014:

- `data.split_type: intrasession` (was `intersubject`)
- `run.group: PRETRAIN_SESSION_EMB_INTRASESSION`
- `run.name` includes `intra_` prefix
- Tags include `intrasession` and `exp015` instead of `intersubject`
and `exp014`

The Hydra sweeper in the base config still varies `model/session_emb`
over `static`, `disabled`, `dynamic` (3 runs).

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
| Max epoch reached            | TBD    | TBD      | TBD     |




### Analysis

TBD

**Analysis script:** `analysis/015_session_emb_intrasession.py`

```bash
uv run python analysis/015_session_emb_intrasession.py
```



### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Compare results directly with exp 014 to quantify the contribution of
session embeddings in seen-vs-unseen session regimes.
- If Static dominates as expected, this validates the interpretation that
exp 014's Disabled advantage is driven by embedding mismatch, not by
session identity being inherently useless.


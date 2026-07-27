# Channel Embedding Ablation for Intersubject Pretraining

**Status:** Draft
**Date started:** 2026-07-27
**Parent experiment:** [Session Embedding Mode Comparison](../experiments/014-session-emb-mode-comparison.md)
**Follow-up experiments:** TBD

## Background

Experiment 014 showed that disabling session embeddings slightly improves
intersubject validation loss, confirming that per-session learned
embeddings do not help with cross-subject generalization. However,
channel embeddings are **session-specific by construction**: the Klinzing
dataset (`KempSleepEDF2013`) prefixes every channel ID with the session
ID via `get_recording_hook` (`uniquify_channel_ids=True`), producing IDs
like `"sub-100_task-Sleep_acq-psg/Fpz"` instead of bare `"Fpz"`. The
`OpenNeuroMultiBrainset` wrapper further prepends the brainset name,
yielding `"klinzing_sleep_ds005555/sub-100_task-Sleep_acq-psg/Fpz"`.
Each of these session-scoped strings gets its own learned vector in the
`InfiniteVocabEmbedding` channel lookup table.

This means channel embeddings carry session identity information by
design — there is no weight sharing between the same electrode across
different sessions. When session embeddings are disabled (exp 014
winner), the model could simply shift session-specific calibration
(amplitude scale, noise floor, impedance) into the channel embeddings,
making the session ablation superficially effective while the same
information leaks through a different path. Channel embeddings are fused
into encoder tokens (via concatenation) and into masked-reconstruction
decoder queries.

This experiment adds a `channel_emb_mode` parameter (analogous to
`session_emb_mode`) with three options: `static` (learned per-channel
lookup, current default), `disabled` (zeros), and `dynamic` (not yet
implemented, raises `NotImplementedError`). We sweep over a 2×2 grid of
session × channel embedding modes to disentangle their contributions.

## Question

Given that channel embeddings are session-scoped (each session has its
own set of channel vectors), does disabling session embeddings in
exp 014 actually remove session identity, or does the information simply
migrate into the channel embeddings?

## Hypothesis

1. **Disabling channel embeddings alone** (session=static,
   channel=disabled) will hurt reconstruction quality because the model
   loses the ability to distinguish which channel was masked — critical
   for per-channel reconstruction queries.
2. **The exp 014 "disabled session" advantage may shrink or
   disappear** when channel embeddings are also disabled
   (session=disabled, channel=disabled), because session-specific
   information stored in the session-scoped channel embeddings is no
   longer available to compensate.
3. **If exp 014's disabled-session advantage persists** even when
   channel embeddings are also disabled, it confirms that session
   identity genuinely hurts intersubject generalization and is not
   merely relocated to channel embeddings.

## Experiment

### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self
  heads, dim_head=128, TemporalBlockMasking (block_size=10,
  mask_ratio=0.5), `zero_output_timestamps: false`,
  `normalize_inputs: true`
- **Data:** Balanced Klinzing subset (`sleep_brainset_small`) — 14
  subjects, 28 recordings, **intersubject** split, fold 0,
  sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=512, lr=1e-4, weight_decay=0.01,
  max_epochs=200, bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining,
  group=PRETRAIN_CHANNEL_EMB_ABLATION
  - `pretrain_chemb_sess-static_ch-static` — run ID `TBD`
  - `pretrain_chemb_sess-static_ch-disabled` — run ID `TBD`
  - `pretrain_chemb_sess-disabled_ch-static` — run ID `TBD`
  - `pretrain_chemb_sess-disabled_ch-disabled` — run ID `TBD`

**Conditions:**

| Condition             | session_emb | channel_emb | Purpose                                 |
| --------------------- | ----------- | ----------- | --------------------------------------- |
| sess-static, ch-static    | `static`    | `static`    | Baseline (exp 014 Static)               |
| sess-static, ch-disabled  | `static`    | `disabled`  | Channel ablation with session identity  |
| sess-disabled, ch-static  | `disabled`  | `static`    | Baseline (exp 014 Disabled — best)      |
| sess-disabled, ch-disabled| `disabled`  | `disabled`  | Full identity ablation                  |

### Launch command

```bash
# SLURM sweep (2×2 grid: session_emb × channel_emb_mode):
uv run python main.py experiment=pretraining/poyo_pretrain_dynamic_session_emb \
    'model/session_emb=static,disabled' \
    'model.channel_emb_mode=static,disabled' \
    run.group=PRETRAIN_CHANNEL_EMB_ABLATION \
    'run.name=pretrain_chemb_sess-${model.session_emb.session_emb_mode}_ch-${model.channel_emb_mode}' \
    'run.tags=[pretraining,mae,masked,channel_emb_ablation,intersubject,exp016]' \
    -m
```

### Key config overrides

Base config:
`configs/experiment/pretraining/poyo_pretrain_dynamic_session_emb.yaml`
(same as exp 014)

Overrides:

- Hydra sweeper now varies both `model/session_emb` (static, disabled)
  and `model.channel_emb_mode` (static, disabled) → 4 runs
- `run.group: PRETRAIN_CHANNEL_EMB_ABLATION`
- `run.name` encodes both session and channel mode
- Tags include `channel_emb_ablation` and `exp016`
- `model.channel_emb_mode` is a new model parameter added in this
  experiment (defaults to `static` for backward compatibility)

## Results

### Summary

TBD

### Metrics

| Metric                       | sess-S ch-S | sess-S ch-D | sess-D ch-S | sess-D ch-D |
| ---------------------------- | ----------- | ----------- | ----------- | ----------- |
| Best val/loss                | TBD         | TBD         | TBD         | TBD         |
| Train loss at best val epoch | TBD         | TBD         | TBD         | TBD         |
| Train-val gap at best val    | TBD         | TBD         | TBD         | TBD         |
| Epoch of best val            | TBD         | TBD         | TBD         | TBD         |
| Max epoch reached            | TBD         | TBD         | TBD         | TBD         |

### Analysis

TBD

**Analysis script:** `analysis/016_channel_emb_ablation.py`

```bash
uv run python analysis/016_channel_emb_ablation.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If channel embeddings do absorb session-specific information, consider
  implementing dynamic channel embeddings that compute
  electrode-position representations from signal characteristics rather
  than a learned lookup table.
- Results here inform whether `channel_emb_mode=disabled` should be
  paired with `session_emb_mode=disabled` for maximum generalization
  in experiment 017 (full dataset pretraining).

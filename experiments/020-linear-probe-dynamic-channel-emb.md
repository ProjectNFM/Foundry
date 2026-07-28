# Linear Probing: Dynamic Channel Embeddings

**Status:** Not Started
**Date started:** 2026-07-28
**Parent experiment:** [Dynamic Channel Embedding Analysis](../experiments/019-dynamic-channel-embedding-analysis.md)
**Follow-up experiments:** TBD

## Background

Experiment 019 visualized the backbone and channel embeddings from the dynamic
channel embedding models (exp 018) and found that:

1. The dynamic model's backbone embeddings show more geometric structure than
   the disabled baseline, but this structure does **not** clearly align with
   sleep stage boundaries (partially refuted).
2. The dynamic channel embeddings do **not** cluster by electrode type — both
   channels are entangled along an arch (refuted).
3. Sleep stage labels show band-like organization along the channel embedding
   arch, suggesting the channel encoder captures brain-state-related statistics
   (partially supported).

Visualization alone cannot determine whether the dynamic model's
representations are actually better for downstream sleep staging. A linear
probe test — freezing the backbone and training only a linear classification
head — directly measures how much discriminative information for sleep staging
exists in the pretrained representations, without the confound of finetuning
overwriting features.

This is the same methodology used in experiment 008, which showed that
CWT-CNN pretraining gave a substantial +15 pp F1 advantage in linear probing,
confirming that reconstruction pretraining can learn sleep-stage-relevant
features.

## Question

Does the dynamic channel embedding model (exp 018) produce backbone
representations with more linearly separable sleep stage information than
the disabled baseline or random initialization?

## Hypothesis

The dynamic model's substantially better reconstruction loss (0.11 vs 0.40)
should translate into at least a modest advantage in linear probing F1, even
though the embedding visualizations were not conclusive. If the dynamic model
shows no linear probe advantage over disabled, it would suggest that the
reconstruction improvement comes from the channel encoder capturing low-level
signal structure rather than learning discriminative brain-state features in
the backbone.

## Experiment

### Setup

- **Model:** POYOEEGModel, embed_dim=256, depth=4, ResampleCNN tokenizer
  (same architecture as exp 018)
- **Data:** KempSleepEDF2013, intersubject split, fold 0 (validation set for
  evaluation)
- **Task:** 5-class sleep staging linear probe — freeze backbone, train only
  a linear classification head
- **Conditions:**

| Condition              | channel_emb_mode | Init       | Source checkpoint                      |
| ---------------------- | ---------------- | ---------- | -------------------------------------- |
| random-ch-disabled     | `disabled`       | Random     | None                                   |
| random-ch-dynamic      | `dynamic`        | Random     | None                                   |
| pretrained-ch-disabled | `disabled`       | Pretrained | exp 018 `pretrain_dynch_ch-disabled`   |
| pretrained-ch-dynamic  | `dynamic`        | Pretrained | exp 018 `pretrain_dynch_ch-dynamic`    |

- **Training:** lr=1e-3, batch_size=512, max_epochs=100, early stopping on
  val F1 (patience=10), only the linear head is trainable
- **Hardware:** 1× GPU (linear probing is lightweight)

### Launch command

```bash
# --- SLURM: 4 jobs total (2 pretrained + 2 random) ---

# Pretrained backbone conditions (2 jobs: ch-disabled, ch-dynamic):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe_dynch \
    run.init_mode=pretrained -m

# Random-init backbone conditions (2 jobs: ch-disabled, ch-dynamic):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe_dynch \
    run.init_mode=random run.pretrained_checkpoint=null -m
```

Each command submits 2 SLURM jobs via Hydra multirun, sweeping
`model.channel_emb_mode` over `disabled` and `dynamic`. The experiment config
(`poyo_kemp_linear_probe_dynch`) selects the matching exp 018 checkpoint per
channel mode for the pretrained runs.

### Key config overrides

- Base experiment: `sleep_staging/poyo_kemp_linear_probe_dynch` (dedicated exp
  020 config; extends the exp 008 linear probe setup)
- `model/tokenizer=per_channel_resample_cnn` (matches exp 018 architecture)
- `model/session_emb=disabled` (matches exp 018 — no session embeddings)
- `model.channel_emb_mode` swept over `disabled` / `dynamic`
- `run.pretrained_transfer_mode=permissive` (checkpoint from
  MaskedPOYOEEGModel loaded into POYOEEGModel; non-matching keys skipped)
- `run.freeze_pretrained=true` / `run.freeze_backbone=true` (freeze backbone
  for linear probing)
- Single fold (fold 0) only

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If the dynamic pretrained backbone shows a clear advantage, consider
  finetuning with progressive unfreezing to avoid catastrophic forgetting.
- If neither pretrained condition outperforms random, the reconstruction
  objective may need rethinking (e.g., adding a contrastive or classification
  auxiliary loss during pretraining).
- Compare with the CWT-CNN linear probe results from experiment 008 to
  assess whether the tokenizer or the channel embedding mode matters more.

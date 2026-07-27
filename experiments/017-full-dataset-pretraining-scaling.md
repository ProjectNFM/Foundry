# Full Dataset Pretraining — Embedding Mode Scaling

**Status:** Draft
**Date started:** 2026-07-27
**Parent experiment:** [Session Embedding Mode Comparison](../experiments/014-session-emb-mode-comparison.md), [Channel Embedding Ablation](../experiments/016-channel-emb-ablation.md)
**Follow-up experiments:** TBD

## Background

Experiments 014 and 016 tested session and channel embedding ablations on
the balanced Klinzing **subset** (`sleep_brainset_small`, 14 subjects, 28
recordings). Key findings:
- Disabled session embeddings slightly outperform static for intersubject
  pretraining (exp 014).
- Channel embedding ablation results (exp 016) will reveal whether
  channel embeddings absorb session-specific information.

However, these results are from a small dataset with short training runs
(16–19 epochs before SLURM wall time). It is unclear whether the same
trends hold at scale — with more subjects, more data diversity, and
longer training:
- Static session embeddings might eventually learn generalizable
  features with enough subjects.
- The channel embedding contribution might change with more diverse
  electrode montages across the full dataset.
- Overfitting dynamics may differ substantially.

This experiment scales the best configurations from exp 014 and 016 to
the **full** Klinzing brainset (`sleep_brainset`) to validate whether
the small-subset findings transfer to realistic pretraining conditions.

## Question

Do the relative rankings of session/channel embedding configurations
established on the small Klinzing subset hold when pretraining on the
full dataset with longer training?

## Hypothesis

1. **Disabled session embeddings will still outperform static** on the
   full dataset, because the embedding mismatch problem is structural
   (unseen sessions always get the padding embedding) and does not
   resolve with more data.
2. **Channel embedding trends will be amplified**: if channel embeddings
   help on the subset, they should help more on the full dataset where
   electrode montage diversity is greater.
3. **Longer training will widen the gap** between configurations,
   because overfitting dynamics differ: modes with fewer learnable
   embeddings should overfit more slowly.

## Experiment

### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self
  heads, dim_head=128, TemporalBlockMasking (block_size=10,
  mask_ratio=0.5), `zero_output_timestamps: false`,
  `normalize_inputs: true`
- **Data:** Full Klinzing brainset (`sleep_brainset`) — all subjects,
  **intersubject** split, fold 0, sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=512, lr=1e-4, weight_decay=0.01,
  max_epochs=200, bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining,
  group=PRETRAIN_FULL_DATASET_SCALING
  - Run names and IDs TBD (depends on exp 014 + 016 results)

**Conditions:**

The exact conditions will be determined by results from experiments 014
and 016. The planned sweep covers the key configurations:

| Condition                  | session_emb | channel_emb | Rationale                     |
| -------------------------- | ----------- | ----------- | ----------------------------- |
| sess-disabled, ch-static   | `disabled`  | `static`    | Exp 014 winner                |
| sess-static, ch-static     | `static`    | `static`    | Reference baseline            |
| sess-disabled, ch-disabled | `disabled`  | `disabled`  | Full identity ablation        |
| (optional more from 016)   | ...         | ...         | Best config from exp 016      |

### Launch command

```bash
# SLURM sweep on full dataset (adjust conditions based on exp 014/016 results):
uv run python main.py experiment=pretraining/poyo_pretrain_dynamic_session_emb \
    data=openneuro/sleep_brainset \
    data.split_type=intersubject \
    data.task_type=null \
    data.pin_memory=false \
    'model/session_emb=static,disabled' \
    'model.channel_emb_mode=static,disabled' \
    run.group=PRETRAIN_FULL_DATASET_SCALING \
    'run.name=pretrain_full_sess-${model.session_emb.session_emb_mode}_ch-${model.channel_emb_mode}' \
    'run.tags=[pretraining,mae,masked,full_dataset,scaling,intersubject,exp017]' \
    -m
```

### Key config overrides

Base config:
`configs/experiment/pretraining/poyo_pretrain_dynamic_session_emb.yaml`
(same as exp 014)

Overrides:

- `data=openneuro/sleep_brainset` (was `openneuro/sleep_brainset_small`)
  — uses the full Klinzing brainset with all subjects
- Hydra sweeper varies `model/session_emb` and `model.channel_emb_mode`
  — exact grid TBD based on exp 014/016 results
- `run.group: PRETRAIN_FULL_DATASET_SCALING`
- Tags include `full_dataset`, `scaling`, and `exp017`

## Results

### Summary

TBD

### Metrics

TBD (metrics table will be filled after runs complete)

### Analysis

TBD

**Analysis script:** `analysis/017_full_dataset_scaling.py`

```bash
uv run python analysis/017_full_dataset_scaling.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If full-dataset results confirm the small-subset findings, the best
  embedding configuration can be adopted as the default for all future
  intersubject pretraining.
- Consider downstream finetuning evaluation: does the best pretraining
  configuration also produce the best pretrained weights for sleep
  staging or other downstream tasks?
- With the full dataset, training may need longer wall time or
  checkpoint-based resumption to reach convergence.

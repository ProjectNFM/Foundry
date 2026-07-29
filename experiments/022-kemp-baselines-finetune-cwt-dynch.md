# KempSleep Baselines and Finetuning: CWT-CNN with Dynamic Channel Embeddings

**Status:** Draft
**Date started:** 2026-07-29
**Parent experiment:** [CWT CNN with Dynamic Channel Embeddings](../experiments/021-cwt-cnn-dynamic-channel-emb.md)
**Follow-up experiments:** TBD

## Background

Experiment 021 pretrained CWT-CNN models with dynamic and disabled channel
embeddings (session embeddings disabled for both). The linear probe results
from experiment 020 (ResampleCNN) showed that the dynamic channel embedding
backbone carries substantially more linearly separable sleep stage information
(+7.3 pp F1 over disabled). Experiment 021 Phase 2 will repeat the linear
probe for CWT-CNN (pending).

However, linear probing only measures what the frozen backbone already
encodes. Full finetuning (unfreezing all parameters) allows the model to
adapt its representations to the downstream task. This experiment establishes:

1. **Scratch baselines** — training from random initialization on KempSleep
   with CWT-CNN + disabled/dynamic channel embeddings and session embeddings
   disabled. These baselines isolate the architectural effect of the
   `RelativeChannelEncoder` without any pretraining benefit.
2. **Finetuned models** — initializing from the exp 021 pretrained checkpoints
   and finetuning all parameters on KempSleep. This measures how much the
   pretrained representations transfer to downstream sleep staging when the
   model is free to adapt.

Together, the 2×2 grid (scratch vs finetuned × disabled vs dynamic) separates
the contribution of the channel embedding architecture from the pretraining
initialization.

## Question

Does finetuning from exp 021 CWT-CNN pretrained checkpoints (with dynamic
and disabled channel embeddings) improve KempSleep 5-class sleep staging F1
over training from scratch, and does the dynamic channel embedding maintain
its advantage over disabled in the finetuned setting?

## Hypothesis

1. **Finetuning will outperform scratch** for both channel embedding modes,
   since the pretrained backbone has already learned useful EEG representations
   from the Klinzing reconstruction task.
2. **Dynamic will outperform disabled** in both scratch and finetuned
   conditions. In scratch, the `RelativeChannelEncoder` provides an
   architectural inductive bias (as shown by the +4.1 pp random-init advantage
   in exp 020). In finetuned, the dynamic model's better pretraining (lower
   reconstruction loss) should give a stronger starting point.
3. **The finetuned-dynamic condition will achieve the best overall F1**, as it
   combines the best architecture (dynamic channel embeddings) with the best
   initialization (CWT-CNN pretraining).

## Experiment

### Setup

- **Model:** POYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, **CWT-CNN tokenizer** (`per_channel_cwt_cnn`),
  `zero_output_timestamps: true`, `normalize_inputs: true`
- **Channel embeddings:** `channel_emb_mode` swept over `disabled` / `dynamic`
- **Session embeddings:** `disabled` for all conditions
- **Data:** KempSleepEDF2013 (`kemp_sleep_edf/allsess`), intersubject split,
  fold 0
- **Task:** 5-class sleep staging, class-weighted cross-entropy (auto weights,
  smoothing=1.0)
- **Training:** lr=1e-4, weight_decay=0.01, batch_size=512,
  sequence_length=2.0s, max_epochs=1000, early stopping on val F1
  (patience=20), bf16-mixed precision
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=KEMP_FINETUNE_CWT_DYNCH

**Conditions:**

| Condition            | channel_emb_mode | Init       | Pretrained checkpoint          |
| -------------------- | ---------------- | ---------- | ------------------------------ |
| scratch-ch-disabled  | `disabled`       | Scratch    | None                           |
| scratch-ch-dynamic   | `dynamic`        | Scratch    | None                           |
| finetune-ch-disabled | `disabled`       | Finetuned  | exp 021 `pretrain_cwt_dynch_ch-disabled` |
| finetune-ch-dynamic  | `dynamic`        | Finetuned  | exp 021 `pretrain_cwt_dynch_ch-dynamic`  |

### Launch command

```bash
# --- Scratch baselines (2 SLURM jobs: ch-disabled, ch-dynamic) ---
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_cwt_dynch \
    run.init_mode=scratch run.pretrained_checkpoint=null -m

# --- Finetuning from exp 021 pretrained (2 SLURM jobs: ch-disabled, ch-dynamic) ---
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_cwt_dynch \
    run.init_mode=finetuned -m
```

Each command submits 2 SLURM jobs via Hydra multirun, sweeping
`model.channel_emb_mode` over `disabled` and `dynamic`.

### Key config overrides

- Base experiment config:
  `configs/experiment/sleep_staging/poyo_kemp_finetune_cwt_dynch.yaml`
  (new config for exp 022)
- Modeled after `poyo_kemp_linear_probe_cwt_dynch.yaml` (exp 021 Phase 2) but
  with `freeze_pretrained: false` / `freeze_backbone: false` (not frozen) for
  full finetuning
- Scratch runs override: `run.pretrained_checkpoint=null`, `run.init_mode=scratch`
- Finetuned runs use default config: checkpoint auto-selected per
  `channel_emb_mode` from the `pretrained_checkpoints` map
- `pretrained_transfer_mode: permissive` (MaskedPOYOEEGModel → POYOEEGModel;
  non-matching keys like the reconstruction head are skipped)

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/022_kemp_baselines_finetune_cwt_dynch.py`

```bash
uv run python analysis/022_kemp_baselines_finetune_cwt_dynch.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If finetuned-dynamic achieves strong results, consider multi-fold validation
  (folds 0, 1, 2) to confirm robustness.
- Compare against the exp 008/009 finetuning results (CWT-CNN pretrained from
  exp 005 without dynamic channel embeddings) to quantify the additive benefit
  of the dynamic channel embedding.
- If the scratch-dynamic baseline is competitive with finetuned-disabled, it
  would suggest the `RelativeChannelEncoder` architecture alone provides most
  of the benefit, reducing the need for pretraining.
- Consider discriminative learning rates (lower LR for pretrained backbone,
  higher for head) if finetuning shows signs of catastrophic forgetting
  (as observed in exp 009).

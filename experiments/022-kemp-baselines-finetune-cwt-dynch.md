# KempSleep Baselines and Finetuning: Dynamic Channel Embeddings × Tokenizer

**Status:** Draft
**Date started:** 2026-07-29
**Parent experiment:** [CWT CNN with Dynamic Channel Embeddings](../experiments/021-cwt-cnn-dynamic-channel-emb.md)
**Follow-up experiments:** [KempSleep 30s-Epoch From-Scratch Baselines](../experiments/023-kemp-30s-baselines.md)

## Background

Experiment 021 pretrained CWT-CNN models with dynamic and disabled channel
embeddings (session embeddings disabled for both). Experiment 018 did the same
with ResampleCNN. The linear probe results from experiment 020 (ResampleCNN)
showed that the dynamic channel embedding backbone carries substantially more
linearly separable sleep stage information (+7.3 pp F1 over disabled).
Experiment 021 Phase 2 will repeat the linear probe for CWT-CNN (pending).

However, linear probing only measures what the frozen backbone already
encodes. Full finetuning (unfreezing all parameters) allows the model to
adapt its representations to the downstream task. This experiment establishes:

1. **Scratch baselines** — training from random initialization on KempSleep
   with disabled/dynamic channel embeddings and session embeddings disabled,
   for both CWT-CNN and ResampleCNN tokenizers. These baselines isolate the
   architectural effect of the `RelativeChannelEncoder` without any
   pretraining benefit.
2. **Finetuned models** — initializing from the pretrained checkpoints
   (exp 021 for CWT-CNN, exp 018 for ResampleCNN) and finetuning all
   parameters on KempSleep. This measures how much the pretrained
   representations transfer to downstream sleep staging when the model is
   free to adapt.

The full 2×2×2 grid (tokenizer × channel_emb_mode × init) separates the
contributions of the tokenizer architecture, channel embedding mechanism,
and pretraining initialization.

## Question

Across both CWT-CNN and ResampleCNN tokenizers:
1. Does finetuning from pretrained checkpoints improve KempSleep 5-class
   sleep staging F1 over training from scratch?
2. Does the dynamic channel embedding maintain its advantage over disabled
   in the finetuned setting?
3. Does CWT-CNN outperform ResampleCNN, and does the tokenizer interact with
   the channel embedding mode?

## Hypothesis

1. **Finetuning will outperform scratch** for both tokenizers and both channel
   embedding modes, since the pretrained backbones have already learned useful
   EEG representations from the Klinzing reconstruction task.
2. **Dynamic will outperform disabled** in all conditions. In scratch, the
   `RelativeChannelEncoder` provides an architectural inductive bias (as shown
   by the +4.1 pp random-init advantage in exp 020). In finetuned, the dynamic
   model's better pretraining should give a stronger starting point.
3. **CWT-CNN will outperform ResampleCNN** in the finetuned conditions,
   consistent with exp 008's finding that CWT-CNN pretraining captures more
   sleep-stage-relevant features.
4. **The finetuned CWT-CNN dynamic condition will achieve the best overall
   F1**, combining the best tokenizer with the best channel embedding and
   pretrained initialization.

## Experiment

### Setup

- **Model:** POYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, `zero_output_timestamps: true`, `normalize_inputs: true`
- **Tokenizer:** swept over `per_channel_cwt_cnn` / `per_channel_resample_cnn`
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

**Conditions (8 total):**

| Condition                      | Tokenizer      | channel_emb_mode | Init      | Pretrained checkpoint                     |
| ------------------------------ | -------------- | ---------------- | --------- | ----------------------------------------- |
| scratch-cwt-ch-disabled        | CWT-CNN        | `disabled`       | Scratch   | None                                      |
| scratch-cwt-ch-dynamic         | CWT-CNN        | `dynamic`        | Scratch   | None                                      |
| scratch-rcnn-ch-disabled       | ResampleCNN    | `disabled`       | Scratch   | None                                      |
| scratch-rcnn-ch-dynamic        | ResampleCNN    | `dynamic`        | Scratch   | None                                      |
| finetune-cwt-ch-disabled       | CWT-CNN        | `disabled`       | Finetuned | exp 021 `pretrain_cwt_dynch_ch-disabled`  |
| finetune-cwt-ch-dynamic        | CWT-CNN        | `dynamic`        | Finetuned | exp 021 `pretrain_cwt_dynch_ch-dynamic`   |
| finetune-rcnn-ch-disabled      | ResampleCNN    | `disabled`       | Finetuned | exp 018 `pretrain_dynch_ch-disabled`      |
| finetune-rcnn-ch-dynamic       | ResampleCNN    | `dynamic`        | Finetuned | exp 018 `pretrain_dynch_ch-dynamic`       |

### Launch command

```bash
# --- Scratch baselines (4 SLURM jobs: 2 tokenizers × 2 channel_emb_modes) ---
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_cwt_dynch \
    run.init_mode=scratch run.pretrained_checkpoint=null -m

# --- Finetuning from pretrained (4 SLURM jobs: 2 tokenizers × 2 channel_emb_modes) ---
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_cwt_dynch \
    run.init_mode=finetuned -m
```

Each command submits 4 SLURM jobs via Hydra multirun, sweeping
`model/tokenizer` over `per_channel_cwt_cnn` / `per_channel_resample_cnn` and
`model.channel_emb_mode` over `disabled` / `dynamic`.

### Key config overrides

- Base experiment config:
  `configs/experiment/sleep_staging/poyo_kemp_finetune_cwt_dynch.yaml`
  (new config for exp 022)
- Checkpoint map keyed by `[tokenizer][channel_emb_mode]`:
  CWT-CNN checkpoints from exp 021, ResampleCNN checkpoints from exp 018
- Scratch runs override: `run.pretrained_checkpoint=null`, `run.init_mode=scratch`
- Finetuned runs use default config: checkpoint auto-selected per tokenizer
  and `channel_emb_mode` from the nested `pretrained_checkpoints` map
- `pretrained_transfer_mode: permissive` (MaskedPOYOEEGModel → POYOEEGModel;
  non-matching keys like the reconstruction head are skipped)
- `model/session_emb=disabled` — session embeddings disabled for all conditions

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

- If finetuned-dynamic achieves strong results for either tokenizer, consider
  multi-fold validation (folds 0, 1, 2) to confirm robustness.
- Compare against the exp 008/009 finetuning results (CWT-CNN pretrained from
  exp 005 without dynamic channel embeddings) to quantify the additive benefit
  of the dynamic channel embedding.
- If the scratch-dynamic baseline is competitive with finetuned-disabled, it
  would suggest the `RelativeChannelEncoder` architecture alone provides most
  of the benefit, reducing the need for pretraining.
- Consider discriminative learning rates (lower LR for pretrained backbone,
  higher for head) if finetuning shows signs of catastrophic forgetting
  (as observed in exp 009).
- The tokenizer × channel_emb interaction will reveal whether CWT-CNN's
  frequency decomposition is redundant with or complementary to the dynamic
  channel encoder's statistical features.

# Pretraining Loss vs Downstream Task Performance

**Status:** Draft
**Date started:** 2026-07-17
**Parent experiment:** [Tokenizer Architecture Comparison for Masked Pretraining](../experiments/005-tokenizer-comparison.md), [Kemp Sleep EDF — Tokenizer Baseline Comparison](../experiments/006-kemp-sleep-tokenizer-baseline.md)
**Follow-up experiments:** TBD

## Background

Experiments 005 and 006 revealed a surprising discrepancy between pretraining
and downstream performance across tokenizer architectures:

- **Pretraining (exp 005):** CWT-CNN achieved dramatically lower reconstruction
  loss (0.0364) compared to ResampleCNN (0.1201) — a 3.3× advantage.
- **Downstream from scratch (exp 006):** ResampleCNN achieved higher sleep
  staging F1 (0.573) than CWT-CNN (0.554) when trained from scratch.

This raises a fundamental question: does the quality of learned representations
during self-supervised pretraining (as measured by reconstruction loss) actually
translate to better downstream task performance when those representations are
used as initialization for finetuning?

This experiment bridges the gap by taking the pretrained checkpoints from
experiment 005 (one per tokenizer) and finetuning them on the Kemp Sleep
staging task from experiment 006. By comparing finetuned performance against
the from-scratch baselines, we can assess whether pretraining provides a
meaningful advantage, and whether the tokenizer that pretrains better also
finetunes better.

A new pretrain-to-finetune weight loading pipeline was implemented to support
this experiment (`foundry/training/pretrained.py`), transferring the shared
backbone, tokenizer, rotary embeddings, and latent embeddings while
re-initializing dataset-specific components (session/channel embeddings, task
heads).

## Question

Does a better pretraining loss correspond to better downstream task
performance when finetuning on Kemp Sleep 5-class staging?

## Hypothesis

Yes — CWT-CNN's substantially lower pretraining loss (0.0364 vs 0.1201)
indicates it has learned richer EEG representations, and these will transfer
to yield higher downstream F1 than ResampleCNN when finetuned. Specifically:

1. Both finetuned models will outperform their from-scratch baselines (exp 006).
2. Finetuned CWT-CNN will outperform finetuned ResampleCNN, reversing the
   from-scratch ranking where ResampleCNN led.

## Experiment

### Setup

- **Model:** POYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, ffn_dropout=0.2, lin_dropout=0.4, atn_dropout=0.2,
  16 latents per 0.1s step, zero_output_timestamps=true, normalize_inputs=true
- **Data:** KempSleepEDF2013, inter-subject split, 3 folds,
  sequence_length=2.0s
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
  (smoothing=1.0)
- **Training:** batch_size=512, lr=1e-4, weight_decay=0.01, max_epochs=1000,
  bf16-mixed precision, early stopping on val/sleep_stage_5class_f1
  (patience=20, mode=max), full finetuning (all parameters trainable)
- **Pretrained checkpoints (from exp 005):**
  - ResampleCNN: `pretrain_tokenizer_per_channel_resample_cnn` (wandb: `vup5m7er`)
  - CWT-CNN: `pretrain_tokenizer_per_channel_cwt_cnn` (wandb: `wlmobz7y`)
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=KEMP_FINETUNE_FROM_PRETRAIN

### Launch command

```bash
# Full sweep (2 tokenizers × 3 folds = 6 runs):
uv run python main.py experiment=sleep_staging/poyo_kemp_finetune_from_pretrain -m
```

### Key config overrides

Uses the new config
`configs/experiment/sleep_staging/poyo_kemp_finetune_from_pretrain.yaml`.

Key differences from the from-scratch baseline
(`poyo_kemp_allsess_tokenizer_sweep.yaml`):
- `pretrained_checkpoints` map: automatically selects the correct exp 005 best
  checkpoint for each tokenizer via
  `${pretrained_checkpoints.${hydra:runtime.choices.model/tokenizer}}`
- `run.freeze_pretrained: false` (full finetuning)
- Sweep over `model/tokenizer: per_channel_resample_cnn, per_channel_cwt_cnn`
  and `hyperparameters.fold_number: 0, 1, 2`

All other hyperparameters match experiment 006 exactly (same lr, batch_size,
early stopping, class weights) to ensure a fair comparison.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/007_pretraining_loss_vs_downstream.py`

```bash
uv run python analysis/007_pretraining_loss_vs_downstream.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If finetuning helps, test whether freezing the backbone
  (`run.freeze_pretrained=true`) and only training the task head (linear
  probing) also improves over from-scratch — this would confirm the
  representations themselves are better, not just the initialization.
- Compare convergence speed: finetuned models may reach peak F1 in fewer
  epochs than from-scratch, even if final performance is similar.
- Test with longer pretraining (exp 005 was time-limited to ~4 epochs) — more
  pretraining may amplify or diminish the gap.
- If CWT-CNN pretraining advantage does not transfer, investigate whether the
  reconstruction objective optimizes for features that are not discriminative
  for sleep staging (e.g., high-frequency detail that aids reconstruction but
  not classification).

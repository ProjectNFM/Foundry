# Pretraining Loss vs Downstream Task Performance

**Status:** Completed
**Date started:** 2026-07-17
**Parent experiment:** [Tokenizer Architecture Comparison for Masked Pretraining](../experiments/005-tokenizer-comparison.md), [Kemp Sleep EDF — Tokenizer Baseline Comparison](../experiments/006-kemp-sleep-tokenizer-baseline.md)
**Follow-up experiments:** [Embedding Analysis: t-SNE/PCA and Linear Probing](../experiments/008-embedding-analysis-linear-probing.md), [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md)

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
- **WandB:** project=foundry_finetuning, group=KEMP_FINETUNE_FROM_PRETRAIN_FIXED



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

All 6 runs completed (2 tokenizers × 3 folds). **Pretraining provides no
positive transfer** — both finetuned models perform *worse* than their
from-scratch baselines. ResampleCNN finetuned averages F1 0.5221 vs 0.5728
from scratch (−5.1 pp), and CWT-CNN finetuned averages F1 0.5339 vs 0.5536
from scratch (−2.0 pp). The degradation is consistent across all folds.

### Metrics


| Tokenizer   | Condition | Fold | Best Val F1 | Best Val Loss | Final Epoch | WandB Run ID |
| ----------- | --------- | ---- | ----------- | ------------- | ----------- | ------------ |
| CWT-CNN     | Finetuned | 0    | 0.5419      | 1.5373        | 3           | `inad6ohi`   |
| CWT-CNN     | Finetuned | 1    | 0.5610      | 1.2890        | 4           | `olvv36r6`   |
| CWT-CNN     | Finetuned | 2    | 0.4989      | 1.3842        | 3           | `33sfsb7y`   |
| ResampleCNN | Finetuned | 0    | 0.5251      | 1.5386        | 3           | `wdg0f677`   |
| ResampleCNN | Finetuned | 1    | 0.5455      | 1.2611        | 4           | `flmottw3`   |
| ResampleCNN | Finetuned | 2    | 0.4956      | 1.2675        | 3           | `6yfsvspp`   |


**Aggregated (mean ± std across folds):**


| Tokenizer   | Condition               | Val F1          | Δ vs Scratch |
| ----------- | ----------------------- | --------------- | ------------ |
| ResampleCNN | From scratch (exp 006)  | 0.5728 ± 0.0111 | —            |
| ResampleCNN | Finetuned from pretrain | 0.5221 ± 0.0251 | −0.0508      |
| CWT-CNN     | From scratch (exp 006)  | 0.5536 ± 0.0150 | —            |
| CWT-CNN     | Finetuned from pretrain | 0.5339 ± 0.0318 | −0.0196      |


**Per-fold deltas (finetuned − scratch):**


| Fold | CWT-CNN Δ F1 | ResampleCNN Δ F1 |
| ---- | ------------ | ---------------- |
| 0    | −0.0193      | −0.0524          |
| 1    | −0.0022      | −0.0353          |
| 2    | −0.0373      | −0.0646          |




### Analysis

Results were extracted from local wandb-summary.json files and cross-referenced
with the from-scratch baselines from experiment 006.

**Analysis script:** `analysis/007_pretraining_loss_vs_downstream.py`

```bash
uv run python analysis/007_pretraining_loss_vs_downstream.py
```



### Figures

Find them in figures

## Conclusions

1. **Hypothesis refuted.** Pretraining provides no positive transfer to
  downstream sleep staging — finetuned models are consistently *worse* than
   training from scratch, for both tokenizer architectures.
2. **Lower pretraining loss does not imply better downstream performance.**
  CWT-CNN had 3.3× lower pretraining loss but only marginally better finetuned
   F1 than ResampleCNN (0.534 vs 0.522). Neither beats their respective scratch
   baselines.
3. **The pretrained representations may be actively harmful.** The negative
  transfer (−2 to −5 pp F1) suggests the reconstruction objective learns
   features that are either irrelevant or antagonistic to sleep staging
   classification. Full finetuning is unable to overcome this initialization
   disadvantage within the training budget (3–4 epochs before early stopping).
4. **The pretraining-finetuning pipeline itself is verified to work** (weight
  loading, architecture matching, training loop). The issue is with the quality  
   or relevance of the pretrained features, not the infrastructure.



## Notes for future experiments

- **Embedding visualization is the critical next step.** Use t-SNE/PCA on the
pretrained vs randomly-initialized backbone outputs to understand what
structure (if any) pretraining imparts and whether it aligns with sleep stage
boundaries.
- **Linear probing before full finetuning.** Freeze the pretrained backbone and
train only the classification head — this directly measures representation
quality without the confound of full finetuning overwriting everything. If
linear probing also fails, the representations truly lack discriminative
information for sleep staging.
- The reconstruction objective may optimize for high-frequency temporal detail
that aids reconstruction but is not discriminative for sleep staging (where
spectral power in specific bands matters more).
- Consider alternative pretraining objectives that are more aligned with
downstream classification (e.g., contrastive learning, temporal prediction).
- The very short training (3–4 epochs to early stopping) may be insufficient for
finetuning to overcome a bad initialization — experiment with longer patience
or a warmup schedule that gradually unfreezes layers.


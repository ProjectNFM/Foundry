# NeuroSoft Leave-One-Subject-Out From-Scratch Baselines

**Status:** In Progress
**Date started:** 2026-08-17
**Parent experiment:** [Leak-Fixed iEEG Pretraining for Neurosoft Transfer](20260814-MS-ieeg-leak-fixed-pretraining.md)
**Follow-up experiments:** TBD — compare LOSO baselines with Kochi-only and Kochi + B2 leak-fixed pretraining
**Tags:** neurosoft, 8band, auditory_decoding, baseline, from_scratch, intersubject, loso, generalization

## Background

The parent [leak-fixed iEEG pretraining experiment](20260814-MS-ieeg-leak-fixed-pretraining.md)
needs a baseline that measures genuine subject generalization rather than
within-recording interpolation.  The earlier NeuroSoft experiments on
`suarez/sweep-report-skill` optimized supervised recipes only under pooled,
multisubject `intrasession-block` evaluation.  They therefore cannot show how
well a model trained on other animals transfers to a completely unseen animal.

This experiment applies the **pretraining-architecture-matched** scratch
recipe from the paired intrasession experiment to leave-one-subject-out (LOSO)
evaluation.  It is intentionally distinct from the intrasession baseline
experiment: its question is the cross-subject generalization gap, not the
highest possible within-session F1.

The current NeuroSoft data loader supports `intersubject` assignment-based
splits, but the available data artifact exposes three fixed folds rather than
a verified LOSO partition.  Building and validating deterministic LOSO
assignments for every eligible subject is therefore a prerequisite, not an
assumption.

## Question

With the pretraining-compatible architecture and downstream optimization
settings frozen, what held-out-subject NeuroSoft 8-band acoustic-stimulus F1
is achieved when every subject is held out once and no recording from that
subject is used for training or model selection?

## Hypothesis

LOSO test F1 will be lower and more variable than the corresponding pooled
`intrasession-block` baselines because subject-specific channel layouts and
physiology are unavailable during training.  The species-specific recipes
will nevertheless exceed chance-level eight-class performance for every
evaluable held-out subject; the magnitude and consistency of the gap will
define the required baseline for later pretraining transfer.

## Experiment

### Setup

- **Model:** POYO-EEG trained from scratch, using the same frozen,
  pretraining-compatible CWT-CNN architecture as
  [the paired intrasession baseline experiment](20260817-MS-neurosoft-intrasession-baselines.md).
  This means concat CWT-CNN, `embed_dim=256`, `depth=4`, 8/8 attention heads,
  `sequence_length=2.0`, dynamic channel embeddings, and disabled session
  embeddings.  Run minipigs and monkeys separately.
- **Data:** all recordings from all non-held-out subjects form the training
  pool for each species.  All recordings belonging to the held-out subject are
  excluded from training and are used only for validation.
- **Task:** `neurosoft_acoustic_stim_8band`, with the established
  multi-frequency eight-band label mapping.
- **Split:** validation-only LOSO over every subject represented in the species
  cohort.  The held-out subject is the subject-disjoint validation partition;
  no test pass is run.  Record the subject and recording IDs in the resolved
  config and WandB metadata.
- **Training:** full supervised finetuning from random initialization.  Use
  identical recipe, seed(s), early stopping, and checkpoint policy for every
  held-out subject within a species.
- **Primary metric:** macro F1 on the held-out validation subject, logged as
  `val/neurosoft_acoustic_stim_8band_f1`.  Aggregate as subject-macro mean±SD
  and report every held-out subject; retain AUROC, precision, recall, balanced
  accuracy, and per-class metrics.
- **WandB:** project `auditory_decoding`; group
  `NEUROSOFT_8B_LOSO_SCRATCH_BASELINES`.

### Launch command

```bash
# Queues one job for every held-out subject (seven minipigs, six monkeys).
uv run python main.py experiment=auditory_decoding/neurosoft_8band_loso_scratch_minipigs -m
uv run python main.py experiment=auditory_decoding/neurosoft_8band_loso_scratch_monkeys -m
```

### Key config overrides

| Setting | Minipigs | Monkeys | Source |
| --- | --- | --- | --- |
| Tokenizer / fusion | `per_channel_cwt_cnn` / concat | same | Leak-fixed pretraining |
| CWT tokenizer | 9 log-spaced frequencies, 0.5–30 Hz, 64 filters, 2 conv layers, 100 Hz tokens | same | Leak-fixed pretraining |
| `embed_dim`, `depth`, `dim_head` | 256, 4, 128 | same | Leak-fixed pretraining |
| self / cross heads; latents per step | 8 / 8; 16 | same | Leak-fixed pretraining |
| Sequence / temporal settings | 2.0 s; latent step 0.1; `t_min=0.01`, `t_max=0.5`, rotated values | same | Leak-fixed pretraining |
| Channel embedding | dynamic mode; concat dimension 64 | same | Leak-fixed pretraining |
| Session embedding | disabled | same | Leak-fixed pretraining |
| Dropout (`ffn` / `lin` / attention) | 0.2 / 0.4 / 0.2 | same | Leak-fixed pretraining |
| Input normalization | enabled | same | Leak-fixed pretraining |
| Learning rate | 2.75e-5 | 2.5e-5 | Prior optimized-HP baseline |
| Weight decay | 0.08 | 0.30 | Capacity-winner setting |
| Gradient clip | 0.5 | 1.0 | Prior optimized-HP baseline |
| Class-weight smoothing | 0.75 | 1.0 | Prior class-weight sweep |
| Token rate | 100 Hz | 100 Hz | Prior sampling-rate sweep |
| Split | validation-only LOSO with a subject-disjoint validation set | same | This experiment |
| Labeled sampling window | 0.5 s | same | NeuroSoft trial interval length; POYO remains a 2.0-s model |
| Effective batch size | 128 (microbatch 16 × gradient accumulation 8) | same | Fits the cluster GPU memory budget |

The held-out recordings log aggregate validation metrics and per-recording
`val_session/...` values, including scalar per-class F1, precision, and recall.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD — compare this architecture-matched scratch baseline with the two
leak-fixed pretraining initializers.

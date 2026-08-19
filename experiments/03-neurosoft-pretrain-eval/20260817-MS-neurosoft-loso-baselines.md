# NeuroSoft Leave-One-Subject-Out From-Scratch Baselines

**Status:** Completed
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
| Effective batch size | 128 (microbatch 128 × gradient accumulation 1) | same | Uses the requested 48-GB L40S GPU memory budget |

The held-out recordings log aggregate validation metrics and per-recording
`val_session/...` values, including scalar per-class F1, precision, and recall.

## Results

### Summary

All 13 LOSO scratch baseline runs (7 minipigs + 6 monkeys) completed
successfully.  Both species hover at or near eight-class chance level
(0.125), with several held-out subjects falling below it.

| Species | Subjects | Mean F1 | Std | Min | Max |
| --- | --- | --- | --- | --- | --- |
| Minipigs | 7 | 0.1241 | 0.0131 | 0.1018 | 0.1395 |
| Monkeys | 6 | 0.1262 | 0.0228 | 0.1055 | 0.1700 |

### Metrics

| Species | Subject | Best val F1 | Run name | State |
| --- | --- | --- | --- | --- |
| Minipigs | sub-01 | 0.1159 | neurosoft_8b_loso_scratch_minipigs_sub-01 | finished |
| Minipigs | sub-02 | 0.1330 | neurosoft_8b_loso_scratch_minipigs_sub-02 | finished |
| Minipigs | sub-03 | 0.1356 | neurosoft_8b_loso_scratch_minipigs_sub-03 | finished |
| Minipigs | sub-04 | 0.1018 | neurosoft_8b_loso_scratch_minipigs_sub-04 | finished |
| Minipigs | sub-05 | 0.1224 | neurosoft_8b_loso_scratch_minipigs_sub-05 | finished |
| Minipigs | sub-06 | 0.1206 | neurosoft_8b_loso_scratch_minipigs_sub-06 | finished |
| Minipigs | sub-07 | 0.1395 | neurosoft_8b_loso_scratch_minipigs_sub-07 | finished |
| Monkeys | sub-01 | 0.1292 | neurosoft_8b_loso_scratch_monkeys_sub-01 | finished |
| Monkeys | sub-02 | 0.1213 | neurosoft_8b_loso_scratch_monkeys_sub-02 | finished |
| Monkeys | sub-03 | 0.1163 | neurosoft_8b_loso_scratch_monkeys_sub-03 | finished |
| Monkeys | sub-04 | 0.1150 | neurosoft_8b_loso_scratch_monkeys_sub-04 | finished |
| Monkeys | sub-05 | 0.1700 | neurosoft_8b_loso_scratch_monkeys_sub-05 | finished |
| Monkeys | sub-06 | 0.1055 | neurosoft_8b_loso_scratch_monkeys_sub-06 | finished |

### Analysis

```bash
uv run python analysis/042_neurosoft_loso_scratch_baselines.py
```

### Figures

![Per-subject LOSO F1](../analysis/figures/042_neurosoft_loso_scratch_subjects.png)

![Training curves](../analysis/figures/042_neurosoft_loso_scratch_curves.png)

## Conclusions

Hypothesis partially confirmed.  As predicted, LOSO test F1 is substantially
lower and more variable than the corresponding intrasession baselines (0.124
vs 0.270 for minipigs; 0.126 vs 0.264 for monkeys).  However, the prediction
that all held-out subjects would exceed chance-level performance (0.125) is
**not met**: several subjects in both species fall below chance (minipig
sub-04 at 0.102, monkey sub-06 at 0.105).  The scratch LOSO models are
effectively at chance, indicating that training on other subjects' data
provides negligible transfer to a completely unseen subject without
pretrained initialization.  These baselines set a floor near chance for the
pretrained LOSO transfer experiments.

## Notes for future experiments

- Compare these at-chance scratch baselines with pretrained LOSO transfer
  (Kochi-only and Kochi+B2 initializers) to determine whether pretraining
  lifts held-out-subject performance meaningfully above chance — the primary
  measure of genuine cross-subject generalization benefit.

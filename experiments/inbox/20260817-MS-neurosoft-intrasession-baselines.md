# NeuroSoft Intrasession Multisubject From-Scratch Baselines

**Status:** In Progress
**Date started:** 2026-08-17
**Parent experiment:** [Leak-Fixed iEEG Pretraining for Neurosoft Transfer](20260814-MS-ieeg-leak-fixed-pretraining.md)
**Follow-up experiments:** TBD — compare these baselines with Kochi-only and Kochi + B2 leak-fixed pretraining
**Tags:** neurosoft, 8band, auditory_decoding, baseline, from_scratch, intrasession, multisubject, block_split

## Background

The parent [leak-fixed iEEG pretraining experiment](20260814-MS-ieeg-leak-fixed-pretraining.md)
produced two NeuroSoft-transfer initializers, but a matched downstream
no-pretraining reference is still missing.  The preceding NeuroSoft work on
`suarez/sweep-report-skill` established the relevant within-recording protocol:
`intrasession-block` on the 8-band acoustic-stimulus task, with all recordings
for a species pooled during training.  Its default-capacity three-fold
multisubject baselines reached F1 0.360±0.003 for minipigs and 0.499±0.010 for
monkeys.

That thread selected useful downstream optimization settings, but its best
capacity runs used Resample-CNN tokenizers, smaller embeddings, and different
channel-fusion schemes.  Those choices are incompatible with the leak-fixed
pretraining checkpoints.  This baseline instead fixes **every transferable
model setting** to the parent pretraining recipe: CWT-CNN tokenizer, concat
fusion, 256-dimensional model, dynamic channel embeddings, and disabled
session embeddings.  Only supervised-training settings that do not change
transferable tensor shapes are inherited from the prior NeuroSoft studies.

## Question

When all recordings for one species are trained together, what three-fold
`intrasession-block` F1 does a pretraining-architecture-matched, from-scratch
POYO model achieve on NeuroSoft 8-band acoustic-stimulus classification?

## Hypothesis

The CWT-CNN transfer-compatible scratch models will provide stable three-fold
references for each species, but will not exceed the prior Resample-CNN
capacity peaks (0.394 for minipigs and 0.538 for monkeys).  This is the
appropriate control because any subsequent change after loading a pretrained
checkpoint can be attributed to initialization rather than a changed
architecture.

## Experiment

### Setup

- **Model:** POYO-EEG trained from scratch, separately for minipigs and
  monkeys.  Match the parent pretraining architecture exactly: concat
  `per_channel_cwt_cnn`, `embed_dim=256`, `depth=4`,
  `dim_head=128`, `self_heads=8`, `cross_heads=8`,
  `num_latents_per_step=16`, `sequence_length=2.0`,
  `channel_emb_mode=dynamic`, and disabled session embeddings.  Do not pool
  species in a single run.
- **Data:** all recordings in each species' `multisess_raw` config, pooled in
  a single multisubject training run per species and fold (41 minipig and 27
  monkey recordings in the current configs).
- **Task:** `neurosoft_acoustic_stim_8band`, retaining the established
  multi-frequency eight-band label mapping.
- **Split:** `intrasession-block`; all three existing block folds (0, 1, 2).
  Every recording contributes only its own train/validation block intervals.
- **Training:** full supervised finetuning from random initialization.  Freeze
  the pretraining-compatible architecture below.  Use species-specific
  optimizer, regularization, and class-weight settings from the preceding
  NeuroSoft studies; use the same seed(s), early-stopping policy, and metric
  across species so fold-level results are comparable.
- **Primary metric:** best validation macro F1,
  `val/neurosoft_acoustic_stim_8band_f1`, reported as mean±standard deviation
  over folds.  Also retain AUROC, precision, recall, balanced accuracy, and
  per-class metrics.
- **WandB:** project `auditory_decoding`; group
  `NEUROSOFT_8B_INTRASESSION_SCRATCH_BASELINES`.

### Launch command

```bash
# Each command queues three independent block-fold jobs.
uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_minipigs -m
uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_monkeys -m
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
| Split / folds | `intrasession-block`; 0, 1, 2 | same | This experiment |
| Labeled sampling window | 0.5 s | same | NeuroSoft trial interval length; POYO remains a 2.0-s model |
| Effective batch size | 128 (microbatch 16 × gradient accumulation 8) | same | Fits the cluster GPU memory budget |

Validation logs include the aggregate task metrics plus
`val_session/<subject_session_acquisition>/...` for each recording.  Each
session emits macro F1, AUROC, precision, recall, balanced accuracy, Cohen's
kappa, and scalar per-class F1/precision/recall.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD — compare this architecture-matched scratch control with the Kochi-only
and Kochi + B2 leak-fixed initializers.

# Leak-Fixed iEEG Pretraining for Neurosoft Transfer

**Status:** In Progress
**Date started:** 2026-08-14
**Parent experiment:** [Information Leak Fixes: Channel Encoder Masking + Signal Zeroing + Tokenizer Comparison](20260812-MS-channel-encoder-leak-fix-impact.md)
**Follow-up experiments:** [NeuroSoft Intrasession Multisubject From-Scratch Baselines](20260817-MS-neurosoft-intrasession-baselines.md), [NeuroSoft Leave-One-Subject-Out From-Scratch Baselines](20260817-MS-neurosoft-loso-baselines.md)
**Tags:** pretraining, mae, ieeg, kochi, neurosoft, data_composition, channel_encoder, information_leak, signal_zeroing, cwt_cnn

## Background

The [information-leak fix experiment](20260812-MS-channel-encoder-leak-fix-impact.md)
showed that masking channel-encoder pooling and zeroing masked raw signal remove
large decoder-side shortcuts.  Both are now the default and must be retained for
all new pretraining.  Although the fixes substantially increase reconstruction
loss, they do not materially change performance on the project's existing EEG
downstream suite.

The earlier [paradigm-diversity experiment](../02-data-scaling/20260807-MS-paradigm-diversity-pretrain.md)
trained Kochi-containing models before these fixes.  It found that Kochi did not
help the usual sleep, motor-imagery, and P300 benchmarks, but those are not the
target of this work.  This experiment instead creates two leak-fixed initializers
for the forthcoming Neurosoft iEEG benchmark: a source-specific Kochi model and
a higher-volume model that combines Kochi with B2's three EEG sources (Klinzing,
Shirazi, and Pavlov).

The decisive outcome will be Neurosoft 8-band acoustic-stimulus F1, measured in
a separate downstream experiment alongside a matched no-pretraining baseline.
Reconstruction loss is recorded only as a training diagnostic; its scale is not
comparable with pre-fix runs.

## Question

For Neurosoft iEEG acoustic-stimulus decoding, does leak-fixed pretraining on
Kochi alone or on Kochi plus the B2 EEG sources provide the strongest transfer
initialization relative to no pretraining?

## Hypothesis

Both leak-fixed pretrained initializations will improve Neurosoft validation F1
over a matched no-pretraining control.  Kochi-only pretraining will provide the
stronger transfer initialization because its variable-channel, iEEG-oriented
source is closer to the intended target than the additional scalp-EEG sources;
the higher-volume Kochi + B2 model may trade this source specificity for broader
signal diversity.

## Experiment

### Setup

- **Model:** Masked POYO EEG with CWT-CNN tokenizer, dynamic channel embeddings,
  disabled session embeddings, and both leak fixes enabled (defaults:
  `disable_channel_encoder_token_mask=false`, `zero_masked_signal=true`).
- **Data:** Kochi Visual Naming alone, or Kochi plus B2's Klinzing, Shirazi, and
  Pavlov datasets.  The mixed arm contains substantially more channel-hours, so
  this comparison tests source configuration rather than modality alone.
- **Task:** MAE masked reconstruction with TemporalBlockMasking
  (`mask_ratio=0.5`, `block_size=10`).
- **Training:** 400k maximum steps; batch size 64; learning rate 1e-4; 2k-step
  warmup followed by cosine decay; bf16 mixed precision; intersubject
  validation; early-stopping patience 10.
- **WandB:** project `foundry_pretraining`, group `IEEG_LEAK_FIXED_PRETRAIN`.
- **Evaluation:** Full supervised finetuning on the paired, architecture-matched
  NeuroSoft `intrasession-block` recipe.  Each initialization is evaluated for
  minipigs and monkeys over folds 0, 1, and 2 (12 downstream jobs total),
  alongside the paired from-scratch baseline experiment.  The primary metric is
  `val/neurosoft_acoustic_stim_8band_f1`; the existing EEG downstream suite is
  intentionally out of scope.
- **Checkpoint selection:** Use the best validation checkpoint from each
  completed pretraining run, not its final training-state checkpoint:

| Initialization | Checkpoint | Strict transfer validation |
| --- | --- | --- |
| Kochi-only | `${oc.env:SCRATCH}/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_fixed/checkpoints/best-step315000-val_loss_0.6941.ckpt` | Verified: 93 loaded; 0 missing or mismatched |
| Kochi + B2 | `${oc.env:SCRATCH}/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_b2_fixed/checkpoints/best-step200000-val_loss_0.2815.ckpt` | Verified: 93 loaded; 0 missing or mismatched |

The downstream configurations preserve the pretraining architecture
(CWT-CNN concat tokenizer, dynamic channels, disabled session embeddings,
256-dimensional backbone). `run.pretrained_transfer_mode=strict` is retained
so a missing or shape-incompatible transferable tensor aborts before training
rather than silently producing a partial initialization.

### Launch command

```bash
# Kochi-only, leak-fixed iEEG pretraining
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/kochi_only \
  run.name=pretrain_ieeg_kochi_fixed \
  run.group=IEEG_LEAK_FIXED_PRETRAIN -m

# Kochi plus the three B2 EEG datasets, with the same leak-fixed setup
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  data.dataset_kwargs.brainsets=[klinzing_sleep_ds005555,shirazi_hbnr1_ds005505,pavlov_verbal_wm_ds003655,kochi_visualnaming_ds006914] \
  run.name=pretrain_ieeg_kochi_b2_fixed \
  run.group=IEEG_LEAK_FIXED_PRETRAIN -m

# Downstream finetuning: each command submits three independent block folds.
# The quoted run.name keeps Hydra's fold interpolation intact in the shell.
uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_minipigs \
  run.pretrained_checkpoint="$SCRATCH/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_fixed/checkpoints/best-step315000-val_loss_0.6941.ckpt" \
  'run.name=neurosoft_8b_intrasession_kochi_fixed_minipigs_fold${hyperparameters.fold_number}' \
  run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS -m

uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_monkeys \
  run.pretrained_checkpoint="$SCRATCH/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_fixed/checkpoints/best-step315000-val_loss_0.6941.ckpt" \
  'run.name=neurosoft_8b_intrasession_kochi_fixed_monkeys_fold${hyperparameters.fold_number}' \
  run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS -m

uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_minipigs \
  run.pretrained_checkpoint="$SCRATCH/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_b2_fixed/checkpoints/best-step200000-val_loss_0.2815.ckpt" \
  'run.name=neurosoft_8b_intrasession_kochi_b2_fixed_minipigs_fold${hyperparameters.fold_number}' \
  run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS -m

uv run python main.py experiment=auditory_decoding/neurosoft_8band_intrasession_scratch_monkeys \
  run.pretrained_checkpoint="$SCRATCH/runs/IEEG_LEAK_FIXED_PRETRAIN/pretrain_ieeg_kochi_b2_fixed/checkpoints/best-step200000-val_loss_0.2815.ckpt" \
  'run.name=neurosoft_8b_intrasession_kochi_b2_fixed_monkeys_fold${hyperparameters.fold_number}' \
  run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS -m
```

### Key config overrides

| Config / override | Purpose |
| --- | --- |
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Leak-fixed CWT-CNN pretraining recipe and standard 400k-step budget |
| `data=openneuro/kochi_only` | Kochi-only source-specific initialization |
| `data=openneuro/three_dataset_pretrain` | B2 source list used as the base for the mixed arm |
| `data.dataset_kwargs.brainsets=[...]` | Adds Kochi to B2 without duplicating a data config |
| `run.group=IEEG_LEAK_FIXED_PRETRAIN` | Isolates the two checkpoints for later retrieval |
| `run.pretrained_checkpoint=<best checkpoint>` | Initializes each downstream finetuning run from the selected pretraining arm |
| `run.pretrained_transfer_mode=strict` | Requires an exact match for every transferable backbone tensor before training |
| `run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS` | Groups the two initializations × three folds for minipigs |
| `run.group=NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS` | Groups the two initializations × three folds for monkeys |

## Results

### Summary

Pretraining jobs submitted after three-batch GPU smoke tests completed cleanly:

- Kochi-only: SLURM job `10375080` (`pretrain_ieeg_kochi_fixed`)
- Kochi + B2: SLURM job `10375081` (`pretrain_ieeg_kochi_b2_fixed`)

Strict downstream checkpoint-load validation passed for both checkpoints and
both species configurations: all 93 transferable tensors load exactly, with
zero missing or shape-mismatched tensors. The 11 excluded tensors are the
pretraining/task-specific components deliberately outside the transfer policy.

Downstream finetuning submitted on 2026-08-18 (three folds per array):

| Initialization | Species | WandB / output group | SLURM array |
| --- | --- | --- | --- |
| Kochi-only | Minipigs | `NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS` | `10402288` (`_0`–`_2`) |
| Kochi-only | Monkeys | `NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS` | `10402329` (`_0`–`_2`) |
| Kochi + B2 | Minipigs | `NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MINIPIGS` | `10402335` (`_0`–`_2`) |
| Kochi + B2 | Monkeys | `NEUROSOFT_8B_INTRASESSION_PRETRAIN_TRANSFER_MONKEYS` | `10402339` (`_0`–`_2`) |

### Metrics

TBD — record each pretraining run's best `val/loss`, WandB run name and ID, and
the Neurosoft benchmark's mean ± standard deviation of
`val/neurosoft_acoustic_stim_8band_f1` for Kochi-only, Kochi + B2, and no
pretraining.

### Analysis

```bash
uv run python analysis/038_ieeg_leak_fixed_pretraining.py
```

The script fetches pretraining history from WandB now. Once the Neurosoft
benchmark has been created, pass its WandB group with `--neurosoft-group` to add
the Kochi-only, Kochi + B2, and no-pretraining comparison.

### Figures

TBD — generated by `analysis/038_ieeg_leak_fixed_pretraining.py`.

## Conclusions

TBD

## Notes for future experiments

TBD — create the Neurosoft benchmark experiment with matched optimization,
split, and seeds for all three initialization conditions. Record the resulting
WandB group and run IDs here.

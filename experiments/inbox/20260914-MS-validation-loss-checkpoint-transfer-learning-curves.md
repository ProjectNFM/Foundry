# Phase 4E -- Validation-Loss Checkpoint Transfer Learning Curves

**Status:** In Progress
**Date started:** 2026-09-14
**Parent experiment:** [Phase 4D -- Transfer Recipe and LR Warmup Screen](20260911-MS-transfer-lr-warmup-screen.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, supervised-pretraining, transfer, validation-loss, checkpoint-selection, data-scaling, learning-curves, phase4e, 8band, mila

## Background

The [Phase 4A transfer gate](20260904-MS-fullpool-finetune-transfer.md)
showed that selecting a source checkpoint by validation F1 after long
pretraining did not improve downstream transfer and was harmful for minipigs.
[Phase 4B](20260910-MS-early-checkpoint-transfer.md) localized the problem:
the 500-step checkpoint was approximately neutral to scratch, while later
checkpoints became progressively less transferable.  [Phase 4D](20260911-MS-transfer-lr-warmup-screen.md)
then found that the transfer deficit was partly an optimization mismatch;
high-LR discriminative fine-tuning (base LR `3e-3`, transferred
frontend/BiGRU LR `3e-4`) was performance-safe relative to scratch at full
target-data scale, although no speed advantage was established.

This experiment combines those lessons into a properly powered transfer
learning-curve test.  Source models are re-pretrained for at most 10,000
optimizer steps, with validation every 100 steps.  The existing milestone
callback retains its default `1%`, `3%`, `10%`, `30%`, and `100%` checkpoints
(steps 100, 300, 1,000, 3,000, and 10,000), rather than every validation
checkpoint.  A separate best-checkpoint callback retains the checkpoint with
minimum source validation loss for transfer.  The downstream model then uses the
Phase-4D high-LR discriminative recipe and is evaluated at the established
5%, 10%, 25%, 50%, and 100% target-training fractions.  All source
checkpoints remain available for later audit or follow-up, but intermediate
source checkpoints are not separately fine-tuned in this experiment.

## Question

Does a source checkpoint selected by minimum validation loss after the revised
10K-step pretraining protocol transfer at least as well as scratch with all
target data, while providing a downstream performance advantage when only a
small fraction of the target training data is available?

## Hypothesis

For each target recording, the minimum-source-validation-loss checkpoint from
the target-excluded, same-species 10K-step source run will be non-inferior to
the matched scratch control at 100% target training data and will outperform
scratch at lower target-data fractions.  The transfer advantage is expected
to be largest at 5--25% data and to shrink toward zero at 100%.

The primary confirmation criteria are prespecified as follows:

- At 100%, transfer is **on par** with scratch if the paired,
  subject-balanced 95% bootstrap lower bound for transfer minus scratch is at
  least `-0.01` and the point estimate is within `0.01` F1 of scratch.
- At least one fraction in `{5%, 10%, 25%, 50%}` must show a positive
  transfer-minus-scratch point estimate with a 95% subject-bootstrap interval
  excluding zero.  The predicted direction at each lower fraction remains
  positive, but failure to meet this stricter criterion at every fraction is
  not treated as a refutation when the overall lower-data trend is positive.

## Experiment

### Setup

- **Model:** Audited train-global-normalized `NeurosoftConvBiGRU` with the
  per-recording session adapter and target-fresh router used in Phases 4A--4D.
- **Source data:** Target-excluded, same-species full pools using the existing
  audited source manifests: 12 target-subject exclusions across the cohort,
  each trained with paired source selection/model seeds `(42,42)`, `(43,43)`,
  and `(44,44)`. Source test partitions remain forbidden during pretraining.
- **Source training:** Keep the validated Phase-4 source optimizer recipe
  (`batch_size=16`, learning rate `2.5e-4`, weight decay `0.018`, causal split,
  train-global normalization), change the budget to `max_steps=10000`, and
  validate every 100 optimizer steps. Disable source early stopping so all
  scheduled milestones are reached. Retain the callback-defined milestone
  checkpoints at `1%`, `3%`, `10%`, `30%`, and `100%` of the 10K budget, plus a
  hash-verified manifest for the separately retained checkpoint with minimum
  `val/loss`; ties are resolved by the earliest optimizer step.
- **Target data:** All 40 eligible minipig and 13 eligible monkey recordings,
  with the established `intrasession-causal` splits and nested target-training
  fractions `0.05`, `0.10`, `0.25`, `0.50`, and `1.00`.
- **Downstream transfer:** Transfer only the source checkpoint selected by
  minimum source validation loss for the matching target-subject exclusion.
  Use `full_finetuning_reset_router`, fresh target session adapter and router,
  base LR `3e-3`, transferred temporal frontend/BiGRU LR `3e-4`, and the
  common Phase-4D phased LR warmup from `1e-4` of peak LR over 10% of the
  estimated optimizer steps. No adapter warmup is used.
- **Downstream checkpoint selection:** For both transfer and scratch, select
  the finetuned checkpoint exclusively by maximum target validation supported
  macro-F1 using the established patience-40 early-stopping recipe. Evaluate
  the held-out target test split once from that selected checkpoint.
- **Seeds:** Use target fine-tuning seeds `42`, `43`, and `44`. Source seed
  provenance and target seed are kept as separate axes and are never selected
  using target test results.
- **Scratch controls:** Run a fresh model for every target recording, target
  fraction, and target seed with the identical normalized data, downstream
  schedule, checkpoint-selection rule, and stopping criteria. Scratch cells
  carry explicit null source provenance.
- **WandB:** Use the existing `neurosoft_supervised_pretraining` project with
  new Phase-4E groups and deterministic compiled cell IDs. Source-pretraining
  and downstream-transfer identities must not overlap earlier phase groups.

### Run matrix

| Population | Conditions | New cells |
|---|---:|---:|
| Source pretraining | 12 target-subject exclusions × 3 source seed pairs | 36 source runs |
| Transfer: minipigs | 40 recordings × 5 fractions × 3 target seeds × 3 source seeds | 1,800 |
| Transfer: monkeys | 13 recordings × 5 fractions × 3 target seeds × 3 source seeds | 585 |
| Scratch: minipigs | 40 recordings × 5 fractions × 3 target seeds | 600 |
| Scratch: monkeys | 13 recordings × 5 fractions × 3 target seeds | 195 |
| **Total downstream** |  | **3,180** |

The 36 source runs each retain the five callback-defined milestone checkpoints
and one separately retained loss-selected checkpoint, but only one
loss-selected checkpoint per source run is referenced by the 2,385
transfer cells.  The source seed axis is averaged before target-session
pairing, then target seeds and sessions are averaged within subject before
species-level summaries.

### Metrics and decision rule

- **Primary metric:** held-out target-test supported macro-F1 from the
  target-validation-selected finetuned checkpoint, summarized by fraction and
  species using session means followed by equal-weight subject means.
- **Primary contrast:** paired transfer minus matched scratch test macro-F1 at
  each target fraction, with source seeds averaged before pairing and 95%
  subject-bootstrap intervals.
- **Secondary metrics:** validation supported macro-F1 trajectories, selected
  target optimizer steps, source validation-loss trajectory and selected
  source step, and the fraction of recordings reaching 80% of their own
  100%-data test-F1 reference.
- **Leakage guard:** source selection uses only source validation loss;
  downstream checkpoint selection uses only target validation F1; target test
  metrics are read only after both selections are complete. No checkpoint,
  fraction, seed, or recipe is selected from target test performance.
- **Interpretation:** report minipig and monkey results separately. A positive
  lower-data effect without a 100% non-inferiority result does not support the
  full hypothesis; conversely, 100% non-inferiority alone does not establish
  low-data transfer benefit.

### Launch command

```bash
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints

# Production launch requires a clean, committed repository.
git status --short

# Verify the existing target-excluded source manifests and generate three
# paired source-seed cells per exclusion.  Each cell has a fixed 10K-step
# budget, validation every 100 steps, no early stopping, retained
# 1/3/10/30/100% milestones, and a separate val/loss-selected best manifest.
uv run python tools/generate_phase4e_source_registry.py \
  --output-dir launch/phase4e

uv run python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs \
  run.group=PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-source-minipigs.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_monkeys \
  run.group=PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-source-monkeys.jsonl \
  hydra.launcher.tasks_per_node=2 hydra.launcher.cpus_per_task=2 \
  hydra.launcher.mem_gb=32 -m

# After all source jobs finish, construct and validate the immutable registry
# from their loss-selected manifests, then compile only the selected checkpoint
# for each downstream transfer cell plus its matched scratch control.
uv run python tools/generate_phase4e_source_registry.py \
  --registry --output-dir launch/checkpoint_sets \
  --run-root /network/scratch/s/sobralm/runs \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"

uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/phase4e-validation-loss.jsonl \
  --recipe configs/downstream_recipes/phase4e_validation_loss_learning_curves.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/phase4e --check

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-transfer-learning-curves-minipigs.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-transfer-learning-curves-monkeys.jsonl \
  hydra.launcher.tasks_per_node=2 hydra.launcher.cpus_per_task=2 \
  hydra.launcher.mem_gb=32 -m
```

The registry generator, source-loss checkpoint callback/configuration, and
downstream recipe must be implemented and committed before any production
submission. Follow the repository snapshot requirements and record every
Slurm job ID and snapshot bundle path here immediately after launch.

### Key config overrides

- Source: `trainer.max_steps=10000`, `trainer.val_check_interval=100`, the
  default milestone callback fractions `[0.01, 0.03, 0.10, 0.30, 1.00]`,
  source checkpoint monitor `val/loss` with `mode=min`, no source early
  stopping, and a hash-verified best-loss manifest in addition to the
  milestone manifests.
- Source optimizer: `batch_size=16`, `learning_rate=0.00025`,
  `weight_decay=0.018`, `intrasession-causal` split, and
  `recording_train_global_zscore` normalization.
- Transfer: `run.pretrained_transfer_regime=full_finetuning_reset_router`,
  `hyperparameters.learning_rate=0.003`,
  `hyperparameters.backbone_learning_rate=0.0003`,
  `hyperparameters.backbone_components=[temporal_frontend,gru]`.
- Transfer schedule: `scheduler_name=phased`, `warmup_fraction=0.1`,
  `start_lr_factor=0.0001`, `scheduler_interval=step`, `hold=0`, and `decay=0`.
- Target fractions: `data.training_fraction` in `{0.05, 0.10, 0.25, 0.50,
  1.00}`; target seeds in `{42,43,44}`; source seed pairs in
  `{(42,42),(43,43),(44,44)}`.
- Downstream checkpoint callback: monitor
  `val/neurosoft_acoustic_stim_8band_supported_f1`, `mode=max`, patience `40`;
  test evaluation only after the selected checkpoint is restored.
- Scratch cells must include explicit null source fields and otherwise match
  the transfer cells exactly.

## Results

TBD

### Source-pretraining launch record

- **Minipigs:** Slurm array `10788617` (6 packed allocations / 21 cells),
  group `PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS`, snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T151025_PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS_edfcb672_2d158bf7`.
- **Monkeys:** Slurm array `10788618` (8 packed allocations / 15 cells),
  group `PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS`, snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T151101_PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS_edfcb672_a4ce7d43`.
- Both source arrays use immutable snapshot commit `edfcb672`, RTX 8000 GPUs
  on `long`, fixed `max_steps=10000`, `val_check_interval=100`, no early
  stopping, and best-checkpoint monitor `val/loss` (`mode=min`).

## Conclusions

TBD

## Notes for future experiments

TBD

# Phase 4E -- Validation-Loss Checkpoint Transfer Learning Curves

**Status:** Completed
**Date started:** 2026-09-14
**Parent experiment:** [Phase 4D -- Transfer Recipe and LR Warmup Screen](20260911-MS-transfer-lr-warmup-screen.md)
**Follow-up experiments:** [Phase 4E Runtime Packing Audit](20260914-MS-phase4e-runtime-packing-audit.md)
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
  fractions `0.05`, `0.10`, `0.25`, `0.50`, and `1.00`. A
  recording/fraction combination is compiled only when every class present in
  that recording retains at least three target-training examples, as
  prespecified by the Phase-0 audit. Unavailable low-fraction combinations are
  omitted without rebalancing; all nine transfer runs and all three matched
  scratch runs for that combination are omitted together.
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
- **Scratch controls:** Run a fresh model for every available target
  recording/fraction/seed combination with the identical normalized data,
  downstream schedule, checkpoint-selection rule, and stopping criteria.
  Scratch cells carry explicit null source provenance and use exactly the same
  availability mask as transfer cells.
- **WandB:** Use the existing `neurosoft_supervised_pretraining` project with
  new Phase-4E groups and deterministic compiled cell IDs. Source-pretraining
  and downstream-transfer identities must not overlap earlier phase groups.

### Run matrix

| Population | Conditions | New cells |
|---|---:|---:|
| Source pretraining | 12 target-subject exclusions × 3 source seed pairs | 36 source runs |
| Transfer: minipigs | 193 available recording/fractions × 3 target seeds × 3 source seeds | 1,737 |
| Transfer: monkeys | 62 available recording/fractions × 3 target seeds × 3 source seeds | 558 |
| Scratch: minipigs | 193 available recording/fractions × 3 target seeds | 579 |
| Scratch: monkeys | 62 available recording/fractions × 3 target seeds | 186 |
| **Total downstream** |  | **3,060** |

The 36 source runs each retain the five callback-defined milestone checkpoints
and one separately retained loss-selected checkpoint, but only one
loss-selected checkpoint per source run is referenced by the 2,295
transfer cells.  The source seed axis is averaged before target-session
pairing, then target seeds and sessions are averaged within subject before
species-level summaries.

The availability mask retains 193 of 200 nominal minipig
recording/fraction combinations (`37`, `37`, `39`, `40`, and `40` recordings
at 5%, 10%, 25%, 50%, and 100%) and 62 of 65 monkey combinations (`11`, `12`,
`13`, `13`, and `13`). The ten omitted combinations remove 120 runs from the
nominal 3,180-cell matrix because each combination expands to nine transfer
and three scratch runs. The omissions are:

- Minipigs `sub-07_ses-04_task-AcousStim_acq-LH_desc-raw` at 5%, 10%, and
  25%; `mid_bass` and/or `low_treble` have fewer than three examples.
- Minipigs `sub-07_ses-04_task-AcousStim_acq-RH_desc-raw` at 5% and 10%;
  `low_treble` has fewer than three examples.
- Minipigs `sub-07_ses-05_task-AcousStim_acq-LH_desc-raw` at 5% and 10%;
  `midrange` and/or `low_treble` have fewer than three examples.
- Monkeys `sub-01_ses-015_task-AcousStim_acq-RH_desc-raw` at 5% and 10%;
  `low_bass` has fewer than three examples.
- Monkeys `sub-01_ses-03_task-AcousStim_acq-RH_desc-raw` at 5%; `low_treble`
  and `mid_treble` each have only two examples.

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
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
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
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
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
  run.group=PHASE4E_VALIDATION_LOSS_PRODUCTION_MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_PRODUCTION_MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-transfer-learning-curves-minipigs-pending.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 \
  hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=PHASE4E_VALIDATION_LOSS_PRODUCTION_MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_PRODUCTION_MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/phase4e-transfer-learning-curves-monkeys-pending.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 \
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
  `{(42,42),(43,43),(44,44)}`. Compile only audit-supported
  recording/fraction combinations, using the same per-cell availability mask
  for transfer and scratch rather than lowering the three-example minimum or
  dropping an otherwise usable recording at all fractions.
- Downstream checkpoint callback: monitor
  `val/neurosoft_acoustic_stim_8band_supported_f1`, `mode=max`, patience `40`;
  test evaluation only after the selected checkpoint is restored.
- Scratch cells must include explicit null source fields and otherwise match
  the transfer cells exactly.

## Results

### Summary

The validation-loss-selected pretrained GRU did not improve upon its matched
scratch GRU. Across the target-data curve, its subject-balanced test-F1 effect
was negative for every minipig fraction and for four of five monkey fractions;
the positive monkey 100% estimate remained inconclusive. The convergence
contrast was also predominantly negative: transfer reached its own stable
90%-of-smoothed-peak validation endpoint later than scratch, particularly at
the larger fractions. Thus neither prespecified low-data benefit nor the
anticipated speed benefit was observed.

The primary plots use the prespecified aggregation: source seeds are averaged
before matching each target seed, target seeds are averaged to recording,
recordings to subject, and subjects receive equal species-level weight. Bands
are 20,000-draw non-parametric 95% bootstrap intervals over subjects. Stable
convergence is the first of three consecutive validation evaluations at or
above 90% of a run's own three-point-median-smoothed peak F1; censored runs use
their final evaluation. Positive values in the paired panels favor transfer.

The absolute comparator uses the completed train-global-z-score EEGNet
baseline (not the raw-input Phase-1 EEGNet), so its preprocessing, causal
split, fractions, and target seeds align with the Phase 4E target matrix. It
is descriptive rather than a paired intervention contrast; in particular,
EEGNet uses its established model-specific training schedule.

### Figures

![Primary paired transfer effects](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_main_transfer_advantage.png)

The main panel shows the desired common sign convention: test-F1 advantage is
transfer minus scratch, and speed advantage is scratch stable steps minus
transfer stable steps. Solid trajectories are minipigs; dashed trajectories
are monkeys.

![Absolute model comparison](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_absolute_comparison.png)

Blue is global-z-score EEGNet, light green is scratch GRU, and dark green is
validation-loss-selected pretrained GRU. Full uncertainty ribbons are used for
test F1. EEGNet's convergence uncertainty is sufficiently broad that the
absolute convergence panel uses pointwise intervals instead, preserving the
readability of all six trajectories.

![Session-level paired transfer effects](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_session_paired_distributions.png)

![Top-half scratch-performance sensitivity analysis](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_top_half_robustness.png)

The sensitivity subset is post hoc: within species, it retains the top half of
recordings ranked by seed-averaged scratch 100%-data test F1 (20 of 40 minipig
recordings and 7 of 13 monkey recordings). It does not reveal a joint
performance-and-speed transfer advantage. This selection is explicitly not a
confirmatory test, because choosing on scratch full-data performance can induce
regression-to-the-mean effects.

### Downstream result-accounting note

All 3,060 downstream Slurm tasks completed successfully. One completed
minipig scratch cell has no final W&B test summary and is excluded from
paired downstream test-metric summaries: `sub-03_ses-07_task-AcousStim_acq-LHanest_desc-raw`,
25% target fraction, target seed `44`, run ID `d2c35668`. The corresponding
transfer runs and the other two scratch seeds remain available. Do not rerun
this cell solely for the final analysis; report the missing datum explicitly
in results accounting.

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
- **Launch outcome:** the started allocations failed before Foundry imported:
  the worker interpreter raised `ModuleNotFoundError: No module named
  'functools'` while Submitit imported Python's `contextlib`. The pending
  monkey allocations were cancelled; no source-training cell began. Correct
  the compute-node Python environment before submitting a fresh source array.
- **Replacement source arrays:** submitted with
  `FOUNDRY_ENV_FILE=/home/mila/s/sobralm/Foundry/.venv/bin/activate`, which
  establishes the project virtual environment on each worker. Minipigs:
  `10788724` (6 packed allocations / 21 cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T151855_PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS_ebf74839_eb03ffd8`.
  Monkeys: `10788727` (8 packed allocations / 15 cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T151937_PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS_ebf74839_fa4da94e`.
  Both arrays were queued cleanly at submission time. Five minipig packed
  allocations then started and failed in 2--3 seconds with the same
  pre-Foundry `functools` import error; the pending remainder of both arrays
  was cancelled. No source-training cell reached Foundry initialization.
- **Root cause:** every failed allocation from both attempts was placed on
  `cn-c004`. The same project interpreter imports `functools` correctly on the
  login node, and `functools.py` is present beside the `contextlib.py` that the
  failing traceback successfully opened. This isolates the failure to
  `cn-c004`'s stale or broken view of the home-hosted Python standard library,
  rather than Foundry, Submitit, the snapshot, or the virtual environment.
- **Node-excluded replacement arrays:** resubmitted on `long` with explicit
  `#SBATCH --exclude=cn-c004`. Minipigs: Slurm array `10788773` (6 packed
  allocations / 21 cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T152741_PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS_7cef83d1_a8229694`.
  Monkeys: Slurm array `10788779` (8 packed allocations / 15 cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T152809_PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS_7cef83d1_c824d86d`.
  Both use immutable snapshot commit `7cef83d1`; their generated submission
  scripts were verified to contain the node exclusion and both arrays were
  pending normally after submission.
- **Final source outcome:** all six minipig and eight monkey packed
  allocations completed with exit code `0`, covering all 36 logical source
  runs. W&B reports 21/21 minipig and 15/15 monkey runs as `finished`. The
  source audit verified all expected milestones and best-loss manifests, all
  216 retained manifest/checkpoint hash pairs, and exact agreement between
  each best manifest and its W&B validation-loss minimum. No non-finite or
  divergent curves were found. Minipig validation loss improved by 16.7% on
  average and monkey validation loss by 31.4%; the later monkey overfitting
  confirms the usefulness of loss-based checkpoint selection. Reproduce with
  `uv run python analysis/20260914-MS-validation-loss-checkpoint-transfer-learning-curves.py`.
- **Downstream compilation status:** the hash-verified 36-row source registry
  was generated at `launch/checkpoint_sets/phase4e-validation-loss.jsonl`.
  Compilation intentionally stopped when the existing compiler encountered
  the first audit-unavailable low-fraction cell. Before launch, implement and
  test an explicit per-cell `skip` policy, update the recipe's expected counts
  to 2,316 minipig and 744 monkey cells, and recompile the complete 3,060-cell
  matrix. No downstream jobs have been submitted yet.

### Downstream-production plan

- The runtime audit completed 48 exact Phase 4E cells (32 minipig and 16
  monkey) under their deterministic production W&B identities. They count
  toward the scientific matrix and are excluded from fresh submissions.
- Pending lists contain 2,284 minipig and 728 monkey cells, 3,012 total. The
  common `tasks_per_node=8`, `cpus_per_task=2`, and `num_workers=1` setup
  creates 286 and 91 packed RTX-8000 allocations, respectively.
- Both submissions use `long`, 32 GB RAM, the shared snapshot root, the
  project virtual environment, and exclude `cn-c004`.

### Downstream-production launch record

- **Minipigs:** Slurm array `10791974` (286 packed allocations / 2,284
  pending cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T185237_PHASE4E_VALIDATION_LOSS_PRODUCTION_MINIPIGS_4da2dd41_87c2a707`.
- **Monkeys:** Slurm array `10792003` (91 packed allocations / 728 pending
  cells), snapshot
  `/network/scratch/s/sobralm/foundry-launches/20260914T185344_PHASE4E_VALIDATION_LOSS_PRODUCTION_MONKEYS_4da2dd41_6cadc1e2`.
- Both sealed snapshots use immutable commit `4da2dd41`, one RTX 8000 per
  packed allocation, `tasks_per_node=8`, `cpus_per_task=2`,
  `hyperparameters.num_workers=1`, `mem_gb=32`, partition `long`, and
  `cn-c004` exclusion. The minipig array started allocations immediately;
  the monkey array was pending normally at submission.

### Source-pretraining figures

![Minipig source validation-loss trajectories](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_minipig_source_val_loss.png)

![Monkey source validation-loss trajectories](../../analysis/figures/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_monkey_source_val_loss.png)

## Conclusions

The hypothesis is not supported. Minimum-source-validation-loss selection at
10K source steps yielded no confirmed low-data transfer improvement and no
convergence-speed advantage over the matched scratch GRU. The paired,
subject-balanced results instead support negative transfer as the primary
interpretation, with the strongest and most consistent speed disadvantages at
larger target-data fractions. The one missing scratch test summary remains
explicitly accounted for below and was not rerun.

## Notes for future experiments

- Keep the Phase 4E figure script and W&B-derived CSV caches as the
reproducible source for all figure revisions; do not hand-edit plotted values.
- Treat the top-half result as a robustness sensitivity analysis, not a basis
  for a new selection rule or a confirmatory claim.

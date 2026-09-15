# Source-validation performance versus downstream transfer

**Status:** In Progress
**Date started:** 2026-09-15
**Parent experiment:** [Adapter-bias reliance across source-pretraining age](20260915-MS-adapter-bias-perturbation.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, supervised-pretraining, checkpoint-age, source-validation, transfer, optimizer-efficiency, 8band, minipigs, monkeys, phase4f

## Background

The [Phase 4E validation-loss checkpoint study](../06-neurosoft-supervised-pretraining/20260914-MS-validation-loss-checkpoint-transfer-learning-curves.md)
trained 36 target-subject-excluded source models for 10,000 optimizer steps and
retained fixed checkpoints at steps 100, 300, 1,000, 3,000, and 10,000, plus a
minimum-source-validation-loss checkpoint. Only the loss-selected checkpoint
was transferred downstream in Phase 4E. Consequently, the study cannot show
how downstream performance changes over the source-validation trajectory.

The earlier [checkpoint-age experiment](../06-neurosoft-supervised-pretraining/20260910-MS-early-checkpoint-transfer.md)
found that a 500-step source checkpoint was approximately neutral to scratch,
while later checkpoints became increasingly harmful for minipigs. That result
used an earlier source-pretraining run, a different downstream optimization
recipe, and sparse checkpoints at 500, 1,500, 5,000, and 15,000 steps. The
relationship should therefore be tested directly using all retained Phase 4E
milestones and the final Phase 4E downstream recipe.

The parent [adapter-bias perturbation experiment](20260915-MS-adapter-bias-perturbation.md)
showed that reliance on the correct recording-specific adapter bias grows with
Phase 4E checkpoint age. That result motivates a qualitative comparison of
trajectories, but adapter-bias derangement is not an endpoint or confirmation
criterion here. This experiment tests whether improving source-validation
performance is associated with worsening downstream transfer; it cannot show
that source improvement or adapter-bias reliance causes the transfer change.

## Question

As Phase 4E source-validation performance improves across retained source
checkpoints, does downstream transfer degrade in both final held-out F1 and
the optimizer steps required to reach a matched level of useful performance?

## Hypothesis

Within each species, lower source-validation cross-entropy across the fixed
100-, 300-, 1,000-, 3,000-, and 10,000-step checkpoints will be associated
with (1) lower downstream held-out test supported macro-F1 and (2) a longer
time, or a lower probability within the downstream budget, of reaching 90% of
the matched scratch run's peak smoothed validation F1.

The joint hypothesis is supported for a species when all of the following
directional relationships hold over the five fixed checkpoints, with 95%
subject-bootstrap intervals excluding zero in the predicted direction:

1. source-validation cross-entropy decreases with `ln(source_step)`;
2. downstream test supported macro-F1 decreases as source-validation
   performance improves; and
3. matched-quality time-to-threshold worsens as source-validation performance
   improves.

If only one downstream dimension meets its criterion, the hypothesis is
partially supported. Minipigs and monkeys are evaluated separately; success
in one species is not treated as success in the other. The irregular
minimum-validation-loss checkpoint is a secondary reference and is excluded
from the ordered primary trend.

## Experiment

### Setup

- **Model:** Existing Phase 4E train-global-normalized
  `NeurosoftConvBiGRU` source checkpoints, transferred with
  `full_finetuning_reset_router`, a fresh target adapter and router, and the
  final Phase 4E high-LR discriminative recipe.
- **Source data:** Existing same-species, target-subject-excluded full source
  pools for seven minipig and five monkey exclusions. No new source training.
- **Target data:** All 53 eligible recordings (40 minipig and 13 monkey), each
  using 100% of its causal target-training split and its unchanged validation
  and test splits.
- **Task:** NeuroSoft eight-band acoustic-stimulus classification.
- **Ordered checkpoints:** Fixed source optimizer steps 100, 300, 1,000,
  3,000, and 10,000.
- **Secondary checkpoint:** The separately retained
  minimum-source-validation-loss checkpoint. Its optimizer step varies by
  source run, so it is shown as a reference but excluded from fixed-step slope
  tests.
- **Seeds:** Full crossing of source seeds `{42,43,44}` with target-finetuning
  seeds `{42,43,44}`.
- **Downstream checkpoint selection:** Maximum target-validation supported
  macro-F1 with the established patience-40 early-stopping recipe; target test
  is evaluated once after restoring that selected checkpoint.
- **Primary downstream performance endpoint:** Held-out target-test
  supported macro-F1.
- **Primary downstream efficiency endpoint:** First optimizer step at which a
  transfer run's trailing-three-evaluation-median validation F1 reaches and
  remains for three evaluations at or above 90% of the matched scratch run's
  smoothed peak validation F1. Matching is by target recording and target
  seed. Runs that do not reach the threshold are right-censored at their final
  validation step.
- **Primary source endpoint:** Source-validation cross-entropy at the exact
  retained checkpoint. Source-validation supported macro-F1 is secondary.
- **Baseline:** Existing Phase 4E scratch runs at 100% target data, matched by
  target recording and target seed.
- **WandB:** New fixed-checkpoint groups
  `20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MINIPIGS` and
  `20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MONKEYS`; reuse the audited Phase 4E
  groups for the loss-selected reference and scratch controls.

### Exact run matrix

Only the five fixed-checkpoint transfer cells are new:

| Species | Target recordings | Fixed checkpoints | Source seeds | Target seeds | New transfer runs |
|---|---:|---:|---:|---:|---:|
| Minipigs | 40 | 5 | 3 | 3 | 1,800 |
| Monkeys | 13 | 5 | 3 | 3 | 585 |
| **Total** | **53** | **5** | **3** | **3** | **2,385** |

The analysis additionally reuses 477 Phase 4E transfer runs at 100% target
data from the minimum-loss checkpoint (360 minipig and 117 monkey) and 159
matched scratch runs (120 minipig and 39 monkey). Thus the complete downstream
analysis contains 2,862 transfer observations across six checkpoint labels
and 159 scratch observations, while requiring exactly 2,385 new training
runs. The 36 existing source runs contribute 180 fixed-checkpoint source
measurements plus 36 loss-selected references.

No lower target-data fractions, new scratch controls, new source-pretraining
runs, bias perturbations, alternate transfer regimes, or hyperparameter sweeps
belong in this experiment.

### Analysis plan

For every target recording and target seed, derive the matched-quality
threshold exclusively from the corresponding scratch validation history.
Apply that fixed threshold to every source-seed/checkpoint transfer history
for the same recording and target seed. Test F1 never enters the efficiency
threshold or downstream checkpoint selection.

For each species and checkpoint, first average target-seed replicates within
recording and source seed, then average recordings within the excluded target
subject, and finally average source seeds. Subjects receive equal weight.
Estimate the ordered checkpoint trends and their 95% intervals by resampling
excluded target subjects with replacement. The efficiency analysis must
retain censoring; report both time-to-threshold and the proportion reaching
the threshold within budget. The implementation should use a censored
time-to-event or restricted-time estimator rather than treating a censored
final step as an observed crossing.

The primary association uses within-source-run improvement in validation
cross-entropy so stable differences in difficulty among exclusion pools do
not drive the result. Report the corresponding trends against
`ln(source_step)` as an interpretable checkpoint-age view. Show the
loss-selected checkpoints only as secondary points at their actual optimizer
steps.

As contextual visualization only, place the already measured species-level
adapter-bias derangement-penalty curve from the parent experiment beside the
new source-performance and downstream trajectories. Do not include that curve
in the statistical confirmation criterion and do not interpret aligned trends
as mediation or causation.

### Launch command

```bash
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints

# Production launch requires a clean, committed repository.
git status --short

# Generate and audit a hash-verified Phase 4E milestone registry. The compiler
# emits exactly 1,800 minipig and 585 monkey transfer cells and no scratch
# cells.
uv run python tools/generate_phase4f_milestone_registry.py \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"

uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/phase4e-fixed-milestones.jsonl \
  --recipe configs/downstream_recipes/phase4f_checkpoint_trajectory.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/phase4f --check

# After the compiled cell lists pass their exact-count and provenance audit:
uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4f/phase4f-checkpoint-trajectory-minipigs.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4f/phase4f-checkpoint-trajectory-monkeys.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 -m
```

### Submission record

Submitted from clean commit
`8b933c9ff269b8561df81c48b215a35b9666b03d` on 2026-09-15.

| Species | Logical runs | Packed allocations | Slurm array | Snapshot bundle |
|---|---:|---:|---|---|
| Minipigs | 1,800 | 225 | `10807331` | `/network/scratch/s/sobralm/foundry-launches/20260915T230308_20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MINIPIGS_8b933c9f_658b9db7` |
| Monkeys | 585 | 74 | `10807340` | `/network/scratch/s/sobralm/foundry-launches/20260915T230501_20260915-MS-SOURCE_VALIDATION_DOWNSTREAM_TRAJECTORY_MONKEYS_8b933c9f_f6bda681` |

Both arrays use the Phase 4E downstream hardware and packing configuration:
partition `long`, one RTX 8000 per packed allocation, eight tasks per node,
two CPUs per task, one dataloader worker per task, 32 GB memory, a three-hour
limit, requeue enabled, and `cn-c004` excluded.

### Key config overrides

- `data.training_fraction=1.0` only.
- `run.pretrained_transfer_regime=full_finetuning_reset_router`.
- `hyperparameters.learning_rate=0.003` and
  `hyperparameters.backbone_learning_rate=0.0003` for
  `[temporal_frontend,gru]`.
- Phased step scheduler with `warmup_fraction=0.1`,
  `start_lr_factor=0.0001`, `hold=0`, and `decay=0`.
- Fresh target adapter and router; no adapter warmup.
- Target seeds `{42,43,44}` crossed with all three source seeds.
- Test evaluation enabled only after validation-based downstream checkpoint
  selection.
- Registry must verify every manifest self-hash, checkpoint SHA-256, excluded
  target subject, species, source-selection seed, source-model seed, fixed
  milestone step, and source Git provenance.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

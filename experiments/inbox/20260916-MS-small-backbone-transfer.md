# Small-backbone checkpoint transfer

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-pretraining architecture viability](20260916-MS-pretraining-architecture-viability.md)
**Follow-up experiments:** [Transfer failure-mode synthesis](20260916-MS-transfer-failure-mode-synthesis.md)
**Tags:** neurosoft, supervised-pretraining, transfer, capacity, small-backbone, checkpoint-age, 8band, minipigs, monkeys

## Background

The parent [source-pretraining architecture viability experiment](20260916-MS-pretraining-architecture-viability.md)
will train a small `NeurosoftConvBiGRU` backbone for every target-subject
exclusion and retain fixed checkpoints at source steps 100, 300, 1,000, 3,000,
and 10,000. This experiment is conditional on those source runs passing their
predeclared readiness gate.

The existing [default-backbone checkpoint trajectory](20260915-MS-source-validation-downstream-trajectory.md)
showed that useful early transfer is progressively erased during source
training. A smaller transferable backbone may constrain source-specific
solutions and preserve that early benefit, but it may instead underfit the
source task or provide insufficient target capacity. Architecture-matched
scratch controls are therefore required.

## Question

Does reducing the transferable backbone to approximately one tenth of its
default parameter count preserve more of the early source-checkpoint transfer
advantage as source training progresses?

## Hypothesis

Relative to the matched batch-128 reference-backbone seed-42 trajectory, the small
backbone will have a less negative subject-balanced change in transfer-minus-
scratch held-out supported macro-F1 from source step 100 to step 10,000. The
hypothesis is evaluated separately by species and is supported when the 95%
excluded-subject bootstrap interval for
`small erosion - default erosion` is strictly positive. Downstream RMST and
threshold attainment are secondary endpoints expected to show the same
preservation.

## Experiment

### Setup

- **Source model:** Small-backbone condition declared downstream-ready by the
  parent experiment; one source seed `(selection=42, model=42)` per excluded
  target subject. Its comparator is the parent's separately trained
  batch-128 reference-backbone condition.
- **Target model:** The same small frontend/GRU architecture as the source,
  with a fresh ordinary recording-specific bias-enabled target adapter and a
  fresh router.
- **Transfer regime:** `full_finetuning_reset_router`; transfer the temporal
  frontend and GRU, exclude all source adapters and the source router.
- **Target data:** All 53 eligible recordings, using 100% of each causal target
  training split and unchanged validation/test splits.
- **Source checkpoints:** Fixed steps 100, 300, 1,000, 3,000, and 10,000 only.
  Loss-selected checkpoints are saved upstream but excluded here.
- **Target seeds:** `{42, 43, 44}`.
- **Scratch comparator:** New small-architecture scratch runs matched by target
  recording and target seed.
- **Primary endpoint:** Held-out target-test supported macro-F1 expressed as
  transfer minus architecture-matched scratch.
- **Secondary endpoints:** Step-10,000 minus step-100 test F1, censoring-aware
  RMST to matched scratch quality, and threshold attainment.
- **WandB:** Planned groups
  `20260916-MS-SMALL-BACKBONE-TRANSFER-MINIPIGS`,
  `20260916-MS-SMALL-BACKBONE-TRANSFER-MONKEYS`, and matched scratch groups;
  exact run names and eight-character IDs will be recorded after launch.

The expected new matrix is:

| Species | Target recordings | Source checkpoints | Source seeds | Target seeds | Transfer runs | Scratch runs |
|---|---:|---:|---:|---:|---:|---:|
| Minipigs | 40 | 5 | 1 | 3 | 600 | 120 |
| Monkeys | 13 | 5 | 1 | 3 | 195 | 39 |
| **Total** | **53** |  |  |  | **795** | **159** |

Use the same validation-selected downstream checkpoint, high-LR
discriminative Phase 4F recipe, test-once policy, and subject-balanced
aggregation as the default trajectory. Comparisons with the default use only
its source seed 42 cells so source initialization multiplicity is matched.

### Launch command

```bash
# Run after the parent source condition passes its gate. Launch only from a
# clean committed snapshot after generating the audited registry and cells.
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
git status --short  # must print nothing

uv run python tools/generate_architecture_viability_registry.py \
  --run-root /network/scratch/s/sobralm/runs \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"
uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/architecture-small_backbone-fixed.jsonl \
  --recipe configs/downstream_recipes/architecture_small_backbone.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/architecture_transfer

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=20260916-MS-SMALL-BACKBONE-TRANSFER-MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-SMALL-BACKBONE-TRANSFER-MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/small-backbone-minipigs.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=20260916-MS-SMALL-BACKBONE-TRANSFER-MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-SMALL-BACKBONE-TRANSFER-MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/small-backbone-monkeys.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m
```

### Key config overrides

- Small capacity parameters exactly matching the source checkpoint.
- `data.training_fraction=1.0`.
- `run.pretrained_transfer_regime=full_finetuning_reset_router`.
- Target seeds `{42,43,44}` and source seed `(42,42)` only.
- Phase 4F learning rates, scheduler, no adapter warmup, and test-once policy.
- Registry and compiler must verify source condition, architecture parameters,
  parameter count, excluded target subject, fixed source step, manifest hash,
  checkpoint hash, and Git provenance.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-small-backbone-transfer_analysis.py`.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

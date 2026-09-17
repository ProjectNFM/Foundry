# Bias-free checkpoint transfer compatibility

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-pretraining architecture viability](20260916-MS-pretraining-architecture-viability.md)
**Follow-up experiments:** [Transfer failure-mode synthesis](20260916-MS-transfer-failure-mode-synthesis.md)
**Tags:** neurosoft, supervised-pretraining, transfer, adapter-bias, target-compatibility, checkpoint-age, 8band, minipigs, monkeys

## Background

The [adapter-bias perturbation experiment](20260915-MS-adapter-bias-perturbation.md)
showed that default source models become increasingly dependent on the correct
recording-specific adapter bias, while warning that removing bias during
training could move its function into the adapter weights or backbone. The
parent [architecture-viability experiment](20260916-MS-pretraining-architecture-viability.md)
therefore retrains each target-subject-excluded source pool with bias-free
recording-specific adapters and retains five fixed checkpoints.

Two downstream treatments are necessary. Loading the bias-free source
backbone into the ordinary target model tests whether source bias removal
improves the transferable representation. Using a fresh bias-free target
adapter tests the matched end-to-end pipeline. The target adapter remains
fresh in both cases because the target recording was excluded from source
pretraining.

## Question

Does bias-free source training preserve checkpoint-age transferability, and
is that preservation stronger when target finetuning also uses a bias-free
recording-specific adapter?

## Hypothesis

Bias-free source checkpoints will exhibit less deterioration in transfer-
minus-scratch held-out supported macro-F1 from source step 100 to step 10,000
than the matched batch-128 reference bias-enabled seed-42 checkpoints, and the preservation
will be greatest with a fresh bias-free target adapter. The primary contrast
is the target-adapter interaction within the bias-free source trajectory:
`bias-free-target erosion - ordinary-target erosion` must have a strictly
positive 95% excluded-subject bootstrap interval. The comparison with the
default source trajectory is required to determine whether either treatment
actually mitigates the original failure mode.

## Experiment

### Setup

- **Source model:** Bias-free recording-specific source adapters with the
  default 1x backbone, declared downstream-ready by the parent experiment.
  The bias-enabled comparator is the parent's batch-128 reference-backbone
  condition.
- **Source seed:** `(selection=42, model=42)` only.
- **Source checkpoints:** Fixed steps 100, 300, 1,000, 3,000, and 10,000;
  loss-selected checkpoints remain saved but excluded.
- **Target treatment A -- source-only intervention:** Fresh ordinary
  recording-specific `Linear(C, 64, bias=True)` target adapter; transfer and
  finetune the source frontend/GRU; reset the router.
- **Target treatment B -- matched bias-free pipeline:** Fresh recording-
  specific `Linear(C, 64, bias=False)` target adapter; transfer and finetune
  the same source frontend/GRU; reset the router.
- **Target data:** All 53 eligible recordings at 100% target training data.
- **Target seeds:** `{42,43,44}`.
- **Scratch comparators:** Existing ordinary Phase 4E scratch runs for
  treatment A and new bias-free target scratch runs for treatment B, each
  matched by recording and target seed.
- **Primary endpoint:** Transfer-minus-treatment-matched-scratch test supported
  macro-F1.
- **Secondary endpoints:** Step-10,000 minus step-100 F1, RMST, attainment,
  convergence curves, and the source-treatment versus target-treatment
  interaction.
- **WandB:** Planned groups
  `20260916-MS-BIAS-FREE-TRANSFER-MINIPIGS` and
  `20260916-MS-BIAS-FREE-TRANSFER-MONKEYS`, with target-adapter treatment in
  every run name and condition record; exact eight-character IDs will be
  recorded after launch.

| Species | Recordings | Checkpoints | Target treatments | Target seeds | Transfer runs | New scratch runs |
|---|---:|---:|---:|---:|---:|---:|
| Minipigs | 40 | 5 | 2 | 3 | 1,200 | 120 |
| Monkeys | 13 | 5 | 2 | 3 | 390 | 39 |
| **Total** | **53** |  |  |  | **1,590** | **159** |

All source adapters remain excluded from checkpoint loading. Treatment B
changes the freshly initialized target adapter architecture; it does not load
any source recording-specific adapter.

### Launch command

```bash
# Run after the parent bias-free source condition passes its gate.
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
git status --short  # must print nothing

uv run python tools/generate_architecture_viability_registry.py \
  --run-root /network/scratch/s/sobralm/runs \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"
uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/architecture-bias_free-fixed.jsonl \
  --recipe configs/downstream_recipes/architecture_bias_free.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/architecture_transfer

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=20260916-MS-BIAS-FREE-TRANSFER-MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-BIAS-FREE-TRANSFER-MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/bias-free-minipigs.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=20260916-MS-BIAS-FREE-TRANSFER-MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-BIAS-FREE-TRANSFER-MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/bias-free-monkeys.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m
```

### Key config overrides

- Source condition fixed to `bias_free` and source seed `(42,42)`.
- Target adapter bias crossed over `{true,false}`.
- `data.training_fraction=1.0` and target seeds `{42,43,44}`.
- `full_finetuning_reset_router` transfers only frontend/GRU for both target
  treatments.
- Phase 4F optimizer and downstream checkpoint selection remain unchanged.
- Compiled cells must identify the treatment-matched scratch comparator and
  audit that source adapter parameters never load.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-bias-free-transfer_analysis.py`.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

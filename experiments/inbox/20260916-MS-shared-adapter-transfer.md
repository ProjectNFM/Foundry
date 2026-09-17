# Shared-adapter checkpoint transfer compatibility

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-pretraining architecture viability](20260916-MS-pretraining-architecture-viability.md)
**Follow-up experiments:** [Transfer failure-mode synthesis](20260916-MS-transfer-failure-mode-synthesis.md)
**Tags:** neurosoft, supervised-pretraining, transfer, shared-adapter, zero-padding, target-compatibility, checkpoint-age, 8band, minipigs, monkeys

## Background

The default model learns one input projection per source recording. The parent
[architecture-viability experiment](20260916-MS-pretraining-architecture-viability.md)
replaces those projections with one shared `Linear(32, 64)` layer. After
modality filtering, channels remain in their existing order and missing
trailing positions are padded with zeros; channel names and anatomical
correspondence are deliberately ignored.

The shared layer has the same shape and semantics for every recording,
including an unseen target. It can therefore be retained during downstream
finetuning, unlike a source recording-specific adapter. Comparing backbone-
only transfer into the ordinary target model with transfer of the shared
adapter and backbone separates improved source representation learning from
the practical value of input-interface continuity.

## Question

Does shared-padded-adapter source training preserve checkpoint-age
transferability, and is additional benefit obtained by retaining the
pretrained shared input projection during target finetuning?

## Hypothesis

Shared-adapter source checkpoints will exhibit less deterioration in transfer-
minus-scratch held-out supported macro-F1 from source step 100 to step 10,000
than the batch-128 reference recording-specific seed-42 checkpoints, and preservation will
be greatest when the pretrained shared adapter is transferred and finetuned.
The primary contrast is
`shared-adapter-retained erosion - ordinary-target erosion`; its 95%
excluded-subject bootstrap interval must be strictly positive. Comparisons
with the default source trajectory establish whether either treatment
mitigates the original transfer erosion.

## Experiment

### Setup

- **Source model:** One shared biased `Linear(32, 64)` input adapter and the
  default 1x backbone, declared downstream-ready by the parent experiment.
  The ordinary-adapter comparator is the parent's batch-128 reference-backbone
  condition.
- **Source seed:** `(selection=42, model=42)` only.
- **Source checkpoints:** Fixed steps 100, 300, 1,000, 3,000, and 10,000;
  loss-selected checkpoints remain saved but excluded.
- **Target treatment A -- backbone-only:** Instantiate a fresh ordinary
  recording-specific bias-enabled target adapter, transfer and finetune the
  source frontend/GRU, and reset the router.
- **Target treatment B -- retained shared interface:** Instantiate the same
  shared-padded `Linear(32, 64, bias=True)` target adapter, load it together
  with the source frontend/GRU, reset the router, and finetune all transferred
  components.
- **Padding:** After modality filtering, retain channel order, right-pad with
  zeros to width 32, ignore channel names, and re-zero time padding after the
  adapter.
- **Target data:** All 53 eligible recordings at 100% target training data.
- **Target seeds:** `{42,43,44}`.
- **Scratch comparators:** Existing ordinary scratch runs for treatment A and
  new shared-padded-adapter scratch runs for treatment B.
- **Primary endpoint:** Transfer-minus-treatment-matched-scratch test supported
  macro-F1.
- **Secondary endpoints:** Step-10,000 minus step-100 F1, RMST, attainment,
  source performance versus target transfer, and target effects stratified by
  filtered channel count.
- **WandB:** Planned groups
  `20260916-MS-SHARED-ADAPTER-TRANSFER-MINIPIGS` and
  `20260916-MS-SHARED-ADAPTER-TRANSFER-MONKEYS`, with target treatment encoded
  in each run; exact names and eight-character IDs will be recorded after
  launch.

| Species | Recordings | Checkpoints | Target treatments | Target seeds | Transfer runs | New scratch runs |
|---|---:|---:|---:|---:|---:|---:|
| Minipigs | 40 | 5 | 2 | 3 | 1,200 | 120 |
| Monkeys | 13 | 5 | 2 | 3 | 390 | 39 |
| **Total** | **53** |  |  |  | **1,590** | **159** |

Treatment B requires a new explicit transfer regime that includes the shared
input adapter, temporal frontend, and GRU while excluding/resetting the router.
The loader must reject this regime for recording-specific source adapters.

### Launch command

```bash
# Run after the parent shared-adapter source condition passes its gate.
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
git status --short  # must print nothing

uv run python tools/generate_architecture_viability_registry.py \
  --run-root /network/scratch/s/sobralm/runs \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"
uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/architecture-shared_padded-fixed.jsonl \
  --recipe configs/downstream_recipes/architecture_shared_adapter.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/architecture_transfer

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=20260916-MS-SHARED-ADAPTER-TRANSFER-MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-SHARED-ADAPTER-TRANSFER-MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/shared-adapter-minipigs.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=20260916-MS-SHARED-ADAPTER-TRANSFER-MONKEYS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-SHARED-ADAPTER-TRANSFER-MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_transfer/shared-adapter-monkeys.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m
```

### Key config overrides

- Source condition fixed to `shared_padded`, source seed `(42,42)`.
- Target treatment crossed over ordinary fresh adapter versus retained shared
  adapter.
- `full_finetuning_retain_shared_adapter_reset_router` includes input adapter,
  `temporal_frontend`, and `gru`, but resets `router`.
- `data.training_fraction=1.0`; target seeds `{42,43,44}`.
- Phase 4F optimizer, scheduler, target checkpoint selection, and test-once
  policy remain unchanged.
- Registry and compiler must verify shared width 32, adapter bias, transfer
  components, source step, checkpoint identity, and matched scratch family.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-shared-adapter-transfer_analysis.py`.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

# Recipe-matched scratch transfer control

**Status:** In Progress
**Date started:** 2026-09-18
**Parent experiment:** [Batch-128 reference transfer replication](20260916-MS-batch128-reference-transfer-replication.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, supervised-pretraining, transfer, scratch-control, recipe-matching, discriminative-lr, checkpoint-age, 8band, minipigs

## Background

The [batch-128 reference transfer replication](20260916-MS-batch128-reference-transfer-replication.md)
compared reference-model checkpoints at source steps 100, 300, 1,000, 3,000,
and 10,000 against a scratch reference model. Its downstream transfer recipe
used a base learning rate of `0.003` for newly initialized target components
and a tenfold lower learning rate of `0.0003` for the transferred temporal
frontend and GRU. The scratch control shared the base learning rate, phased
scheduler, target sessions, target-data fraction, and seeds, but set
`backbone_learning_rate=null` and `backbone_components=null`; consequently,
its randomly initialized temporal frontend and GRU trained at `0.003` rather
than under the transfer arm's discriminative learning-rate schedule.

This optimization mismatch is a confound in the reported transfer-minus-
scratch effects. The existing scratch control cannot distinguish a benefit of
source initialization from a benefit of training the temporal frontend and
GRU at a lower learning rate. This follow-up removes that confound by training
a new minipig-only scratch control with the exact downstream optimizer and
scheduler recipe used by the completed transfer runs. The five transfer
checkpoint conditions will be reused rather than relaunched.

## Question

After replacing the original scratch baseline with a scratch reference model
trained under the exact downstream transfer recipe, does any pretrained source
checkpoint yield statistically higher minipig test supported macro-F1?

## Hypothesis

At least one of the five pretrained checkpoints will outperform the new
recipe-matched scratch control. In particular, the early 100- or 300-step
checkpoint is expected to have higher subject-balanced test supported
macro-F1 because useful early source initialization should compensate for the
slower `0.0003` optimization of the temporal frontend and GRU, whereas a fully
random model must learn those components from scratch at that same rate.

The confirmatory criterion is that at least one transfer-minus-new-scratch
contrast has a family-wise simultaneous 95% whole-subject bootstrap interval
entirely above zero across the five checkpoint comparisons. Failure of every
simultaneous interval to exclude zero will provide no evidence of transfer
improvement under the recipe-matched comparison; it will not establish
equivalence.

## Experiment

### Setup

- **Model:** Batch-128 reference `NeurosoftConvBiGRU` architecture
  (`temporal_channels=128`, `gru_hidden_size=128`), initialized entirely from
  scratch for the new control.
- **Data:** All 40 eligible minipig recordings, each at 100% of its causal
  target training split with its existing validation and test partitions.
- **Task:** NeuroSoft 8-band acoustic-stimulus decoding.
- **Training:** Target seeds `{42,43,44}`; base learning rate `0.003` and
  backbone learning rate `0.0003` for `[temporal_frontend,gru]`; the same
  phased step scheduler, 200-epoch maximum, early-stopping patience, validation
  checkpoint selection, and one-time test evaluation as the parent transfer
  cells. The only intended difference from transfer is random initialization
  with no source checkpoint. Expected new matrix: 40 recordings x 3 seeds =
  120 scratch runs.
- **Comparators:** Reuse the parent's completed reference transfer runs at
  source steps `{100,300,1000,3000,10000}` without relaunching them.
- **Primary metric:** Subject-balanced test supported macro-F1. Average target
  seeds within recording, recordings within subject, and weight the seven
  minipig subjects equally.
- **Primary inference:** Five paired transfer-minus-scratch contrasts with
  simultaneous 95% whole-subject bootstrap intervals controlling family-wise
  error across checkpoints.
- **Secondary analysis:** Pointwise intervals, absolute F1, subject-level
  trajectories, and optimization-efficiency endpoints are descriptive.
- **Main plot:** Extend the parent's checkpoint trajectory and recenter every
  checkpoint effect on the new recipe-matched scratch control at zero. Show
  subject-level trajectories plus the equal-subject mean and simultaneous
  95% intervals; retain the original-scratch comparison only as a clearly
  labeled sensitivity analysis.
- **WandB:** Group `20260918-MS-RECIPE-MATCHED-SCRATCH-MINIPIGS`; every
  human-readable run name and deterministic eight-character run ID is recorded
  in the immutable compiled matrix and will be copied into the analysis
  coverage CSV.

### Launch command

```bash
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints

git status --short  # must print nothing

uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/architecture-reference_backbone-fixed.jsonl \
  --recipe configs/downstream_recipes/architecture_reference_recipe_matched_scratch.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/recipe_matched_scratch

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=20260918-MS-RECIPE-MATCHED-SCRATCH-MINIPIGS \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260918-MS-RECIPE-MATCHED-SCRATCH-MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/recipe_matched_scratch/recipe-matched-scratch-minipigs.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 \
  -m
```

### Submission record

TBD immediately after submission: record the clean launch commit, Slurm array
ID, allocation count, and immutable snapshot bundle path.

### Key config overrides

- `model.temporal_channels=128`
- `model.gru_hidden_size=128`
- `hyperparameters.learning_rate=0.003`
- `hyperparameters.backbone_learning_rate=0.0003`
- `hyperparameters.backbone_components=[temporal_frontend,gru]`
- `hyperparameters.warmup_fraction=0.1`
- `hyperparameters.start_lr_factor=0.0001`
- `hyperparameters.scheduler_name=phased`
- `hyperparameters.scheduler_interval=step`
- `hyperparameters.hold=0`
- `hyperparameters.decay=0`
- `hyperparameters.adapter_warmup_steps=0`
- `data.training_fraction=1.0`
- `run.pretrained_transfer_regime=null`
- `run.evaluate_test=true`

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

The analysis scaffold is
[`analysis/20260918-MS-recipe-matched-scratch-control_analysis.py`](../../analysis/20260918-MS-recipe-matched-scratch-control_analysis.py).
It must fetch the exact parent transfer and new scratch runs through the W&B
API, audit their compiled identities, and write CSV and figure artifacts using
this report's filename stem.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

# Phase 4D -- Transfer Recipe and LR Warmup Screen

**Status:** In Progress
**Date started:** 2026-09-11
**Parent experiment:** [Phase 4C -- 500-Step Head Reset and Backbone Freezing](20260911-MS-500step-head-reset-transfer.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, supervised-pretraining, transfer, learning-rate, warmup, session-adapter, phase4d, 8band, mila

## Background

[Phase 4C](20260911-MS-500step-head-reset-transfer.md) found that resetting the
source router did not produce a reliable advantage over scratch, while frozen
500-step transfer was substantially below both scratch and full fine-tuning.
The learned frozen representation was also no better than the frozen-random
control.  This may reflect an optimization/interface problem rather than the
absence of transferable source information: the target session adapter is
fresh, the transferred encoder is updated with the same relatively large LR as
fresh parameters, and the existing target recipe was inherited from earlier
experiments rather than calibrated for this transfer boundary.

This follow-up keeps the source checkpoint and target task fixed while testing a
small, deliberately bounded set of downstream optimization recipes.  It also
replaces the previous target-seed replication axis with three plausible base
learning rates, so the experiment can determine whether the apparent transfer
failure is sensitive to ordinary optimizer choices without launching a full
hyperparameter sweep for every method.

## Question

Across the four proposed downstream adaptation recipes, how strongly does the
base target learning rate affect 500-step source-checkpoint transfer, and does
adapter/router calibration plus a conservative LR for transferred parameters
produce a simple transfer recipe that is performance-safe relative to scratch?

## Hypothesis

The current uniform-LR recipe is unnecessarily destructive or poorly matched
to the fresh target session adapter.  A recipe that first calibrates the fresh
adapter/router for 500 target optimizer steps and then uses a lower LR for the
transferred frontend/GRU will produce the most reliable transfer.  At least
one of the three base LRs will be performance-safe relative to a matched
scratch control (lower 95% subject-bootstrap bound at least -0.01) and will
avoid the large frozen-transfer deficit, while the best recipe/LR combination
will show no worse stable validation convergence than scratch at the same LR.

## Experiment

### Setup

- **Model:** The audited train-global-normalized `NeurosoftConvBiGRU` with
  per-recording session-specific linear adapters, target-fresh router, and
  full target fine-tuning after the selected warmup.
- **Source checkpoint:** The existing audited 500-step full-pool checkpoint
  for source selection/model seed pair `(42,42)`, target-excluded and same
  species, transferred with `full_finetuning_reset_router`. No new source
  pretraining is launched.
- **Target data:** All 40 eligible minipig and 13 eligible monkey recordings,
  using the established causal intrasession splits and full target training
  fraction.
- **Target seed:** Fix target seed to `42` for every new cell. This is an
  optimizer-recipe/LR screen, not a source-seed or target-seed robustness
  claim.
- **Common optimizer schedule:** Use a fixed NeuralBench-inspired LR warmup:
  linearly ramp each active optimizer group from `1e-4` of its peak LR to its
  peak over the first 10% of that cell's estimated optimizer steps, then hold
  the peak LR constant for the remainder. There is no cosine or other decay.
  The schedule warmup is common to every arm and is distinct from the
  recipe-level adapter warmup below.

### Recipe matrix

| Recipe | First 500 target steps | After step 500 |
|---|---|---|
| Uniform, no adapter warmup | All target parameters trainable | All parameters use the base LR |
| Discriminative, no adapter warmup | All target parameters trainable | Fresh adapter/router use the base LR; transferred frontend/GRU use `0.1 ×` base LR |
| Adapter warmup + uniform | Freeze transferred frontend/GRU; train fresh adapter/router | Unfreeze all parameters; all use the base LR |
| Adapter warmup + discriminative | Freeze transferred frontend/GRU; train fresh adapter/router | Unfreeze all; fresh adapter/router use base LR and transferred frontend/GRU use `0.1 ×` base LR |

The adapter warmup trains both the fresh target session adapter and fresh
router, since both are required before the transferred encoder can be used
effectively. Frozen frontend/GRU modules must remain in evaluation mode during
this phase so their dropout does not inject stochasticity into the fixed
representation.

### LR matrix

The searched base learning rates are `3e-4`, `1.5e-3`, and `3e-3`.  The middle
value is the existing matched target recipe; the other two provide one lower
and one higher reasonable alternative.  The discriminative arms derive the
transferred-encoder LR mechanically as 10% of the selected base LR rather
than introducing a second search axis.

### Scratch controls

New scratch controls use target seed 42 and the same three base LRs:

- **Scratch uniform:** fresh model, all parameters use the base LR.
- **Scratch adapter warmup:** random frontend/GRU are frozen for 500 steps
  while the fresh adapter/router train, then all parameters use the base LR.

The discriminative scratch duplicates are omitted because there are no
transferred parameters to receive a distinct low LR.  Existing Phase-2/Phase-4
scratch and Phase-4C transfer runs remain historical references but are not
treated as exact controls for the new common LR-warmup schedule.

### Run matrix

| Population | Conditions | New cells |
|---|---:|---:|
| Pretrained transfer: minipigs | 40 sessions × 4 recipes × 3 LRs | 480 |
| Pretrained transfer: monkeys | 13 sessions × 4 recipes × 3 LRs | 156 |
| Scratch: minipigs | 40 sessions × 2 recipes × 3 LRs | 240 |
| Scratch: monkeys | 13 sessions × 2 recipes × 3 LRs | 78 |
| **Total** |  | **954** |

### Metrics and decision rule

- **Primary metric:** target-test supported macro-F1, selected only through the
  target validation checkpoint and aggregated by session, then subject, then
  species.
- **Secondary metrics:** validation supported macro-F1 at fixed optimizer
  steps, stable time-to-90%-of-peak validation F1, target-adapter parameter
  movement during the first 500 steps, and transferred frontend/GRU parameter
  movement after unfreezing.
- **Primary comparison:** each transfer recipe/LR is compared with scratch at
  the same base LR and common LR-warmup schedule. Subject-balanced bootstrap
  intervals are reported, with the limitation that the fixed target seed does
  not estimate seed variance.
- **Working-recipe gate:** a recipe/LR is favorable when it is performance-safe
  relative to its matched scratch control, has finite/stable optimization, and
  does not require a materially later stable validation endpoint. Among
  favorable arms, prefer the simplest recipe with the smallest base LR that
  does not sacrifice convergence.
- **Interaction analysis:** report whether the recipe ranking changes across
  the three base LRs. Do not interpret a single winning LR as a general
  hyperparameter optimum or make a multi-seed robustness claim.

### Launch command

```bash
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints

# Requires a clean, committed repository before the snapshot is created.
git status --short

uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/phase4a-mila-early-checkpoints.jsonl \
  --recipe configs/downstream_recipes/phase4d_transfer_lr_warmup.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/phase4d \
  --check

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4d/phase4d-transfer-lr-warmup-minipigs.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 \
  hydra.launcher.cell_list=launch/phase4d/phase4d-transfer-lr-warmup-monkeys.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hydra.launcher.mem_gb=32 -m
```

### Key config overrides

- Fixed Phase-4C 500-step source manifest, source selection/model seeds
  `(42,42)`, and `run.pretrained_transfer_regime=full_finetuning_reset_router`
  for all transfer cells; scratch cells carry explicit null source fields.
- `run.seed=42` for all new cells.
- Base `hyperparameters.learning_rate` in `{0.0003, 0.0015, 0.003}`.
- Discriminative arms set `hyperparameters.backbone_components` to
  `[temporal_frontend,gru]` and use a transferred frontend/GRU LR equal to 10%
  of the selected base LR; fresh adapter/router parameters use the base LR.
- Common optimizer LR schedule: `scheduler_name=phased`,
  `warmup_fraction=0.1`, `scheduler_interval=step`, followed by constant LR
  with no decay.
- Adapter-warmup arms set `adapter_warmup_steps=500`; only
  `session_adapter` and `router` are trainable during that phase, with the
  frontend/GRU kept in evaluation mode before being unfrozen.
- Scratch cells carry explicit null source provenance and use the same target
  data, seed, LR values, and common schedule.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

- If adapter warmup and discriminative LR both fail across the LR ladder,
  prioritize an architectural interface test rather than expanding the
  optimizer search.
- If one recipe is clearly stable and performance-safe, rerun only that
  recipe at target seeds 43 and 44 before making a seed-robust transfer claim.
- If scratch is highly LR-sensitive, freeze the selected target recipe before
  evaluating any architectural change so transfer comparisons remain fair.

# Large-backbone checkpoint transfer

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-pretraining architecture viability](20260916-MS-pretraining-architecture-viability.md)
**Follow-up experiments:** [Transfer failure-mode synthesis](20260916-MS-transfer-failure-mode-synthesis.md)
**Tags:** neurosoft, supervised-pretraining, transfer, capacity, large-backbone, checkpoint-age, 8band, minipigs, monkeys

## Background

The parent [source-pretraining architecture viability experiment](20260916-MS-pretraining-architecture-viability.md)
will train an approximately ten-times-larger transferable backbone for every
target-subject exclusion. This experiment begins only if those source runs are
declared ready for downstream evaluation.

The existing [default-backbone checkpoint trajectory](20260915-MS-source-validation-downstream-trajectory.md)
showed an early transfer benefit that eroded as same-recording source
validation improved. If excess shared capacity permits increasingly
source-specific representations, a larger backbone should lose transferability
faster. Conversely, greater capacity could learn more generalizable features,
so the result must be measured against architecture-matched scratch controls
rather than inferred from absolute target F1.

## Question

Does increasing the transferable backbone to approximately ten times its
default parameter count accelerate or amplify checkpoint-age transfer erosion?

## Hypothesis

Relative to the matched batch-128 reference-backbone seed-42 trajectory, the large
backbone will have a more negative subject-balanced change in transfer-minus-
scratch held-out supported macro-F1 from source step 100 to step 10,000. The
hypothesis is evaluated separately by species and is supported when the 95%
excluded-subject bootstrap interval for
`large erosion - default erosion` is strictly negative. RMST and attainment
are secondary endpoints expected to deteriorate more strongly with checkpoint
age.

## Experiment

### Setup

- **Source model:** Large-backbone condition declared downstream-ready by the
  parent experiment; one source seed `(selection=42, model=42)` per exclusion.
  Its comparator is the parent's separately trained batch-128 reference
  backbone.
- **Target model:** The same large frontend/GRU architecture as the source,
  with a fresh ordinary recording-specific bias-enabled target adapter and a
  fresh router.
- **Transfer regime:** `full_finetuning_reset_router`; transfer frontend and
  GRU only.
- **Target data:** All 53 eligible recordings at 100% target training data.
- **Source checkpoints:** Fixed steps 100, 300, 1,000, 3,000, and 10,000 only;
  loss-selected checkpoints are excluded.
- **Target seeds:** `{42, 43, 44}`.
- **Scratch comparator:** New large-architecture scratch runs matched by target
  recording and target seed.
- **Primary endpoint:** Transfer-minus-matched-scratch held-out supported
  macro-F1.
- **Secondary endpoints:** Step-10,000 minus step-100 F1, RMST, attainment,
  runtime, peak memory, and downstream compute.
- **WandB:** Planned groups
  `20260916-MS-LARGE-BACKBONE-TRANSFER-MINIPIGS`,
  `20260916-MS-LARGE-BACKBONE-TRANSFER-MONKEYS`, and matched scratch groups;
  exact run names and eight-character IDs will be recorded after launch.

| Species | Target recordings | Source checkpoints | Source seeds | Target seeds | Transfer runs | Scratch runs |
|---|---:|---:|---:|---:|---:|---:|
| Minipigs | 40 | 5 | 1 | 3 | 600 | 120 |
| Monkeys | 13 | 5 | 1 | 3 | 195 | 39 |
| **Total** | **53** |  |  |  | **795** | **159** |

The downstream recipe is held fixed to Phase 4F. Instability under that
prespecified optimizer is an experiment result, not permission to silently
retune the large model. Any later learning-rate recovery must be documented as
a separate follow-up.

### Launch command

```bash
# Run after the parent source condition passes its gate. Launch only from a
# clean committed snapshot after generating the audited registry and cells.
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
git status --short  # must print nothing

uv run python tools/generate_architecture_viability_registry.py \
  --run-root /network/scratch/s/sobralm/runs \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT"
uv run python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/architecture-large_backbone-fixed.jsonl \
  --recipe configs/downstream_recipes/architecture_large_backbone.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root "$FOUNDRY_CHECKPOINT_ROOT" \
  --output-dir launch/architecture_transfer

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.cell_list=launch/architecture_transfer/large-backbone-minipigs.jsonl \
  -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.cell_list=launch/architecture_transfer/large-backbone-monkeys.jsonl \
  -m
```

### Key config overrides

- Large capacity parameters exactly matching the source checkpoint.
- `data.training_fraction=1.0`.
- `run.pretrained_transfer_regime=full_finetuning_reset_router`.
- Target seeds `{42,43,44}` and source seed `(42,42)` only.
- Phase 4F optimizer, scheduler, adapter warmup, and checkpoint-selection
  settings unchanged.
- Registry and compiler must verify architecture and parameter-count metadata
  in addition to the normal checkpoint provenance.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-large-backbone-transfer_analysis.py`.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

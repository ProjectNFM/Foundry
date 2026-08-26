# HERO Motor Imagery small-subset memorization diagnostic

**Status:** Draft
**Date started:** 2026-08-26
**Parent experiment:** [Position-conditioned channel values for HERO Motor Imagery learnability](20260826-MS-hero-position-value-mi-learnability.md)
**Follow-up experiments:** TBD — determined by the diagnostic decision table below.
**Tags:** neuralbench, hero, eegnet, motor_imagery, memorization, learnability, diagnostic, validation_only, from_scratch

## Background

Flat HERO is near chance on four-class NeuralBench Motor Imagery (MI), despite
matched EEGNet reaching substantially higher performance. The
[spatial-slot](20260824-MS-hero-spatial-slots.md),
[relational-context](20260825-MS-hero-relational-context-sufficiency.md),
[delayed-fusion](20260826-MS-hero-delayed-fusion-mi-learnability.md), and
[position-value](20260826-MS-hero-position-value-mi-learnability.md)
experiments have not produced a meaningful HERO gain. The delayed-fusion pilot
reduced training cross-entropy below the uniform four-class value but did not
demonstrate strong train-set learning, so the current results cannot cleanly
distinguish an inability to fit MI from a failure to generalize.

This diagnostic uses one fixed, tiny stratified training subset and removes
early stopping and explicit weight decay. It asks the simplest discriminating
question before further architectural sweeps: can HERO memorize exactly the
same examples that a known-good EEGNet can memorize? Validation is retained
only as a secondary curve; it must not select checkpoints or be used for test
evaluation.

## Question

Can flat one-slot HERO reach near-perfect training balanced accuracy on a
fixed 64-example NeuralBench MI subset when matched EEGNet is trained on the
identical examples under the same diagnostic schedule?

## Hypothesis

EEGNet will reach at least 0.95 peak training balanced accuracy and clearly
below-uniform training cross-entropy. The result diagnoses HERO as follows:

1. **HERO peak train balanced accuracy >= 0.95:** HERO can fit this MI subset;
   the leading next question is generalization/inductive bias rather than a
   basic data or optimization failure.
2. **HERO < 0.95 while EEGNet >= 0.95:** HERO has a learnability bottleneck
   under the established data/target path. Investigate temporal feature
   extraction, optimization, and the order of temporal versus spatial fusion
   before more position or routing variants.
3. **Both models < 0.95:** inconclusive. Audit data/target and optimization
   contracts before attributing the failure to HERO.

The 0.95 gate is intentionally a training-only diagnostic. No validation or
test threshold authorizes an architecture change from this experiment.

## Experiment

### Setup

- **Models:** from-scratch flat HERO with one spatial slot, and from-scratch
  EEGNet. HERO uses the simpler one-slot control because eight slots did not
  help MI in the parent spatial-slot ablation. EEGNet retains the matched MI
  architecture but has dropout disabled for this capacity diagnostic.
- **Data:** `Schalk2004Bci2000` from the canonical NeuralBench MI split. Both
  runs use the exact same deterministic, split-local, stratified subsets:
  64 train examples (one batch; expected 16 per class) and 256 validation
  examples, selected with `subset_seed=33`. The test split is not loaded for
  evaluation.
- **Task:** Four-class Motor Imagery, seed 33.
- **Training:** 500 epochs with the prior MI AdamW/OneCycle schedule
  (`lr=1e-4`, batch size 64, gradient clipping 1.0), but `weight_decay=0`, an
  inert early-stopping callback (`patience=501`, exceeding the epoch cap), and
  no best-validation checkpoint selection. HERO uses `16-mixed`; EEGNet uses
  `32-true`, matching its established stable setting.
- **Evaluation:** `run.evaluate_test=false`. Record peak and final training
  balanced accuracy/loss, validation curves and confusion matrices, learning
  rate, the final checkpoint, subset size, and subset index SHA-256 from the
  launcher log. Do not use validation to stop or select a model.
- **WandB:** project `foundry-neuralbench`, group
  `NB_MI_MEMORIZATION_DIAGNOSTIC`.

### Launch command

Before launching, commit all experiment files and require a clean repository:

```bash
git status --short
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py \
  experiment=neuralbench/mi_hero_memorization_diagnostic -m

uv run python main.py \
  experiment=neuralbench/mi_eegnet_memorization_diagnostic -m
```

These are local-GPU single-run diagnostics. Record both W&B run names/IDs and
the immutable snapshot bundle paths after completion. Do not run a test
evaluation or a production Slurm sweep from this experiment.

### Key config overrides

Implemented Hydra configs:

- `configs/experiment/neuralbench/mi_hero_memorization_diagnostic.yaml`
- `configs/experiment/neuralbench/mi_eegnet_memorization_diagnostic.yaml`

| Setting | HERO | EEGNet |
|---|---:|---:|
| Train / validation subset | 64 / 256 | 64 / 256 |
| Subset seed / run seed | 33 / 33 | 33 / 33 |
| Batch size / epochs | 64 / 500 | 64 / 500 |
| Explicit weight decay | 0 | 0 |
| Early stopping | inert (`patience=501`) | inert (`patience=501`) |
| Test evaluation | disabled | disabled |
| Precision | 16-mixed | 32-true |
| HERO temporal mode / spatial slots | flat / 1 | n/a |
| EEGNet dropout | n/a | 0.0 |

## Results

TBD — run both fixed-subset diagnostics and fetch validation-only W&B metrics.

## Conclusions

TBD — apply the pre-registered training-only decision table in **Hypothesis**.

## Notes for future experiments

TBD

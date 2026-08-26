# Causal delayed fusion for HERO Motor Imagery learnability

**Status:** Draft
**Date started:** 2026-08-26
**Parent experiment:** [HERO relational-context sufficiency for Motor Imagery](20260825-MS-hero-relational-context-sufficiency.md)
**Follow-up experiments:** TBD
**Tags:** neuralbench, hero, motor_imagery, delayed_fusion, causal_convolution, absolute_position, learnability, validation_only, from_scratch

## Background

The [HERO spatial-slot ablation](20260824-MS-hero-spatial-slots.md) found that
eight spatial slots did not improve over one-slot pooling on NeuralBench Motor
Imagery (MI): both conditions remained near chance and far below matched
EEGNet. The follow-up [relational-context sufficiency
experiment](20260825-MS-hero-relational-context-sufficiency.md) likewise found
that signal-only, local-context, relational-context, and shuffled-context HERO
all remained near chance. Absolute electrode position produced only a small
mean increase, from approximately 0.302 to 0.317 test balanced accuracy, which
was insufficient to make the task learnable.

These results do not establish that absolute position is uninformative. In the
current model, position affects only spatial-routing logits, while the routed
values come from a three-layer causal channel encoder with kernel size 7. At
the 128 Hz canonical rate, that encoder has a receptive field of only 19
samples, or approximately 148 ms. Channels are then fused, after which their
explicit identities are unavailable to the flat temporal encoder. Position
may therefore be selecting channel-local values before those values contain
MI-relevant trial-scale rhythmic features such as lateralized mu suppression
or beta rebound.

This experiment tests whether preserving independent channel streams through
a longer causal temporal receptive field makes MI learnable. Absolute
position is a secondary factor tested only after introducing delayed fusion.
The test split must not be evaluated during this architecture-design stage.

## Question

Does extending causal channel-wise temporal processing to approximately 700
ms before spatial fusion make NeuralBench Motor Imagery learnable by HERO, and
does absolute electrode position provide an additional validation improvement
once the delayed-fusion model learns?

## Hypothesis

The delayed-fusion HERO condition without absolute position will make MI
learnable if all of the following pre-registered criteria hold:

1. on a deterministic reduced-data, single-seed local pilot, it reaches at
   least 0.95 training balanced accuracy;
2. on the full training split, it reaches at least 0.40 mean best-validation
   balanced accuracy across seeds 33, 34, and 35;
3. it exceeds the matched early-fusion control by at least 0.05 mean
   best-validation balanced accuracy and wins for all three matched seeds; and
4. its training cross-entropy moves clearly below the uniform four-class loss
   of 1.386.

Absolute position is a secondary hypothesis. It is supported if the
delayed-fusion + position condition exceeds delayed fusion without position by
at least 0.02 mean best-validation balanced accuracy, wins for at least two of
three matched seeds, and has no matched-seed regression larger than 0.02.

No held-out test metric is part of either hypothesis. Test evaluation is
deferred until an architecture has been selected using validation results.

## Experiment

### Setup

- **Model:** HERO from scratch with `temporal_mode=flat`, eight spatial slots,
  `embed_dim=64`, eight attention heads, and the existing task decoder.
- **Task:** Four-class NeuralBench Motor Imagery on
  `Schalk2004Bci2000` (64 channels, 4-second trials), using the canonical
  NeuralBench train/validation split.
- **Primary independent variable:** pre-fusion channel-wise temporal receptive
  field.
  - **Early fusion:** current three-layer, kernel-7 causal channel encoder;
    receptive field 19 samples, approximately 148 ms at 128 Hz.
  - **Delayed fusion:** four causal convolution blocks with kernel size 7 and
    dilations `1,2,4,8`; receptive field 91 samples, approximately 711 ms at
    128 Hz. All channels use the same encoder weights, remain separate through
    this stack, and retain one feature vector per timestamp.
- **Secondary independent variable:** absolute electrode position, applied
  only to the delayed-fusion model and only through spatial-routing logits.
- **Full comparison conditions:**

| Condition | Pre-fusion encoder | Absolute position | Purpose |
|---|---|:---:|---|
| Early fusion | Current ~148 ms causal encoder | No | Matched learnability control |
| Delayed fusion | Dilated ~711 ms causal encoder | No | Primary condition |
| Delayed fusion + position | Dilated ~711 ms causal encoder | Yes | Secondary position test |

- **Local pilot:** seed 33 on deterministic, stratified reduced subsets. Use
  512 training trials and 256 validation trials when class counts permit,
  retaining the original split boundary and at least one example of every
  class. Run the early-fusion and delayed-fusion conditions first; add the
  position condition only after delayed fusion passes the 0.95 training
  balanced-accuracy gate. The subset indices must be fixed by the run seed and
  logged or saved for exact reproduction.
- **Full experiment:** the three conditions above on the complete train and
  validation splits with matched seeds 33, 34, and 35 (nine runs total).
- **Training:** initially match the parent experiment: AdamW (`lr=1e-4`,
  `weight_decay=0.05`), step-wise cosine OneCycleLR with `pct_start=0.1`, batch
  size 64, gradient clipping 1.0, 40-epoch cap, and validation-balanced-accuracy
  early stopping with patience 10. The local pilot may use a longer epoch cap
  because its purpose is an overfit/learnability gate.
- **Evaluation:** select and report the best-validation-balanced-accuracy
  checkpoint. Set `run.evaluate_test=false` for every pilot and full run. Do
  not fetch, log, or compare test metrics during architecture development.
- **Diagnostics:** log train and validation loss/balanced accuracy, selected
  epoch, per-class validation recall and confusion counts, position gate values
  and gradient norms, position/content routing-logit RMS, routing entropy and
  attention by electrode, parameter count, peak memory, and wall-clock time.
- **External reference:** matched EEGNet validation performance may be shown as
  context, but it is not a formal success criterion and no EEGNet test metric
  should be used for model selection.
- **WandB:** project `foundry-neuralbench`; planned groups
  `NB_MI_HERO_DELAYED_FUSION_PILOT` and `NB_MI_HERO_DELAYED_FUSION_FULL`.

The NeuralBench data module now exposes `train_subset_size`,
`val_subset_size`, and `subset_seed`. These controls construct exact-size,
seeded stratified subsets without changing the canonical split. The test split
is never subsetted. Do not use `trainer.limit_train_batches` as the reduced-data
mechanism because shuffling would expose a changing subset across epochs.

### Launch command

Local pilot for the primary early-versus-delayed comparison:

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py \
  experiment=neuralbench/mi_hero_delayed_fusion_pilot -m
```

After delayed fusion passes the local training-learnability gate, run the
single-seed position condition locally:

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py \
  experiment=neuralbench/mi_hero_delayed_fusion_pilot \
  hero_delayed_fusion_condition=delayed_fusion_position seed=33 -m
```

Full three-seed Slurm launch, only after the local pilot passes its gate:

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py experiment=neuralbench/mi_hero_delayed_fusion \
  -m
```

Before the full launch, require a clean, committed repository, retain the
normal immutable-snapshot workflow, and use the `long` Slurm partition. Record
the Slurm job ID and snapshot bundle path in this report after submission.

### Key config overrides

Implemented Hydra configs:

- `configs/experiment/neuralbench/mi_hero_delayed_fusion_pilot.yaml`
- `configs/experiment/neuralbench/mi_hero_delayed_fusion.yaml`
- `configs/hero_delayed_fusion_condition/{early_fusion,delayed_fusion,delayed_fusion_position}.yaml`

| Setting | Pilot | Full |
|---|---|---|
| `model.temporal_mode` | `flat` | `flat` |
| `model.num_spatial_slots` | `8` | `8` |
| pre-fusion condition | early, delayed; then delayed + position | early, delayed, delayed + position |
| `model.channel_encoder_dilations` | early: `[1,1,1]`; delayed: `[1,2,4,8]` | same |
| delayed kernel size | `7` | `7` |
| `data.train_subset_size` | `512` | `null` (full) |
| `data.val_subset_size` | `256` | `null` (full) |
| `data.subset_seed` | `${seed}` | `${seed}` (inactive with full data) |
| seed | `33` | `33,34,35` |
| `run.evaluate_test` | `false` | `false` |
| Hydra launcher | `local_gpu` | `slurm_default`, partition `long` |

Pilot and full runs share the same condition definitions. They differ only in
dataset size, seed sweep, launcher, worker count, and the explicitly documented
pilot epoch cap and early-stopping patience.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

Run the validation-only W&B analysis after either stage completes:

```bash
uv run python analysis/20260826-MS-hero-delayed-fusion-mi-learnability_analysis.py
```

The script fetches both experiment groups, saves per-run and aggregate CSVs,
evaluates the pre-registered pilot/full criteria when the required cells are
available, and writes the validation balanced-accuracy figure. It does not
request or consume test metrics.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD

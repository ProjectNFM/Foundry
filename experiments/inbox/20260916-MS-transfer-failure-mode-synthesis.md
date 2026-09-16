# Transfer success and failure modes across architecture interventions

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-pretraining architecture viability](20260916-MS-pretraining-architecture-viability.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, supervised-pretraining, synthesis, transfer, capacity, adapter-bias, shared-adapter, checkpoint-age, failure-modes, 8band

## Background

The parent [architecture-viability experiment](20260916-MS-pretraining-architecture-viability.md)
creates a common source-training comparison and fixed-checkpoint inventory for
four model interventions. Its downstream children test the interventions
independently:

- [small-backbone checkpoint transfer](20260916-MS-small-backbone-transfer.md);
- [large-backbone checkpoint transfer](20260916-MS-large-backbone-transfer.md);
- [bias-free checkpoint transfer](20260916-MS-bias-free-transfer.md); and
- [shared-adapter checkpoint transfer](20260916-MS-shared-adapter-transfer.md).

Those reports retain separate hypotheses, scratch comparators, and target
treatments. A final integrated analysis is still necessary because the
central mechanistic question is comparative: whether the original late-stage
transfer erosion is better explained or mitigated by shared backbone capacity,
recording-specific input adaptation, or compatibility between the source and
target input interfaces.

This experiment launches no training. It independently fetches and audits all
source, transfer, and scratch runs, reconstructs a common subject-balanced
table, and classifies the observed success and failure modes without replacing
the verdicts in the child reports.

## Question

Across backbone-capacity and input-adapter interventions, which source-learning
and target-compatibility patterns distinguish preserved transfer, ordinary
underfitting, target-interface mismatch, and late source specialization?

## Hypothesis

Constraining recording-specific input adaptation will preserve late-checkpoint
transfer more consistently than changing shared backbone capacity or selecting
the condition with the lowest same-recording source-validation loss. The
hypothesis is supported when both matched adapter interventions (bias-free
source/target and retained shared adapter) have a less negative step-10,000-
minus-step-100 transfer-minus-scratch F1 contrast than the default trajectory
in their predicted species, while neither capacity intervention provides an
equally consistent improvement across species. All comparisons use 95%
excluded-subject bootstrap intervals and retain species-specific verdicts.

## Experiment

### Setup

- **Runs:** No new training. Reuse the complete fixed-checkpoint results from
  the parent and four linked downstream experiments plus the existing default
  seed-42 Phase 4F trajectory and matched scratch controls.
- **Checkpoints:** Fixed source steps 100, 300, 1,000, 3,000, and 10,000 only.
  Loss-selected checkpoints are excluded even though they remain stored.
- **Common unit:** Excluded target subject, preserving its full checkpoint
  trajectory, target recordings, and target seeds during aggregation and
  bootstrap resampling.
- **Primary response:** Late-transfer preservation, defined as the
  subject-balanced change in transfer-minus-treatment-matched-scratch test F1
  from source step 100 to step 10,000.
- **Secondary responses:** Absolute transfer-minus-scratch F1 at each step,
  RMST, attainment, source CE/F1 improvement, train-validation gap, compute,
  and filtered channel count.
- **WandB:** Reuse all definitive groups from the linked experiments; no new
  group is created. Exact group names, run names, and eight-character run IDs
  will be enumerated in Results.

The integrated analysis uses the following failure-mode definitions:

| Failure mode | Source behavior | Downstream behavior |
|---|---|---|
| Preserved transfer | Meaningful source learning | Transfer benefit persists through late checkpoints |
| Late specialization | Source validation improves | Transfer-minus-scratch and/or efficiency deteriorates with checkpoint age |
| Underfitting | Weak or collapsed source learning | Poor transfer at early and late checkpoints |
| Interface mismatch | Source learning is viable | Backbone-only transfer fails but matched input-interface transfer succeeds |
| Target architecture effect | Source learning is viable | Transfer and matched scratch change together |
| Capacity-sensitive optimization | Capacity changes source/target stability | Both scratch and transfer degrade under the fixed optimizer |

The synthesis must distinguish descriptive cross-condition associations from
causal intervention contrasts. It must not treat the 12 overlapping source
exclusion pools as independent biological datasets or use target test metrics
to select a condition, checkpoint, or analysis subset.

### Launch command

```bash
# No training launch. Run only after all linked reports identify their
# definitive W&B groups and have complete fixed-checkpoint matrices.
uv run python analysis/20260916-MS-transfer-failure-mode-synthesis_analysis.py \
  --entity poyo-eeg
```

### Key config overrides

- None; this is an analysis-only experiment.
- The analysis must require a locked expected matrix covering source condition,
  species, excluded target subject, fixed source step, target adapter mode,
  target seed, target recording, transfer regime, and matched scratch family.
- Every source/target cell must carry exact checkpoint and manifest hashes.
- Missing or duplicate cells are fatal unless documented prospectively in the
  relevant child report.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-transfer-failure-mode-synthesis_analysis.py`.

### Figures

TBD. Planned figures include source improvement versus transfer erosion,
early-versus-late transfer effects, target-interface interactions, compute
versus transfer benefit, and a condition-by-failure-mode summary.

## Conclusions

TBD

## Notes for future experiments

TBD

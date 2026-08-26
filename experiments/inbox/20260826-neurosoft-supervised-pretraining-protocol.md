# NeuroSoft Supervised-Pretraining Protocol and Data Audit

**Status:** Completed
**Date started:** 2026-08-26
**Parent experiment:** [Causal Split Baselines](../02-neurosoft-intrasession-multisubj/20260805-LS-causal-split.md)
**Follow-up experiments:** TBD (Phase 1 EEGNet learning curves)
**Tags:** neurosoft, protocol, preregistration, data-audit, 8band, intrasession-causal, supervised-pretraining

## Background

The [supervised-pretraining roadmap](../../docs/neurosoft-supervised-pretraining-roadmap.md)
requires a data and protocol audit before model training. The parent experiment
established the `intrasession-causal` split as a harder fixed evaluation
protocol. This Phase 0 work freezes the eligibility rule, random seeds, nested
subset format, leakage-free source pools, feasible source-volume/diversity
bins, and staged run-count estimates before any Phase 1 results exist.

## Question

Can the processed minipig and monkey data support the planned causal 8-band
learning curves and leakage-free leave-one-subject-out source pools, with every
unsupported session/fraction cell made explicit before training?

## Hypothesis

The split and manifest validations will pass determinism, nesting,
disjointness, block-local chronology, and target-exclusion checks. Some
recordings and some low-fraction cells will be unavailable because fewer than
six mapped classes are represented or the requested subset cannot retain the
required support for every represented class.

## Experiment

### Setup

- **Data:** all 41 configured minipig and 27 configured monkey recordings.
- **Task:** 25 raw stimulus frequencies mapped to eight ordered bands by
  `configs/tasks/neurosoft_acoustic_stim_8band.yaml`.
- **Split:** `intrasession-causal`; train, validation, and test are ordered
  within each disjoint recording-domain/stimulus block rather than by one
  global session cut.
- **Eligibility:** a recording loads, at least six of eight mapped classes are
  represented, the represented-class set is identical in causal
  train/validation/test, every represented class has at least three
  causal-train trials, and causal validation/test are non-empty.
- **Target fractions:** 5%, 10%, 25%, 50%, and 100%.
- **Seeds:** 42, 43, and 44 control initialization and recording-specific
  subset permutations.
- **Phase 4/5 fixed source composition:** same-species. Phase 6 separately
  compares minipig-only, monkey-only, and equal-total-volume 50/50 mixed data.
- **Checkpoint selections used in Phase 4 planning:** 1%, 3%, 10%, 30%, 100%
  of planned source compute, plus the best source-validation checkpoint.
- **WandB:** not applicable; Phase 0 performs no model training.

The fraction manifest stores the recording ID, seed, requested/realized
fraction, represented/absent class lists, per-class selected and total counts,
positional indices, stable content-derived interval IDs, a source-interval
hash, availability/failure reason, and a manifest hash. An absent mapped class
does not invalidate a fraction. Per-class permutations use a stable digest of
the recording ID in addition to the seed and class ID, so recordings with
equal class counts do not reuse the same permutation.

### Launch command

```bash
uv run python tools/audit_neurosoft_sessions.py \
  --data-root "$SCRATCH/brainsets/processed" \
  --min-class-support 3 \
  --min-present-classes 6 \
  --output-json docs/neurosoft-phase0-audit.json \
  --output-markdown docs/neurosoft-phase0-audit.md

uv run python tools/validate_neurosoft_splits.py \
  --data-root "$SCRATCH/brainsets/processed" \
  --output-json docs/neurosoft-phase0-split-validation.json

uv run python tools/validate_neurosoft_fractions.py \
  --data-root "$SCRATCH/brainsets/processed" \
  --min-class-support 3 \
  --min-present-classes 6 \
  --output-json docs/neurosoft-phase0-fraction-validation.json
```

These are read-only data audits, not Slurm training launches. Phase 1
production submissions must use a clean committed repository, set
`FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches`, use the
normal `python main.py ... -m` workflow, and default to the `long` partition.

### Key config overrides

- `split_type=intrasession-causal`
- `task_type=acoustic_stim`
- `min_class_support=3`
- `min_present_classes=6`
- `fractions=[0.05,0.10,0.25,0.50,1.00]`
- `seeds=[42,43,44]`
- `source_volume_fractions=[0.10,0.25,0.50,1.00]`

## Results

### Summary

Phase 0 passes. The full inventory is in the generated
[human-readable audit](../../docs/neurosoft-phase0-audit.md), with the
[content-hashed JSON manifest](../../docs/neurosoft-phase0-audit.json),
[split-validation record](../../docs/neurosoft-phase0-split-validation.json),
and [fraction-validation record](../../docs/neurosoft-phase0-fraction-validation.json).

The audit finds 40/41 eligible minipig recordings from all seven subjects and
13/27 eligible monkey recordings from five subjects: 53 recordings across 12
target subjects. The exclusions are protocol driven: fewer than six bands are
represented, a represented band has inadequate causal-training support, or the
represented set differs across splits. Ten of 265 otherwise planned
session/fraction cells are unavailable under the three-example support rule;
with three seeds this leaves 255 supported cells and 765 Phase 1 jobs.
The eligible class-count mix is 8/19/13 minipig and 2/3/8 monkey recordings
with 6/7/8 represented classes, respectively.

### Metrics

| Species | Configured sessions | Eligible sessions | Eligible target subjects | 6/7/8-class sessions | Signal hours | Channels |
|---|---:|---:|---:|---|---:|---:|
| Minipigs | 41 | 40 | 7 | 8/19/13 | 18.53 | 32 |
| Monkeys | 27 | 13 | 5 | 2/3/8 | 11.26 | 32 |

| Validation | Result |
|---|---|
| Block splits | 3 folds × both species: exact coverage, non-empty, disjoint |
| Causal splits | 68 recordings: exact coverage and disjoint; chronological within 812 domain segments |
| LOSO | 13 configured held-out subjects: exact subject isolation; test empty by design |
| Fraction manifests | deterministic, nested, unique stable interval IDs, validation/test invariant |
| Scientific cells | 765 available; 30 explicitly unavailable seed cells across five recordings |
| Source leakage | zero target-subject recordings in every target/composition source pool |

Target-specific same-species Phase 4 volume caps and Phase 5 diversity caps
are tabulated in the audit. After target exclusion, minipig targets support
diversity bins `[1, 2, 4, 6]` and monkey targets support `[1, 2, 4]`.
Partial-class sessions create a second design variable: the represented-class
union/intersection of each source set must be logged, and diversity comparisons
must match it where feasible or report and stratify the difference.

The corrected staged planning counts are 765 Phase 1 jobs, 765 Phase 2 jobs,
10 Phase 3 smoke jobs, and up to 3,960 / 11,160 / 18,504 cumulative Phase 4 jobs
at the 100%-only / three-fraction / full-grid gates. Later-phase counts and the
explicit timing assumptions are in the audit. GPU-hour values are planning
estimates and must be recalibrated with measured Phase 1 and Phase 3 pilots.

### Analysis

The audit script derives all counts directly from the processed artifacts and
task/data YAMLs; no run values are hardcoded. The JSON artifact includes hashes
of those YAMLs, every recording split, every target-specific source pool, and
the complete audit payload. Phase 0 has no WandB analysis because it launches
no model runs.

### Figures

None. The complete session inventory and source/run-planning tables are more
appropriate than plots for this protocol audit.

## Conclusions

The Phase 0 exit criteria are satisfied:

- all 68 configured recordings have an explicit eligible/ineligible status;
- all eligible fractions either meet support or are explicitly unavailable;
- deterministic nested manifests are tied to immutable interval identities;
- every target-specific source manifest proves target-subject exclusion;
- source-volume caps, feasible diversity bins, and equal-volume composition
  budgets are frozen from data support rather than model outcomes; and
- later-phase scientific run counts are staged and reviewed before submission.

The audit also prevents material design errors in the original draft: global
rather than target-specific source totals, a mixed condition with twice the
single-species example budget, an unnecessarily strict all-eight-class
eligibility rule, and metric aggregation that could penalize an undefined
absent class as though it were a model error.

## Notes for future experiments

- Create the Phase 1 one-hypothesis experiment file and link it here before
  launching EEGNet learning curves.
- Recalibrate GPU-hour assumptions after Phase 1 and Phase 3 pilots; retain the
  job-count formulas and measured inputs in the audit artifact.
- Implement and unit-test supported-class metric aggregation before Phase 1;
  retain eight output logits and treat predictions of absent classes as errors.
- Log source represented-class union/intersection in Phase 5, and match or
  stratify source-label coverage rather than confounding it with diversity.
- Commit this protocol, code, tests, and generated manifests before any Slurm
  submission so the launcher can create an immutable snapshot.

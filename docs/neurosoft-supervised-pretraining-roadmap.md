# NeuroSoft supervised pretraining roadmap

**Status:** In progress (Phase 0 completed)<br>
**Owner initials:** MS  
**Scope:** NeuroSoft minipigs and monkeys  
**Out of scope:** self-supervised pretraining and POYO architecture studies

## Objective

Determine when supervised pretraining becomes useful for single-session
NeuroSoft 8-band acoustic-stimulus decoding. The program will measure three
distinct benefits:

1. whether pretraining improves absolute downstream performance;
2. whether pretraining reduces the labeled data or optimization compute needed
   for downstream adaptation; and
3. whether the reduction in downstream compute ever amortizes the compute spent
   on pretraining, and after how many independent session adaptations.

The immediate objective of this branch is infrastructure, not pretrained model
results. The first runnable scientific phase is a protocol-matched EEGNet
learning-curve baseline over all eligible sessions.

## Research questions

The roadmap is organized around separate, falsifiable questions. Each question
that reaches execution should receive its own MS-authored experiment file in
`experiments/inbox/`, linked to the session-level EEGNet/GRU baseline lineage.

1. **EEGNet data efficiency:** How does single-session EEGNet performance and
   optimization efficiency change with 5%, 10%, 25%, 50%, and 100% of the
   causal training partition?
2. **Matched GRU scratch baseline:** What learning curve does the shared
   convolution--bidirectional-GRU architecture achieve without pretraining?
3. **Pretraining volume:** With source diversity fixed, how does the amount of
   supervised source data affect target-session adaptation?
4. **Pretraining diversity:** With source-example volume held approximately
   fixed, how does the number of source subjects affect adaptation?
5. **Species composition:** At matched labeled-data volume and compute, how do
   minipig-only, monkey-only, and 50/50 mixed-species pretraining transfer to
   each target species?
6. **Model scale:** Once the base recipe is stable, how do GRU width and depth
   alter the performance and compute break-even point?
7. **Compute amortization:** For every useful pretrained checkpoint, how many
   independent session adaptations are required to recover its pretraining
   cost relative to matched scratch training?

These questions must not be collapsed into a single omnibus experiment. The
staged design below keeps each comparison interpretable and limits the number
of runs launched before the pipeline is trustworthy.

## Fixed evaluation protocol

### Target construction

- Evaluate both minipigs and monkeys.
- Leave out one target subject at a time.
- Exclude every recording from the target subject from supervised pretraining,
  pretraining validation, checkpoint selection, and recipe selection.
- Adapt independently to every eligible session of the held-out subject. Every
  adaptation starts from the same source checkpoint; weights never carry from
  one target session into the next.
- Use `intrasession-causal` train/validation/test partitions for downstream
  adaptation. The stimulation protocol makes these splits only approximately
  causal because frequencies were presented in blocks, but the split remains
  the fixed protocol for this roadmap.
- Determine session eligibility before model training using data integrity and
  label coverage and minimum per-class support. A session is eligible when at
  least 6 of the 8 mapped bands are represented, the represented band set is
  identical in causal train/validation/test, and every represented band has at
  least three causal-training examples. Never exclude a session because its
  model metrics look poor or unusual.

### Downstream data fractions

Use nested training fractions:

```text
5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%
```

Fractions are drawn only from the causal training partition. Validation and
test intervals remain identical across fractions and seeds.

The subset builder must be deterministic and class-aware. It should sample each
represented class separately, keep the subsets nested, preserve approximately
the causal training distribution, and enforce a configured minimum number of
examples per represented class. An absent class is recorded in the manifest but
does not make the session unavailable. Each run must log the requested and
realized fraction, represented/absent class lists, per-class counts, selected
interval identifiers, and any eligibility failure. A fraction that cannot meet
the minimum support is reported as unavailable rather than being silently
rebalanced with validation/test data or replacement examples.

### Seeds

- Start with three seeds for every complete comparison. Use a single declared
  seed list everywhere (initial proposal: 42, 43, and 44).
- A seed controls parameter initialization and downstream subset selection.
- Diversity experiments must also replicate source-subject selection; do not
  treat three initializations of one favorable subject subset as three
  diversity replicates.
- Add seeds only after a preliminary story identifies important effects or a
  compute break-even boundary that needs stronger evidence.

### Metrics and checkpoint selection

- Keep the model output at eight logits. Do not mask absent-class logits at
  inference: predicting an absent class remains an error.
- Compute per-class metrics first, then macro-average only over classes with
  positive support in that session's evaluation split. This avoids treating an
  undefined no-positive AUROC as zero. Log the exact aggregation mask.
- Select downstream checkpoints using validation macro-F1 over the
  validation-supported classes.
- Report test macro-F1 over test-supported classes from the selected checkpoint
  as the primary absolute performance metric.
- Also report balanced accuracy, macro-AUROC, macro precision/recall, per-class
  metrics, class support, confusion matrices, represented/absent class lists,
  and `num_present_classes` for every session.
- Report results separately for 6-, 7-, and 8-class sessions, plus an 8/8-only
  sensitivity analysis. Do not interpret an unstratified absolute metric mean
  as if every session had the same task cardinality.
- Keep validation and test metrics distinct. Do not report the maximum test
  metric over epochs.
- Select the best pretraining checkpoint using the unweighted mean of each
  source session's supported-class validation macro-F1. This prevents the
  longest recordings from dominating selection while avoiding undefined
  absent-class metrics.
- Retain a best checkpoint for every pretraining data scale and species
  composition in addition to the fixed compute-milestone checkpoints.

For species summaries, report both:

1. a subject-balanced result (average sessions within each subject, then
   average subjects), as the primary species-level estimate; and
2. the unweighted distribution across sessions, for continuity with the
   existing session-level baseline report.

All pretrained-versus-scratch claims should be paired on target subject,
session, represented class set, data fraction, and seed.

## Efficiency definitions

### Absolute performance

For target session `s` and downstream fraction `d`:

```text
absolute_gain(s, d) = test_macro_F1_pretrained(s, d)
                    - test_macro_F1_scratch(s, d)
```

The primary scratch control uses the exact same convolution--BiGRU
architecture, downstream recipe, and data subset as the pretrained model.
EEGNet is an external reference and an early pipeline validator, not a
substitute for the matched scratch control.

### Labeled-data efficiency

Define the session-specific target as 80% of the mean full-data scratch
performance for that session:

```text
target_F1(s) = 0.8 * mean_seed(test_macro_F1_scratch(s, 100%))
```

For scratch and each pretrained condition, report the smallest tested data
fraction whose mean test macro-F1 reaches `target_F1(s)`. Treat unreached
targets as right-censored; do not extrapolate beyond 100%. Report the discrete
fraction directly. A later analysis may interpolate the learning curve, but it
must retain the observed-bin result.

### Optimization efficiency

For each run, report the cumulative downstream optimizer steps, processed
examples/windows, estimated FLOPs, and wall time required to first reach 80% of
that run's eventual best validation macro-F1. Also report the same quantities
at the selected best checkpoint. Epoch count is supplementary because epochs
contain different numbers of examples at different data fractions.

### Compute accounting

Every training run must record at least:

- parameter count and trainable parameter count;
- forward and training-step FLOP estimates for the realized input shape;
- optimizer steps and effective batch size;
- examples/windows and signal seconds processed;
- cumulative estimated FLOPs at every validation event and checkpoint;
- GPU type, numeric precision, wall time, and peak accelerator memory; and
- best-checkpoint step, examples, FLOPs, and elapsed time.

FLOPs are the primary hardware-independent compute measure. Wall time remains a
secondary operational measure and must only be compared directly on compatible
hardware. Before using FLOPs in scientific claims, validate profiler coverage
for the convolution and recurrent operations and document any analytic
correction. Store raw counters so the estimates can be recomputed later.

### Amortization

For a pretrained checkpoint `p`, let `C_pre(p)` be the compute used to create
it. At a fixed downstream performance target, estimate the paired mean costs
`C_scratch` and `C_ft(p)` per independent target-session adaptation.

```text
C_total_scratch(K) = K * C_scratch
C_total_pretrain(K, p) = C_pre(p) + K * C_ft(p)
```

When `C_ft(p) < C_scratch`, the nominal break-even count is:

```text
K_break_even(p) = ceil(C_pre(p) / (C_scratch - C_ft(p)))
```

Report a variable-`K` curve and mark the number of eligible target sessions in
the benchmark. If finetuning does not save compute at matched performance,
report that the checkpoint never amortizes under the measured regime. Keep
performance-improving but non-amortizing checkpoints on the performance/compute
Pareto frontier rather than declaring them failures.

Compute amortization and labeled-data efficiency are separate. The initial
analysis counts training FLOPs, not the monetary or experimental cost of
collecting and labeling data.

## Data-scale design

Run a data audit before freezing exact bins. The audit must inventory subjects,
sessions, usable causal-train examples, signal duration, per-class support, and
channel count for both species. It may adjust bins only for documented support
constraints.

### Source volume axis

- Fix the full eligible source-subject set.
- Use 10%, 25%, 50%, and 100% of its available causal training examples.
- Sample per source and class using deterministic nested manifests.
- Use source validation data only for early stopping and checkpoint selection;
  do not scale validation data with the training fraction.

### Source diversity axis

- Compare 1, 2, 4, and all available source subjects.
- Hold total labeled source examples approximately constant across diversity
  bins, capped by what the smallest valid bin can supply.
- Balance the allocation across selected subjects and classes.
- Replicate which source subjects are selected. Record the complete selection
  manifest in every run.
- Record the represented-class union and intersection for each selected source
  set. Match aggregate source-label coverage across diversity bins when
  feasible; otherwise stratify by it and report it as a design difference, so
  gains from broader label coverage are not attributed to subject diversity.

### Species-composition axis

At equal labeled-example volume and approximately equal estimated compute,
compare:

- minipig-only pretraining;
- monkey-only pretraining; and
- 50/50 minipig/monkey pretraining.

For a minipig target, minipig-only and mixed conditions exclude that minipig;
the monkey-only condition may use eligible monkeys. Apply the symmetric rule
for monkey targets. Allocate the mixed condition by examples/windows rather
than raw session count. Only add 25/75 and 75/25 mixtures if the 50/50 result
shows a reason to study the ratio.

## Model and recipe boundaries

The shared model is intentionally underspecified here because its detailed
design is being developed separately. The roadmap assumes only:

- a simple convolutional frontend;
- a bidirectional multilayer GRU;
- support for session-specific input heads when channel counts differ;
- a shared 8-band classification output; and
- configurable width and depth with parameter/FLOP reporting.

The model must support initializing a new target-session input head while
loading the shared pretrained weights. Downstream finetuning updates the new
input head and the shared model. More detailed convolution, routing, and
initialization choices belong in the model implementation document.

Avoid broad hyperparameter sweeps initially. Start from the existing
session-level recipes, make only changes required for stable execution, and
freeze the recipe before scientific comparisons.

Initial fixed downstream references are:

| Setting | EEGNet | Existing GRU starting point |
|---|---:|---:|
| Batch size | 16 | 16 |
| Learning rate | 0.015 | 0.0015 |
| Weight decay | 0.018 | 0.018 |
| Maximum epochs | 1000 | 1000 |
| Early-stopping monitor | validation macro-F1 | validation macro-F1 |
| Early-stopping patience | 20 | 20 |
| Sequence length | 0.5 s | 0.5 s |

The colleague-owned convolution--BiGRU may replace the existing GRU
architecture while retaining a single declared starting recipe. Record that
recipe in a Hydra config before running scratch/pretrained comparisons. If
smoke tests show marked instability or strong recipe dependence, pause and
create a separate targeted recipe experiment; do not tune each data fraction
or pretraining condition independently.

Pretraining should use the same principle: one declared optimizer/scheduler,
maximum budget, early-stopping rule, and source-session validation aggregator.
The initial endpoint may be early stopping, but retain checkpoints near 1%, 3%,
10%, 30%, and 100% of the planned full compute budget, plus the best checkpoint.
Record realized rather than nominal compute at each milestone.

## Execution phases

### Phase 0 -- Protocol and data audit

**Status:** Completed 2026-08-26. See the
[protocol and results](../experiments/inbox/20260826-neurosoft-supervised-pretraining-protocol.md),
[full data audit](neurosoft-phase0-audit.md), and machine-readable
[audit](neurosoft-phase0-audit.json),
[split validation](neurosoft-phase0-split-validation.json), and
[fraction validation](neurosoft-phase0-fraction-validation.json) artifacts.

Audit-adjusted decisions: sessions with at least 6/8 represented bands are
eligible, yielding 40/41 minipig and 13/27 monkey recordings across 12 target
subjects. The eligible class-count mix is 8/19/13 minipig and 2/3/8 monkey
sessions with 6/7/8 represented classes, respectively. Ten fraction cells
across five otherwise eligible recordings remain
unavailable under the three-example support rule, leaving 255 supported
session/fraction cells and 765 three-seed Phase 1 jobs. Phase 4/5 hold species
composition fixed to same-species sources. After target exclusion, minipigs
support diversity bins 1/2/4/all (all=6) and monkeys support 1/2/4 (all=4).

Deliverables:

- session/subject/class/channel inventory for both species;
- preregistered eligibility and minimum-class-support rules;
- fixed seeds and nested subset manifest format;
- final source-volume caps and diversity bins; and
- run-count and compute estimate for each later phase.

Exit criteria:

- every proposed target session has a documented eligible/ineligible status;
- every 5% target subset either satisfies support constraints or fails loudly;
- manifests prove that target subjects cannot appear in source data; and
- scientific run counts are reviewed before submission.

### Phase 1 -- Downstream pipeline and EEGNet learning curves

Implement and validate:

- class-aware nested within-session training fractions;
- causal single-session configs for both species;
- three-seed Hydra multiruns over all eligible sessions and five fractions;
- immutable subset and split manifests;
- best-validation checkpoint test evaluation;
- FLOP/example/step/wall-time instrumentation; and
- WandB metadata and an API-backed analysis script.

Run EEGNet at 5%, 10%, 25%, 50%, and 100% on every eligible session. Produce
absolute performance, data-to-80%, optimization-to-80%, and time/compute-to-best
tables and curves.

Exit criteria:

- repeated manifests are deterministic and nested;
- validation/test sets are invariant across fractions;
- all metrics and compute counters are recoverable from WandB;
- supported-class metric aggregation matches a hand-computed reference and
  every run logs its represented class set and class count;
- 6-, 7-, and 8-class summaries and the 8/8 sensitivity analysis can be
  regenerated from the same run records;
- aggregate results can be regenerated without hardcoded run values; and
- failures and ineligible cells are explicit rather than silently omitted.

This is the first experiment to create under `experiments/inbox/`.

### Phase 2 -- Matched convolution--BiGRU scratch baseline

After the colleague-owned model is available:

- test variable channel counts and session-head routing;
- test fresh target-head initialization and strict/shared checkpoint loading;
- expose width/depth and compute metadata without launching a scale sweep;
- lock one starting training recipe; and
- reproduce the Phase 1 fraction/seed/session matrix from scratch.

EEGNet remains the external reference. The Phase 2 model becomes the matched
scratch control for all claims about pretraining.

Exit criteria:

- the same data and seed manifests can drive EEGNet and GRU runs;
- model transfer cannot accidentally load a source-session input head into an
  incompatible target session;
- full-data scratch training is stable across the three seeds; and
- no hyperparameter sweep is needed to obtain interpretable learning curves.

### Phase 3 -- Supervised pretraining pipeline

Build source training with:

- leave-one-subject-out source manifests;
- multisession, intrasession-causal source train/validation data;
- session-specific input heads and a shared model;
- unweighted source-session validation macro-F1;
- fixed compute-milestone checkpoints and a best checkpoint;
- permissive loading of shared weights plus a fresh target-session input head;
  and
- independent target-session finetuning jobs for every fraction and seed.

Run small end-to-end smoke tests only. No scientific pretraining claim should be
made until checkpoint provenance, leakage tests, and compute accounting pass.

### Phase 4 -- Pretraining volume

With source diversity and species composition fixed, compare the four source
volume bins and their intermediate/best checkpoints. Evaluate a staged subset
of downstream conditions first:

1. 100% downstream data on all eligible sessions;
2. 5%, 25%, and 100% downstream data if transfer is non-degenerate; and
3. the full downstream fraction grid only for informative source scales.

This phase establishes whether additional source examples improve absolute
performance, downstream data efficiency, optimization efficiency, or compute
amortization.

### Phase 5 -- Source-subject diversity

At matched source-example volume, compare 1, 2, 4, and all available subjects.
Replicate source-subject selection. Keep architecture, recipe, species
composition, and downstream evaluation fixed. This phase must remain separate
from the volume experiment so diversity is not confounded with more examples.
It must also match source-label coverage across bins or report and stratify the
coverage difference explicitly.

### Phase 6 -- Species composition and transfer direction

At matched source volume and compute, run the matrix:

| Pretraining source | Minipig target | Monkey target |
|---|---:|---:|
| Minipigs only | yes | yes |
| Monkeys only | yes | yes |
| 50/50 mixed | yes | yes |

Report same-species, cross-species, and mixed-species effects separately. A
mixed model is useful only if its improvement is not explained solely by
extra examples or compute.

### Phase 7 -- Width/depth and optional follow-ups

Only after one pretraining condition transfers reliably, compare a small set of
declared model scales. Prefer three coherent presets (small/default/large) over
an immediate Cartesian sweep. Each preset must report parameters, FLOPs,
absolute performance, data-to-80%, and break-even `K`.

Potential follow-ups, triggered only by evidence:

- 25/75 and 75/25 species mixtures;
- additional seeds around break-even boundaries;
- a targeted learning-rate or regularization study if the fixed recipe is
  demonstrably unstable; and
- denser downstream data fractions around the observed 80% crossing.

## Required configuration and code work

The setup branch should leave the following reusable pieces:

1. **Fraction manifest builder** -- deterministic, nested, class-aware causal
   training subsets with audit output.
2. **Eligibility audit** -- data-quality, per-class support, signal duration,
   and channel-count checks before launch.
3. **Experiment config family** -- species, target subject/session, seed,
   downstream fraction, source manifest, source composition, source volume,
   diversity bin, architecture preset, and checkpoint milestone exposed as
   Hydra dimensions.
4. **Transfer policy** -- explicit shared-weight loading and fresh target input
   head initialization, with tests for variable channel counts.
5. **Checkpoint schedule** -- realized-compute milestones plus best source
   validation checkpoint.
6. **Compute callback** -- cumulative steps/examples/signal seconds/FLOPs/wall
   time and best-checkpoint annotations.
7. **Evaluation path** -- test metrics evaluated once from the selected
   validation checkpoint.
8. **Analysis schema** -- stable WandB keys and API-backed analysis scripts for
   supported-class metrics, class-count-stratified paired learning curves,
   time-to-80%, time-to-best, Pareto plots, and variable-`K` amortization.
9. **Validation tests** -- target leakage, subset nesting, determinism, class
   support, split invariance, checkpoint provenance, and compute-counter
   monotonicity.

Avoid encoding large subject/session lists directly into experiment logic.
Generate and version small manifests so every scientific cell has inspectable
data provenance.

## Run metadata

Every run should log enough information to reconstruct its scientific cell:

- target species, subject, and session;
- source species composition and source subject/session manifest hash;
- represented-class union/intersection and class count for the source manifest;
- requested and realized source/target fractions and class counts;
- represented/absent target classes, `num_present_classes`, and the metric
  aggregation mask;
- model architecture ID, width/depth preset, and parameter counts;
- seed and all subset-selection seeds;
- split type and task;
- pretraining run/checkpoint ID, checkpoint selection type, and source compute;
- downstream best step/epoch/examples/FLOPs/time;
- hardware and precision; and
- Git commit and launch snapshot bundle.

Use stable WandB groups for each one-hypothesis experiment and human-readable
run names that include target species/subject/session, data fraction, seed, and
checkpoint scale without relying on the name as the sole source of metadata.

## Statistical analysis

- Treat subjects and sessions as experimental units; seeds measure training
  variability, not additional biological samples.
- Use paired comparisons within target session and seed.
- Report effect sizes and uncertainty, not only grand means.
- Use a hierarchical bootstrap (subjects, then sessions within subjects) for
  species-level confidence intervals where practical.
- Keep minipig and monkey results separate before presenting a combined view.
- Stratify absolute metrics by 6/7/8 represented classes and include the 8/8
  sensitivity analysis; class count is a session property, not a seed-level
  replicate.
- Report the fraction of sessions helped/hurt as well as the mean effect.
- For source-diversity experiments, include variability across selected source
  subject sets.
- Mark missing, failed, unsupported, and target-not-reached cells explicitly.

## Launch discipline

Before production submissions:

1. require `git status --short` to be empty and commit the relevant configs,
   code, manifests, and experiment hypothesis file;
2. set `FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches`;
3. use the normal `python main.py ... -m` Hydra workflow;
4. submit Slurm jobs to the `long` partition unless another partition is
   explicitly chosen; and
5. record the Slurm job ID and immutable snapshot bundle path in the experiment
   file immediately after submission.

Do not disable launch snapshots or bypass the clean-Git requirement for
production jobs.

## Decision gates

Stop or narrow the matrix when any of the following occurs:

- **Pipeline gate:** class support, leakage, test evaluation, or compute
  accounting cannot be verified.
- **Recipe gate:** the fixed recipe is unstable enough that seed variance
  obscures the scientific factor; create a small targeted recipe experiment.
- **Transfer gate:** pretrained models fail to match scratch at 100% downstream
  data; diagnose before evaluating every smaller fraction.
- **Compute gate:** a checkpoint cannot save downstream compute and does not
  improve performance; do not expand it to more mixtures or model scales.
- **Scale gate:** mixed-species gains disappear under equal-volume/equal-compute
  controls; do not launch mixture-ratio follow-ups.

## Completion criteria

This roadmap is complete when the repository can reproducibly answer, for each
species and pretraining composition:

- whether test macro-F1 improves over matched scratch;
- how much causal-session labeled data is needed to reach 80% of the full-data
  scratch reference;
- how much downstream compute is needed to reach 80% of eventual best and the
  selected best checkpoint;
- how performance changes with source volume and source-subject diversity;
- whether same-species, cross-species, or mixed-species sources transfer best;
- the performance/compute Pareto frontier across pretraining checkpoints and
  model scales; and
- whether pretraining amortizes, the corresponding `K`, or a clear statement
  that it does not amortize in the measured regime.

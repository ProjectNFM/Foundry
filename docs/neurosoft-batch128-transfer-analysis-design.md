# Batch-128 NeuroSoft transfer analysis design

## Purpose

This document is the implementation specification for analyzing the completed
batch-128 NeuroSoft supervised-pretraining and downstream-transfer runs. It is
intended for the agent that will reorganize the experiment markdowns, implement
the shared analysis code, fetch and audit W&B results, render figures and
tables, and fill the experiment reports.

The scientific program has two co-primary downstream questions:

1. **Final performance:** does transfer change held-out test supported-class
   macro-F1 relative to an exactly matched scratch model?
2. **Optimization efficiency:** does transfer reach matched scratch validation
   quality in fewer downstream optimizer steps?

The batch-128 reference is the definitive comparator. The older batch-16 Phase
4 trajectory is historical replication context only. Do not pool batch-16 and
batch-128 runs or use the batch-16 estimates as controls for a new intervention.

This design deliberately avoids hard hypothesis-verdict thresholds. Report the
direction, magnitude, uncertainty, subject consistency, and trajectory shape.

## Scope and non-goals

- Use source checkpoints at optimizer steps `100`, `300`, `1_000`, `3_000`,
  and `10_000`.
- Analyze minipigs and monkeys separately.
- Use only the prespecified target-test supported macro-F1 and validation-F1
  histories. Do not select a source checkpoint from target-test performance.
- Keep the completed source-pretraining architecture-viability experiment and
  its analysis mostly intact.
- Do not create a new hypothesis-bearing synthesis experiment.
- Do not report p-values or label intervals as significant/non-significant.
- Do not use wall time, validation-curve AUC, selected-best-checkpoint step, or
  time to a run's own peak as optimization-efficiency endpoints.

## Final experiment organization

The final downstream program should contain four hypothesis-bearing experiment
files plus the existing source-pretraining viability experiment.

### 1. Source-pretraining architecture viability (retain)

Keep
`experiments/07-neurosoft-transferability/20260916-MS-pretraining-architecture-viability.md` and
`analysis/20260916-MS-pretraining-architecture-viability_analysis.py` mostly
intact. This experiment remains the source-training prerequisite and reference:
all five source conditions completed, learned non-degenerate source-task
solutions, and published the fixed checkpoint inventory.

Required markdown changes:

- Set the dedicated batch-128 reference replication as its downstream
  follow-up. The three intervention experiments follow from that reference.
- Remove links to the deleted synthesis experiment and superseded separate
  small/large reports.
- Keep the existing source CE/F1, readiness, provenance, parameter-count, and
  compute results.
- Its source figures may be referenced from downstream reports rather than
  duplicated.

### 2. Batch-128 reference transfer replication

Create a dedicated experiment file with stem:

`20260916-MS-batch128-reference-transfer-replication`

Suggested question:

> Does the batch-128 reference reproduce the earlier qualitative result that
> early source checkpoints provide positive downstream transfer and that this
> benefit changes or erodes with continued source training?

Use the batch-128 reference transfer and reference scratch groups. The
historical batch-16 Phase 4F trajectory is shown only as a labeled contextual
overlay.

Set the source-pretraining viability experiment as this file's parent. Set the
model-scale, adapter-bias, and shared-adapter experiments as its follow-ups.

### 3. Model-scale transfer

Merge the current small- and large-backbone experiment files into one file with
stem:

`20260916-MS-model-scale-transfer`

Suggested question:

> How does transferable-backbone scale change absolute target performance,
> transfer benefit, and downstream optimization efficiency across source
> checkpoint age?

The conditions are small, batch-128 reference, and large. Treat scale as one
ordered intervention, not two independent experiments.

Set the batch-128 reference replication as this file's parent. Link the source
viability experiment in Background as the origin of all three source models.

### 4. Adapter-bias transfer

Retain and revise the current file with stem:

`20260916-MS-bias-free-transfer`

Suggested question:

> Does removing recording-specific adapter bias during source training preserve
> downstream transfer, and does matching the target adapter to the bias-free
> source architecture add further benefit?

The three displayed conditions are:

1. batch-128 reference;
2. bias-free source with a fresh ordinary biased target adapter; and
3. bias-free source with a fresh bias-free target adapter.

These treatments belong in one experiment because the source intervention and
source-target adapter interaction use the same run matrix, estimands, and plots.

Set the batch-128 reference replication as this file's parent. Retain the older
adapter-bias perturbation report as motivating context rather than its parent.

### 5. Shared-adapter transfer

Retain and revise the current file with stem:

`20260916-MS-shared-adapter-transfer`

Suggested question:

> Does pretraining through a shared padded input adapter preserve downstream
> transfer, and is additional benefit obtained by retaining the pretrained
> shared input interface during target finetuning?

The three displayed conditions are:

1. batch-128 reference;
2. shared-adapter source with a fresh ordinary target adapter; and
3. shared-adapter source with the pretrained shared adapter retained.

Set the batch-128 reference replication as this file's parent. Link the source
viability report for the shared-adapter source-training evidence.

### Files to remove after content and links are migrated

- `experiments/inbox/20260916-MS-small-backbone-transfer.md`
- `experiments/inbox/20260916-MS-large-backbone-transfer.md`
- `experiments/inbox/20260916-MS-transfer-failure-mode-synthesis.md`
- Their superseded analysis stubs, after the model-scale script and shared core
  replace them.

Update every parent/follow-up link affected by the moves. Do not leave a
hypothesis-bearing synthesis report. A future archived group README may provide
navigation and a non-inferential summary after the individual reports are
complete.

## Run sources and expected matrices

The immutable compiled cell lists are the source of truth for expected run
identity, matching, treatment labels, checkpoints, and W&B IDs:

- `launch/architecture_transfer/reference-backbone-{minipigs,monkeys}.jsonl`
- `launch/architecture_transfer/small-backbone-{minipigs,monkeys}.jsonl`
- `launch/architecture_transfer/large-backbone-{minipigs,monkeys}.jsonl`
- `launch/architecture_transfer/bias-free-{minipigs,monkeys}.jsonl`
- `launch/architecture_transfer/shared-adapter-{minipigs,monkeys}.jsonl`

Use the adjacent lock files to verify file hashes, recipe hashes, registry
hashes, counts, and condition inventories.

Expected counts across both species:

| Matrix | Transfer runs | New scratch runs | Total cells |
|---|---:|---:|---:|
| Reference | 795 | 159 | 954 |
| Small | 795 | 159 | 954 |
| Large | 795 | 159 | 954 |
| Bias-free | 1,590 across two target treatments | 159 bias-free scratch | 1,749 |
| Shared adapter | 1,590 across two target treatments | 159 shared scratch | 1,749 |
| **Total compiled cells** |  |  | **6,360** |

The ordinary-target bias-free and shared-source treatments use the appropriate
ordinary/reference scratch controls in the analysis even though those controls
are stored in the reference matrix rather than duplicated in their JSONL files.

Exact W&B groups are encoded in the JSONL cells. Do not maintain a second
handwritten group list when the run IDs are already available in the compiled
design.

Historical batch-16 replication context comes from:

- `experiments/07-neurosoft-transferability/20260915-MS-source-validation-downstream-trajectory.md`
- `analysis/20260915-MS-source-validation-downstream-trajectory.py`
- its generated non-versioned CSV caches.

## Shared analysis architecture

Implement one reusable module, suggested path:

`analysis/_architecture_transfer_analysis.py`

and four thin experiment entry points:

- `analysis/20260916-MS-batch128-reference-transfer-replication_analysis.py`
- `analysis/20260916-MS-model-scale-transfer_analysis.py`
- `analysis/20260916-MS-bias-free-transfer_analysis.py`
- `analysis/20260916-MS-shared-adapter-transfer_analysis.py`

Each entry point must be runnable directly with `uv run python`, fetch results
through `wandb.Api()`, print its principal tables, and write artifacts with its
own experiment stem. It may call the shared module; it must not import the main
`foundry` package.

### Responsibilities of the shared module

1. Load and hash-audit compiled JSONL and lock files.
2. Resolve exact run IDs from compiled cells; never discover the scientific
   matrix by a broad W&B group query.
3. Verify for every run:
   - run ID, name, group, condition, species, recording, subject, target seed;
   - source condition, source checkpoint step, manifest hash, checkpoint hash;
   - transfer regime, target adapter treatment, scratch family;
   - finished state and complete final test/validation history.
4. Fail on missing or duplicate expected cells. Any exception must be
   documented in the relevant experiment markdown before analysis continues.
5. Recover a missing public summary metric from that run's exact synced W&B
   summary file only when the run is otherwise complete. Never hardcode it.
6. Cache raw run endpoints and validation histories under `analysis/csv/`.
7. Construct exact transfer-to-scratch matches from `matched_scratch_id` plus
   recording, target seed, architecture/treatment, and target fraction.
8. Produce the common performance, efficiency, uncertainty, sensitivity, and
   plotting data described below.
9. Apply one shared color, label, sign, axis, and bootstrap convention.

### Canonical extracted columns

At minimum, retain:

- provenance: `cell_id`, `run_id`, `run_name`, `group`, `state`;
- design: species, excluded subject, target recording, target seed, source
  condition, source step, target treatment, transfer regime, scratch family;
- hashes: checkpoint set, checkpoint ID, manifest hash, checkpoint hash;
- endpoint: validation-selected held-out test supported macro-F1;
- validation history: optimizer step, epoch if present, validation supported
  macro-F1, processed windows, and cumulative FLOPs when available;
- training policy: maximum epochs, early-stopping settings, effective batch
  size, FLOPs per window, and architecture parameter counts.

Write a machine-readable coverage table before computing any scientific result.

## Common statistical unit and uncertainty

### Hierarchical aggregation

For each transfer condition and source checkpoint:

1. Match transfer and scratch within target recording and target seed.
2. Compute the paired endpoint at target-seed level.
3. Average target seeds within each recording.
4. Average recordings within each excluded subject.
5. Weight excluded subjects equally in the species summary.

Target seeds are optimization replicates, not biological samples. Never treat
recordings, target seeds, source checkpoints, or overlapping source pools as
independent subjects.

### Bootstrap

- Use 20,000 non-parametric whole-subject resamples with a fixed seed.
- Resample an excluded subject with its complete checkpoint trajectory and all
  treatments, preserving paired intervention contrasts.
- Report pointwise 95% intervals.
- Analyze minipigs and monkeys separately.
- Describe intervals as stability across the available excluded subjects, not
  precise population inference: there are only seven minipig and five monkey
  exclusion units and the source pools overlap.
- Do not show p-values or significance stars.

### Consistent subject identity

Assign every subject a consistent color within species across all experiment
files and figures. Subject trajectories are thin and translucent. Condition
means are visually dominant and have bootstrap bands. Use identical y-limits
when conditions are intended to be compared.

## Final-performance estimands

### Primary endpoint

Use held-out test
`neurosoft_acoustic_stim_8band_supported_f1` from the downstream checkpoint
selected only by target-validation supported F1.

Do not promote accuracy, test cross-entropy, or validation F1 to co-primary
endpoints. Predictions into absent classes remain errors under the established
supported-F1 implementation.

### Canonical transfer effect

For each target recording and seed:

`transfer test supported F1 - exact treatment-matched scratch test supported F1`

Scratch matching is:

| Transfer condition | Scratch comparator |
|---|---|
| Reference | Reference architecture scratch |
| Small | Small architecture scratch |
| Large | Large architecture scratch |
| Bias-free source -> ordinary biased target | Ordinary/reference target scratch |
| Bias-free source -> bias-free target | Bias-free target scratch |
| Shared source -> fresh ordinary target | Ordinary/reference target scratch |
| Shared source -> retained shared target | Shared-padded target scratch |

This contrast isolates the contribution of pretrained initialization from the
target architecture's scratch performance.

### Descriptive checkpoint changes

Report, but do not use as hard gates:

- step 300 minus step 100 transfer effect (early change);
- step 10,000 minus step 100 transfer effect (full-budget change).

The full five-checkpoint trajectory is the evidence. A step-10,000 decline may
reflect ordinary source overfitting and must not be described as proof that the
architecture or representation is intrinsically bad.

### Direct intervention contrasts

Compute paired, subject-level differences of transfer effects across every
source checkpoint:

- Scale: small minus reference; large minus reference.
- Bias: bias-free-source/ordinary-target minus reference; bias-free-target
  minus ordinary-target within the bias-free source condition.
- Shared: shared-source/ordinary-target minus reference; retained shared target
  minus ordinary target within the shared source condition.

These full trajectories replace a step-10,000-only intervention forest plot.

## Optimization-efficiency estimands

The efficiency question is operational:

> Under the established downstream training policy, how many optimizer steps
> does transfer save when adapting to a new session before reaching matched
> scratch validation quality?

### Per-seed matched target

For each exact scratch run:

1. Sort validation history by optimizer step and deduplicate steps, keeping the
   last logged value.
2. Smooth validation supported F1 with a trailing three-evaluation median.
3. Define the primary target as `0.90 * maximum(smoothed scratch F1)`.
4. Apply that identical absolute target to the scratch run and every exactly
   matched transfer run.
5. Define crossing as the first observed validation step whose smoothed F1 is
   at or above the target. Do not interpolate and do not require additional
   consecutive crossings.

Repeat the complete calculation at targets `0.80` and `0.95` for sensitivity
analysis. Detect crossing per target seed before any seed averaging.

### Planned-budget cap for non-attainment

An early-stopped run that never reaches the target must not look efficient.
Assign it the full planned optimizer-step budget for its recording.

Reconstruct the planned budget as follows:

1. Read `trainer.max_epochs` from the run config; it should be 200.
2. Infer optimizer steps per epoch from the validation-history step/epoch
   cadence or compute-history counters. Use the median positive step increment
   across complete consecutive epochs.
3. Multiply by the configured maximum epochs, respecting any documented final
   partial-batch behavior.
4. Verify that all runs sharing recording, target seed, target fraction, and
   data policy yield the same planned budget regardless of transfer condition.
5. Verify the reconstruction against runs that completed the full 200 epochs.
6. Treat an inability to reconstruct a unique budget as an audit failure, not
   an invitation to use the observed early-stopping step.

The cap is recording-specific because recordings have different steps per
epoch. Preserve both the absolute optimizer-step value and its fraction of the
planned recording budget.

### Primary efficiency values

For each run and threshold:

- `capped_time_to_target = crossing_step` if attained;
- `capped_time_to_target = planned_step_budget` otherwise;
- `attained = 1` if crossed, else `0`.

For a matched pair:

- `steps_saved = scratch capped time - transfer capped time`;
- `fraction_budget_saved = steps_saved / planned_step_budget`.

Positive values favor transfer. Aggregate seeds, recordings, subjects, and
bootstrap subjects using the same hierarchy as performance.

Use the plain-language names **capped steps to matched scratch quality** and
**optimizer steps saved versus scratch**. Do not foreground RMST or
Kaplan-Meier terminology.

### Finetuning-dynamics curves

The secondary dynamics view should mean:

> What was the best smoothed validation quality available by a given optimizer
> budget?

For each run:

1. Use the trailing-three-evaluation median series.
2. Take its cumulative maximum.
3. Forward-fill between validation observations on a shared step grid.
4. After early stopping, carry the best available value forward through that
   recording's planned budget.
5. Normalize by the exact matched scratch maximum:
   `normalized_progress = cumulative_best_f1 / matched_scratch_maximum`.

Thus `1.0` is the matched scratch run's best smoothed validation F1. Show the
primary efficiency target as a horizontal line at `0.9`. Average seeds,
recordings, and subjects in the established order and bootstrap subjects. This
plot is descriptive; do not reduce it to an AUC.

### Compute-adjusted efficiency for scale

Only the model-scale experiment needs compute-to-target annex metrics:

- windows processed to matched quality;
- cumulative training FLOPs to matched quality;
- paired windows and FLOPs saved versus architecture-matched scratch.

Use the same crossing and planned-budget cap. Interpolate cumulative counters
only by exact step-proportional accounting when the per-step/window quantity is
constant and audited. Do not analyze or plot wall-clock time.

## Common visual grammar

### Performance figures

**P1. Subject-level transfer-effect trajectories (main).**

- Rows: species.
- Columns: experiment conditions.
- X: source step on a log scale with all five checkpoints labeled.
- Y: transfer minus matched scratch test supported macro-F1, preferably shown
  in percentage points.
- Thin colored lines: subjects; thick neutral line and band: mean and 95% CI.
- Horizontal zero line.

**P2. Condition comparison and intervention contrast (main).**

- Upper portion: species-specific condition mean trajectories overlaid without
  subject lines.
- Lower portion: direct paired intervention-contrast trajectories described
  above.
- Use consistent condition colors and the same source-step axis as P1.

**P3. Absolute performance (annex).**

- Absolute subject-balanced transfer test F1 across source checkpoints.
- Matched scratch shown as the target-architecture reference.
- Mean and bootstrap interval; no individual subject lines required.
- Purpose: distinguish transfer effects from target-architecture effects.

**P4. Recording heterogeneity heatmap (annex).**

- Rows: recordings grouped and labeled by subject.
- Columns: source checkpoints.
- Fill: transfer minus matched scratch F1.
- Separate species and conditions; use one diverging scale centered at zero.

**P5. Target-seed dispersion (annex).**

- Show the distribution of paired seed-level effects after centering within
  recording.
- Do not attach biological confidence intervals to seed distributions.

**P6. Classwise effects (annex).**

- Rows: eight stimulus-frequency bands.
- Columns: source checkpoints.
- Fill: transfer minus matched scratch per-class F1 reconstructed from test
  confusion counts.
- Annotate support in subjects or recordings; exclude unsupported cells rather
  than treating them as zero.
- Keep compact and descriptive.

**P7. Source-to-downstream association (annex).**

- Relate source validation CE or supported F1 to downstream transfer effect.
- Connect checkpoints within excluded subject and separate species.
- Label as descriptive, not causal, and never use it to select a checkpoint.

### Efficiency figures

**E1. Subject-level efficiency trajectories (main).**

- Same row/column/source-step grammar as P1.
- Upper panel: optimizer steps saved versus matched scratch.
- Lower panel: percentage of matched recording-seed runs attaining the target
  within budget, aggregated through subjects.
- Zero line for steps saved; 0--100% scale for attainment.

**E2. Performance-efficiency trade-off (main).**

- X: transfer minus scratch test F1.
- Y: optimizer steps saved to matched scratch quality.
- Zero lines form four labeled quadrants: faster/better, faster/worse,
  slower/better, slower/worse.
- Connect the five source checkpoints in order within condition.
- Separate species; show uncertainty without obscuring the trajectory.

**E3. Normalized finetuning dynamics (secondary).**

- A 2 x 5 grid: species rows, source-checkpoint columns.
- X: downstream optimizer step.
- Y: normalized cumulative-best validation progress.
- Horizontal target line at `1.0`.
- Solid transfer and dashed matched-scratch curves with consistent condition
  colors and subject-bootstrap bands.

**E4. Cumulative target attainment (annex).**

- A 2 x 5 grid: species rows, source-checkpoint columns.
- X: downstream optimizer step.
- Y: cumulative percentage attaining the matched target.
- Non-attainers remain in the denominator through planned budget.
- Show conditions and matched scratch consistently.

**E5. Threshold sensitivity (annex).**

- Compact comparison of mean steps saved and final attainment for 80%, 90%,
  and 95% scratch targets.
- The 90% result remains primary.

## Common tables

### Coverage and provenance table

For every experiment, print and cache expected/found counts by species, group,
condition, source step, treatment, target seed, and state. Include missing,
duplicate, hash-mismatched, or history-incomplete cells. Scientific plotting
must not proceed when this audit is nonempty.

### Performance table

For every species, condition, and source checkpoint:

- mean subject-level transfer-minus-scratch F1;
- pointwise 95% whole-subject bootstrap interval;
- subject median, minimum, and maximum;
- number of subjects above scratch;
- absolute transfer F1;
- absolute matched-scratch F1.

Also report the descriptive 300-minus-100 and 10,000-minus-100 changes.

### Efficiency table

For every species, condition, and source checkpoint at the 90% target:

- mean capped scratch time-to-target;
- mean capped transfer time-to-target;
- mean optimizer steps saved;
- percent of planned budget saved;
- pointwise 95% whole-subject bootstrap interval;
- subject median, minimum, and maximum;
- number of subjects with positive mean steps saved;
- transfer attainment rate.

Put the corresponding 80% and 95% summaries in an annex table.

## Per-experiment figure and analysis design

### A. Batch-128 reference replication

#### Main scientific story

Establish the batch-128 reference trajectory before interpreting any
intervention. Ask whether transfer is positive relative to scratch at early
checkpoints, how it changes across the complete source-training trajectory,
and whether its qualitative behavior resembles the historical batch-16 result.

#### Main figures

1. **P1 reference subject trajectories:** two species panels showing all
   subject-level transfer effects over the five source checkpoints.
2. **Historical replication overlay:** batch-128 mean trajectory in color and
   batch-16 Phase 4F mean trajectory in gray, separately by species. Each is
   normalized to its own matched scratch control. Do not pool them or imply
   independent experiments.
3. **E1 reference efficiency:** per-subject steps saved plus attainment across
   source checkpoints.
4. **E2 reference trade-off:** checkpoint trajectory through final-performance
   and steps-saved space.
5. **E3 reference dynamics:** full 2 x 5 normalized-progress grid.

#### Annex

- P3 absolute reference transfer versus reference scratch.
- P4 recording heatmap.
- P5 target-seed dispersion.
- P6 classwise heatmap.
- P7 source CE/F1 versus transfer effect.
- E4 cumulative attainment and E5 threshold sensitivity.

#### Interpretation requirements

- Use graded language: replicated direction, magnitude, subject consistency,
  and trajectory shape.
- Do not require both species' intervals to exclude zero.
- Explicitly describe batch size and source-seed differences from Phase 4F.
- Do not declare step 10,000 intrinsically bad; describe full-budget behavior.

### B. Model-scale transfer

#### Main scientific story

Separate three questions:

1. Does target architecture scale change absolute scratch and transfer F1?
2. Does pretrained initialization add benefit within each scale?
3. Does scale change how transfer and efficiency evolve with source training?

#### Main figures

1. **P1 faceted subject trajectories:** small, reference, and large columns;
   species rows.
2. **P2 scale comparison:** overlaid condition means plus small-minus-reference
   and large-minus-reference transfer-effect trajectories.
3. **E1 scale efficiency:** subject steps saved and attainment for all scales.
4. **E2 scale trade-off:** performance versus steps saved, with checkpoint
   paths for all three scales.
5. **E3 scale dynamics:** 2 x 5 grid using solid transfer and dashed
   architecture-matched scratch curves.

#### Annex

- P3 absolute scratch and transfer F1 by scale; this is essential for separating
  a target-capacity effect from a transfer effect.
- P4, P5, P6, and P7 common diagnostics.
- Optimizer-step versus cumulative-source-FLOP rendering of the transfer-effect
  trajectory. Source step is primary because it equalizes batch exposure;
  source FLOPs are the scale-specific alternative axis.
- Windows and FLOPs saved to matched quality.
- Source parameter count, source validation CE/F1, throughput, and peak-memory
  context referenced from the viability report.
- E4 and E5.

#### Interpretation requirements

- Never infer a transfer advantage from higher absolute F1 alone.
- Equal source steps mean equal batch-128 update counts, not equal compute.
- Discuss capacity-sensitive scratch optimization separately from pretrained
  initialization.
- Treat non-monotonic checkpoint behavior as evidence, not noise to be replaced
  by an endpoint-only contrast.

### C. Adapter-bias transfer

#### Main scientific story

Distinguish the source intervention from the target-architecture interaction:

1. Bias-free source -> ordinary target versus batch-128 reference tests whether
   bias-free source learning changes the transferable backbone.
2. Bias-free source -> bias-free target versus bias-free source -> ordinary
   target tests whether matching the target adapter provides additional benefit.

#### Main figures

1. **P1 faceted subject trajectories:** reference, bias-free source/ordinary
   target, and bias-free source/bias-free target.
2. **P2 bias contrasts:** condition means plus the two direct full-checkpoint
   contrasts above.
3. **E1 bias efficiency:** subject steps saved and attainment for all three
   treatments using their treatment-matched scratch controls.
4. **E2 bias trade-off:** performance versus steps saved.
5. **E3 bias dynamics:** full 2 x 5 normalized-progress grid.

#### Annex

- P3 must show ordinary and bias-free target scratch separately.
- P4, P5, P6, and P7.
- Refer to the older adapter-bias perturbation experiment as mechanistic
  motivation only. Do not treat its batch-16 perturbation association as proof
  that bias causes the new downstream result.
- E4 and E5.

#### Interpretation requirements

- A change shared by bias-free transfer and bias-free scratch is a target
  architecture effect, not a pretraining benefit.
- A source-only improvement with no extra matched-target improvement supports a
  source-representation account but not target-interface matching.
- An additional matched-target improvement supports a source-target
  compatibility interaction.

### D. Shared-adapter transfer

#### Main scientific story

Distinguish representation learning through a shared source interface from the
practical benefit of retaining that interface downstream:

1. Shared source -> fresh ordinary target versus batch-128 reference tests the
   transferred backbone after shared-interface source training.
2. Retained shared adapter versus fresh ordinary target tests interface
   continuity.

#### Main figures

1. **P1 faceted subject trajectories:** reference, shared source/ordinary
   target, and retained shared interface.
2. **P2 shared-interface contrasts:** condition means plus the two direct
   full-checkpoint contrasts above.
3. **E1 shared efficiency:** subject steps saved and attainment.
4. **E2 shared trade-off:** performance versus steps saved.
5. **E3 shared dynamics:** full 2 x 5 normalized-progress grid.

#### Annex

- P3 must show ordinary and shared-padded scratch separately.
- P4, P5, P6, and P7.
- **Channel-count diagnostic:** recording-level transfer-minus-scratch F1 versus
  filtered real-channel count, with backbone-only and retained-interface
  treatments distinguished, species separated, and points colored by subject.
  This tests sensitivity to zero padding and is specific to this experiment.
- If available, add a descriptive source-performance versus channel-count view
  by reference to the viability analysis.
- E4 and E5.

#### Interpretation requirements

- A retained-interface advantage over backbone-only transfer is the direct
  evidence for interface continuity.
- A shared scratch improvement is a target architecture effect.
- Stratify channel-count conclusions by species and do not treat recordings as
  independent biological replicates.

## Required implementation invariants

Add automated assertions or focused tests for the following before trusting
any rendered result:

- Every compiled cell ID and W&B run ID is unique within its source matrix.
- Every transfer cell resolves to exactly one comparator with identical
  species, target recording, target seed, target fraction, and required target
  architecture/treatment.
- Reference, small, large, bias-free-target, and shared-target scratch runs are
  never substituted across scratch families.
- A scratch run paired with itself has zero performance effect and zero steps
  saved.
- Every scratch run attains its own 80%, 90%, and 95% smoothed-peak target.
- A transfer run that does not attain a target receives the reconstructed
  planned budget, never its observed early-stopping step.
- Planned budgets agree across all matched conditions for a recording and seed.
- Aggregated recording tables contain three target-seed replicates unless the
  compiled design explicitly documents otherwise.
- Subject tables contain exactly seven minipig and five monkey subjects for
  every complete condition/checkpoint trajectory.
- Intervention contrasts use the intersection of paired subjects, recordings,
  target seeds, and source checkpoints and never compare unmatched marginal
  means.
- Bootstrap resamples preserve the complete set of conditions and checkpoints
  for each sampled subject.
- Performance and steps-saved sign conventions are checked with synthetic
  examples: positive always favors transfer.
- All sensitivity calculations rerun threshold detection from the histories;
  they are not algebraic rescalings of the 90% result.

## Artifact naming

Follow the experiment stem for every artifact. Suggested suffixes are:

- `_coverage.csv`
- `_run_endpoints.csv`
- `_validation_histories.csv`
- `_paired_performance_seed.csv`
- `_paired_performance_recording.csv`
- `_paired_performance_subject.csv`
- `_performance_summary.csv`
- `_efficiency_seed.csv`
- `_efficiency_recording.csv`
- `_efficiency_subject.csv`
- `_efficiency_summary.csv`
- `_threshold_sensitivity.csv`
- `_classwise_effects.csv`
- `_subject_performance_trajectories.png`
- `_condition_performance_contrasts.png`
- `_absolute_performance.png`
- `_recording_effect_heatmap.png`
- `_seed_dispersion.png`
- `_classwise_effects.png`
- `_source_downstream_association.png`
- `_subject_efficiency_trajectories.png`
- `_performance_efficiency_tradeoff.png`
- `_normalized_finetuning_dynamics.png`
- `_cumulative_attainment.png`
- `_threshold_sensitivity.png`

Experiment-specific artifacts may add `_historical_replication`,
`_source_flops`, `_compute_to_target`, or `_channel_count`.

Do not commit CSV caches. Figures and updated experiment markdowns are intended
to be committed.

## Markdown-writing requirements

Each final experiment markdown must contain:

1. One clear motivating question and one coherent mechanism-level hypothesis.
2. Exact definitive W&B groups, human-readable run names, and eight-character
   run IDs through the generated inventory artifact.
3. Expected and observed run counts and a statement that the immutable compiled
   matrix passed provenance/coverage audit.
4. The canonical aggregation order and whole-subject bootstrap description.
5. A concise performance table and efficiency table.
6. The small main figure set described above.
7. An annex/supporting-figures subsection linking the diagnostic figures without
   letting them dominate the narrative.
8. Conclusions that separately state:
   - final-performance evidence;
   - optimization-efficiency evidence;
   - absolute target-architecture behavior;
   - source-learning/mechanistic context;
   - limitations and unresolved alternatives.
9. The exact reproducible analysis command.

Do not copy large result tables between experiment files. Reference the source
viability report for source-only results and keep each downstream file focused
on its own intervention.

## Implementation order

1. Reorganize/rename experiment markdowns and repair their DAG links without
   changing results yet.
2. Implement the shared loader, provenance audit, caches, aggregation, and
   plotting primitives with tests on compiled-cell metadata.
3. Complete the batch-128 reference analysis first. Its outputs establish the
   canonical visual style and validate scratch matching and planned-budget
   reconstruction.
4. Implement model scale, then bias, then shared adapter using the same core.
5. Render all main and annex artifacts.
6. Inspect figures for consistent axes, labels, sign conventions, subject
   colors, and species separation.
7. Present generated tables and figures for scientific interpretation before
   writing conclusions, as required by the experiment-analysis workflow.
8. Fill Results and Conclusions only after the user validates the interpretation.
9. Mark the four downstream experiment files Completed only after all coverage,
   provenance, histories, tables, figures, and interpretation checks pass.

## Completion checklist

- [ ] Source-pretraining viability experiment remains intact and correctly linked.
- [ ] Dedicated batch-128 reference replication exists.
- [ ] Small and large reports are merged into one scale experiment.
- [ ] Bias and shared reports each contain one coherent interaction hypothesis.
- [ ] Synthesis experiment is removed and all links are repaired.
- [ ] All 6,360 compiled cells are accounted for exactly.
- [ ] Every transfer run maps to one exact treatment-matched scratch run.
- [ ] Performance aggregation is seed -> recording -> subject -> species.
- [ ] Efficiency thresholds are defined per matched scratch seed before averaging.
- [ ] Planned budgets are reconstructed and audited independently of early stop.
- [ ] Primary 90% and sensitivity 80%/95% results are generated.
- [ ] Subject bootstrap preserves complete checkpoint and treatment trajectories.
- [ ] Main performance, efficiency, dynamics, and trade-off figures are complete.
- [ ] Common and experiment-specific annex diagnostics are complete.
- [ ] No checkpoint is recommended using target-test results.
- [ ] No conclusions are written before user interpretation review.

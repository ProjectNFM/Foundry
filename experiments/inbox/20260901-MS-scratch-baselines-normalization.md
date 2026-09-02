# Scratch Baselines Normalization

**Status:** In Progress
**Date started:** 2026-09-01
**Parent experiment:** [Phase 3 -- NeuroSoft Input-Normalization Seed Replication](20260901-MS-neurosoft-input-normalization-replication.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, scratch, baselines, eegnet, convolution-bigru, input-normalization, learning-curves, 8band, intrasession-causal

## Background

The completed [Phase 3 input-normalization replication](20260901-MS-neurosoft-input-normalization-replication.md)
established on representative causal minipig and monkey recordings that
train-split-only global z-scoring is necessary or strongly beneficial for the
Conv--BiGRU, while raw input remains the strongest EEGNet condition.  It did
not establish whether those rankings generalize across the full audited
session/fraction matrix.

[Phase 1 EEGNet learning curves](20260826-MS-neurosoft-eegnet-learning-curves.md)
already provide the raw-input EEGNet reference at the five nested causal
training fractions.  This experiment adds two from-scratch, global-normalized
baselines over that identical matrix: EEGNet, to quantify the normalization
cost relative to Phase 1, and Conv--BiGRU, to compare architectures without
an input-normalization confound.  The Phase-1 raw EEGNet results are reused;
they are not rerun.

## Question

Across eligible NeuroSoft sessions and nested causal training fractions, how
do global-normalized from-scratch EEGNet and Conv--BiGRU compare with the
existing raw-input EEGNet learning curves in test supported macro-F1 and the
fraction of sessions reaching 80% of their own full-data performance?

## Hypothesis

Train-global z-scoring will produce lower test supported macro-F1 for EEGNet
than the Phase-1 raw-input reference at most training fractions.  With
normalization held fixed, Conv--BiGRU will achieve higher test supported
macro-F1 than EEGNet overall and will have an equal or greater share of
sessions reaching 80% of its own full-data test performance at lower training
fractions.  Effects must be reported separately by species and paired by
recording, fraction, and seed.

## Experiment

### Setup

- **Data:** the Phase-0-audited NeuroSoft cohort: 53 eligible recordings (40
  minipig, 13 monkey), yielding 255 supported recording/fraction cells.
- **Task:** `neurosoft_acoustic_stim_8band` (25 stimuli mapped to eight
  frequency bands).
- **Split:** `intrasession-causal`; train-global normalization statistics are
  fitted only on each causal training partition.  Validation and test
  intervals are invariant across fractions.
- **Fractions and seeds:** nested 5%, 10%, 25%, 50%, and 100% training
  fractions; seeds 42, 43, and 44 control initialization and the deterministic
  class-aware subset manifests.
- **New from-scratch conditions:**
  - **EEGNet global:** the Phase-1 EEGNet recipe with
    `recording_train_global_zscore` in place of raw input.
  - **Conv--BiGRU global:** the validated Phase-3 Conv--BiGRU scratch recipe
    with `recording_train_global_zscore`.
- **Reference condition:** completed Phase-1 raw-input EEGNet learning curves.
  Its one missing monkey 100%/seed-44 test result is excluded from paired
  contrasts and recorded explicitly rather than imputed.
- **Scope:** 1,530 new training jobs (255 supported cells x 3 seeds x 2 new
  conditions), plus the existing Phase-1 EEGNet reference runs.
- **Checkpoint selection:** best validation
  `val/neurosoft_acoustic_stim_8band_supported_f1`; consume test metrics only
  from that selected checkpoint.
- **Primary metric:** test supported-class macro-F1, summarized both
  subject-balanced (sessions -> subjects -> species) and across sessions.
- **Data-efficiency metric:** for each condition and session,
  `0.8 x mean_seed(test supported macro-F1 at 100%)`; report the cumulative
  percentage of sessions reaching that condition-specific threshold at each
  tested fraction.  Do not extrapolate or assign a value to unreached targets.
- **WandB:** project `neurosoft_supervised_pretraining`; use distinct,
  species-specific groups for the two new conditions and retain the Phase-1
  groups as a read-only reference in the analysis.

### Launch command

The four production Hydra configs reuse the audited Phase-1 cell sweep with
global normalization and model-specific recipes. Once they are committed,
submit four independent one-node Clariden pools using the snapshot workflow:

```bash
git status --short  # must print nothing
export FOUNDRY_SNAPSHOT_ROOT=<mounted-clariden-shared-path>/foundry-launches
scripts/launch_clariden_normalization.sh
```

Record every Slurm job ID and snapshot bundle path here immediately after
submission.

The first throughput canary uses the EEGNet minipig pool so its 579 queued
cells can keep 192 MPS workers busy across multiple waves:

- **Submitted:** 2026-09-01 at 22:37:13 CEST.
- **Slurm job:** `3257306` (`debug`, one exclusive GH200 node).
- **Start:** 2026-09-01 at 22:37:14 CEST on `nid007512` (no queue wait).
- **Expected end:** approximately 22:57:14 CEST after the 20-minute limit.
- **Concurrency:** `hydra.launcher.jobs_per_gpu=48` (192 workers/node).
- **Canary overrides:** `timeout_min=20`, `drain_guard_min=1`, and
  `signal_delay_s=60`.
- **Snapshot:**
  `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T203654_NORM_GLOBAL_EEGNET_MINIPIGS_a7ce1664_b4d52560`
- **Git commit:** `a7ce166451d3db48a0a5cae72b1ac55c79734197`.

The subsequent production submissions did not yield usable experiment cells:

| Slurm job | Pool | Snapshot | Outcome |
| --- | --- | --- | --- |
| `3257481` | EEGNet minipigs | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T210135_NORM_GLOBAL_EEGNET_MINIPIGS_f23b8c75_40829b94` | Child processes inherited Slurm rank variables and failed when configuring WandB. Superseded by `cceed1a`, which clears those variables. |
| `3257560` | EEGNet minipigs | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T212243_NORM_GLOBAL_EEGNET_MINIPIGS_cceed1a3_d0378503` | All workers stopped during MPS NUMA-domain detection. |
| `3257632` | EEGNet minipigs | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T213448_NORM_GLOBAL_EEGNET_MINIPIGS_cceed1a3_dec0c317` | All workers stopped during MPS NUMA-domain detection. |
| `3257647` | Conv--BiGRU minipigs | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T213705_NORM_GLOBAL_CONV_BIGRU_MINIPIGS_cceed1a3_2a435564` | All workers stopped during MPS NUMA-domain detection. |
| `3257662` | EEGNet monkeys | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T213929_NORM_GLOBAL_EEGNET_MONKEYS_cceed1a3_48b6f364` | All workers stopped during MPS NUMA-domain detection. |
| `3257667` | Conv--BiGRU monkeys | `/capstor/scratch/cscs/milosobral/foundry-launches/20260901T214040_NORM_GLOBAL_CONV_BIGRU_MONKEYS_cceed1a3_f689014b` | All workers stopped during MPS NUMA-domain detection. |

The MPS bug was that `hwloc-bind --taskset` produces Linux physical CPU
indexes, but the launcher passed that mask to `hwloc-calc` as logical indexes.
The resulting false multi-NUMA result halted every rank before it could claim
or execute a cell.  The verified fix uses `--physical-input` and
`--physical-output`; replacement jobs must use a snapshot at or after Git
commit `fe55162`.

### Key config overrides

- Reuse the Phase-0 eligibility manifest, nested-fraction resolver, causal
  split, task, seed list, test evaluation, and supported-F1 checkpoint policy
  from the Phase-1 EEGNet configs.
- Set `data.input_normalization.mode=recording_train_global_zscore` for both
  new conditions; retain raw input only in the existing Phase-1 reference.
- Keep the EEGNet model and optimization recipe from Phase 1, and the
  Conv--BiGRU architecture and scratch recipe validated in Phase 3.  Do not
  substitute a common optimizer recipe merely to make the architectures look
  comparable.
- Require validated model/input-shape-specific FLOP metadata before any
  production submission.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD — create `analysis/20260901-MS-scratch-baselines-normalization_analysis.py`
after runs are available.  It must fetch all three conditions through the
WandB API, produce paired raw-vs-global EEGNet and global EEGNet-vs-BiGRU
tables, and regenerate both requested learning-curve figures.

### Figures

TBD — subject-balanced test supported macro-F1 by training fraction, and the
cumulative percentage of sessions reaching 80% of condition-specific full-data
test performance.

## Conclusions

TBD

## Notes for future experiments

TBD

# Phase 4A -- Full-Pool Pretraining Full-Finetuning Transfer Gate

**Status:** Draft
**Date started:** 2026-09-04
**Parent experiment:** [NeuroSoft Supervised Pretraining Pipeline](20260903-MS-neurosoft-supervised-pretraining-pipeline.md)
**Follow-up experiments:** Frozen-representation transfer gate (TBD); Phase 4 source-volume study (TBD)
**Tags:** neurosoft, supervised-pretraining, transfer, full-finetuning, full-pool, phase4, 8band, intrasession-causal, clariden

## Background

The [Phase 3 pipeline experiment](20260903-MS-neurosoft-supervised-pretraining-pipeline.md)
establishes the scientific and operational prerequisites for transfer: target-
excluded source manifests, source-test prohibition, recording-level train-global
normalization, strict shared-component loading into a fresh target adapter, and
hash-verified checkpoint manifests.  Phase 2 provides the matched normalized
Conv--BiGRU scratch full-finetuning runs; it is the control for this experiment,
not EEGNet or an unpaired historical run.

The roadmap's full Phase 4 study varies source volume (10/25/50/100%) and
eventually evaluates intermediate compute checkpoints and lower target-data
fractions.  That is too large to be the first scientific claim.  This gate fixes
the source condition at the complete same-species, target-excluded pool and
uses only the source-validation-selected best checkpoint.  It asks whether full
finetuning benefits at all before expanding the matrix.

Phase 4A intentionally excludes frozen-representation transfer.  That is a
separate follow-up hypothesis requiring its own frozen-random control.  It also
does not select source checkpoints by target outcomes.

## Question

When every target subject is pretrained on all eligible same-species source
data excluding that subject, does initialization from the source
validation-selected checkpoint improve full, 100%-data target-session
finetuning relative to the existing matched Conv--BiGRU scratch baseline?

## Hypothesis

For both species, full finetuning from a target-excluded full-pool checkpoint
will have a positive paired, subject-balanced test supported macro-F1 effect
relative to scratch and will reach its own validation convergence threshold in
fewer optimizer steps, processed windows, FLOPs, and wall-clock time on
average.  The final-F1 effect may be modest; faster optimization is an
independent expected benefit.

## Experiment

### Setup

- **Model:** The Phase-2 train-global-normalized `NeurosoftConvBiGRU` recipe;
  no Phase-4-specific hyperparameter tuning.
- **Source data:** The audited `source_volume` manifests at `fraction-1.00`,
  same species as the target, with every recording of the target subject
  excluded.  Source training sees causal train/validation intervals only; its
  test loader must remain forbidden.
- **Target data:** Every Phase-0-eligible BIDS recording -- the established
  Phase-1 downstream unit -- comprising 40 minipig and 13 monkey targets.
  Each is adapted independently with its full causal train split; validation
  and test intervals remain fixed.  This gate does not split a recording's
  disjoint domain segments into additional downstream targets.
- **Seeds:** Source selection/model seeds are paired as `42`, `43`, and `44`.
  For every source checkpoint, independently run target finetuning seeds `42`,
  `43`, and `44`; do not pair the two seed axes downstream.
- **Transfer:** `full_finetuning` only.  Strictly load the declared shared
  temporal frontend/GRU/router components, exclude every source adapter, and
  initialize a new target adapter.
- **Checkpoint selection:** Select each source checkpoint only by
  `val/source_session_mean_supported_f1`.  Retain all compute milestones for
  provenance, but transfer only the best checkpoint in this gate.  Select each
  target checkpoint by
  `val/neurosoft_acoustic_stim_8band_supported_f1`, then evaluate target test
  exactly once.
- **WandB:** project `neurosoft_supervised_pretraining`; use four dedicated
  immutable groups: `PHASE4A_FULLPOOL_SOURCE_MINIPIGS`,
  `PHASE4A_FULLPOOL_SOURCE_MONKEYS`,
  `PHASE4A_FULL_FINETUNE_MINIPIGS`, and
  `PHASE4A_FULL_FINETUNE_MONKEYS`.

### Run matrix

The Phase-0 audit fixes seven minipig and five monkey target subjects, and 40
and 13 eligible target sessions respectively.

| Stage | Minipigs | Monkeys | Total new runs |
|---|---:|---:|---:|
| Source pretraining: target subject x paired source seed | 7 x 3 = 21 | 5 x 3 = 15 | 36 |
| Full finetuning: target session x source seed x target seed | 40 x 3 x 3 = 360 | 13 x 3 x 3 = 117 | 477 |
| **New Phase 4A work** | **381** | **132** | **513** |

The matched 100%-fraction scratch full-finetuning baseline is reused, not
rerun: 53 sessions x 3 target seeds = 159 completed Phase-2 controls.  This
design therefore estimates each session/target-seed effect from three source
pretraining replicates.  Analyses must average those three source-seed effects
before treating the session/target-seed pair as an inferential replicate.

### Pipeline and infrastructure requirements

1. Generate or verify the 36 immutable `fraction-1.00` manifests, including
   target exclusion, nested-selection hash, represented-class summary, and
   source train/validation interval identities.  Do not use the source-pool
   catalog as training input.
2. Submit source cells first.  A source cell writes best and milestone
   checkpoint manifests under a shared `/capstor` checkpoint root, with source
   manifest hash, checkpoint SHA-256, source model/selection seed, source
   compute counters, Git SHA, and snapshot bundle path.
3. Validate all 36 best manifests before fan-out: finite non-collapsed source
   validation, no target leakage, matching manifest/source hash, and readable
   checkpoint path from Clariden workers.
4. Compile the 477 downstream cells from those validated manifests.  Every
   cell record must explicitly carry target species/subject/session, source
   target subject, source selection/model seed, target finetuning seed,
   checkpoint-manifest path/hash, and `full_finetuning` regime.  The fan-out
   must not infer provenance from a W&B run name.
5. Run those cells in the Clariden durable node pool.  Queue source and target
   stages separately; a target cell may not be claimed until its declared
   source manifest has passed validation.  Resume from the original snapshot
   and queue state without rerunning successful source or target cells.
6. Before production, use a `debug` canary to validate container/venv, four
   GPU bindings, data and checkpoint access, W&B authentication, and one real
   source-to-target handoff.  On `normal`, benchmark per-model concurrency,
   beginning at `jobs_per_gpu=1`; use MPS only at a separately validated value.
7. Use a pinned ARM64 EDF, compute-node-built persistent venv, and same-path
   read-only mounts for `/capstor` data, snapshot root, checkpoint root,
   output root, and the mode-0600 W&B application env file.  The snapshot and
   checkpoint roots must be visible both at submission and on workers.

### Launch command

The job graph has a true dependency, so this is two normal Hydra multiruns,
not one static sweep.  A committed cell-list generator should emit exact
source and downstream override vectors and preserve the paired source seeds.
It must be run only after the Phase-4A config, generator, manifests, and this
report are committed and `git status --short` is empty.

```bash
# Clariden production environment; paths are mounted and visible to workers.
export CSCS_ACCOUNT=<project-account>
export PROJECT=/capstor/store/cscs/swissai/a0091
export FOUNDRY_DATA_ROOT=${PROJECT}/processed
export FOUNDRY_SNAPSHOT_ROOT=<shared-capstor-path>/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=<shared-capstor-path>/foundry-checkpoints
export FOUNDRY_CLARIDEN_VENV=<shared-capstor-path>/foundry-venv
export FOUNDRY_ENV_FILE=<absolute-path>/clariden.env
export FOUNDRY_CLARIDEN_EDF=<absolute-path>/clariden-foundry.toml

git status --short  # must print nothing before every production submission

# Stage A: one source multirun per species and paired source seed.
# The generated comma-separated list contains one fraction-1.00 manifest for
# every target subject of the indicated species.  Repeat for seed 42, 43, 44.
python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs \
  hydra/launcher=slurm_clariden \
  source_manifest=<generated-minipig-manifest-list-for-seed-42> \
  run.seed=42 \
  run.group=PHASE4A_FULLPOOL_SOURCE_MINIPIGS \
  trainer.max_steps=<phase3-calibrated-full-pool-budget> \
  trainer.val_check_interval=<phase3-calibrated-validation-interval> -m

# After the source queue has succeeded and every best manifest is verified,
# compile the 477 target cells.  Each cell contains one manifest path, target
# recording, source seed, and target seed; it performs exactly one test pass.
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  hydra/launcher=slurm_clariden \
  data.dataset_kwargs.recording_ids=[<target-recording>] \
  data.training_fraction=1.0 \
  run.seed=<target-finetuning-seed> \
  +run.source_selection_seed=<source-pretraining-seed> \
  run.pretrained_checkpoint_manifest=<verified-best-manifest.json> \
  run.pretrained_transfer_regime=full_finetuning \
  run.evaluate_test=true \
  run.group=PHASE4A_FULL_FINETUNE_MINIPIGS -m
```

Run the symmetric monkey source/target configs with their dedicated groups.
The placeholders are deliberate: Phase 3 must supply the measured full-pool
step/validation budget and the fan-out generator must supply checkpoint paths;
hard-coding either before Phase-3 calibration would make the Phase-4 scientific
matrix irreproducible.

### Key config overrides

- `source_manifest`: only audited, target-specific
  `source_volume/.../fraction-1.00/selection-<seed>.json` files.
- `run.seed`: source seed in Stage A; independent target finetuning seed in
  Stage B.
- `data.training_fraction=1.0` and `training_fraction_seed=${run.seed}` for
  all target cells.
- `run.pretrained_checkpoint_manifest`: the verified best manifest produced
  by the matching target-excluded source cell.
- `run.pretrained_transfer_regime=full_finetuning` and
  `run.evaluate_test=true` for all target cells.
- Clariden: `hydra/launcher=slurm_clariden`, production `normal` partition,
  immutable snapshots enabled, and only benchmark-validated `jobs_per_gpu`.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD -- use `analysis/20260904-MS-fullpool-finetune-transfer_analysis.py` to
fetch the dedicated W&B groups, check all planned cells, and calculate paired
test-F1 and convergence-cost effects against the Phase-2 scratch controls.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Advance to the frozen-representation experiment only as a separately
  controlled hypothesis, with a frozen-random representation baseline.
- Advance to the 10/25/50/100% source-volume study only if this gate passes
  source/target provenance and produces an interpretable transfer estimate.
- If pretrained full finetuning fails to match scratch at 100% target data,
  stop the low-data and checkpoint-milestone expansion and diagnose the
  transfer recipe first.

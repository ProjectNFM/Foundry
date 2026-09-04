# NeuroSoft Supervised Pretraining Pipeline

**Status:** In Progress
**Date started:** 2026-09-03
**Parent experiment:** None (infrastructure validation)
**Follow-up experiments:** Phase 4 volume, Phase 5 diversity, Phase 6 composition
**Tags:** neurosoft, supervised-pretraining, transfer, infrastructure, phase3, 8band, intrasession-causal

## Background

Phase 3 of the NeuroSoft supervised pretraining roadmap validates the complete
manifest-to-pretraining-to-transfer pipeline. Phases 0--2 established content-
hashed session eligibility, causal splits, and normalized Conv--BiGRU scratch
baselines. Phase 3 adds:

- immutable, leakage-free source-data manifests for all Phase 4--6 conditions;
- multi-session source pretraining with unweighted session-mean checkpoint
  selection;
- fixed optimizer-step milestone and best checkpoint manifests with verified
  provenance;
- strict component-scoped transfer with fresh target adapters; and
- a ten-job, two-stage minipig/monkey smoke matrix.

The implementation plan is
[neurosoft-supervised-pretraining-phase3-implementation.md](../../docs/neurosoft-supervised-pretraining-phase3-implementation.md).

## Question

Does the manifest-to-pretraining-to-transfer pipeline pass all declared
correctness, leakage, provenance, checkpoint, and compute gates for both
minipig and monkey species?

## Hypothesis

The pipeline passes every gate: source manifests are immutable and leakage-
free; source pretraining produces finite, non-collapsed validation metrics;
milestone and best checkpoints have verified JSON/Markdown manifests; strict
full-finetuning and frozen-representation transfer load the declared shared
components, exclude source adapters, and leave target adapters bitwise fresh;
and downstream target adaptation produces finite metrics with a single
validation-selected test evaluation.

No performance claim is made. Target F1 may be lower than scratch without
failing this gate.

## Experiment

### Setup

- **Data:** NeuroSoft Phase-0-audited recordings, selected through committed
  source manifests in `manifests/neurosoft_supervised/v1/`.
- **Task:** `neurosoft_acoustic_stim_8band`.
- **Split:** `intrasession-causal`; source test access forbidden.
- **Source recipe:** the exact Phase 2 normalized Conv--BiGRU recipe (see
  implementation plan §Fixed source recipe).
- **Model seed:** 42 for all Phase 3 runs.
- **Target fraction:** 100% causal target training data; target fraction
  seed 42.
- **Transfer regimes:** `full_finetuning` and `frozen_representation`.
- **Checkpoint selection:** source uses
  `val/source_session_mean_supported_f1` (max, patience 40); target uses
  `val/neurosoft_acoustic_stim_8band_supported_f1` (max, patience 40).
- **WandB project:** `neurosoft_supervised_pretraining`.

### Canonical ten-job matrix

#### Stage A -- two-session smoke (6 jobs)

| # | Species | Run | Input | Budget |
|--:|---------|-----|-------|--------|
| 1 | Minipigs | source pretraining | `smoke_minipigs_target-sub-06` | 500 steps |
| 2 | Monkeys | source pretraining | `smoke_monkeys_target-sub-01` | 500 steps |
| 3 | Minipigs | full finetuning | job 1 best manifest | target test once |
| 4 | Minipigs | frozen representation | job 1 best manifest | target test once |
| 5 | Monkeys | full finetuning | job 2 best manifest | target test once |
| 6 | Monkeys | frozen representation | job 2 best manifest | target test once |

- Smoke targets: minipig `sub-06_ses-02_task-AcousStim_acq-LH_desc-raw`;
  monkey `sub-01_ses-04_task-AcousStim_acq-RH_desc-raw`.

#### Stage B -- full-pool canary (4 jobs)

| # | Species | Run | Input | Budget |
|--:|---------|-----|-------|--------|
| 7 | Minipigs | source pretraining | full same-species pool (38 recordings) | 5,000 steps |
| 8 | Monkeys | source pretraining | full same-species pool (4 recordings) | 5,000 steps |
| 9 | Minipigs | full finetuning | job 7 best manifest | target test once |
| 10 | Monkeys | full finetuning | job 8 best manifest | target test once |

### Gate criteria

Stage A:
- Both source jobs are finite and predict more than one validation class.
- Source-session means match logged values (hand-checked).
- All milestone and best manifests validate (hash + SHA-256).
- Downstream transfer reports contain the exact expected components.
- Frozen/trainable parameter sets are correct.
- Target jobs are finite, select on validation, and evaluate test once.

Stage B (repeats Stage A gates at full-pool scale):
- Normalization, adapter construction, per-session validation, compute
  accounting, and checkpoint manifests work for complete source pools.

### Launch commands

```bash
# Stage A -- source pretraining
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints

# Minipig smoke source
python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs \
  source_manifest=manifests/neurosoft_supervised/v1/phase3_smoke/minipigs/target-sub-06.json \
  run.seed=42 trainer.max_steps=500 trainer.val_check_interval=100 -m

# Monkey smoke source
python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_monkeys \
  source_manifest=manifests/neurosoft_supervised/v1/phase3_smoke/monkeys/target-sub-01.json \
  run.seed=42 trainer.max_steps=500 trainer.val_check_interval=100 -m

# Stage A -- downstream transfer (after source manifests exist)
# Minipig full finetuning
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw] \
  run.pretrained_checkpoint_manifest=<path-to-job1-best-manifest.json> \
  run.pretrained_transfer_regime=full_finetuning \
  run.evaluate_test=true -m

# Minipig frozen representation
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  data.dataset_kwargs.recording_ids=[sub-06_ses-02_task-AcousStim_acq-LH_desc-raw] \
  run.pretrained_checkpoint_manifest=<path-to-job1-best-manifest.json> \
  run.pretrained_transfer_regime=frozen_representation \
  run.evaluate_test=true -m

# Monkey full finetuning
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  data.dataset_kwargs.recording_ids=[sub-01_ses-04_task-AcousStim_acq-RH_desc-raw] \
  run.pretrained_checkpoint_manifest=<path-to-job2-best-manifest.json> \
  run.pretrained_transfer_regime=full_finetuning \
  run.evaluate_test=true -m

# Monkey frozen representation
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  data.dataset_kwargs.recording_ids=[sub-01_ses-04_task-AcousStim_acq-RH_desc-raw] \
  run.pretrained_checkpoint_manifest=<path-to-job2-best-manifest.json> \
  run.pretrained_transfer_regime=frozen_representation \
  run.evaluate_test=true -m

# Stage B -- source pretraining (full same-species pools, local GPU)
# `source_pools/` contains pool catalogs, not runnable selection manifests.
# These overrides apply to source pretraining only. Downstream transfer keeps
# the established finetuning recipe below.
export CUDA_VISIBLE_DEVICES=0
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed

# Run this first; run the monkey command after it completes when using one GPU.
python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs \
  source_manifest=manifests/neurosoft_supervised/v1/source_volume/minipigs/target-sub-06/fraction-1.00/selection-42.json \
  run.name=stageb_src_mp_fullpool_b128_m42_retry1 \
  run.seed=42 trainer.max_steps=5000 trainer.val_check_interval=500 \
  +trainer.check_val_every_n_epoch=null \
  hyperparameters.batch_size=128 \
  hyperparameters.learning_rate=0.00025 \
  hyperparameters.weight_decay=0.01 \
  data.root=${FOUNDRY_DATA_ROOT}

python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_monkeys \
  source_manifest=manifests/neurosoft_supervised/v1/source_volume/monkeys/target-sub-01/fraction-1.00/selection-42.json \
  run.name=stageb_src_mk_fullpool_b128_m42_retry1 \
  run.seed=42 trainer.max_steps=5000 trainer.val_check_interval=500 \
  +trainer.check_val_every_n_epoch=null \
  hyperparameters.batch_size=128 \
  hyperparameters.learning_rate=0.00025 \
  hyperparameters.weight_decay=0.01 \
  data.root=${FOUNDRY_DATA_ROOT}
```

### Key config overrides

Stage B source pretraining uses `batch_size=128`, `learning_rate=0.00025`, and
`weight_decay=0.01`. These are not downstream fine-tuning overrides; jobs 9 and
10 retain the established downstream recipe.

### Execution log

| # | Execution | Output | Status |
|--:|-----------|--------|--------|
| 7 | Slurm `10654257` | snapshot `20260904T181358_NEUROSOFT_SOURCE_PRETRAINING_MINIPIGS_74069bf5_91b14432` | Failed before training: snapshot-relative `data.root` had no recordings. |
| 8 | Slurm `10654259` | snapshot `20260904T181436_NEUROSOFT_SOURCE_PRETRAINING_MONKEYS_74069bf5_fb644db8` | Failed before training: snapshot-relative `data.root` had no recordings. |
| 7--8 retry | Slurm `10654323`, `10654325` | fresh snapshots | Cancelled before execution; local execution was requested. |
| 7--8 local attempt | GPU 0 | `stageb_src_{mp,mk}_fullpool_b128_m42` | Failed at trainer setup: the 500-step validation interval exceeded a single epoch (226 minipig; 51 monkey batches). |
| 7 | Local GPU 0; W&B `stageb_src_mp_fullpool_b128_m42_retry1` (`348shpfe`) | `/network/scratch/s/sobralm/runs/NEUROSOFT_SOURCE_PRETRAINING_MINIPIGS/stageb_src_mp_fullpool_b128_m42_retry1` | Interrupted externally after 2,500 optimizer steps (not a model exception). Its validation-selected checkpoint, `best-epoch011-1.6531.ckpt` (source F1 0.304606), was preserved in the hash-verified recovery manifest `manifests/best-epoch011-1.6531.json`. |
| 8 | Not run | — | Intentionally omitted when Stage B was cut short; the user requested minipigs-only downstream validation. |
| 9 | Local GPU 0; W&B `stageb_tgt_mp_from_interrupted_step2500_full_finetuning_m42` (`ea1iftnv`) | `/network/scratch/s/sobralm/runs/NEUROSOFT_TRANSFER_MINIPIGS/stageb_tgt_mp_from_interrupted_step2500_full_finetuning_m42` | Finished. Full fine-tuning from the recovered job-7 checkpoint selected `best-epoch026-2.1096.ckpt` (validation supported F1 0.241902) and evaluated the held-out target test split once (supported F1 0.225325). |
| 10 | Not run | — | Intentionally omitted with the monkey source run. |

## Results

### Partial Stage B downstream result

The requested minipig-only downstream check completed locally from the
interrupted source run's verified best checkpoint. Transfer loaded the 26
shared temporal-frontend/GRU/router tensors in strict mode and deliberately
excluded 188 session-adapter tensors, including the target adapter. Full
fine-tuning selected its checkpoint on validation at optimizer step 1,809 and
ran the held-out test split exactly once. The run is finite: validation
supported F1 was 0.241902 and test supported F1 was 0.225325. The remaining
Stage B monkey runs were deliberately not launched.

## Analysis

Analysis script: `analysis/20260903-MS-neurosoft-supervised-pretraining-pipeline_analysis.py`

```bash
GOMAXPROCS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python analysis/20260903-MS-neurosoft-supervised-pretraining-pipeline_analysis.py
```

## Conclusions

(Pending execution and analysis.)

## Notes for future experiments

- Phase 4 volume experiments use the same pipeline with full source pools
  and the committed `source_volume` manifest family.
- Reuse the measured full-pool canary FLOPs to calibrate Phase 4 compute
  budgets.

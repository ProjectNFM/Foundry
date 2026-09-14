# Phase 4E Runtime Packing Audit

**Status:** In Progress
**Date started:** 2026-09-14
**Parent experiment:** [Phase 4E -- Validation-Loss Checkpoint Transfer Learning Curves](20260914-MS-validation-loss-checkpoint-transfer-learning-curves.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, phase4e, performance, dataloader, gpu-packing, mila

## Background

Phase 4A measured 11.89 minipig cells/GPU-hour with four tasks per RTX 8000,
but monkey throughput peaked at 9.22 cells/GPU-hour with two tasks and fell to
5.76 with four. Those canaries used only full target data, `num_workers=1`,
and different cells at each density. Phase 4E adds 5--100% learning curves,
so worker startup and validation overhead may change the best packing density.

## Question

Can Phase 4E aggregate downstream throughput be improved by disabling the
single persistent DataLoader worker and packing more independent tasks on each
RTX 8000, and does the answer differ between 5% and 100% target data?

## Hypothesis

Because one downstream process allocates about 0.13 GB of CUDA memory and
historical four-way packs used under 3% of a 48 GB RTX 8000, memory will not be
the limiting resource. `num_workers=0` should match or improve 5%-data
throughput by avoiding three split-specific worker processes, while denser
packing should improve minipig aggregate throughput. Monkey four-way packing
will only be retained if it reverses the Phase 4A throughput regression under
the lower-CPU worker configuration.

## Experiment

### Setup

- **Model:** Phase 4E `NeurosoftConvBiGRU` transfer/scratch downstream cells.
- **Data:** Two median-size minipig recordings and one representative monkey
  recording, at target fractions 0.05 and 1.00.
- **Task:** Compare aggregate completed cells per GPU-hour, individual runtime,
  GPU utilization/memory, CPU/RAM, and disk/network pressure.
- **Training:** 48 exact, disjoint cells from the compiled scientific matrix.
  Baselines are minipig `(tpn=4,cpu=1,nw=1)` and monkey
  `(tpn=2,cpu=2,nw=1)`. Worker ablations hold packing fixed with `nw=0`;
  density candidates use minipig `tpn=8` and monkey `tpn=4`, both with
  `cpu=1,nw=0`.
- **WandB:** Existing deterministic Phase 4E group/run identities are retained;
  the audit manifest selects exact IDs rather than changing provenance.

Accept a candidate only if it completes without OOM/requeue, improves
same-species/same-fraction aggregate throughput by at least 10%, keeps peak GPU
memory below 40 GB and allocation RAM below 28 GB, and does not show persistent
loader/I/O stalls or more than 2x per-cell slowdown. Stop any allocation that
OOMs, thrashes, or sustains near-zero throughput.

### Launch command

```bash
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_ENV_FILE=/home/mila/s/sobralm/Foundry/.venv/bin/activate

# Required immediately before each submission block.
git status --short

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN4_CPU1_NW1 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN4_CPU1_NW1 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/minipigs-tpn4-cpu1-nw1.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN4_CPU1_NW0 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN4_CPU1_NW0 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/minipigs-tpn4-cpu1-nw0.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hyperparameters.num_workers=0 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  run.group=PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN8_CPU1_NW0 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MINIPIGS_TPN8_CPU1_NW0 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/minipigs-tpn8-cpu1-nw0.jsonl \
  hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=1 \
  hyperparameters.num_workers=0 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN2_CPU2_NW1 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN2_CPU2_NW1 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/monkeys-tpn2-cpu2-nw1.jsonl \
  hydra.launcher.tasks_per_node=2 hydra.launcher.cpus_per_task=2 \
  hyperparameters.num_workers=1 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN2_CPU1_NW0 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN2_CPU1_NW0 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/monkeys-tpn2-cpu1-nw0.jsonl \
  hydra.launcher.tasks_per_node=2 hydra.launcher.cpus_per_task=1 \
  hyperparameters.num_workers=0 hydra.launcher.mem_gb=32 -m

uv run python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_monkeys \
  run.group=PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN4_CPU1_NW0 \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/PHASE4E_RUNTIME_AUDIT_MONKEYS_TPN4_CPU1_NW0 \
  'hydra.sweep.subdir=${run.name}' hydra/launcher=slurm_default \
  hydra.launcher.partition=main hydra.launcher.gres=gpu:rtx8000:1 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/phase4e/runtime-audit/monkeys-tpn4-cpu1-nw0.jsonl \
  hydra.launcher.tasks_per_node=4 hydra.launcher.cpus_per_task=1 \
  hyperparameters.num_workers=0 hydra.launcher.mem_gb=32 -m
```

### Key config overrides

- `hydra/launcher=slurm_default`, `hydra.launcher.partition=main`
- `hydra.launcher.gres=gpu:rtx8000:1`, `mem_gb=32`
- `+hydra.launcher.additional_parameters.exclude=cn-c004`
- Candidate-specific `tasks_per_node`, `cpus_per_task`, and
  `hyperparameters.num_workers` from
  `launch/phase4e/runtime-audit/manifest.json`.

### Launch record

All arrays use immutable commit `58dd5e00`, Mila `main`, one RTX 8000, 32 GB,
and `cn-c004` excluded. Each array contains two packed allocations, ordered as
one 5% pack followed by one 100% pack:

| Configuration | Slurm array | Snapshot bundle suffix |
|---|---|---|
| Minipig 4/1/1 | `10791170_[0-1]` | `20260914T175743_..._b5a94673` |
| Minipig 4/1/0 | `10791172_[0-1]` | `20260914T175802_..._e6d2b94e` |
| Minipig 8/1/0 | `10791204_[0-1]` | `20260914T175837_..._b4032400` |
| Monkey 2/2/1 | `10791209_[0-1]` | `20260914T175918_..._661e0ffb` |
| Monkey 2/1/0 | `10791214_[0-1]` | `20260914T180002_..._80f5691a` |
| Monkey 4/1/0 | `10791215_[0-1]` | `20260914T180031_..._8766f503` |

The full absolute snapshot paths are machine-recorded in the audit manifest.
`10791170_0` completed and `10791170_1` was already running on `main` when
the account QOS left the other five arrays serialized. On explicit direction,
only those five still-pending arrays were cancelled before starting and will be
resubmitted unchanged to `long`; no production array was launched.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

Production remains on `long` and must exclude completed audit cells from any
restart list. Every eventual launch requires a clean committed repository and
an immutable snapshot under shared `FOUNDRY_SNAPSHOT_ROOT`.

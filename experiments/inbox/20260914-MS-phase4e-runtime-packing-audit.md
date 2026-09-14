# Phase 4E Runtime Packing Audit

**Status:** Complete
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
resubmitted unchanged to `long` as arrays `10791382`--`10791386`; no production
array was launched.

## Results

All 12 packed allocations completed successfully; there were no OOMs or
requeues. The analysis queried the exact deterministic W&B runs named in
[`manifest.json`](../../launch/phase4e/runtime-audit/manifest.json). Per-run
measurements are retained in ignored local artifacts
`analysis/csv/20260914-MS-phase4e-runtime-packing-audit_{runs,allocations}.csv`.

| Species / fraction | Candidate | Cells/GPU-hour | Median / max cell runtime | GPU memory peak | Decision |
|---|---:|---:|---:|---:|---|
| Minipig / 5% | 4 tasks, 1 CPU, 1 worker | 95.4 | 116 / 151 s | 1.44 GB | baseline |
| Minipig / 5% | 8 tasks, 1 CPU, 0 workers | **165.5** | 125 / 174 s | 2.87 GB | retain |
| Minipig / 100% | 4 tasks, 1 CPU, 1 worker | 10.4 | 911 / 1,381 s | 1.44 GB | reject |
| Minipig / 100% | 8 tasks, 1 CPU, 0 workers | **28.9** | 716 / 997 s | 2.87 GB | retain |
| Monkey / 5% | 2 tasks, 2 CPU, 1 worker | **34.0** | 195 / 212 s | 0.72 GB | retain (conservative) |
| Monkey / 5% | 4 tasks, 1 CPU, 0 workers | 36.4 | 295 / 396 s | 1.44 GB | reject: only +7%, slower cells |
| Monkey / 100% | 2 tasks, 2 CPU, 1 worker | **9.14** | 759 / 788 s | 0.72 GB | retain |
| Monkey / 100% | 4 tasks, 1 CPU, 0 workers | 8.45 | 1,391 / 1,705 s | 1.44 GB | reject |

The unlisted worker-only comparisons point in the same direction. At 100%,
minipig 4/1/0 achieved 18.95 cells/GPU-hour versus 10.43 for 4/1/1; monkey
2/1/0 achieved 4.59 versus 9.14 for 2/2/1. Therefore the result is not simply
that workers are intrinsically bad: a worker must have CPU capacity, and the
monkey loader benefits materially from that pairing.

Peak model CUDA allocation was only 0.121 GB/run (reserved 0.132 GB/run).
The W&B whole-device peaks above scale almost exactly with task count and are
below 6% of the 48 GB RTX 8000. Process RSS was 1.81--1.83 GB/run. The densest
minipig allocation reached 29% host memory and the densest monkey allocation
24%; its observed GPU and RAM headroom is substantial within the 32 GB request.
The W&B event stream reports modest local disk writes and network counters that
grow with runtime; it does not directly measure shared HDF5 server latency.

At 100%, minipig 8/1/0 had 97.0% median GPU utilization and 86.9% host CPU;
it is GPU-saturated rather than waiting for a loader. Minipig 4/1/1 had 55.6%
GPU / 46.8% host CPU and a much slower tail, consistent with worker/HDF5
overhead under a one-CPU request. Monkey 2/2/1 reached 66.5% GPU / 38.6% host
CPU at 100%; removing the worker drove host CPU to 86.4% while GPU utilization
fell to 34.7%, an input/CPU-bound pattern. Four workerless monkey tasks raised
GPU utilization but reduced aggregate throughput and lengthened the slowest
cell 2.2x.

## Conclusions

The audit does not support a biological or dataset-intrinsic species claim:
it uses two minipig recordings and one monkey recording, and the monkey
density comparison also changes worker/CPU allocation. It does establish that
memory is not the limiter, and that a one-CPU task must not also own a
persistent worker. If minimizing the number of Slurm allocations is the
priority, use one common eight-way production setting rather than a
species-specific plan. This produces 290 minipig plus 93 monkey allocations,
383 total, rather than 662 for the prior 8-way/2-way plan.

| Species | Fractions | tasks/node | CPUs/task | `num_workers` | Memory basis |
|---|---|---:|---:|---:|---|
| Minipigs | 5%, 10%, 25%, 50%, 100% | 8 | 2 | 1 | 2.87 GB observed at 8-way; worker does not add CUDA model memory |
| Monkeys | 5%, 10%, 25%, 50%, 100% | 8 | 2 | 1 | 1.44 GB observed at 4-way; linear 8-way estimate about 2.9 GB |

Use `hydra/launcher=slurm_default hydra.launcher.partition=long`, one
`gpu:rtx8000:1`, `hydra.launcher.mem_gb=32`, and
`+hydra.launcher.additional_parameters.exclude=cn-c004` for both. The single
set of workload overrides is:

```text
hydra.launcher.tasks_per_node=8 hydra.launcher.cpus_per_task=2 hyperparameters.num_workers=1
```

This is a resource-safe, minimum-allocation recommendation, not a claim that
the exact universal 8/2/1 combination has already won a throughput benchmark
for both species. No further audit jobs are needed or planned. The completed
evidence makes the two components of this choice conservative: eight-way
packing is already the fastest observed minipig density with very large GPU
headroom, and two CPUs plus one worker is the fastest observed monkey loader
pairing. Across the full matrix, recording/session variance is expected to
average rather than determine the production decision.

Do not launch the original complete lists unchanged: 32 minipig and 16 monkey
cells were completed by this audit with their production W&B identities. Before
an eventual production submission, derive pending lists with
`tools/exclude_downstream_cells.py` using the corresponding audit lists, commit
those lists, confirm a clean worktree, then submit from a fresh immutable
snapshot. This preserves IDs, source provenance, transfer/scratch pairing, and
restart semantics without appending duplicate metrics to completed W&B runs.

Rollback a running production pack to the prior audited species setting if any
allocation OOMs, GPU device memory exceeds 40 GB, host RSS/requested-memory
exceeds 28 GB, or the first 8 completed comparable cells show more than 10%
lower aggregate cells/GPU-hour. Also roll back minipig 8-way packing if median
GPU utilization falls below 75% together with sustained host CPU above 90% or
repeated HDF5/I/O stalls. The only defensible next canary would be an I/O
stress sample across multiple large recordings if production monitoring shows
that signal; none is needed before this recommendation.

## Notes for future experiments

Production remains on `long` and must exclude completed audit cells from any
restart list. Every eventual launch requires a clean committed repository and
an immutable snapshot under shared `FOUNDRY_SNAPSHOT_ROOT`.

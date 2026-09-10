# Clariden Node-Pool Hydra Launcher: Implementation Handoff

**Status:** Proposed implementation plan  
**Date:** 2026-09-01  
**Scope:** reusable CSCS Clariden launch environment and Hydra launcher for
high-density independent Foundry training sweeps.

## Decision Summary

Clariden GH200 production partitions allocate complete nodes exclusively,
including when a submitted workload uses only one GPU.  Foundry therefore
needs to acquire an exclusive node and schedule many independent Hydra sweep
cells inside that allocation.

The desired user-facing interface remains a normal Hydra multirun:

```bash
python main.py experiment=<experiment> hydra/launcher=slurm_clariden -m
```

The Clariden launcher must:

1. preserve Foundry's immutable Git-archive snapshot workflow;
2. use a containerized, ARM64-compatible Python environment;
3. allocate whole GH200 nodes on the `normal` partition;
4. run a dynamic pool of independent workers inside each allocated node;
5. support `jobs_per_gpu` from 1 through 48, with MPS required above 1;
6. record enough state to resume only unstarted/failed cells from the original
   source snapshot; and
7. select model-specific production concurrency from measured throughput, not
   a guessed GPU-utilization target.

For the immediate normalization experiment, submit four independent one-node
allocations, each capped at four hours:

| Pool | Sweep cells | Allocation |
|---|---:|---|
| EEGNet, minipigs | 579 | one GH200 node, at most 4 h |
| Conv--BiGRU, minipigs | 579 | one GH200 node, at most 4 h |
| EEGNet, monkeys | 186 | one GH200 node, at most 4 h |
| Conv--BiGRU, monkeys | 186 | one GH200 node, at most 4 h |

Submit these as four one-node jobs rather than one four-node Slurm job.  This
gives the scheduler more independent backfill opportunities and lets the
smaller monkey pools finish independently.  It does not guarantee a shorter
queue wait; actual priority is governed by Slurm fair-share and availability.

## Facts and Constraints

### Clariden resources and partitions

- A GH200 node has four GPUs and four CPU domains, each with approximately 72
  CPU cores and local host memory.
- `normal` is the production partition and has a 12-hour job limit.
- `debug` is only for genuine short canaries.  It has a 90 node-minute job
  limit, one running job per user, and two submitted jobs per user.
- `normal`, `debug`, and `low` allocate nodes exclusively.  A one-GPU request
  still reserves the node, so one-GPU Slurm jobs are not an efficient
  production pattern here.
- Node sharing between users is not currently available.  Resource subdivision
  must happen with job steps inside Foundry's exclusive allocation.

Authoritative references:

- [Clariden overview and partition policy](https://docs.cscs.ch/clusters/clariden/)
- [GH200 Slurm and node-oversubscription guide](https://docs.cscs.ch/running/slurm/)
- [Container Engine guide](https://docs.cscs.ch/software/container-engine/run/)

### Repository constraints

- Do not bypass launch snapshots or clean-Git enforcement for production.
- Before a production submission, `git status --short` must produce no output.
- A snapshot root must be visible on login and compute nodes and mounted inside
  the container.  The historic `/network/scratch/...` root is not a Clariden
  default; use a mounted Clariden shared path instead.
- The existing `slurm_cscs.yaml` and `jobs/cscs_sweep_hyperqueue.sh` are
  historical references, not the implementation base.  They have legacy
  resource assumptions and hard-coded account/path behavior.
- The existing `PackedSubmititLauncher` statically groups sweep cells.  It is
  insufficient for high-density work because a worker that finishes early
  leaves its GPU slot idle.

### Workload target

The 1,530 new cells are split across four pools above.  With four nodes running
for four hours, the maximum unshared capacity is 64 GPU-hours.  This is enough
only if the average work per cell is sufficiently small.  Oversubscribing a GPU
can recover idle time from CPU/data/kernel gaps but cannot create more GPU
compute capacity.

For the two minipig pools, 579 cells on four GPUs over four hours require an
average of about 100 seconds per cell in exclusive-GPU-equivalent work.  For
the monkey pools, the equivalent budget is about 310 seconds.  Validate these
assumptions with the benchmark protocol below before production submission.

## Architecture

```text
Hydra multirun (`main.py ... -m`)
        |
        v
SlurmClaridenNodePoolLauncher
  - expand Hydra overrides once
  - create one immutable source snapshot
  - create durable queue/attempt manifest beside snapshot
  - submit one or more exclusive-node allocations
        |
        v
Clariden allocation (one GH200 node)
  - enter EDF with `srun --environment=...`
  - start four MPS services when jobs_per_gpu > 1
  - start 4 * jobs_per_gpu persistent workers
        |
        v
shared durable work queue
  - atomically claim next pending override
  - launch one snapshot-resident Foundry cell
  - persist outcome and provenance
  - repeat until queue empty or time guard reached
```

The scheduler sees a small number of whole-node jobs.  The queue runner, not
Slurm array batching, fills freed worker slots.  Every invoked training cell
must execute from the same snapshot bundle produced at submission time.

## Configuration Contract

Add `configs/hydra/launcher/slurm_clariden.yaml`.  Register a new launcher
target, for example:

```yaml
defaults:
  - foundry_submitit_slurm

_target_: hydra_plugins.foundry_launcher.clariden_launcher.ClaridenNodePoolLauncher

account: ${oc.env:CSCS_ACCOUNT,???}
partition: normal
nodes: 1
timeout_min: 240
exclusive: true
mem_gb: 450
gpus_per_node: 4

# Logical Foundry concurrency, not a Slurm request for fractional node sharing.
jobs_per_gpu: 1
workers_per_node: ${eval:'4 * ${hydra.launcher.jobs_per_gpu}'}
cpus_per_worker: ${eval:'72 // ${hydra.launcher.jobs_per_gpu}'}
memory_per_worker_gb: ${eval:'450 // (4 * ${hydra.launcher.jobs_per_gpu})'}

# Do not count cells started this close to timeout; preserve them for resume.
drain_guard_min: 10

container_environment: ${oc.env:FOUNDRY_CLARIDEN_EDF,???}
application_environment_file: ${oc.env:FOUNDRY_ENV_FILE,.env}
snapshot:
  enabled: true
  root: ${oc.env:FOUNDRY_SNAPSHOT_ROOT,???}
  require_clean_git: true
  verify_on_worker: true
  environment_file: ${hydra.launcher.application_environment_file}
```

Use a supported resolver rather than the illustrative `eval` expressions if
the project does not already provide one.  Resolve and validate all derived
values in Python before submitting Slurm work.

Required validation:

- `jobs_per_gpu` is an integer in `[1, 48]`. This conservative ceiling remains
  compatible with the MPS client-context limit in CUDA 13.0 and earlier; raise
  it only together with a pinned newer CUDA environment and a validated limit.
- `cpus_per_worker >= 1` and `memory_per_worker_gb >= 1`.
- `nodes == 1` in the first implementation.  Multi-node pool orchestration is
  explicitly out of scope.
- `partition` is `normal` by default; a dedicated `slurm_clariden_debug`
  profile may select `debug` with a timeout at or below 90 minutes.
- `container_environment`, `CSCS_ACCOUNT`, and `FOUNDRY_SNAPSHOT_ROOT` are
  required and must resolve to absolute paths where applicable.
- The EDF and snapshot root must be readable before submission.  Do not print
  secret values.
- `jobs_per_gpu > 1` requires the EDF MPS annotation described below.

Each experiment may override only the operational choices it owns:

```yaml
hydra:
  launcher:
    timeout_min: 240
    jobs_per_gpu: 4  # replace only after benchmark validation
```

Do not bury account IDs, EDF paths, or user-specific filesystem roots in
experiment YAML files.

## Slurm Resource Semantics

### One job per GPU (`jobs_per_gpu: 1`)

This is the baseline and must use explicit Slurm GPU binding:

```bash
srun --exclusive \
  --ntasks=4 \
  --gpus-per-task=1 \
  --cpus-per-task=72 \
  --mem=112G \
  --environment="$FOUNDRY_CLARIDEN_EDF" \
  <worker-command>
```

Actual memory values may reserve a small node-level margin.  Verify that each
rank receives exactly one GPU by recording `SLURM_LOCALID`, CPU affinity, and
`CUDA_VISIBLE_DEVICES`.

### Multiple jobs per GPU (`jobs_per_gpu > 1`)

Multiple CUDA processes on a GPU must use NVIDIA MPS.  Do not run independent
workers with an unset `CUDA_VISIBLE_DEVICES`; on GH200 default compute mode
that can silently make all workers use GPU 0.

The EDF must enable the CSCS Container Engine MPS hook:

```toml
[annotations]
com.hooks.nvidia_cuda_mps.enabled = "true"
```

Use one MPS service per physical GPU.  The implementation may use the CSCS
Container Engine hook or the CSCS native MPS wrapper, but must prove in the
canary that four distinct GPU services are active and that workers map to the
intended GPU/NUMA domain.  Do not use one MPS daemon for the whole node.

For each worker, allocate:

```text
CPU cores = floor(72 / jobs_per_gpu)
host-memory limit = configured safe share of 450 GB / (4 * jobs_per_gpu)
GPU = assigned MPS service for its NUMA/GPU domain
```

The actual launch strategy must prevent Slurm from treating the same GPU as
unavailable after the first worker claims it.  A supported approach is a
single MPS-aware `srun` launch of `4 * jobs_per_gpu` persistent worker ranks,
with CPU affinity used to map ranks to GPU domains.  Do not launch multiple
independent `srun --gpus-per-task=1` steps and expect them to share a GPU.

MPS does not provide a memory quota or performance isolation.  The queue
runner must fail a cell cleanly on OOM, record it, and permit retry at a lower
concurrency.  It must never silently retry an OOM indefinitely.

## Container Environment

### Required environment layout

Create a pinned container environment specifically for Clariden's ARM64
compute nodes.  A login-node virtualenv or the repository's existing `.venv`
must not be assumed compatible.

Use a CSCS Alps-extended PyTorch image or a compatible NGC PyTorch base.  Pin
the image digest/tag and record it in the EDF and launch provenance.  Build
and install the Foundry environment from a compute allocation inside that
container:

1. Store container images under shared scratch, e.g. `$SCRATCH/ce-images/`.
2. Create a persistent venv on mounted scratch, e.g.
   `$SCRATCH/foundry-envs/<environment-id>/`.
3. Inside a compute-node container shell, create the venv and install from the
   committed `uv.lock` / `pyproject.toml` using the container interpreter.
4. Run a smoke test inside the EDF: `python -c 'import torch, hydra, foundry'`
   and print Python, Torch, CUDA, and platform details.
5. Do not activate or modify this venv from outside the container.

The EDF must mount, at the same absolute paths where practical:

- the snapshot root;
- the source data and any project cache/data-cache paths;
- the persistent virtualenv;
- the output/sweep directory;
- the credentials file only if required and trusted.

The source snapshot must be first on `PYTHONPATH`, and the venv must be
activated before `main.py` runs.  The container filesystem is shared among all
Slurm tasks on a node; workers must write only to distinct per-cell output and
temporary directories.

Use `--environment=<edf>` on `srun`, not as an `#SBATCH` option.  Do not nest
an EDF in both sbatch and srun.

Keep these concepts separate in code and documentation:

| Name | Purpose | May contain secrets? |
|---|---|---|
| EDF (`FOUNDRY_CLARIDEN_EDF`) | image, mounts, hooks, container environment | no by default |
| application env file (`FOUNDRY_ENV_FILE`) | W&B/API credentials and application variables | yes |
| snapshot bundle | immutable Git source and non-secret manifest | no |

The snapshot should reference the application env file by absolute path; it
must never copy its contents into the bundle or logs.

### W&B authentication contract

Do not execute `wandb login` interactively in every worker.  Instead, create
one user-readable, non-versioned application environment file, for example:

```bash
# /users/<user>/.config/foundry/clariden.env (mode 0600)
WANDB_API_KEY=<secret>
WANDB_ENTITY=poyo-eeg
PROJECT=/capstor/store/cscs/swissai/a0091
FOUNDRY_DATA_ROOT=/capstor/store/cscs/swissai/a0091/processed
```

The exact `WANDB_ENTITY` is optional if the account default is correct, but
setting it explicitly avoids writing runs to an unintended entity.  The
existing `build_setup_commands()` behavior uses `set -a; source
<absolute-environment-file>; set +a`; preserve that behavior in the Clariden
launcher so `WANDB_API_KEY` is exported before Lightning constructs its
`WandbLogger`.

Because CSCS discourages mounting an entire home directory into containers,
the EDF must bind mount this one credentials file read-only at the same
absolute path, or bind mount a dedicated non-secret/secret configuration
directory read-only.  The launcher must validate that the file is readable
inside the container without printing its contents.  The env file is not part
of the source snapshot, queue manifest, Hydra output, or W&B config.  Its
absolute path and a non-secret fingerprint may be recorded in provenance.

The canary must prove authentication with a harmless online W&B initialization
in the intended project.  It must finish the canary run and must not log the
API key.  `WANDB_MODE=offline` is acceptable only for an explicitly requested
offline test; it is not the production default.

### Data-path contract

The NeuroSoft processed data currently lives at:

```text
/capstor/store/cscs/swissai/a0091/processed
```

This is the correct initial source of truth.  It is project store, not part of
the snapshot, and must be mounted into the EDF at the same path.  Exporting
`PROJECT=/capstor/store/cscs/swissai/a0091` preserves the existing
`configs/cluster/cscs.yaml` composition:

```yaml
data:
  root: ${oc.env:PROJECT}/processed/
stage:
  source_root: ${oc.env:PROJECT}/processed/
```

Before worker ranks start, the runner must enter the container and verify:

```bash
test -r /capstor/store/cscs/swissai/a0091/processed
test -x /capstor/store/cscs/swissai/a0091/processed
```

It must then resolve the composed Hydra configuration and log the resulting
`data.root` and `stage.source_root`.  This confirms that the process sees the
same path that was used at submission, while revealing no credentials.

For the first functional canary, set `stage.mode=direct`.  This avoids a
large, unmeasured staging operation and proves that all workers can read the
mounted project store correctly.  It is also the right correctness fallback.

For production throughput, benchmark data placement as an independent factor:

1. direct reads from the mounted Capstor project store;
2. a deliberate, once-per-node staging scheme to node-local storage, if
   available and large enough; and
3. a project-managed copy in `/iopsstor/datacache/cscs/<organization>/<project>`
   if direct reads are the bottleneck.

Do not let every high-density worker independently copy the same recording to
node-local storage.  A node-pool staging leader must populate a keyed cache
once, verify completion, and only then release workers.  Keep data-source and
staging choices in the experiment/run provenance.  Do not create or copy a
project datacache dataset until the user explicitly authorizes that storage
operation.

## Dynamic Queue and Resume Design

Create queue state inside the snapshot bundle, for example:

```text
<snapshot>/manifests/
  submission.json
  clariden-queue.jsonl
  clariden-attempts.jsonl
  clariden-launch.json
```

Use an atomic, filesystem-safe claim mechanism.  Prefer SQLite with WAL mode
only after verifying the selected shared filesystem supports its locking
semantics; otherwise use atomic file creation/rename of one record per cell.
Do not rely on in-memory state, one shared JSON rewrite, or advisory locking
whose reliability has not been verified on the target filesystem.

Every cell record must include:

- canonical cell ID: stable hash of ordered Hydra overrides;
- complete override vector;
- state: `pending`, `running`, `succeeded`, `failed`, or `not_started_due_to_drain`;
- attempt number and timestamps;
- allocation ID, node hostname, worker rank, GPU identifier, and
  `jobs_per_gpu`;
- Hydra output directory and W&B run identity when available;
- snapshot bundle ID, Git SHA, source digest, environment fingerprint;
- exit status and a short failure classification.

Worker behavior:

1. Atomically claim a `pending` cell.
2. Refuse a new cell if remaining Slurm time is below `drain_guard_min` plus
   the configured minimum-start budget; mark it
   `not_started_due_to_drain`.
3. Launch the snapshot-resident `main.py` with exactly the recorded overrides.
4. On normal exit, mark `succeeded` only after the child process returns zero.
5. On failure, record `failed` with logs and do not hide the failure from the
   overall allocation result.
6. Continue claiming cells until the queue is exhausted or draining starts.

Resume behavior:

- Resume takes an existing snapshot bundle path/ID, never the current checkout
  as the source of truth.
- It requeues only `failed` cells requested by policy and all
  `not_started_due_to_drain` cells.  It must not rerun `succeeded` cells.
- A retry uses the original snapshot, resolved overrides, container identity,
  and application environment reference.
- The launcher emits the submitted Slurm job IDs and snapshot path immediately
  and appends them to the submission manifest.

## Experiment Configs for the Immediate Sweep

Add four committed experiment YAML files, one for each species/model pool.
They should reuse Phase-1 cell resolution, audit manifest, causal split,
fraction/seed sweep, and test/checkpoint policy.  They should differ only in:

- model and its validated scratch recipe;
- species;
- `data.input_normalization.mode=recording_train_global_zscore`;
- distinct W&B group/name/tag values;
- validated, model/input-shape-specific FLOP metadata.

Expected sweep sizes:

```text
minipigs: 193 supported recording/fraction cells * 3 seeds = 579
monkeys:   62 supported recording/fraction cells * 3 seeds = 186
```

The existing raw-input Phase-1 EEGNet runs are read-only reference data and
must not be rerun.  Do not use a common optimizer merely to make EEGNet and
Conv--BiGRU look symmetrical; preserve each validated recipe.

Provide a small, explicit launcher script or documented sequence that submits
the four independent configs.  It must only submit after all configs are
committed and the worktree is clean.  Do not use `long` on Clariden; use
`normal` for production.

## Benchmark Protocol

The selected `jobs_per_gpu` must optimize completed valid cells per wall-clock
hour, not merely average GPU utilization.

### Phase A: environment and binding canary

Use `debug` only for a genuine short validation allocation.  Confirm:

1. the EDF starts successfully;
2. Python imports Foundry, Hydra, Torch, and dependencies from the intended
   snapshot/venv paths;
3. CUDA is available;
4. one-worker-per-GPU mode produces four distinct GPU bindings;
5. source snapshot verification and provenance logging succeed;
6. data, output directories, and W&B credentials are accessible;
7. one short real train/evaluation cell succeeds.

For MPS modes, also confirm that MPS is active and every worker reports its
intended GPU/NUMA assignment.  Capture `nvidia-smi` and Slurm affinity details
without exposing credentials.

### Phase B: throughput measurement

On `normal`, choose representative full cells for both models and both
species, biased toward the slowest expected minipig cases.  Measure
`jobs_per_gpu` values:

```text
1, 2, 4, 8, 16, 32, 48
```

Stop increasing concurrency for a model when any of the following occurs:

- cell throughput no longer improves materially;
- an OOM, repeated CUDA/MPS failure, or unacceptable failure rate occurs;
- CPU/data-loading contention dominates;
- training correctness/metric behavior differs from the one-job baseline;
- the configuration cannot finish representative cells within the pool's
  planned drain policy.

Record for every setting:

- cells completed per wall-clock hour and per GPU-hour;
- per-cell wall time distribution and startup overhead;
- peak GPU memory and GPU utilization;
- CPU utilization/affinity and host-memory usage;
- OOM/error count;
- test/validation metric parity against the one-job baseline.

Select separate production defaults for EEGNet and Conv--BiGRU when warranted.
The launcher supports 1--48; experiment configs set only values proven by this
benchmark.

## Implementation Sequence

1. **Document and correct cluster assumptions.** Update repository launch
   guidance to distinguish the legacy cluster from Clariden: `normal`, full
   exclusive GH200 nodes, CSCS shared paths, and debug-only canaries.
2. **Implement and validate the EDF/venv.** Add a non-secret EDF template and
   an environment bootstrap document/script.  Do not commit credentials or
   user-specific secret configuration.
3. **Implement `ClaridenNodePoolLauncher`.** Reuse snapshot preparation,
   verification, and environment-fingerprint code from the current launcher;
   do not duplicate or disable it.
4. **Implement queue persistence and resume.** Add a snapshot-resident worker
   runner and cell-state model before adding MPS.
5. **Implement single-job-per-GPU mode.** Validate with the debug canary.
6. **Implement MPS mode.** Add EDF validation, worker/GPU affinity handling,
   and high-density canaries.
7. **Add four normalization experiment configs.** Validate composition and
   FLOP metadata before any production submission.
8. **Benchmark concurrency.** Select values independently for the two models.
9. **Production launch.** Commit all changes; ensure a clean worktree; set a
   compute-visible `FOUNDRY_SNAPSHOT_ROOT`; submit the four pools; record job
   IDs and snapshot bundle paths in the experiment report.

## Tests and Acceptance Criteria

### Unit tests

- Config validation rejects missing account, EDF, snapshot root, invalid
  `jobs_per_gpu`, and invalid resource-derived values.
- Queue claim is race-safe under concurrent local worker processes.
- A completed cell is never claimed again by resume.
- A failed/unstarted cell is resumed from the original snapshot only.
- Queue records include all required source, environment, resource, and
  scheduler provenance.
- Container/application environment references are not embedded or logged as
  secret content.
- Launcher fails before submission on a dirty repository.
- Snapshot source resolution uses the staged bundle, never the live checkout.

### Integration/canary tests

- A `debug` one-worker-per-GPU canary proves four distinct GPU bindings.
- A `debug` MPS canary proves intentional multiple-worker-per-GPU binding and
  clean worker shutdown.
- A `normal` throughput run demonstrates a valid non-default concurrency
  value for at least one representative model.
- A controlled time-drain/resume test leaves succeeded cells untouched and
  processes only the remaining cells in a second allocation.
- An EDF container test proves the interpreter is from the mounted venv, not
  the login-node `.venv`.

### Production definition of done

- `python main.py ... hydra/launcher=slurm_clariden -m` creates a committed,
  immutable snapshot and submits a valid Clariden allocation.
- All queued/running workers execute from that snapshot and record provenance.
- `jobs_per_gpu=1` works with explicit GPU binding.
- A validated MPS value greater than one works with dynamic queue filling.
- Four model/species normalization pools can be submitted independently with a
  four-hour cap and resumed without rerunning successful cells.
- The experiment report contains every Slurm job ID and snapshot bundle path.

## Non-goals for the First Release

- Multi-node queue coordination inside one Slurm allocation.
- Automatic environment-image rebuilds on every code commit.
- Automatic selection of `jobs_per_gpu` without a measured benchmark.
- Treating high GPU utilization as sufficient evidence of throughput or model
  correctness.
- Retrying failed training cells indefinitely or silently changing a failed
  cell's hyperparameters.

# Clariden node-pool launcher

`slurm_clariden` submits one exclusive GH200 node and runs a dynamic queue of
Hydra cells inside it. Each launch uses the repository's immutable Git archive;
workers execute `main.py` from that archive, not from the live checkout.

Clariden production uses `normal`. The `slurm_clariden_debug` profile is only
for short canaries. These defaults follow the current [Clariden partition
policy](https://docs.cscs.ch/clusters/clariden/) and [GH200 binding
guidance](https://docs.cscs.ch/running/slurm/).

## One-time environment setup

1. Copy `containers/clariden-foundry.toml.example` outside the repository,
   replace the image placeholder with a pinned ARM64-compatible PyTorch image,
   and keep the same-path mounts. Do not put credentials in the EDF.
2. Create a mode-0600 application env file containing `WANDB_API_KEY`,
   `WANDB_ENTITY`, `PROJECT`, and `FOUNDRY_DATA_ROOT`. The initial project data
   root is `/capstor/store/cscs/swissai/a0091/processed`.
3. From a compute allocation, enter the configured EDF and run
   `scripts/bootstrap_clariden_environment.sh` with a persistent shared venv
   path. Do not create or modify that venv from the login-node interpreter.
4. For concurrency above one job per GPU, configure the separate MPS EDF from
   `containers/clariden-foundry-mps.toml.example`. The launcher checks for the
   [CSCS MPS hook](https://docs.cscs.ch/software/container-engine/resource-hook/)
   before submitting.

The required submission environment is:

```bash
export CSCS_ACCOUNT=<project-account>
export PROJECT=/capstor/store/cscs/swissai/a0091
export FOUNDRY_DATA_ROOT=${PROJECT}/processed
export FOUNDRY_SNAPSHOT_ROOT=<shared-absolute-path>/foundry-launches
export FOUNDRY_CLARIDEN_VENV=<shared-absolute-path>/foundry-venv
export FOUNDRY_ENV_FILE=<absolute-path>/clariden.env
export FOUNDRY_CLARIDEN_EDF=<absolute-path>/clariden-foundry.toml
```

The EDF uses the [CSCS Container Engine](https://docs.cscs.ch/software/container-engine/run/)
through `srun --environment=...`. The application env file is sourced once by
the Slurm setup and is referenced by path; its contents are never copied into
the snapshot or queue.

## Canary and production

Start with one job per GPU and direct data reads:

```bash
git status --short  # must print nothing
python main.py experiment=<short-canary> \
  hydra/launcher=slurm_clariden_debug -m
```

Verify four distinct `SLURM_LOCALID`/GPU bindings, CPU affinity, snapshot
provenance, the container interpreter, data access, and W&B authentication.
For MPS, select the MPS EDF and explicitly test the intended concurrency. Do
not raise `jobs_per_gpu` based only on utilization; compare completed valid
cells per wall-clock hour and stop on OOMs, failures, or metric drift.
The portable launcher ceiling is 48 MPS clients per GPU. MPS workers validate
that their CPU masks belong to exactly one physical NUMA domain and select the
adjacent GPU from that domain; the container must provide `hwloc-bind` and
`hwloc-calc`.

The four normalization pools can be submitted after committing all changes:

```bash
scripts/launch_clariden_normalization.sh
```

Each command prints the Slurm job ID and snapshot bundle immediately. The same
information is stored in `manifests/submission.json` inside the bundle. Every
W&B-enabled cell attempt receives a deterministic run ID derived from the
snapshot, cell, and attempt number; that exact ID is stored in both the queue
record and attempt history.

## Resume

Resume uses the original snapshot and queue. It never re-expands cells from
the current checkout and never reruns successful cells:

```bash
python main.py experiment=<original-experiment> \
  hydra/launcher=slurm_clariden \
  hydra.launcher.resume_snapshot=/absolute/path/to/original/bundle \
  hydra.launcher.jobs_per_gpu=1 -m
```

Failed cells are retried once per submitted resume when `retry_failed=true`;
drained and interrupted cells are always made pending. OOMs remain recorded so
concurrency can be lowered deliberately rather than retried indefinitely.

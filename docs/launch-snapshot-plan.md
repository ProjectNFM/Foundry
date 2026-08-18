# Plan: Immutable source snapshots for queued Hydra jobs

**Status:** Proposed implementation plan (no code changes in this document)

## Objective

Ensure that every Hydra multirun launched through Foundry's local or SLURM
launcher executes the exact committed source and resolved configuration that
existed when it was submitted. In particular, a job that starts hours later
must be unaffected if the developer subsequently changes branches, edits files,
or works in the normal checkout.

The intended user experience is unchanged:

```bash
uv run python main.py experiment=... -m
```

The launcher performs all staging automatically. Users do not create, switch,
or clean up Git worktrees manually.

## Why this is needed in the current codebase

`hydra_plugins/foundry_launcher/packed_launcher.py` submits Slurm arrays using
Submitit, but it does not capture a source revision. The worker therefore
imports/runs code from the checkout visible when the allocation begins. This is
unsafe for a queued sweep: the checkout can point to another branch by then.

There is already a partial local-only version of this idea in
`hydra_plugins/foundry_launcher/local_gpu_launcher.py`:

- `LocalGpuLauncher._snapshot_launch_context()` copies selected source and
  config paths into `${hydra.sweep.dir}/.launch_snapshot`.
- Its subprocesses run `snapshot/main.py` with `snapshot` prepended to
  `PYTHONPATH`.

That implementation demonstrates the desired runtime boundary, but is not a
complete provenance mechanism: it is copy-based rather than Git-addressable,
does not require a clean checkout, does not create a manifest, and does not
cover the Slurm launcher.

The masking-sweep failure report is the motivating incident. Its W&B evidence
does not prove that every failure came from code drift, but it shows why source
identity needs to be visible per run. In particular, the early DataLoader
metadata errors cannot currently be tied unambiguously to one source revision.

## Design decisions

### 1. Stage a Git archive, not a live worktree

The baseline design will create a source tree using `git archive <commit>` at
submission time. This is preferable to a live worktree because it has no
mutable `.git` state and cannot be changed by later branch operations.

The staged tree contains only tracked files at the submitted commit. That is a
feature: the launcher should reject a dirty checkout by default, rather than
silently include uncommitted changes.

An automatically-created detached worktree can remain a future option if a
workflow truly requires `.git` at job time. It should not be the default.

### 2. Create one bundle per Hydra multirun, not per array task

All tasks submitted by one `-m` invocation share the same source revision and
base configuration. Stage that source once, then give every task a reference to
the same immutable bundle. Each Hydra task retains its own output directory and
resolved task configuration.

### 3. Put the bundle on storage visible to compute nodes

The bundle root must be readable from the Slurm worker nodes for the full queue
and retry lifetime. It must not be `hydra.sweep.dir` unless that directory is
known to be shared on every target cluster.

Introduce one explicit launcher setting, for example
`hydra.launcher.snapshot_root`. Configure it per target environment:

- regular Slurm: a shared project filesystem such as
  `/shared/foundry-launches`;
- CSCS: a persistent path under the appropriate shared `$SCRATCH` location;
- local GPU: the normal sweep directory is acceptable because the jobs execute
  immediately on the same machine.

Do not use `/tmp`, `SLURM_TMPDIR`, or a login-node-only path as the primary
bundle location.

### 4. Make provenance a first-class artifact

Each bundle gets a manifest that records the inputs required to identify and
reproduce the launch. Each task logs the same identity locally and sends it to
W&B.

### 5. Do not copy credentials or data into the bundle

The Git archive intentionally excludes `.env` and other ignored files. Jobs
continue to source credentials from a configured external, read-only location.
Data and pretrained checkpoints also remain in their existing external paths.
The manifest stores references and hashes where safe, never secrets.

## Bundle layout

Use a predictable, collision-resistant directory name:

```text
<snapshot_root>/
  <UTC timestamp>_<sweep-name>_<short-git-sha>_<random-suffix>/
    source/                 # git archive of exactly one commit
      main.py
      foundry/
      hydra_plugins/
      configs/
      pyproject.toml
      uv.lock               # if tracked
      ...
    manifests/
      launch.json
      resolved-base-config.yaml
      submitted-overrides.txt
      source-files.sha256   # optional tree/content digest
    task-configs/           # resolved configuration per Hydra task
    logs/                   # launcher-side staging and submission log
```

`sweep-name` should be derived from a safe, human-readable value such as
`run.group`, `run.name`, or the experiment name, with a generic fallback. The
timestamp, full SHA in the manifest, and random suffix make the path unique
when the same commit is launched more than once.

After successful staging, make `source/` read-only. The job output/checkpoint
directories must stay outside `source/`; Hydra already supplies separate output
directories for multirun tasks.

## Proposed implementation shape

### New shared module

Add a small module owned by the launcher plugin, for example:

```text
hydra_plugins/foundry_launcher/launch_snapshot.py
```

It should be dependency-light (standard library plus OmegaConf where necessary)
and expose a narrow API such as:

- `prepare_snapshot(...) -> LaunchSnapshot`
- `write_task_provenance(...)`
- `verify_snapshot(...)`
- `build_worker_environment(...)`

`LaunchSnapshot` should contain absolute paths and immutable identifiers:

```text
bundle_dir, source_dir, manifest_path, git_sha, source_digest,
base_config_path, environment_fingerprint
```

Keep Git subprocess calls and filesystem behavior in this module, rather than
duplicating them in `packed_launcher.py` and `local_gpu_launcher.py`.

### Changes to `PackedSubmititLauncher`

Modify `hydra_plugins/foundry_launcher/packed_launcher.py` as follows:

1. At the beginning of `launch()`, determine the project root from
   `sys.argv[0]` and call `prepare_snapshot()` exactly once.
2. Pass a serializable snapshot descriptor with each submitted task, alongside
   the existing Hydra override list and singleton state.
3. Update `launch_batch()` so that, before invoking Hydra's task function, it:
   - changes working directory to `snapshot.source_dir`;
   - prepends that directory to `PYTHONPATH` and `sys.path`;
   - verifies the manifest's source identity;
   - writes task-specific provenance beside the Hydra output;
   - invokes the task with the original task overrides.
4. Make the Submitit worker process resolve its project imports from the
   snapshot. This must be tested carefully: Submitit unpickles the launcher
   class before `launch_batch()` runs. If changing `sys.path` inside
   `launch_batch()` is too late for a given Submitit version, use a tiny
   snapshot-resident bootstrap entry point in the worker command instead. The
   bootstrap must import and execute `snapshot/main.py`, not live
   `/home/.../Foundry/main.py`.
5. Append the snapshot startup commands to (rather than overwrite) the
   configured Submitit `setup` commands.

The last point is important for existing launcher configs. `slurm_default.yaml`
currently sources `.env` relative to the worker's working directory, and
`slurm_cscs.yaml` explicitly changes to `${oc.env:SCRATCH}/Foundry`. Both would
reintroduce the live-checkout dependency if left unchanged.

The implementation should establish this order in the worker shell:

```text
source the configured external environment file (if present)
export PYTHONPATH=<bundle>/source:${PYTHONPATH}
cd <bundle>/source
start the Submitit/Hydra task
```

Do not put the `.env` file in `source/`, and do not use a bare `source .env`
after changing directory. Replace the current relative setup command with a
documented, explicit environment-file setting (or a clearly scoped absolute
path resolved at submission time).

### Changes to `LocalGpuLauncher`

Replace `_snapshot_launch_context()` in
`hydra_plugins/foundry_launcher/local_gpu_launcher.py` with the shared
`prepare_snapshot()` flow. Preserve its useful behavior: run
`snapshot/source/main.py` and prepend `snapshot/source` to `PYTHONPATH`.

The local launcher should apply the same clean-repository validation and write
the same manifest. This prevents “local test worked, queued run changed” from
being a different provenance model.

### Configuration additions

Add common launcher configuration fields to:

- `configs/hydra/launcher/slurm_default.yaml`
- `configs/hydra/launcher/slurm_cscs.yaml`
- `configs/hydra/launcher/local_gpu.yaml`

Suggested initial fields:

```yaml
snapshot:
  enabled: true
  root: <cluster-specific shared path>
  require_clean_git: true
  verify_on_worker: true
  retain_source: true
  environment_file: ${oc.env:FOUNDRY_ENV_FILE,".env"}
```

Use configuration names consistent with Hydra's structured launcher schema;
the exact nesting can be adjusted while implementing. The crucial requirements
are that the root is explicit and that CSCS and standard Slurm may use distinct
values.

Add a temporary opt-out such as `hydra.launcher.snapshot.enabled=false` only
for rollout/debugging. Once validated, production sweeps should use the safe
default and large multiruns should retain the clean-Git guard.

### Runtime provenance in `main.py`

Add a small startup hook near the existing Slurm job logging in `main.py` to:

1. Read snapshot identity from explicitly named environment variables or a
   manifest path supplied by the launcher.
2. Verify that `main.py` and the imported `foundry` package resolve under the
   snapshot source directory.
3. Log the experiment ID, Git SHA, source digest, manifest path, Slurm job ID,
   array task ID, and restart count.
4. Save a copy/link of the task provenance in Hydra's output directory.
5. Add non-secret provenance keys to the W&B config before
   `_log_config_to_wandb()` runs.

Example W&B fields:

```text
provenance.experiment_id
provenance.git_sha
provenance.source_digest
provenance.manifest_path
provenance.slurm_job_id
provenance.slurm_array_task_id
provenance.environment_lock_hash
```

This must not change existing W&B IDs or resume behavior. A requeued job keeps
the same W&B run ID but reports the same immutable source identity and its
incremented restart count.

## Detailed submission process

The following is the desired behavior of one ordinary command.

### 1. Discover the checkout and validate it

From the path of `main.py`, determine the repository root. Run Git checks with
that directory as the working directory:

1. `git rev-parse --show-toplevel` — confirm this is a Git worktree.
2. `git rev-parse HEAD` — capture the full 40-character commit SHA.
3. `git diff --quiet` and `git diff --cached --quiet` — reject modified and
   staged-but-uncommitted tracked changes.
4. `git status --porcelain --untracked-files=all` — reject untracked files by
   default, or at least display them and document an explicit policy.
5. `git submodule status --recursive` — record submodule revisions if present.

The rejection message should name the affected paths and recommend committing,
stashing, or launching from a clean experiment branch. It must not make a
destructive Git change on the user's behalf.

### 2. Resolve launch inputs before submitting jobs

Construct the snapshot ID, create the bundle directory with restrictive
permissions, and write a staging-in-progress marker. Resolve and store:

- full Git SHA and branch-at-submission;
- exact command-line arguments, preserving quoting separately from display
  text;
- the original Hydra multirun overrides and the generated per-task overrides;
- resolved base config and resolved config for each task;
- hashes of `pyproject.toml`, `uv.lock` (if present), and selected config files;
- Python executable/version and package/environment fingerprint;
- current launcher config, including timeout, array parallelism, container, and
  snapshot settings;
- references to external data roots and checkpoint paths, with no credentials.

The resolved task configs are especially useful: a multirun override can expand
into many folds/parameter combinations, and the manifest must say exactly which
one a given scheduler array task used.

### 3. Materialize and seal the source

Create `source/` from `git archive --format=tar <full-sha>` and extract it into
the bundle. Do not use `copytree` from the live checkout for the production
Slurm path. Verify that the staged top-level layout includes `main.py`,
`foundry/`, `hydra_plugins/`, and `configs/`.

Compute a deterministic content digest of the staged tracked files, write it to
the manifest, then remove write permissions from `source/`. Replace the
staging-in-progress marker with a completion marker only after every manifest
write and verification succeeds. Jobs must refuse a bundle without that marker.

### 4. Submit with immutable absolute paths

Pass only absolute snapshot paths to Submitit. For every packed or array task,
include:

- the absolute manifest path;
- the source directory;
- the expected full Git SHA and content digest;
- the task configuration/provenance path;
- original Hydra task overrides;
- an experiment bundle ID.

Log the snapshot ID and paths once at submission, along with all submitted
Slurm job IDs. Include those IDs in the bundle manifest after submission if
possible; otherwise write an append-only `submission.json` next to it.

### 5. Boot and verify on the worker

Before training, each worker should:

1. Confirm the completion marker and manifest exist and are readable.
2. Confirm the source directory is the expected absolute directory.
3. Verify the source digest (at least once per array allocation; per task in
   the initial safety-focused release).
4. Set `PYTHONPATH` with the snapshot source first and change to that directory.
5. Verify Python imported `main`, `foundry`, and `hydra_plugins` from snapshot
   paths, not the live repository.
6. Write `provenance.json` into that task's Hydra output directory before model
   construction starts.
7. Start training.

Any mismatch must fail before a W&B run is initialized or a checkpoint is
written. The error should name the expected and actual source paths/SHA.

### 6. Retry and resume behavior

Slurm requeues and manual reruns must reuse the original bundle. A retry command
should take a manifest or bundle ID, not create a fresh snapshot from the
currently checked-out branch. The task provenance records the retry/restart
count separately from the immutable source identity.

## Environment and container considerations

The source snapshot prevents checkout drift, but not dependency drift. Record
an environment fingerprint from the submission context in the first release.

For the current project, this should at least include:

- `uv.lock` hash (when tracked);
- `pyproject.toml` hash;
- `python --version` and `sys.executable`;
- versions/commit metadata for Git-sourced dependencies where available;
- the CSCS container-environment path or image identifier.

The next hardening step can be a pinned container image or a dedicated,
lockfile-derived virtual environment per release. Do not conflate that larger
project with the source-snapshot change: the latter should land and be tested
first.

For `slurm_cscs.yaml`, keep the existing container interpreter handling in
`PackedSubmititLauncher.launch()`, but replace its static
`cd ${oc.env:SCRATCH}/Foundry` behavior with the dynamically supplied snapshot
directory. The job still needs access to the configured container environment;
only its application source location changes.

## Tests and acceptance criteria

### Unit tests

Add tests around the shared snapshot module for:

- clean checkout succeeds and records the exact SHA;
- staged/unstaged/untracked changes are rejected under the default policy;
- a staged archive contains required project paths;
- manifests contain required non-secret fields;
- source digest is stable and detects a modified staged file;
- incomplete bundles are rejected;
- task provenance identifies the intended override/config;
- `.env` and other ignored files are not copied;
- a configured external environment file is referenced but not embedded.

Use a temporary Git repository in tests rather than depending on the developer's
actual checkout state.

### Launcher tests

Extend/add tests around the launcher behavior (the existing test suite contains
launcher-adjacent tests under `tests/test_models/test_orchestration_helpers.py`
and config validation tests):

- local launcher invokes `source/main.py`, not the original checkout's
  `main.py`;
- local queued subprocesses keep working after the parent checkout is edited;
- packed launcher task setup receives absolute snapshot paths;
- packed jobs with `tasks_per_node > 1` give every inner task the same bundle
  but distinct per-task configs;
- standard Slurm setup and CSCS setup preserve required environment setup while
  removing their live-checkout `cd` dependency;
- `snapshot.enabled=false` preserves today's behavior during transition.

Mock Submitit and subprocess calls; do not submit real Slurm jobs in unit tests.

### Manual canary sequence

1. Launch a single local-GPU multirun with snapshots enabled. Inspect the
   manifest, process command, Hydra output `provenance.json`, and W&B config.
2. Submit one short Slurm job with a deliberately long queue delay (or hold it
   if the scheduler permits).
3. After submission, change the normal checkout to another branch and make a
   harmless edit there.
4. Release the job and verify its startup logs/import paths/SHA point only to
   the original bundle.
5. Test a requeue or controlled restart and verify that it reuses the same
   bundle and W&B identity.
6. Run a small packed array with `tasks_per_node > 1`.
7. Only then enable snapshotting for a production-sized sweep.

### Definition of done

The work is complete when all of the following are true:

- A normal `uv run python main.py ... -m` launch needs no manual worktree step.
- A dirty checkout is rejected before any scheduler job is submitted by
  default.
- Every local and Slurm task imports application code from one immutable,
  submission-time source bundle.
- Changing branches or editing the live checkout after submission cannot change
  a queued job's application code.
- Each Hydra output and W&B run exposes an unambiguous source SHA, bundle ID,
  resolved config identity, and scheduler identity.
- Retries reuse the original source bundle.
- The special CSCS/container launcher behaves correctly without relying on
  `${SCRATCH}/Foundry` as its code directory.

## Rollout and operational policy

### Phase 1: observability

Implement manifest creation and worker logging in an opt-in mode. Continue
using existing launch behavior only long enough to compare recorded identity
with actual import paths. This phase should be brief: it does not eliminate the
race.

### Phase 2: local enforcement

Enable snapshot execution and clean-Git validation by default for
`local_gpu`. This makes it easy to debug without consuming cluster resources.

### Phase 3: Slurm canaries

Enable the standard Slurm launcher for small canaries, then validate the CSCS
container path separately. Do not roll out to large array jobs until both have
passed the manual canary sequence.

### Phase 4: production default

Make snapshots enabled and clean Git required in `slurm_default` and
`slurm_cscs`. Keep an explicitly named emergency opt-out temporarily, emit a
prominent warning whenever it is used, and remove it after the workflow has
stabilized.

### Retention and cleanup

Retain bundles until all associated jobs and intended retries are complete and
results have been reviewed. A conservative policy is to retain for 30–90 days,
then remove only bundles whose manifest reports no active/recent jobs.

Implement cleanup as a separate, explicit command that lists candidates before
deleting them. It must never prune the live checkout or an unresolved active
bundle. Initial implementation can rely on manual review; automation is not a
prerequisite for safe snapshots.

## Non-goals for the first implementation

- Automatically committing, stashing, or switching the user's Git branches.
- Copying datasets, checkpoints, WandB credentials, or `.env` into bundles.
- Changing the training algorithm, checkpoint policy, Slurm timeout values, or
  the independent timeout/SIGTERM issues documented in the masking-sweep
  report.
- Guaranteeing bitwise reproducibility across GPU drivers or changing package
  versions; the first version records environment identity and makes source
  identity immutable.

## Risks to resolve during implementation

1. **Submitit import timing:** Confirm whether snapshot `PYTHONPATH` is in
   effect before the pickled launcher/task is unpickled. Use a
   snapshot-resident bootstrap if it is not.
2. **Shared storage visibility:** Verify the configured snapshot roots are
   mounted and readable from login and compute nodes on every target cluster.
3. **Hydra config discovery:** Verify that running the staged `main.py` finds
   `configs/` and custom `hydra_plugins/` exactly as it does from the checkout.
4. **Credential source:** Replace relative `.env` sourcing with a robust,
   documented external-path policy without logging secret values.
5. **External source dependencies:** Record how Git-sourced dependencies in
   `pyproject.toml` are pinned by `uv.lock`; if they are not adequately pinned,
   address that as a follow-up.
6. **Bundle size and staging speed:** Measure archive size and startup cost.
   One bundle per multirun should keep this small compared with model training.


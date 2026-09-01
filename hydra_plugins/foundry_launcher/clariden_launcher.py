"""Hydra launcher for dynamic, single-node Clariden worker pools."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from hydra.core.utils import configure_log, filter_overrides, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.foundry_launcher.clariden_queue import (
    ClaridenFileQueue,
    canonical_cell_id,
)
from hydra_plugins.foundry_launcher.launch_snapshot import (
    LaunchSnapshot,
    build_setup_commands,
    build_worker_environment,
    get_slurm_job_identifiers,
    prepare_snapshot,
    verify_import_paths,
    verify_snapshot,
)

log = logging.getLogger(__name__)

# CUDA 13.0 and earlier support at most 48 Volta+ MPS client contexts per
# device.  Keep the launcher compatible with the unspecified pinned CUDA image
# rather than advertising the newer CUDA 13.1+ limit of 60.
MAX_MPS_CLIENTS_PER_GPU = 48


class ClaridenResources(NamedTuple):
    jobs_per_gpu: int
    workers_per_node: int
    cpus_per_worker: int
    memory_per_worker_gb: int


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON without exposing a partially written manifest."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2))
    os.replace(temporary, path)


def _required_absolute_file(value: Any, label: str) -> Path:
    if value is None or str(value).strip() in {"", "???", "null", "None"}:
        raise ValueError(f"Clariden launcher requires {label}")
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"{label} is not a readable file: {path}")
    return path


def _mps_annotation_enabled(edf: Path) -> bool:
    """Accept the CSCS dotted annotation in either TOML representation."""
    import tomllib

    parsed = tomllib.loads(edf.read_text())
    annotations = parsed.get("annotations", {})
    direct = annotations.get("com.hooks.nvidia_cuda_mps.enabled")
    if str(direct).lower() == "true":
        return True
    current: Any = annotations
    for key in ("com", "hooks", "nvidia_cuda_mps", "enabled"):
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return str(current).lower() == "true"


def validate_clariden_config(params: Mapping[str, Any]) -> ClaridenResources:
    """Validate submission inputs and return Python-derived worker resources."""
    account = str(params.get("account") or "").strip()
    if not account or account == "???":
        raise ValueError("Clariden launcher requires CSCS_ACCOUNT")

    nodes = int(params.get("nodes", 1))
    if nodes != 1:
        raise ValueError("Clariden launcher currently supports nodes == 1 only")

    partition = str(params.get("partition", "normal"))
    if partition not in {"normal", "debug"}:
        raise ValueError("Clariden partition must be 'normal' or 'debug'")
    timeout_min = int(params.get("timeout_min", 240))
    if timeout_min < 1:
        raise ValueError("timeout_min must be positive")
    if partition == "debug" and timeout_min > 90:
        raise ValueError("Clariden debug jobs may not exceed 90 minutes")
    if partition == "normal" and timeout_min > 720:
        raise ValueError("Clariden normal jobs may not exceed 720 minutes")

    drain_guard_min = int(params.get("drain_guard_min", 10))
    minimum_start_budget_min = int(params.get("minimum_start_budget_min", 0))
    if drain_guard_min < 0 or minimum_start_budget_min < 0:
        raise ValueError(
            "drain_guard_min and minimum_start_budget_min must be non-negative"
        )
    if drain_guard_min + minimum_start_budget_min >= timeout_min:
        raise ValueError(
            "drain guard plus minimum start budget must be less than timeout_min"
        )

    if not bool(params.get("exclusive", True)):
        raise ValueError("Clariden allocations must be exclusive")
    if int(params.get("gpus_per_node", 4)) != 4:
        raise ValueError("Clariden GH200 pools require gpus_per_node == 4")

    jobs_per_gpu_raw = params.get("jobs_per_gpu", 1)
    try:
        jobs_per_gpu = int(jobs_per_gpu_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"jobs_per_gpu must be an integer in [1, {MAX_MPS_CLIENTS_PER_GPU}]"
        ) from error
    if (
        isinstance(jobs_per_gpu_raw, float)
        and not jobs_per_gpu_raw.is_integer()
    ):
        raise ValueError(
            f"jobs_per_gpu must be an integer in [1, {MAX_MPS_CLIENTS_PER_GPU}]"
        )
    if (
        isinstance(jobs_per_gpu_raw, str)
        and str(jobs_per_gpu) != jobs_per_gpu_raw
    ):
        raise ValueError(
            f"jobs_per_gpu must be an integer in [1, {MAX_MPS_CLIENTS_PER_GPU}]"
        )
    if not 1 <= jobs_per_gpu <= MAX_MPS_CLIENTS_PER_GPU:
        raise ValueError(
            f"jobs_per_gpu must be an integer in [1, {MAX_MPS_CLIENTS_PER_GPU}]"
        )
    workers = 4 * jobs_per_gpu
    cpus = 72 // jobs_per_gpu
    memory = int(params.get("mem_gb", 450)) // workers
    if cpus < 1 or memory < 1:
        raise ValueError("derived CPU and memory shares must be at least one")

    for key, expected in {
        "workers_per_node": workers,
        "cpus_per_worker": cpus,
        "memory_per_worker_gb": memory,
    }.items():
        configured = params.get(key)
        if configured is not None and int(configured) != expected:
            raise ValueError(
                f"{key}={configured} does not match derived value {expected}"
            )

    edf = _required_absolute_file(
        params.get("container_environment"), "container_environment"
    )
    _required_absolute_file(
        params.get("application_environment_file"),
        "application_environment_file",
    )

    snapshot_cfg = dict(_plain(params.get("snapshot") or {}))
    if not snapshot_cfg.get("enabled", False):
        raise ValueError("Clariden launch snapshots may not be disabled")
    if not snapshot_cfg.get("require_clean_git", False):
        raise ValueError("Clariden requires snapshot.require_clean_git=true")
    snapshot_env = snapshot_cfg.get("environment_file")
    if (
        snapshot_env
        and Path(str(snapshot_env)).resolve()
        != Path(str(params.get("application_environment_file"))).resolve()
    ):
        raise ValueError(
            "snapshot.environment_file must reference "
            "application_environment_file"
        )
    root_raw = snapshot_cfg.get("root")
    if root_raw is None or str(root_raw).strip() in {"", "???", "null"}:
        raise ValueError("Clariden launcher requires FOUNDRY_SNAPSHOT_ROOT")
    root = Path(str(root_raw))
    if not root.is_absolute():
        raise ValueError("FOUNDRY_SNAPSHOT_ROOT must be an absolute path")
    if not root.is_dir() or not os.access(root, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(
            "FOUNDRY_SNAPSHOT_ROOT must be an accessible, writable directory: "
            f"{root}"
        )

    if jobs_per_gpu > 1 and not _mps_annotation_enabled(edf):
        raise ValueError(
            "jobs_per_gpu > 1 requires EDF annotation "
            'com.hooks.nvidia_cuda_mps.enabled = "true"'
        )

    return ClaridenResources(jobs_per_gpu, workers, cpus, memory)


def _resolve_bundle(value: str | Path) -> LaunchSnapshot:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "manifests" / "snapshot-descriptor.json"
    snapshot = LaunchSnapshot.from_file(path)
    verify_snapshot(snapshot)
    return snapshot


def _sweep_name(config: DictConfig) -> str:
    """Resolve a batch label without touching cell-only interpolations."""
    group = OmegaConf.select(
        config, "run.group", default=None, throw_on_missing=False
    )
    if group is not None:
        return str(group)

    name = OmegaConf.select(
        config, "run.name", default=None, throw_on_missing=False
    )
    return "clariden-sweep" if name is None else str(name)


def _remaining_seconds(timeout_min: int, started: float) -> float:
    raw_end = os.environ.get("SLURM_JOB_END_TIME")
    if raw_end:
        try:
            return float(raw_end) - time.time()
        except ValueError:
            pass
    return timeout_min * 60.0 - (time.monotonic() - started)


def _numa_node_for_current_affinity() -> int:
    """Return the single physical NUMA node covered by this rank's CPU mask."""
    try:
        mask_result = subprocess.run(
            ["hwloc-bind", "--get", "--taskset"],
            capture_output=True,
            text=True,
            check=True,
        )
        numa_result = subprocess.run(
            [
                "hwloc-calc",
                "--physical",
                "--intersect",
                "NUMAnode",
                mask_result.stdout.strip(),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "MPS worker affinity validation requires working hwloc-bind and "
            "hwloc-calc commands inside the container"
        ) from error

    numa_nodes = numa_result.stdout.split()
    if len(numa_nodes) != 1 or not numa_nodes[0].isdigit():
        raise RuntimeError(
            "MPS worker CPU affinity must belong to exactly one physical NUMA "
            f"node; mask={mask_result.stdout.strip()!r}, "
            f"NUMA nodes={numa_result.stdout.strip()!r}"
        )
    numa_node = int(numa_nodes[0])
    if not 0 <= numa_node < 4:
        raise RuntimeError(
            f"Expected a GH200 GPU/NUMA domain in [0, 3], got {numa_node}"
        )
    return numa_node


def _worker_identity(jobs_per_gpu: int) -> tuple[dict[str, Any], str]:
    local_rank = int(
        os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0"))
    )
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [
        item.strip() for item in original_visible.split(",") if item.strip()
    ]

    numa_node: int | None = None
    if jobs_per_gpu == 1:
        gpu_slot = local_rank
        if len(visible_devices) != 1:
            raise RuntimeError(
                "One-job-per-GPU mode requires exactly one CUDA-visible device "
                f"per rank, got {original_visible!r}"
            )
        assigned = visible_devices[0]
    else:
        numa_node = _numa_node_for_current_affinity()
        gpu_slot = numa_node
        if len(visible_devices) >= 4:
            assigned = visible_devices[gpu_slot]
        elif not visible_devices:
            assigned = str(gpu_slot)
        else:
            raise RuntimeError(
                "MPS mode requires all four GPUs to be visible (or "
                "CUDA_VISIBLE_DEVICES to be unset) before NUMA binding; got "
                f"{original_visible!r}"
            )

    try:
        cpu_affinity: list[int] | None = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpu_affinity = None

    scheduler = get_slurm_job_identifiers()
    identity: dict[str, Any] = {
        "allocation_id": scheduler.get("slurm_job_id"),
        "node_hostname": socket.gethostname(),
        "worker_rank": os.environ.get("SLURM_PROCID", str(local_rank)),
        "local_rank": local_rank,
        "numa_node": numa_node,
        "gpu_slot": gpu_slot,
        "gpu_identifier": assigned,
        "cuda_visible_devices_at_start": original_visible,
        "jobs_per_gpu": jobs_per_gpu,
        "cpu_affinity": cpu_affinity,
    }
    return identity, assigned


def _classify_failure(returncode: int, log_path: Path) -> str:
    try:
        tail = log_path.read_bytes()[-65536:].decode(errors="replace").lower()
    except OSError:
        tail = ""
    if "out of memory" in tail or "cuda error: memory" in tail:
        return "oom"
    if returncode in {-9, 137}:
        return "killed_or_oom"
    return "nonzero_exit"


def _wandb_run_identity(
    snapshot: LaunchSnapshot, cell_id: str, attempt: int
) -> str:
    payload = f"{snapshot.bundle_id}:{cell_id}:attempt-{attempt}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _run_claimed_cell(
    snapshot: LaunchSnapshot,
    queue: ClaridenFileQueue,
    cell: Mapping[str, Any],
) -> bool:
    """Run one claimed cell and always make ordinary worker errors terminal."""
    attempt = int(cell["attempt"])
    output_dir = Path(str(cell["output_root"])) / f"attempt-{attempt:03d}"
    log_dir = (
        Path(snapshot.bundle_dir) / "logs" / "clariden" / str(cell["cell_id"])
    )
    log_path = log_dir / f"attempt-{attempt:03d}.log"
    wandb_run_id = (
        _wandb_run_identity(snapshot, str(cell["cell_id"]), attempt)
        if bool(cell.get("wandb_enabled", False))
        else None
    )

    exit_status = 1
    failure: str | None = "worker_internal_error"
    error_detail: str | None = None
    try:
        if wandb_run_id is not None:
            cell = queue.update_running(
                str(cell["cell_id"]), {"wandb_run_identity": wandb_run_id}
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        overrides = list(cell["overrides"])
        overrides.append(f"hydra.run.dir={output_dir}")
        command = [sys.executable, str(Path(snapshot.source_dir) / "main.py")]
        command.extend(overrides)
        child_env = dict(os.environ)
        # Each cell is an independent single-GPU run, not part of a
        # distributed group.  Clear the rank variables inherited from the
        # parent srun task so that Lightning treats every subprocess as
        # rank 0 (required for WandB logging and checkpoint callbacks).
        for _rank_key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NTASKS",
            "SLURM_NPROCS",
            "PMI_RANK",
            "PMI_SIZE",
        ):
            child_env.pop(_rank_key, None)
        child_env["FOUNDRY_SNAPSHOT_TASK_INDEX"] = str(cell["position"])
        child_env["FOUNDRY_CLARIDEN_CELL_ID"] = str(cell["cell_id"])
        child_env["FOUNDRY_CLARIDEN_ATTEMPT"] = str(attempt)
        if wandb_run_id is not None:
            child_env["FOUNDRY_WANDB_RUN_ID"] = wandb_run_id

        with log_path.open("wb") as stream:
            result = subprocess.run(
                command,
                cwd=snapshot.source_dir,
                env=child_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_status = result.returncode
        failure = (
            None
            if exit_status == 0
            else _classify_failure(exit_status, log_path)
        )
    except Exception as error:
        error_detail = f"{type(error).__name__}: {error}"[:1000]
        log.exception(
            "Clariden worker failed while executing cell %s attempt %d",
            cell["cell_id"],
            attempt,
        )

    succeeded = exit_status == 0
    queue.finish(
        str(cell["cell_id"]),
        succeeded=succeeded,
        exit_status=exit_status,
        failure_classification=failure,
        extra={
            "hydra_output_directory": str(output_dir),
            "worker_log": str(log_path),
            "wandb_run_identity": wandb_run_id,
            "worker_error": error_detail,
        },
    )
    return succeeded


def run_clariden_pool(snapshot_json: str, launch_manifest_path: str) -> None:
    """Submitit worker entry point; every Slurm rank runs this queue loop."""
    started = time.monotonic()
    snapshot = LaunchSnapshot.from_json(snapshot_json)
    os.chdir(snapshot.source_dir)
    sys.path.insert(0, snapshot.source_dir)
    os.environ["PYTHONPATH"] = (
        snapshot.source_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    launch = json.loads(Path(launch_manifest_path).read_text())
    if launch.get("verify_on_worker", True):
        verify_snapshot(snapshot)
        import foundry  # noqa: F401

        verify_import_paths(snapshot.source_dir)

    for key, value in build_worker_environment(snapshot).items():
        os.environ[key] = value
    os.environ["FOUNDRY_SNAPSHOT_VERIFY_ON_WORKER"] = str(
        int(launch.get("verify_on_worker", True))
    )

    env_file = Path(launch["application_environment_file"])
    if not env_file.is_file() or not os.access(env_file, os.R_OK):
        raise RuntimeError(
            "Application environment file is not readable inside the container: "
            f"{env_file}"
        )
    data_root = Path(launch["data_root"])
    if not data_root.is_dir() or not os.access(data_root, os.R_OK | os.X_OK):
        raise RuntimeError(
            f"Configured data root is not readable inside the container: {data_root}"
        )
    stage_source_root = Path(launch["stage_source_root"])
    if not stage_source_root.is_dir() or not os.access(
        stage_source_root, os.R_OK | os.X_OK
    ):
        raise RuntimeError(
            "Configured stage source root is not readable inside the container: "
            f"{stage_source_root}"
        )

    jobs_per_gpu = int(launch["resources"]["jobs_per_gpu"])
    identity, assigned_gpu = _worker_identity(jobs_per_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu
    os.environ["OMP_NUM_THREADS"] = str(launch["resources"]["cpus_per_worker"])
    log.info(
        "Clariden worker binding: rank=%s local_rank=%s numa=%s gpu=%s "
        "affinity=%s",
        identity["worker_rank"],
        identity["local_rank"],
        identity["numa_node"],
        assigned_gpu,
        identity["cpu_affinity"],
    )
    log.info(
        "Resolved data paths: data.root=%s stage.source_root=%s",
        launch["data_root"],
        launch["stage_source_root"],
    )

    queue = ClaridenFileQueue(Path(snapshot.bundle_dir) / "manifests")
    drain_seconds = 60 * (
        int(launch["drain_guard_min"]) + int(launch["minimum_start_budget_min"])
    )
    worker_had_failure = False

    while True:
        if (
            _remaining_seconds(int(launch["timeout_min"]), started)
            <= drain_seconds
        ):
            queue.drain_pending()
            break

        cell = queue.claim(identity)
        if cell is None:
            break

        succeeded = _run_claimed_cell(snapshot, queue, cell)
        worker_had_failure = worker_had_failure or not succeeded

    # Do not let one rank fail the Slurm step while other ranks still own cells.
    while queue.has_records("running"):
        remaining = _remaining_seconds(int(launch["timeout_min"]), started)
        if remaining <= 1:
            log.warning(
                "Stopping the final worker wait with running cells still "
                "recorded because the allocation is expiring"
            )
            break
        time.sleep(min(1, remaining))
    if worker_had_failure or queue.has_records("failed"):
        raise RuntimeError(
            "One or more Clariden pool cells failed; inspect clariden-queue/failed"
        )


class ClaridenNodePoolLauncher(Launcher):
    """Submit one exclusive GH200 node and dynamically fill its worker ranks."""

    def __init__(self, **params: Any) -> None:
        self.params = {key: _plain(value) for key, value in params.items()}
        self.config: DictConfig | None = None
        self.hydra_context: HydraContext | None = None
        self.task_function: TaskFunction | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.config = config

    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[Any]:
        import submitit

        del initial_job_idx
        setup_globals()
        assert self.config is not None
        configure_log(
            self.config.hydra.hydra_logging, self.config.hydra.verbose
        )
        resources = validate_clariden_config(self.params)

        snapshot_cfg = dict(self.params["snapshot"])
        edf = Path(self.params["container_environment"]).resolve()
        app_env = Path(self.params["application_environment_file"]).resolve()
        resume_raw = self.params.get("resume_snapshot")

        if resume_raw:
            snapshot = _resolve_bundle(str(resume_raw))
            launch_path = (
                Path(snapshot.bundle_dir) / "manifests" / "clariden-launch.json"
            )
            if not launch_path.is_file():
                raise RuntimeError(
                    f"Snapshot has no Clariden launch manifest: {launch_path}"
                )
            launch_manifest = json.loads(launch_path.read_text())
            if launch_manifest["container_environment_sha256"] != _sha256_file(
                edf
            ):
                raise ValueError(
                    "Resume EDF differs from the original submission"
                )
            if launch_manifest["application_environment_file"] != str(app_env):
                raise ValueError(
                    "Resume application environment reference differs from the original"
                )
            queue = ClaridenFileQueue(Path(snapshot.bundle_dir) / "manifests")
            resumed = queue.requeue_for_resume(
                retry_failed=bool(self.params.get("retry_failed", True))
            )
            log.info("Resuming original Clariden queue: %s", resumed)
        else:
            if not job_overrides:
                raise ValueError("Clariden launcher received an empty sweep")
            project_root = Path(sys.argv[0]).resolve().parent
            sweep_name = _sweep_name(self.config)
            snapshot = prepare_snapshot(
                project_root=project_root,
                snapshot_root=Path(snapshot_cfg["root"]),
                sweep_name=sweep_name,
                job_overrides=job_overrides,
                hydra_cfg=self.config,
                require_clean_git=True,
            )

            launch_token = uuid.uuid4().hex[:12]
            sweep_dir = Path(str(self.config.hydra.sweep.dir)).resolve()
            output_base = sweep_dir / "clariden-pools" / launch_token
            source = {
                "snapshot_bundle_id": snapshot.bundle_id,
                "git_sha": snapshot.git_sha,
                "source_digest": snapshot.source_digest,
                "environment_fingerprint": snapshot.environment_fingerprint,
                "container_environment_sha256": _sha256_file(edf),
                "container_environment": str(edf),
                "application_environment_file": str(app_env),
                "application_environment_file_sha256": _sha256_file(app_env),
                "wandb_enabled": "WandbLogger"
                in str(
                    OmegaConf.select(self.config, "logger._target_", default="")
                ),
                "resources": resources._asdict(),
            }
            records = []
            for position, overrides in enumerate(job_overrides):
                cell_overrides = list(overrides)
                cell_id = canonical_cell_id(cell_overrides)
                records.append(
                    {
                        **source,
                        "cell_id": cell_id,
                        "position": position,
                        "overrides": cell_overrides,
                        "output_root": str(output_base / cell_id),
                        "hydra_output_directory": None,
                        "wandb_run_identity": None,
                    }
                )
                log.info(
                    "\t#%d [%s]: %s",
                    position,
                    cell_id,
                    " ".join(filter_overrides(cell_overrides)),
                )
            queue = ClaridenFileQueue(Path(snapshot.bundle_dir) / "manifests")
            queue.initialize(records)

            data_root = str(
                Path(
                    str(
                        self.params.get(
                            "data_root",
                            OmegaConf.select(self.config, "data.root"),
                        )
                    )
                ).resolve()
            )
            stage_source = str(
                Path(
                    str(
                        OmegaConf.select(
                            self.config, "stage.source_root", default=data_root
                        )
                    )
                ).resolve()
            )
            launch_path = (
                Path(snapshot.bundle_dir) / "manifests" / "clariden-launch.json"
            )
            launch_manifest = {
                "snapshot_bundle_id": snapshot.bundle_id,
                "container_environment": str(edf),
                "container_environment_sha256": _sha256_file(edf),
                "application_environment_file": str(app_env),
                "application_environment_file_sha256": _sha256_file(app_env),
                "data_root": data_root,
                "stage_source_root": stage_source,
                "resources": resources._asdict(),
                "partition": self.params["partition"],
                "timeout_min": int(self.params["timeout_min"]),
                "drain_guard_min": int(self.params["drain_guard_min"]),
                "minimum_start_budget_min": int(
                    self.params.get("minimum_start_budget_min", 0)
                ),
                "verify_on_worker": bool(
                    snapshot_cfg.get("verify_on_worker", True)
                ),
                "launches": [],
            }
            _atomic_json_write(launch_path, launch_manifest)

        # A resume may deliberately lower concurrency, but all immutable
        # environment and source references remain those of the first launch.
        launch_manifest["resources"] = resources._asdict()
        launch_manifest["partition"] = self.params["partition"]
        launch_manifest["timeout_min"] = int(self.params["timeout_min"])
        launch_manifest["drain_guard_min"] = int(self.params["drain_guard_min"])
        launch_manifest["minimum_start_budget_min"] = int(
            self.params.get("minimum_start_budget_min", 0)
        )
        _atomic_json_write(launch_path, launch_manifest)

        setup_commands = build_setup_commands(
            snapshot,
            environment_file=str(app_env),
            existing_setup=list(self.params.get("setup") or []),
            verify_on_worker=snapshot_cfg.get("verify_on_worker", True),
        )
        executor = submitit.AutoExecutor(
            folder=self.params["submitit_folder"],
            cluster="slurm",
            slurm_max_num_timeout=0,
        )
        inner = getattr(executor, "_executor", executor)
        if hasattr(inner, "python"):
            inner.python = "python"
        if hasattr(inner, "_python"):
            inner._python = "python"

        memory_per_cpu_mb = max(
            1,
            resources.memory_per_worker_gb * 1024 // resources.cpus_per_worker,
        )
        srun_args = [
            "--exclusive",
            f"--ntasks={resources.workers_per_node}",
            f"--cpus-per-task={resources.cpus_per_worker}",
            f"--mem-per-cpu={memory_per_cpu_mb}M",
            "--cpu-bind=cores",
            "--mpi=none",
            "--network=disable_rdzv_get",
            f"--environment={edf}",
        ]
        additional_parameters = dict(
            self.params.get("additional_parameters") or {}
        )
        additional_parameters.setdefault("no-requeue", True)
        update: dict[str, Any] = {
            "timeout_min": int(self.params["timeout_min"]),
            "nodes": 1,
            "tasks_per_node": resources.workers_per_node,
            "cpus_per_task": resources.cpus_per_worker,
            "name": self.params.get("name", "foundry-clariden-pool"),
            "stderr_to_stdout": bool(self.params.get("stderr_to_stdout", True)),
            "slurm_partition": self.params["partition"],
            "slurm_account": self.params["account"],
            "slurm_exclusive": True,
            "slurm_mem_per_cpu": f"{memory_per_cpu_mb}M",
            "slurm_signal_delay_s": int(self.params.get("signal_delay_s", 300)),
            "slurm_srun_args": srun_args,
            "slurm_setup": setup_commands,
            "slurm_additional_parameters": additional_parameters,
        }
        if resources.jobs_per_gpu == 1:
            update["slurm_gpus_per_task"] = 1
            srun_args.append("--gpus-per-task=1")
        executor.update_parameters(**update)

        job = executor.submit(
            run_clariden_pool, snapshot.to_json(), str(launch_path)
        )
        launch_entry = {
            "slurm_job_id": str(job.job_id),
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "resources": resources._asdict(),
            "partition": self.params["partition"],
        }
        launch_manifest.setdefault("launches", []).append(launch_entry)
        _atomic_json_write(launch_path, launch_manifest)
        submission_path = (
            Path(snapshot.bundle_dir) / "manifests" / "submission.json"
        )
        _atomic_json_write(
            submission_path,
            {
                "slurm_job_ids": [
                    item["slurm_job_id"] for item in launch_manifest["launches"]
                ],
                "snapshot_bundle": snapshot.bundle_dir,
                "launches": launch_manifest["launches"],
            },
        )
        log.info("Submitted Clariden node pool: %s", job.job_id)
        log.info("Snapshot bundle: %s", snapshot.bundle_dir)
        return []

"""Local multi-GPU launcher for Hydra multirun.

Spawns each sweep job as a separate subprocess pinned to a distinct GPU via
``CUDA_VISIBLE_DEVICES``.  GPUs are managed as a pool: when more jobs than
GPUs are requested the excess jobs queue until a GPU becomes free.

Usage (from an experiment YAML)::

    defaults:
      - override /hydra/launcher: local_gpu

    hydra:
      sweeper:
        params:
          model/tokenizer: a,b,c
          hyperparameters.fold_number: "0,1"

Then launch with ``-m``::

    uv run python main.py experiment=my_experiment -m
"""

import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

log = logging.getLogger(__name__)

FREE_MEM_THRESHOLD_MIB = 1024


class LocalGpuLauncher(Launcher):
    """Hydra launcher that runs multirun jobs in parallel across local GPUs."""

    def __init__(
        self,
        gpus: Optional[List[int]] = None,
        stop_on_failure: bool = False,
        only_free_gpus: bool = True,
    ) -> None:
        self.gpus = gpus
        self.stop_on_failure = stop_on_failure
        self.only_free_gpus = only_free_gpus
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    # ------------------------------------------------------------------

    @staticmethod
    def _query_gpu_status() -> List[tuple[int, int]]:
        """Return ``[(gpu_index, memory_used_mib), ...]`` for all GPUs."""
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            results = []
            for line in out.strip().splitlines():
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                results.append((int(parts[0].strip()), int(parts[1].strip())))
            return results
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return [(0, 0)]

    @classmethod
    def _detect_free_gpus(cls) -> List[int]:
        """Return indices of GPUs with memory usage below threshold."""
        statuses = cls._query_gpu_status()
        free = [idx for idx, mem in statuses if mem < FREE_MEM_THRESHOLD_MIB]
        busy = [
            (idx, mem) for idx, mem in statuses if mem >= FREE_MEM_THRESHOLD_MIB
        ]
        if busy:
            log.info(
                "Skipping busy GPUs: %s",
                ", ".join(f"{idx} ({mem} MiB)" for idx, mem in busy),
            )
        return free

    @classmethod
    def _detect_all_gpus(cls) -> List[int]:
        return [idx for idx, _ in cls._query_gpu_status()]

    # ------------------------------------------------------------------

    def _resolve_gpus(self) -> List[int]:
        if self.gpus:
            candidates = list(self.gpus)
        elif self.only_free_gpus:
            candidates = self._detect_free_gpus()
        else:
            candidates = self._detect_all_gpus()

        if not candidates:
            raise RuntimeError(
                "LocalGpuLauncher: no GPUs available. All GPUs are in use. "
                "Either wait for running jobs to finish or explicitly set "
                "hydra.launcher.gpus=[...] to target specific GPUs."
            )
        return candidates

    def _snapshot_launch_context(
        self, sweep_dir: Path
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Snapshot configs and source code into the sweep directory so that
        every subprocess -- even ones queued for later -- uses the code and
        config as they existed at launch time.

        Returns ``(config_snapshot, code_snapshot)`` absolute paths, either
        of which may be ``None`` if the corresponding source wasn't found.
        """
        assert self.config is not None
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

        config_snapshot: Optional[Path] = None
        for src in self.config.hydra.runtime.config_sources:
            if src.schema != "file" or not src.path:
                continue
            src_path = Path(src.path)
            if not src_path.is_dir():
                continue
            config_snapshot = sweep_dir / ".config_snapshot"
            if config_snapshot.exists():
                shutil.rmtree(config_snapshot)
            shutil.copytree(src_path, config_snapshot, ignore=ignore)
            config_snapshot = config_snapshot.resolve()
            log.info("Config snapshot: %s", config_snapshot)
            break

        project_root = Path(sys.argv[0]).resolve().parent
        code_snapshot = sweep_dir / ".code_snapshot"
        if code_snapshot.exists():
            shutil.rmtree(code_snapshot)
        code_snapshot.mkdir()

        for pkg in ("foundry", "hydra_plugins"):
            pkg_dir = project_root / pkg
            if pkg_dir.is_dir():
                shutil.copytree(pkg_dir, code_snapshot / pkg, ignore=ignore)

        for py_file in project_root.glob("*.py"):
            shutil.copy2(py_file, code_snapshot / py_file.name)

        code_snapshot = code_snapshot.resolve()
        log.info("Code snapshot: %s", code_snapshot)

        return config_snapshot, code_snapshot

    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[JobReturn]:
        setup_globals()
        assert self.config is not None

        configure_log(
            self.config.hydra.hydra_logging, self.config.hydra.verbose
        )

        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot, code_snapshot = self._snapshot_launch_context(
            sweep_dir
        )

        gpus = self._resolve_gpus()
        num_jobs = len(job_overrides)

        log.info(
            "LocalGpuLauncher: dispatching %d job(s) across GPUs %s",
            num_jobs,
            gpus,
        )

        gpu_pool: queue.Queue[int] = queue.Queue()
        for gpu_id in gpus:
            gpu_pool.put(gpu_id)

        active_procs: list[subprocess.Popen] = []
        proc_lock = threading.Lock()
        failure_event = threading.Event()

        script = sys.argv[0]
        if code_snapshot is not None:
            script = str(code_snapshot / Path(sys.argv[0]).name)

        def _run_one(idx: int, overrides: list[str]) -> int:
            gpu_id = gpu_pool.get()
            try:
                if self.stop_on_failure and failure_event.is_set():
                    return -1

                cmd = [sys.executable, script]
                if config_snapshot is not None:
                    cmd += ["--config-dir", str(config_snapshot)]
                cmd += overrides
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                if code_snapshot is not None:
                    env["PYTHONPATH"] = (
                        str(code_snapshot)
                        + os.pathsep
                        + env.get("PYTHONPATH", "")
                    )
                desc = " ".join(filter_overrides(overrides))
                log.info("  #%d [GPU %d]: %s", idx, gpu_id, desc)

                proc = subprocess.Popen(cmd, env=env)
                with proc_lock:
                    active_procs.append(proc)

                rc = proc.wait()
                if rc != 0:
                    failure_event.set()
                return rc
            finally:
                gpu_pool.put(gpu_id)

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _terminate_all(signum: int, _frame) -> None:  # type: ignore[override]
            with proc_lock:
                for p in active_procs:
                    try:
                        p.terminate()
                    except OSError:
                        pass
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, _terminate_all)
        signal.signal(signal.SIGTERM, _terminate_all)

        try:
            with ThreadPoolExecutor(max_workers=max(num_jobs, 1)) as pool:
                futures = {
                    pool.submit(_run_one, initial_job_idx + i, list(ov)): i
                    for i, ov in enumerate(job_overrides)
                }

                failed: list[int] = []
                for fut in as_completed(futures):
                    i = futures[fut]
                    job_id = initial_job_idx + i
                    try:
                        rc = fut.result()
                    except Exception as exc:
                        log.error("Job #%d raised: %s", job_id, exc)
                        failed.append(job_id)
                        continue

                    if rc == 0:
                        log.info("Job #%d finished successfully.", job_id)
                    else:
                        log.error("Job #%d exited with code %d.", job_id, rc)
                        failed.append(job_id)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        if failed:
            raise RuntimeError(
                f"LocalGpuLauncher: job(s) {sorted(failed)} failed"
            )

        return []

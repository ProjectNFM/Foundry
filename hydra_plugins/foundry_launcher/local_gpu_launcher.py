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


class LocalGpuLauncher(Launcher):
    """Hydra launcher that runs multirun jobs in parallel across local GPUs."""

    def __init__(
        self,
        gpus: Optional[List[int]] = None,
        stop_on_failure: bool = False,
    ) -> None:
        self.gpus = gpus
        self.stop_on_failure = stop_on_failure
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
    def _detect_gpus() -> List[int]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            return [
                int(tok)
                for tok in out.strip().splitlines()
                if tok.strip().isdigit()
            ]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return [0]

    # ------------------------------------------------------------------

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

        gpus = list(self.gpus) if self.gpus else self._detect_gpus()
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

        def _run_one(idx: int, overrides: list[str]) -> int:
            gpu_id = gpu_pool.get()
            try:
                if self.stop_on_failure and failure_event.is_set():
                    return -1

                cmd = [sys.executable, sys.argv[0]] + overrides
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
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

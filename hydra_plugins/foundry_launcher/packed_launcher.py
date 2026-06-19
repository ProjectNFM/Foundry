"""Submitit launcher with support for packing multiple tasks per SLURM node.

Based on https://gist.github.com/dapatil211/5ac70004610c8a3c8412d86bd2bfbcdf

Placed under ``hydra_plugins/`` so Hydra discovers it automatically -- no
monkey-patching of the plugin system is needed.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    BaseSubmititLauncher,
)
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _batch(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


class PackedSubmititLauncher(BaseSubmititLauncher):
    """Extends the stock submitit launcher to pack *tasks_per_node* sweep jobs
    onto a single SLURM allocation, saving cluster quota when individual jobs
    are small enough to share a node.
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    def launch_batch(
        self,
        sweep_overrides: List[List[str]],
        job_dir_key: List[str],
        job_num: List[int],
        job_id: List[str],
        singleton_state: List[Dict[type, Singleton]],
    ) -> JobReturn:
        import submitit

        task_id = submitit.JobEnvironment().global_rank
        return self(
            sweep_overrides[task_id],
            job_dir_key[task_id],
            job_num[task_id],
            job_id[task_id],
            singleton_state[task_id],
        )

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params

        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}
        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        # When the Slurm job runs inside a CSCS Container Engine image
        # (i.e. ``--environment=...`` is on either the ``#SBATCH`` directives
        # or the per-step ``srun`` arguments), submitit's default Python path
        # is wrong: it bakes ``sys.executable`` from the *submission* shell
        # (typically the host ``.venv/bin/python``) into the sbatch script,
        # but inside the container that interpreter either doesn't exist or
        # doesn't see the image's site-packages. Override ``_python`` so the
        # generated ``srun ... python -u -m submitit.core._submit ...`` line
        # resolves ``python`` from the container's ``PATH`` instead.
        if self._EXECUTOR == "slurm":
            additional = self.params.get("additional_parameters") or {}
            try:
                additional = dict(additional)
            except (TypeError, ValueError):
                additional = (
                    OmegaConf.to_container(additional, resolve=True) or {}
                )

            srun_args_raw = self.params.get("srun_args") or []
            try:
                srun_args_list = list(srun_args_raw)
            except TypeError:
                srun_args_list = (
                    OmegaConf.to_container(srun_args_raw, resolve=True) or []
                )
            srun_args_list = [str(a) for a in srun_args_list]

            container_in_additional = (
                "environment" in additional or "container-image" in additional
            )
            container_in_srun = any(
                a.startswith("--environment=")
                or a.startswith("--container-image=")
                for a in srun_args_list
            )
            uses_container = container_in_additional or container_in_srun

            if uses_container:
                inner = getattr(executor, "_executor", executor)
                # submitit >=1.4 uses ``SlurmExecutor.python`` for the ``srun ...`` line
                # (see ``_submitit_command_str``). Older builds used ``_python``; set both.
                submitit_python = "python"
                if hasattr(inner, "python"):
                    inner.python = submitit_python
                if hasattr(inner, "_python"):
                    inner._python = submitit_python
                env_value = additional.get(
                    "environment", additional.get("container-image")
                )
                if env_value is None:
                    for a in srun_args_list:
                        if a.startswith("--environment=") or a.startswith(
                            "--container-image="
                        ):
                            env_value = a.split("=", 1)[1]
                            break
                log.info(
                    "Detected container env (--environment=%s); overriding "
                    "submitit Slurm python to %r so the container interpreter is used "
                    "(not the login-node venv path).",
                    env_value,
                    submitit_python,
                )

        log.info(
            "Submitit '%s' sweep output dir: %s",
            self._EXECUTOR,
            self.config.hydra.sweep.dir,
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info("\t#%d : %s", idx, lst)
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        tasks_per_node = self.params.get("tasks_per_node", 1)
        jobs = executor.map_array(
            self.launch_batch,
            *list(_batch(jps, tasks_per_node) for jps in zip(*job_params)),
        )

        results = []
        for j in jobs:
            if j.state != "DONE":
                try:
                    j.wait()
                except Exception as e:
                    log.error("Job %s failed: %s", j.job_id, e)
                    raise
            results.extend(j.results())
        return results


class SlurmLauncher(PackedSubmititLauncher):
    _EXECUTOR = "slurm"


class LocalLauncher(PackedSubmititLauncher):
    _EXECUTOR = "local"
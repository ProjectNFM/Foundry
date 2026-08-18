"""Submitit launcher with support for packing multiple tasks per SLURM node.

Based on https://gist.github.com/dapatil211/5ac70004610c8a3c8412d86bd2bfbcdf

Placed under ``hydra_plugins/`` so Hydra discovers it automatically -- no
monkey-patching of the plugin system is needed.
"""

import logging
import os
import sys
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
        snapshot_json: List[str | None] | str | None = None,
    ) -> JobReturn:
        import submitit

        # When tasks_per_node > 1, map_array batches all params into lists.
        # All entries are identical — extract the first one.
        if isinstance(snapshot_json, (list, tuple)):
            snapshot_json = snapshot_json[0] if snapshot_json else None

        if snapshot_json:
            from hydra_plugins.foundry_launcher.launch_snapshot import (
                LaunchSnapshot,
                verify_import_paths,
                verify_snapshot,
            )

            snapshot = LaunchSnapshot.from_json(snapshot_json)
            os.chdir(snapshot.source_dir)
            sys.path.insert(0, snapshot.source_dir)
            os.environ["PYTHONPATH"] = (
                snapshot.source_dir
                + os.pathsep
                + os.environ.get("PYTHONPATH", "")
            )

            snapshot_env = {
                "FOUNDRY_SNAPSHOT_BUNDLE_DIR": snapshot.bundle_dir,
                "FOUNDRY_SNAPSHOT_SOURCE_DIR": snapshot.source_dir,
                "FOUNDRY_SNAPSHOT_MANIFEST": snapshot.manifest_path,
                "FOUNDRY_SNAPSHOT_GIT_SHA": snapshot.git_sha,
                "FOUNDRY_SNAPSHOT_SOURCE_DIGEST": snapshot.source_digest,
                "FOUNDRY_SNAPSHOT_BUNDLE_ID": snapshot.bundle_id,
            }
            os.environ.update(snapshot_env)

            if self._get_snapshot_config().get("verify_on_worker", True):
                verify_snapshot(snapshot)
                import foundry  # noqa: F401

                verify_import_paths(snapshot.source_dir)

        task_id = submitit.JobEnvironment().global_rank

        return self(
            sweep_overrides[task_id],
            job_dir_key[task_id],
            job_num[task_id],
            job_id[task_id],
            singleton_state[task_id],
        )

    def _get_snapshot_config(self) -> dict[str, Any]:
        """Extract snapshot settings from launcher params."""
        raw = self.params.get("snapshot", None)
        if raw is None:
            return {"enabled": False}
        try:
            return dict(OmegaConf.to_container(raw, resolve=True))
        except Exception:
            return dict(raw) if raw else {"enabled": False}

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params

        snap_cfg = self._get_snapshot_config()
        snapshot_enabled = snap_cfg.get("enabled", False)
        snapshot = None
        snapshot_json: str | None = None

        if snapshot_enabled:
            from hydra_plugins.foundry_launcher.launch_snapshot import (
                build_setup_commands,
                prepare_snapshot,
            )

            project_root = Path(sys.argv[0]).resolve().parent
            raw_root = snap_cfg.get("root")
            if raw_root and str(raw_root) != "null":
                snapshot_root = Path(str(raw_root))
            else:
                snapshot_root = project_root / ".snapshots"

            sweep_name = str(
                OmegaConf.select(
                    self.config,
                    "run.group",
                    default=OmegaConf.select(
                        self.config, "run.name", default="sweep"
                    ),
                )
            )

            snapshot = prepare_snapshot(
                project_root=project_root,
                snapshot_root=snapshot_root,
                sweep_name=sweep_name,
                job_overrides=job_overrides,
                hydra_cfg=self.config,
                require_clean_git=snap_cfg.get("require_clean_git", True),
            )
            snapshot_json = snapshot.to_json()

            env_file = snap_cfg.get("environment_file", None)
            if env_file and env_file != "null":
                resolved_env = str(
                    Path(env_file).resolve()
                    if not os.path.isabs(env_file)
                    else Path(env_file)
                )
            else:
                resolved_env = None

            existing_setup = list(params.get("setup", []) or [])
            try:
                existing_setup = [str(s) for s in existing_setup]
            except Exception:
                existing_setup = list(
                    OmegaConf.to_container(
                        params.get("setup", []), resolve=True
                    )
                    or []
                )

            new_setup = build_setup_commands(
                snapshot,
                environment_file=resolved_env,
                existing_setup=existing_setup,
                verify_on_worker=snap_cfg.get("verify_on_worker", True),
            )
            params = dict(params)
            params["setup"] = new_setup

        init_params = {"folder": params["submitit_folder"]}
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
        excluded_keys = init_keys | {"snapshot"}
        filtered_params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in excluded_keys
        }
        executor.update_parameters(**filtered_params)

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
            additional = params.get("additional_parameters") or {}
            try:
                additional = dict(additional)
            except (TypeError, ValueError):
                additional = (
                    OmegaConf.to_container(additional, resolve=True) or {}
                )

            srun_args_raw = params.get("srun_args") or []
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
                    snapshot_json,
                )
            )

        tasks_per_node = params.get("tasks_per_node", 1)
        jobs = executor.map_array(
            self.launch_batch,
            *list(_batch(jps, tasks_per_node) for jps in zip(*job_params)),
        )

        job_ids = [j.job_id for j in jobs]
        log.info(
            "Submitted %d Slurm job(s): %s",
            len(job_ids),
            ", ".join(str(jid) for jid in job_ids),
        )
        log.info(
            "Logs directory: %s",
            params["submitit_folder"],
        )

        if snapshot:
            import json

            submission_path = (
                Path(snapshot.bundle_dir) / "manifests" / "submission.json"
            )
            submission_path.write_text(
                json.dumps(
                    {
                        "slurm_job_ids": [str(jid) for jid in job_ids],
                        "num_tasks": num_jobs,
                        "tasks_per_node": tasks_per_node,
                    },
                    indent=2,
                )
            )
            log.info("Snapshot bundle: %s", snapshot.bundle_dir)

        return []


class SlurmLauncher(PackedSubmititLauncher):
    _EXECUTOR = "slurm"


class LocalLauncher(PackedSubmititLauncher):
    _EXECUTOR = "local"

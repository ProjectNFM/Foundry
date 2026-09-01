"""Hydra configuration schemas for Foundry's custom launchers."""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf


@dataclass
class SnapshotConf:
    """Configuration for immutable source snapshots."""

    enabled: bool = True
    root: Optional[str] = None
    require_clean_git: bool = True
    verify_on_worker: bool = True
    environment_file: Optional[str] = None


@dataclass
class FoundrySlurmQueueConf(SlurmQueueConf):
    """Submitit Slurm schema extended with Foundry snapshot settings."""

    snapshot: SnapshotConf = field(default_factory=SnapshotConf)


@dataclass
class ClaridenSlurmQueueConf(FoundrySlurmQueueConf):
    """Configuration contract for one-node Clariden worker pools."""

    _target_: str = (
        "hydra_plugins.foundry_launcher.clariden_launcher."
        "ClaridenNodePoolLauncher"
    )
    exclusive: bool = True
    jobs_per_gpu: int = 1
    workers_per_node: Optional[int] = None
    cpus_per_worker: Optional[int] = None
    memory_per_worker_gb: Optional[int] = None
    drain_guard_min: int = 10
    minimum_start_budget_min: int = 0
    container_environment: Optional[str] = None
    application_environment_file: Optional[str] = None
    data_root: Optional[str] = None
    resume_snapshot: Optional[str] = None
    retry_failed: bool = True


ConfigStore.instance().store(
    group="hydra/launcher",
    name="foundry_submitit_slurm",
    node=FoundrySlurmQueueConf(),
    provider="foundry_launcher",
)

ConfigStore.instance().store(
    group="hydra/launcher",
    name="foundry_clariden_slurm",
    node=ClaridenSlurmQueueConf(),
    provider="foundry_launcher",
)

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


ConfigStore.instance().store(
    group="hydra/launcher",
    name="foundry_submitit_slurm",
    node=FoundrySlurmQueueConf(),
    provider="foundry_launcher",
)

"""Tests for immutable launcher source snapshots."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hydra_plugins.foundry_launcher.launch_snapshot import (
    COMPLETION_MARKER,
    _validate_clean_repo,
    build_setup_commands,
    prepare_snapshot,
    verify_snapshot,
)
from hydra_plugins.foundry_launcher.packed_launcher import (
    PackedSubmititLauncher,
)


def test_hydra_plugin_discovery_imports_snapshot_module() -> None:
    """Hydra executes plugin modules without always inserting sys.modules."""
    cmd = [
        sys.executable,
        "main.py",
        "--cfg",
        "job",
        "experiment=pretraining/poyo_data_scaling_base",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MNE_DONTWRITE_HOME": "true"},
    )

    assert result.returncode == 0, result.stderr


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def project_repo(tmp_path: Path) -> Path:
    """Create a minimal committed repository that can be Git-archived."""
    repo = tmp_path / "project"
    repo.mkdir()
    for relative, contents in {
        "main.py": "print('snapshot')\n",
        "foundry/__init__.py": "",
        "hydra_plugins/__init__.py": "",
        "hydra_plugins/foundry_launcher/__init__.py": "",
        "configs/config.yaml": "value: 1\n",
        "pyproject.toml": "[project]\nname = 'snapshot-test'\n",
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "tests@example.invalid")
    _run_git(repo, "config", "user.name", "Snapshot Tests")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-qm", "initial snapshot source")
    return repo


def test_clean_validation_rejects_untracked_files(project_repo: Path) -> None:
    (project_repo / "new_experiment.py").write_text("# not committed\n")

    with pytest.raises(RuntimeError, match="untracked files"):
        _validate_clean_repo(project_repo)


def test_prepare_snapshot_seals_complete_git_archive(
    project_repo: Path, tmp_path: Path
) -> None:
    snapshot = prepare_snapshot(
        project_root=project_repo,
        snapshot_root=tmp_path / "bundles",
        sweep_name="test sweep",
        job_overrides=[["fold=0"], ["fold=1"]],
        hydra_cfg=OmegaConf.create({"run": {"name": "test"}}),
    )

    bundle = Path(snapshot.bundle_dir)
    source = Path(snapshot.source_dir)
    assert (bundle / COMPLETION_MARKER).is_file()
    assert (source / "main.py").read_text() == "print('snapshot')\n"
    assert (source / ".git").exists() is False
    assert (bundle / "manifests" / "snapshot-descriptor.json").is_file()
    assert (bundle / "task-configs" / "task_0001.json").is_file()
    assert not (source / "main.py").stat().st_mode & 0o222
    verify_snapshot(snapshot)

    descriptor = json.loads(
        (bundle / "manifests" / "snapshot-descriptor.json").read_text()
    )
    assert descriptor["bundle_id"] == snapshot.bundle_id
    assert descriptor["git_sha"] == snapshot.git_sha


def test_prepare_snapshot_preserves_deferred_hydra_values(
    project_repo: Path, tmp_path: Path
) -> None:
    snapshot = prepare_snapshot(
        project_root=project_repo,
        snapshot_root=tmp_path / "bundles",
        sweep_name="deferred-values",
        job_overrides=[["fold=0"]],
        hydra_cfg=OmegaConf.create({"hydra": {"job": {"num": "???"}}}),
    )

    base_config = Path(snapshot.base_config_path).read_text()
    assert "num: ???" in base_config


def test_worker_setup_uses_snapshot_and_explicit_environment_file(
    project_repo: Path, tmp_path: Path
) -> None:
    snapshot = prepare_snapshot(
        project_root=project_repo,
        snapshot_root=tmp_path / "bundles",
        sweep_name="test",
        job_overrides=[["fold=0"]],
        hydra_cfg=OmegaConf.create({"run": {"name": "test"}}),
    )
    env_file = tmp_path / "credentials.env"
    commands = build_setup_commands(
        snapshot,
        environment_file=str(env_file),
        existing_setup=[
            "source .env || true",
            "cd /tmp/Foundry",
            "export RANK=0",
        ],
        verify_on_worker=False,
    )

    assert commands[0] == (
        f'if [ -f "{env_file}" ]; then set -a; source "{env_file}"; set +a; fi'
    )
    assert "source .env || true" not in commands
    assert "cd /tmp/Foundry" not in commands
    assert f'cd "{snapshot.source_dir}"' in commands
    assert 'export FOUNDRY_SNAPSHOT_VERIFY_ON_WORKER="0"' in commands


def test_packed_launcher_accepts_submitit_tuple_snapshot_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitit map-array batches each argument as a tuple for one task."""
    captured = {}

    def fake_call(self, *args):
        captured["args"] = args
        return "called"

    monkeypatch.setattr(PackedSubmititLauncher, "__call__", fake_call)
    monkeypatch.setattr(
        "submitit.JobEnvironment",
        lambda: type("Job", (), {"global_rank": 0})(),
    )

    launcher = object.__new__(PackedSubmititLauncher)
    result = launcher.launch_batch(
        [["fold=0"]],
        ["hydra.sweep.dir"],
        [0],
        ["job_id_for_0"],
        [{}],
        (None,),
    )

    assert result == "called"
    assert captured["args"][0] == ["fold=0"]

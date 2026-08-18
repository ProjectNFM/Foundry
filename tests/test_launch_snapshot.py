"""Tests for immutable launcher source snapshots."""

import json
import subprocess
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

    assert f'source "{env_file}" || true' in commands
    assert "source .env || true" not in commands
    assert "cd /tmp/Foundry" not in commands
    assert f'cd "{snapshot.source_dir}"' in commands
    assert 'export FOUNDRY_SNAPSHOT_VERIFY_ON_WORKER="0"' in commands

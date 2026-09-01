"""Tests for the Clariden dynamic node-pool launcher."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import hydra_plugins.foundry_launcher.clariden_launcher as clariden_launcher
from hydra_plugins.foundry_launcher.clariden_launcher import (
    MAX_MPS_CLIENTS_PER_GPU,
    _run_claimed_cell,
    _worker_identity,
    validate_clariden_config,
)
from hydra_plugins.foundry_launcher.clariden_queue import (
    ClaridenFileQueue,
    canonical_cell_id,
)
from hydra_plugins.foundry_launcher.launch_snapshot import LaunchSnapshot


def _consume_queue(manifests_dir: str) -> None:
    queue = ClaridenFileQueue(manifests_dir)
    while cell := queue.claim(
        {"worker_rank": str(multiprocessing.current_process().pid)}
    ):
        queue.finish(cell["cell_id"], succeeded=True, exit_status=0)


def _launcher_params(tmp_path: Path, *, mps: bool = False) -> dict:
    edf = tmp_path / "foundry.toml"
    annotation = (
        '\n[annotations]\ncom.hooks.nvidia_cuda_mps.enabled = "true"\n'
        if mps
        else ""
    )
    edf.write_text('image = "example.invalid/pinned:image"\n' + annotation)
    env_file = tmp_path / "application.env"
    env_file.write_text("WANDB_API_KEY=test-only\n")
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    return {
        "account": "a-test",
        "partition": "normal",
        "nodes": 1,
        "timeout_min": 240,
        "exclusive": True,
        "mem_gb": 450,
        "gpus_per_node": 4,
        "jobs_per_gpu": 1,
        "workers_per_node": None,
        "cpus_per_worker": None,
        "memory_per_worker_gb": None,
        "container_environment": str(edf),
        "application_environment_file": str(env_file),
        "snapshot": {
            "enabled": True,
            "root": str(snapshot_root),
            "require_clean_git": True,
            "verify_on_worker": True,
            "environment_file": str(env_file),
        },
    }


def test_clariden_resources_are_derived_in_python(tmp_path: Path) -> None:
    params = _launcher_params(tmp_path, mps=True)
    params["jobs_per_gpu"] = 8

    resources = validate_clariden_config(params)

    assert resources.workers_per_node == 32
    assert resources.cpus_per_worker == 9
    assert resources.memory_per_worker_gb == 14


def test_clariden_accepts_conservative_mps_client_ceiling(
    tmp_path: Path,
) -> None:
    params = _launcher_params(tmp_path, mps=True)
    params["jobs_per_gpu"] = MAX_MPS_CLIENTS_PER_GPU

    resources = validate_clariden_config(params)

    assert resources.workers_per_node == 4 * MAX_MPS_CLIENTS_PER_GPU


@pytest.mark.parametrize(
    "jobs_per_gpu", [0, MAX_MPS_CLIENTS_PER_GPU + 1, 1.5, "two"]
)
def test_clariden_rejects_invalid_jobs_per_gpu(
    tmp_path: Path, jobs_per_gpu: object
) -> None:
    params = _launcher_params(tmp_path, mps=True)
    params["jobs_per_gpu"] = jobs_per_gpu

    with pytest.raises(ValueError, match="jobs_per_gpu"):
        validate_clariden_config(params)


def test_clariden_rejects_oversubscription_without_mps(tmp_path: Path) -> None:
    params = _launcher_params(tmp_path)
    params["jobs_per_gpu"] = 2

    with pytest.raises(ValueError, match="nvidia_cuda_mps"):
        validate_clariden_config(params)


def test_mps_worker_uses_actual_numa_domain_for_gpu_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_PROCID", "17")
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "GPU-zero,GPU-one,GPU-two,GPU-three"
    )
    monkeypatch.setattr(
        clariden_launcher, "_numa_node_for_current_affinity", lambda: 2
    )

    identity, assigned = _worker_identity(jobs_per_gpu=8)

    assert assigned == "GPU-two"
    assert identity["numa_node"] == 2
    assert identity["gpu_slot"] == 2


def test_mps_worker_rejects_partial_gpu_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-zero")
    monkeypatch.setattr(
        clariden_launcher, "_numa_node_for_current_affinity", lambda: 0
    )

    with pytest.raises(RuntimeError, match="all four GPUs"):
        _worker_identity(jobs_per_gpu=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"account": ""}, "CSCS_ACCOUNT"),
        ({"nodes": 2}, "nodes == 1"),
        ({"partition": "long"}, "partition"),
        ({"workers_per_node": 5}, "workers_per_node"),
    ],
)
def test_clariden_rejects_invalid_submission_config(
    tmp_path: Path, mutation: dict, message: str
) -> None:
    params = _launcher_params(tmp_path)
    params.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate_clariden_config(params)


@pytest.mark.parametrize(
    "mutation",
    [
        {"drain_guard_min": -1},
        {"minimum_start_budget_min": -1},
        {"timeout_min": 10, "drain_guard_min": 10},
        {
            "timeout_min": 10,
            "drain_guard_min": 5,
            "minimum_start_budget_min": 5,
        },
    ],
)
def test_clariden_rejects_invalid_drain_budget(
    tmp_path: Path, mutation: dict
) -> None:
    params = _launcher_params(tmp_path)
    params.update(mutation)

    with pytest.raises(ValueError, match="drain|start budget"):
        validate_clariden_config(params)


def test_queue_claim_is_race_safe_across_processes(tmp_path: Path) -> None:
    queue = ClaridenFileQueue(tmp_path)
    records = []
    for index in range(60):
        overrides = [f"cell={index}", "seed=42"]
        records.append(
            {
                "cell_id": canonical_cell_id(overrides),
                "overrides": overrides,
            }
        )
    queue.initialize(records)

    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_consume_queue, args=(str(tmp_path),))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert queue.counts() == {
        "pending": 0,
        "running": 0,
        "succeeded": 60,
        "failed": 0,
        "not_started_due_to_drain": 0,
    }
    assert len({record["cell_id"] for record in queue.records()}) == 60


def test_resume_never_requeues_a_successful_cell(tmp_path: Path) -> None:
    queue = ClaridenFileQueue(tmp_path)
    records = [
        {
            "cell_id": canonical_cell_id([f"cell={i}"]),
            "overrides": [f"cell={i}"],
        }
        for i in range(3)
    ]
    queue.initialize(records)

    succeeded = queue.claim({"worker_rank": "0"})
    assert succeeded is not None
    queue.finish(succeeded["cell_id"], succeeded=True, exit_status=0)
    failed = queue.claim({"worker_rank": "1"})
    assert failed is not None
    queue.finish(
        failed["cell_id"],
        succeeded=False,
        exit_status=137,
        failure_classification="oom",
    )
    queue.drain_pending()

    counts = queue.requeue_for_resume(retry_failed=True)

    assert counts == {
        "not_started_due_to_drain": 1,
        "running": 0,
        "failed": 1,
    }
    assert queue.counts()["succeeded"] == 1
    assert queue.counts()["pending"] == 2
    assert queue.records("succeeded")[0]["cell_id"] == succeeded["cell_id"]


def test_resume_archives_interrupted_attempt_provenance(tmp_path: Path) -> None:
    queue = ClaridenFileQueue(tmp_path)
    overrides = ["cell=0"]
    queue.initialize(
        [{"cell_id": canonical_cell_id(overrides), "overrides": overrides}]
    )
    running = queue.claim({"worker_rank": "0"})
    assert running is not None
    queue.update_running(
        running["cell_id"], {"wandb_run_identity": "interrupted-run-id"}
    )

    queue.requeue_for_resume(retry_failed=True)

    attempt_path = (
        tmp_path
        / "clariden-attempts"
        / f"{running['cell_id']}-attempt-001.json"
    )
    attempt = json.loads(attempt_path.read_text())
    assert (
        attempt["failure_classification"] == "previous_allocation_interrupted"
    )
    assert attempt["wandb_run_identity"] == "interrupted-run-id"


def test_queue_records_required_provenance(tmp_path: Path) -> None:
    queue = ClaridenFileQueue(tmp_path)
    overrides = ["experiment=example", "seed=42"]
    queue.initialize(
        [
            {
                "cell_id": canonical_cell_id(overrides),
                "overrides": overrides,
                "snapshot_bundle_id": "bundle",
                "git_sha": "abc",
                "source_digest": "digest",
                "environment_fingerprint": "environment",
                "container_environment_sha256": "edf",
                "application_environment_file_sha256": "env",
                "resources": {"jobs_per_gpu": 1},
            }
        ]
    )
    cell = queue.claim(
        {
            "allocation_id": "123",
            "node_hostname": "node1",
            "worker_rank": "0",
            "gpu_identifier": "GPU-0",
            "jobs_per_gpu": 1,
        }
    )
    assert cell is not None

    for key in (
        "snapshot_bundle_id",
        "git_sha",
        "source_digest",
        "environment_fingerprint",
        "allocation_id",
        "node_hostname",
        "worker_rank",
        "gpu_identifier",
        "jobs_per_gpu",
    ):
        assert key in cell
    assert "WANDB_API_KEY" not in str(cell)


def _snapshot(tmp_path: Path) -> LaunchSnapshot:
    source = tmp_path / "source"
    source.mkdir()
    return LaunchSnapshot(
        bundle_dir=str(tmp_path),
        source_dir=str(source),
        manifest_path=str(tmp_path / "manifest.json"),
        git_sha="abc",
        git_branch="test",
        source_digest="digest",
        base_config_path=str(tmp_path / "config.yaml"),
        environment_fingerprint="environment",
        bundle_id="bundle-id",
    )


def _claimed_cell(tmp_path: Path) -> tuple[ClaridenFileQueue, dict]:
    queue = ClaridenFileQueue(tmp_path / "manifests")
    overrides = ["experiment=example", "run.seed=42"]
    queue.initialize(
        [
            {
                "cell_id": canonical_cell_id(overrides),
                "overrides": overrides,
                "output_root": str(tmp_path / "outputs"),
                "wandb_enabled": True,
            }
        ]
    )
    cell = queue.claim({"worker_rank": "0"})
    assert cell is not None
    return queue, cell


def test_claimed_cell_records_worker_exception_instead_of_staying_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    queue, cell = _claimed_cell(tmp_path)

    def fail_to_start(*args, **kwargs):
        raise OSError("test subprocess failure")

    monkeypatch.setattr(subprocess, "run", fail_to_start)

    assert not _run_claimed_cell(snapshot, queue, cell)
    assert queue.counts()["running"] == 0
    failed = queue.records("failed")[0]
    assert failed["failure_classification"] == "worker_internal_error"
    assert failed["worker_error"] == "OSError: test subprocess failure"
    assert len(failed["wandb_run_identity"]) == 32


def test_claimed_cell_passes_and_records_deterministic_wandb_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    queue, cell = _claimed_cell(tmp_path)
    child_environments: list[dict[str, str]] = []

    def succeed(*args, **kwargs):
        child_environments.append(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(subprocess, "run", succeed)

    assert _run_claimed_cell(snapshot, queue, cell)
    succeeded = queue.records("succeeded")[0]
    wandb_id = succeeded["wandb_run_identity"]
    assert len(wandb_id) == 32
    assert child_environments[0]["FOUNDRY_WANDB_RUN_ID"] == wandb_id
    assert queue.records("succeeded")[0]["wandb_run_identity"] == wandb_id


def test_main_uses_clariden_wandb_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from main import _configure_wandb

    cfg = OmegaConf.create(
        {
            "logger": {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "id": "config-id-that-must-not-be-reused-by-the-pool",
                "save_dir": None,
            },
            "run": {"name": "test", "resume_wandb_if_name_matches": False},
        }
    )
    monkeypatch.setenv("FOUNDRY_WANDB_RUN_ID", "deterministic-id")

    _configure_wandb(cfg, str(tmp_path))

    assert cfg.logger.id == "deterministic-id"

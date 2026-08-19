import multiprocessing
import queue
from pathlib import Path

import fcntl
import pytest
from omegaconf import OmegaConf

import main as training_main
from foundry.tools.stage_data import destination_lock


def _hold_destination_lock(path: str, acquired, release) -> None:
    with destination_lock(path):
        acquired.put(True)
        release.wait(timeout=5)


def test_node_local_staging_uses_task_config_and_rebases_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"root": "./data/processed", "task_marker": "exact-task"},
            "stage": {
                "mode": "node_local",
                "source_root": "/shared/processed",
                "compressed_root": "/shared/compressed",
                "destination_root": str(tmp_path),
                "compress": True,
            },
        }
    )
    calls = []

    def fake_stage_data(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "brainsets" / "processed")

    monkeypatch.setattr(training_main, "stage_data", fake_stage_data)

    training_main._stage_data_if_needed(cfg)

    assert len(calls) == 1
    assert calls[0]["data_cfg"] is cfg.data
    assert calls[0]["data_cfg"].task_marker == "exact-task"
    assert calls[0]["source_root"] == "/shared/processed"
    assert calls[0]["compressed_root"] == "/shared/compressed"
    assert calls[0]["dest_root"] == str(tmp_path)
    assert calls[0]["compress"] is True
    assert cfg.data.root == str(tmp_path / "brainsets" / "processed")


def test_direct_mode_never_stages_or_changes_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"root": "/shared/processed"},
            "stage": {"mode": "direct"},
        }
    )
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        training_main,
        "stage_data",
        lambda **kwargs: pytest.fail("direct mode must not stage data"),
    )

    training_main._stage_data_if_needed(cfg)

    assert cfg.data.root == "/shared/processed"


def test_node_local_mode_without_destination_uses_configured_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"root": "./data/processed"},
            "stage": {"mode": "node_local", "destination_root": None},
        }
    )
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setattr(
        training_main,
        "stage_data",
        lambda **kwargs: pytest.fail(
            "staging requires an explicit destination"
        ),
    )

    training_main._stage_data_if_needed(cfg)

    assert cfg.data.root == "./data/processed"


def test_destination_lock_uses_exclusive_flock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        fcntl, "flock", lambda fd, operation: calls.append(operation)
    )

    with destination_lock(tmp_path):
        assert (tmp_path / ".foundry-stage.lock").is_file()

    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_destination_lock_serializes_packed_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    first_acquired = context.Queue()
    second_acquired = context.Queue()
    release_first = context.Event()
    release_second = context.Event()
    first = context.Process(
        target=_hold_destination_lock,
        args=(str(tmp_path), first_acquired, release_first),
    )
    second = context.Process(
        target=_hold_destination_lock,
        args=(str(tmp_path), second_acquired, release_second),
    )

    try:
        first.start()
        assert first_acquired.get(timeout=3) is True
        second.start()
        with pytest.raises(queue.Empty):
            second_acquired.get(timeout=0.2)

        release_first.set()
        assert second_acquired.get(timeout=3) is True
    finally:
        release_first.set()
        release_second.set()
        first.join(timeout=3)
        second.join(timeout=3)
        if first.is_alive():
            first.terminate()
        if second.is_alive():
            second.terminate()

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_experiment_configs_use_unified_staging_contract() -> None:
    config_root = Path(__file__).parents[1] / "configs" / "experiment"
    violations = []

    for path in config_root.rglob("*.yaml"):
        text = path.read_text()
        for forbidden in (
            "foundry.tools.stage_data",
            "stage.skip",
            "skip: true",
            "skip: false",
            "${oc.env:SLURM_TMPDIR}/brainsets/processed",
        ):
            if forbidden in text:
                violations.append(
                    f"{path.relative_to(config_root)}: {forbidden}"
                )

    assert violations == []

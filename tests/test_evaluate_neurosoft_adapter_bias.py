from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import evaluate_neurosoft_adapter_bias as evaluator


class FakeAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "active:a": torch.nn.Linear(1, 64),
                "active:b": torch.nn.Linear(1, 64),
                "active:c": torch.nn.Linear(1, 64),
                "unused": torch.nn.Linear(1, 64),
            }
        )


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.session_adapter = FakeAdapter()


def test_five_derangements_are_deterministic_distinct_and_fixed_point_free() -> (
    None
):
    ids = [f"species:recording-{index}" for index in range(8)]
    first = evaluator.generate_derangements(
        ids, "species=x|excluded=y|seeds=42"
    )
    second = evaluator.generate_derangements(
        ids, "species=x|excluded=y|seeds=42"
    )
    assert first == second
    assert len({evaluator.stable_json_hash(item) for item in first}) == 5
    for mapping in first:
        assert set(mapping) == set(ids)
        assert set(mapping.values()) == set(ids)
        assert all(source != target for source, target in mapping.items())


def test_source_model_seed_parser() -> None:
    assert evaluator._parse_seed_option("42, 44") == [42, 44]
    with pytest.raises(Exception, match="comma-separated integers"):
        evaluator._parse_seed_option("42,nope")


def test_progress_duration_formatting() -> None:
    assert evaluator.format_duration(7.4) == "7s"
    assert evaluator.format_duration(65) == "1m 05s"
    assert evaluator.format_duration(3_661) == "1h 01m 01s"


def test_only_active_biases_enter_mean_and_biases_restore_exactly() -> None:
    model = FakeModel()
    active = ["active:a", "active:b", "active:c"]
    with torch.no_grad():
        for index, recording_id in enumerate(active, start=1):
            model.session_adapter.layers[recording_id].bias.fill_(index)
        model.session_adapter.layers["unused"].bias.fill_(1000)
    learned = evaluator.active_biases(model, active)
    mappings = evaluator.generate_derangements(active, "stable")
    evaluator.apply_bias_condition(model, learned, "mean", mappings)
    for recording_id in active:
        assert torch.equal(
            model.session_adapter.layers[recording_id].bias,
            torch.full((64,), 2.0),
        )
    assert torch.equal(
        model.session_adapter.layers["unused"].bias,
        torch.full((64,), 1000.0),
    )
    evaluator.apply_bias_condition(model, learned, "zero", mappings)
    evaluator.apply_bias_condition(model, learned, "intact", mappings)
    for recording_id in active:
        assert torch.equal(
            model.session_adapter.layers[recording_id].bias,
            learned[recording_id],
        )


def test_cross_entropy_is_target_weighted_not_batch_weighted() -> None:
    accumulator = evaluator.MetricAccumulator(num_classes=2)
    accumulator.update(
        torch.tensor([[4.0, 0.0]]), torch.tensor([0]), [1], ["a"]
    )
    accumulator.update(
        torch.tensor([[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]),
        torch.tensor([0, 0, 0]),
        [3],
        ["b"],
    )
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]),
        torch.tensor([0, 0, 0, 0]),
    ).item()
    assert accumulator.summary()["cross_entropy"] == pytest.approx(expected)


def test_supported_f1_and_recording_mean() -> None:
    unsupported_class_matrix = np.array([[2, 0], [3, 0]])
    # Class 1 has positive support here, so both classes are included.
    assert evaluator.supported_macro_f1(
        unsupported_class_matrix
    ) == pytest.approx(2 / 7)
    only_class_zero = np.array([[2, 0], [0, 0]])
    assert evaluator.supported_macro_f1(only_class_zero) == pytest.approx(1.0)

    accumulator = evaluator.MetricAccumulator(num_classes=2)
    accumulator.update(
        torch.tensor([[5.0, 0.0]]), torch.tensor([0]), [1], ["short"]
    )
    accumulator.update(
        torch.tensor([[0.0, 5.0]] * 9),
        torch.tensor([0] * 9),
        [9],
        ["long"],
    )
    summary = accumulator.summary()
    assert summary["recording_mean_supported_f1"] == pytest.approx(0.5)
    assert summary["pooled_supported_f1"] == pytest.approx(2 / 11)


def _manifest(path: Path, step: int | None) -> None:
    checkpoint_name = "best.ckpt" if step is None else f"step{step}.ckpt"
    checkpoint = path.parent.parent / "checkpoints" / checkpoint_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{step}".encode())
    payload = {
        "schema": "neurosoft-pretraining-checkpoint",
        "version": 1,
        "checkpoint": {
            "kind": "best" if step is None else "milestone",
            "path": f"checkpoints/{checkpoint_name}",
            "sha256": evaluator.sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "trained_on": {"optimizer_steps": 777 if step is None else step},
        "selection": {},
        "compute": {},
        "recipe": {},
        "normalization_artifact_hashes": {},
        "git_sha": "abc",
        "snapshot_bundle": "bundle",
        "slurm_job_id": "1",
        "wandb": {},
    }
    payload["manifest_hash"] = evaluator.stable_json_hash({})
    # Use the production private canonicalizer to make a valid test manifest.
    from foundry.training.checkpoint_manifest import _compute_manifest_hash

    payload["manifest_hash"] = _compute_manifest_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_checkpoint_discovery_returns_five_milestones_and_best(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    for step in evaluator.MILESTONE_STEPS:
        _manifest(run / "manifests" / f"milestone-step{step}.json", step)
    _manifest(run / "manifests" / "best-loss.json", None)
    result = evaluator.discover_checkpoints(run)
    assert [item["step"] for item in result] == [
        100,
        300,
        1000,
        3000,
        10000,
        None,
    ]


def _rows(count: int, signature: str = "sig") -> list[dict[str, str]]:
    condition_count = len(evaluator.ALL_CONDITIONS)
    return [
        {
            "source_run_name": "source",
            "checkpoint_sha256": f"checkpoint-{index // condition_count}",
            "checkpoint_manifest_hash": f"manifest-{index // condition_count}",
            "condition": evaluator.ALL_CONDITIONS[index % condition_count],
            "evaluation_signature": signature,
        }
        for index in range(count)
    ]


def test_resume_skips_complete_source_and_recomputes_incomplete() -> None:
    checkpoints = [
        {
            "checkpoint_sha256": f"checkpoint-{index}",
            "manifest_hash": f"manifest-{index}",
        }
        for index in range(6)
    ]
    expected_count = 6 * len(evaluator.ALL_CONDITIONS)
    complete = _rows(expected_count)
    assert evaluator.source_complete(
        complete, "source", checkpoints, evaluator.ALL_CONDITIONS, "sig"
    )
    assert not evaluator.source_complete(
        complete[:-1], "source", checkpoints, evaluator.ALL_CONDITIONS, "sig"
    )
    replacement = [{"source_run_name": "source", "condition": "new"}]
    combined = evaluator.replace_source_rows(
        complete[:-1] + [{"source_run_name": "other", "condition": "keep"}],
        "source",
        replacement,
    )
    assert combined == [
        {"source_run_name": "other", "condition": "keep"},
        replacement[0],
    ]


def test_atomic_csv_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "results.csv"
    destination.write_text("original\n")

    def fail_replace(source: object, target: object) -> None:
        raise OSError("injected failure")

    monkeypatch.setattr(evaluator.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        evaluator.atomic_write_csv(destination, [])
    assert destination.read_text() == "original\n"
    assert not list(tmp_path.glob("*.tmp"))

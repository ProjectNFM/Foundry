from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from foundry.models.neurosoft_models import NeurosoftConvBiGRU
from foundry.tasks.config import TaskConfig
from foundry.training import PretrainedTransferError, load_pretrained_weights
from foundry.training.checkpoint_manifest import write_checkpoint_manifest
from tools.compile_downstream_cells import compile_cells
from tools.generate_architecture_viability_registry import (
    MILESTONES,
    build_registry,
)
from tools.generate_architecture_viability_source_cells import (
    CONDITIONS,
    GROUPS,
    TARGETS,
    build_records,
    source_run_name,
)


ROOT = Path(__file__).resolve().parents[1]


def _tasks() -> dict[str, TaskConfig]:
    return {
        "task": TaskConfig.from_dict(
            {
                "name": "task",
                "head": {
                    "_target_": "foundry.tasks.heads.ReadoutHead",
                    "output_dim": 8,
                },
                "target_extractor": {
                    "_target_": "foundry.tasks.targets.TargetExtractor",
                    "timestamp_key": "task.timestamps",
                    "value_key": "task.values",
                },
                "loss": {
                    "_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"
                },
                "metrics": {
                    "_target_": "foundry.tasks.metrics.classification_metrics",
                    "num_classes": 8,
                },
                "class_names": [str(index) for index in range(8)],
            }
        )
    }


def _model(**overrides) -> NeurosoftConvBiGRU:
    kwargs = {
        "task_configs": _tasks(),
        "session_configs": {"a": 3, "b": 5},
        "adapter_dim": 8,
        "temporal_channels": 12,
        "gru_hidden_size": 8,
        "dropout_rate": 0.0,
    }
    kwargs.update(overrides)
    return NeurosoftConvBiGRU(**kwargs)


def _checkpoint(model: NeurosoftConvBiGRU, path: Path) -> None:
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value
                for key, value in model.state_dict().items()
            }
        },
        path,
    )


def test_adapter_variants_padding_and_backward() -> None:
    biased = _model(input_adapter_bias=True)
    bias_free = _model(input_adapter_bias=False)
    assert biased.session_adapter.layers["a"].bias is not None
    assert bias_free.session_adapter.layers["a"].bias is None

    shared = _model(input_adapter_mode="shared_padded", shared_input_channels=6)
    shared.session_adapter.shared.bias.data.fill_(2.0)
    values = torch.randn(2, 5, 80)
    output = shared.session_adapter(
        values,
        input_session_ids=["a", "b"],
        input_channel_counts=[3, 5],
        input_seq_len=[70, 80],
    )
    manual = torch.zeros(80, 6)
    manual[:, :3] = values[0, :3].T
    expected = shared.session_adapter.shared(manual).T
    expected[:, 70:] = 0
    torch.testing.assert_close(output[0], expected)
    assert torch.count_nonzero(output[0, :, 70:]) == 0
    assert torch.count_nonzero(manual[:, 3:]) == 0

    logits = shared(
        input_values=values,
        task_index=torch.ones(2, 1, dtype=torch.long),
        input_session_ids=["a", "b"],
        input_channel_counts=[3, 5],
        input_seq_len=[70, 80],
    )["task"]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in shared.parameters()
    )


def test_shared_width_rejection_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="wider than 4"):
        _model(input_adapter_mode="shared_padded", shared_input_channels=4)
    source = _model(input_adapter_mode="shared_padded", shared_input_channels=6)
    path = tmp_path / "shared.ckpt"
    _checkpoint(source, path)
    target = _model(input_adapter_mode="shared_padded", shared_input_channels=6)
    fresh_router = {
        key: value.clone() for key, value in target.router.state_dict().items()
    }
    report = load_pretrained_weights(
        target,
        path,
        components=target.transferable_components_for_mode(
            "full_finetuning_retain_shared_adapter_reset_router"
        ),
    )
    assert any(
        key.startswith("session_adapter.shared") for key in report.loaded
    )
    assert all(
        key.startswith(("session_adapter.", "temporal_frontend.", "gru."))
        for key in report.loaded
    )
    assert all(
        torch.equal(value, target.router.state_dict()[key])
        for key, value in fresh_router.items()
    )
    assert all(parameter.requires_grad for parameter in target.parameters())

    ordinary = _model()
    ordinary_path = tmp_path / "ordinary.ckpt"
    _checkpoint(ordinary, ordinary_path)
    retained_target = _model(
        input_adapter_mode="shared_padded", shared_input_channels=6
    )
    with pytest.raises(PretrainedTransferError, match="Missing in checkpoint"):
        load_pretrained_weights(
            retained_target,
            ordinary_path,
            components=retained_target.transferable_components_for_mode(
                "full_finetuning_retain_shared_adapter_reset_router"
            ),
        )


def test_exact_transferable_parameter_counts() -> None:
    def count(width: int) -> int:
        model = NeurosoftConvBiGRU(
            task_configs=_tasks(),
            session_configs={"a": 3},
            temporal_channels=width,
            gru_hidden_size=width,
        )
        return sum(
            parameter.numel()
            for component in (model.temporal_frontend, model.gru)
            for parameter in component.parameters()
        )

    assert count(38) == 51066
    assert count(128) == 507456
    assert count(410) == 5084598


@pytest.mark.parametrize(
    "source_kwargs,target_kwargs,regime",
    [
        (
            {"temporal_channels": 38, "gru_hidden_size": 38},
            {"temporal_channels": 38, "gru_hidden_size": 38},
            "full_finetuning_reset_router",
        ),
        (
            {"temporal_channels": 410, "gru_hidden_size": 410},
            {"temporal_channels": 410, "gru_hidden_size": 410},
            "full_finetuning_reset_router",
        ),
        (
            {"input_adapter_bias": False},
            {"input_adapter_bias": True},
            "full_finetuning_reset_router",
        ),
        (
            {"input_adapter_bias": False},
            {"input_adapter_bias": False},
            "full_finetuning_reset_router",
        ),
        (
            {"input_adapter_mode": "shared_padded", "shared_input_channels": 6},
            {"input_adapter_mode": "per_session"},
            "full_finetuning_reset_router",
        ),
        (
            {"input_adapter_mode": "shared_padded", "shared_input_channels": 6},
            {"input_adapter_mode": "shared_padded", "shared_input_channels": 6},
            "full_finetuning_retain_shared_adapter_reset_router",
        ),
    ],
)
def test_all_six_transfer_paths_execute_optimizer_step(
    tmp_path: Path,
    source_kwargs: dict,
    target_kwargs: dict,
    regime: str,
) -> None:
    common = {
        "task_configs": _tasks(),
        "adapter_dim": 64,
        "dropout_rate": 0.0,
    }
    source = NeurosoftConvBiGRU(
        **common, session_configs={"source": 3}, **source_kwargs
    )
    target = NeurosoftConvBiGRU(
        **common, session_configs={"target": 5}, **target_kwargs
    )
    checkpoint = tmp_path / f"{regime}-{len(source_kwargs)}.ckpt"
    _checkpoint(source, checkpoint)
    fresh_router = {
        key: value.clone() for key, value in target.router.state_dict().items()
    }
    fresh_adapter = {
        key: value.clone()
        for key, value in target.session_adapter.state_dict().items()
    }
    report = load_pretrained_weights(
        target,
        checkpoint,
        components=target.transferable_components_for_mode(regime),
    )
    assert all(
        torch.equal(value, target.router.state_dict()[key])
        for key, value in fresh_router.items()
    )
    if regime == "full_finetuning_retain_shared_adapter_reset_router":
        assert any(
            key.startswith("session_adapter.shared") for key in report.loaded
        )
    else:
        assert not any(
            key.startswith("session_adapter") for key in report.loaded
        )
        assert all(
            torch.equal(value, target.session_adapter.state_dict()[key])
            for key, value in fresh_adapter.items()
        )
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    logits = target(
        input_values=torch.randn(1, 5, 80),
        task_index=torch.ones(1, 1, dtype=torch.long),
        input_session_ids=["target"],
        input_channel_counts=[5],
        input_seq_len=[80],
    )["task"]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_source_generator_exact_matrix() -> None:
    minipigs = build_records(ROOT, "minipigs")
    monkeys = build_records(ROOT, "monkeys")
    assert (len(minipigs), len(monkeys)) == (35, 25)
    rows = minipigs + monkeys
    assert len({row["cell_id"] for row in rows}) == 60
    assert {row["source_selection_seed"] for row in rows} == {42}
    assert {row["source_model_seed"] for row in rows} == {42}
    assert all(
        row["fixed_milestones"] == [100, 300, 1000, 3000, 10000] for row in rows
    )
    assert all(
        "hyperparameters.batch_size=128" in row["overrides"] for row in rows
    )
    assert all(
        "+trainer.check_val_every_n_epoch=null" in row["overrides"]
        for row in rows
    )


def _fake_source_outputs(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "runs"
    checkpoint_root = tmp_path / "checkpoints"
    for condition, metadata in CONDITIONS.items():
        for species, numbers in TARGETS.items():
            for number in numbers:
                subject = f"sub-{number:02d}"
                run_name = source_run_name(condition, species, subject)
                manifest_dir = (
                    run_root / GROUPS[species] / run_name / "manifests"
                )
                for step, kind in MILESTONES.items():
                    relative = (
                        Path(condition)
                        / species
                        / subject
                        / f"{kind}-step{step}.ckpt"
                    )
                    checkpoint = checkpoint_root / relative
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(
                        f"{condition}-{species}-{subject}-{step}".encode()
                    )
                    write_checkpoint_manifest(
                        checkpoint,
                        manifest_dir,
                        kind=kind,
                        trained_on={
                            "source_selection_id": f"architecture_{species}_{subject}_sel42",
                            "source_selection_seed": 42,
                            "source_model_seed": 42,
                            "excluded_target": {
                                "species": species,
                                "subject": subject,
                            },
                            "optimizer_steps": step,
                        },
                        selection={
                            "monitor": "not_selected_milestone",
                            "monitor_value": None,
                        },
                        compute={},
                        recipe={
                            "model_metadata": {
                                **metadata,
                                "source_condition": condition,
                                "total_parameter_count": metadata[
                                    "transferable_parameter_count"
                                ]
                                + 1,
                            }
                        },
                        normalization_artifact_hashes={},
                        git_sha="a" * 40,
                        snapshot_bundle="/shared/snapshot",
                        slurm_job_id="1",
                        wandb_info={
                            "project": "test",
                            "group": GROUPS[species],
                            "run_id": "abcdefgh",
                        },
                        checkpoint_relative_path=relative.as_posix(),
                    )
    return run_root, checkpoint_root


def test_registry_and_compiler_exact_global_counts(tmp_path: Path) -> None:
    run_root, checkpoint_root = _fake_source_outputs(tmp_path)
    recipes = {
        "small_backbone": ("architecture_small_backbone.yaml", 795, 159),
        "large_backbone": ("architecture_large_backbone.yaml", 795, 159),
        "bias_free": ("architecture_bias_free.yaml", 1590, 159),
        "shared_padded": ("architecture_shared_adapter.yaml", 1590, 159),
    }
    total_transfer = total_scratch = 0
    for condition, (
        recipe_name,
        expected_transfer,
        expected_scratch,
    ) in recipes.items():
        rows = build_registry(run_root, checkpoint_root, condition)
        assert len(rows) == 60
        registry = tmp_path / f"{condition}.jsonl"
        registry.write_text("".join(json.dumps(row) + "\n" for row in rows))
        cells, _ = compile_cells(
            registry,
            ROOT / "configs/downstream_recipes" / recipe_name,
            ROOT / "docs/neurosoft-phase0-audit.json",
            checkpoint_root,
        )
        flat = [row for species_rows in cells.values() for row in species_rows]
        transfer = [row for row in flat if row["checkpoint_id"] is not None]
        scratch = [row for row in flat if row["checkpoint_id"] is None]
        assert (len(transfer), len(scratch)) == (
            expected_transfer,
            expected_scratch,
        )
        assert len({row["cell_id"] for row in flat}) == len(flat)
        assert all(row["target_fraction"] == 1.0 for row in flat)
        assert {row["target_finetuning_seed"] for row in flat} == {42, 43, 44}
        scratch_ids = {row["cell_id"] for row in scratch}
        assert all(row["matched_scratch_id"] in scratch_ids for row in transfer)
        assert {row["source_model_seed"] for row in transfer} == {42}
        assert {row["source_selection_seed"] for row in transfer} == {42}
        assert {
            row["source_condition"]["milestone_step"] for row in transfer
        } == set(MILESTONES)
        total_transfer += len(transfer)
        total_scratch += len(scratch)
    assert (total_transfer, total_scratch) == (4770, 636)

"""Tests for Phase 1 EEGNet learning-curves infrastructure."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from torchmetrics import Metric, MetricCollection

import foundry.config_resolvers as config_resolvers
from foundry.config_resolvers import (
    _load_neurosoft_audit,
    _neurosoft_supported_cell_sweep_choices,
    _phase1_cell_fraction,
    _phase1_cell_recording,
    register_resolvers,
)
from foundry.tasks.metrics import (
    classification_metrics,
    supported_classification_metrics,
)
from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.fraction_manifest import _canonical_hash
from foundry.training.callbacks.compute import ComputeTrackingCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = str(PROJECT_ROOT / "docs/neurosoft-phase0-audit.json")
FRACTION_VALIDATION_PATH = (
    PROJECT_ROOT / "docs/neurosoft-phase0-fraction-validation.json"
)
PHASE1_SEEDS = 3


def load_phase1_analysis_module():
    """Import the date-prefixed standalone analysis script for unit tests."""
    analysis_dir = PROJECT_ROOT / "analysis"
    sys.path.insert(0, str(analysis_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "phase1_analysis",
            analysis_dir
            / "20260826-MS-neurosoft-eegnet-learning-curves_analysis.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(analysis_dir))


def parse_sweep_cells(choices_str: str) -> list[str]:
    """Parse a comma-separated string of quoted cell values."""
    cells = []
    for part in choices_str.split("','"):
        cell = part.strip().strip("'")
        if cell:
            cells.append(cell)
    return cells


def load_unavailable_cells(
    fraction_validation_path: Path, audit: dict
) -> set[str]:
    """Return ``recording_id|fraction`` cells marked unavailable in validation."""
    with open(fraction_validation_path) as f:
        validation = json.load(f)

    rec_by_id = {rec["recording_id"]: rec for rec in audit["recordings"]}
    unavailable: set[str] = set()
    for item in validation["unavailable_scientific_cells"]:
        recording_id = item["recording_id"]
        requested_fraction = float(item["requested_fraction"])
        recording = rec_by_id.get(recording_id)
        if recording is None:
            continue
        for frac_str, frac_info in recording["fraction_availability"].items():
            if abs(float(frac_str) - requested_fraction) < 1e-9:
                if not frac_info.get("available", False):
                    unavailable.add(f"{recording_id}|{frac_str}")
                break
    return unavailable


@pytest.fixture(autouse=True)
def _clear_audit_cache() -> None:
    config_resolvers._audit_cache.clear()
    yield
    config_resolvers._audit_cache.clear()


@pytest.fixture(scope="module")
def audit() -> dict:
    return _load_neurosoft_audit(AUDIT_PATH)


class TestConfigResolvers:
    def test_audit_loads_and_verifies_hash(self) -> None:
        audit = _load_neurosoft_audit(AUDIT_PATH)
        assert "artifact_sha256" in audit
        assert audit["recordings"]

    def test_audit_caching(self) -> None:
        first = _load_neurosoft_audit(AUDIT_PATH)
        second = _load_neurosoft_audit(AUDIT_PATH)
        assert first is second

    def test_minipig_supported_cells_count(self) -> None:
        choices = _neurosoft_supported_cell_sweep_choices(
            AUDIT_PATH, "minipigs"
        )
        cells = parse_sweep_cells(choices)
        assert len(cells) == 193

    def test_monkey_supported_cells_count(self) -> None:
        choices = _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
        cells = parse_sweep_cells(choices)
        assert len(cells) == 62

    def test_total_seeded_jobs(self) -> None:
        minipig_cells = len(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
            )
        )
        monkey_cells = len(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
            )
        )

        assert minipig_cells == 193
        assert monkey_cells == 62
        assert minipig_cells + monkey_cells == 255
        assert minipig_cells * PHASE1_SEEDS == 579
        assert monkey_cells * PHASE1_SEEDS == 186
        assert (minipig_cells + monkey_cells) * PHASE1_SEEDS == 765

    def test_cell_decode_recording(self) -> None:
        cell = "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw|0.25"
        assert _phase1_cell_recording(cell) == (
            "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw"
        )

    def test_cell_decode_fraction(self) -> None:
        cell = "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw|0.25"
        assert _phase1_cell_fraction(cell) == 0.25

    def test_unavailable_cells_excluded(self, audit: dict) -> None:
        unavailable = load_unavailable_cells(FRACTION_VALIDATION_PATH, audit)
        assert unavailable

        sweep_cells = set(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
            )
        )
        sweep_cells.update(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
            )
        )

        assert unavailable.isdisjoint(sweep_cells)

    def test_deterministic_order(self) -> None:
        first = _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
        second = _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
        assert first == second


class TestSupportedMetrics:
    def test_supported_metrics_collection_keys(self) -> None:
        metrics = supported_classification_metrics(num_classes=8)

        assert isinstance(metrics, MetricCollection)
        assert set(metrics.keys()) == {
            *classification_metrics(num_classes=8).keys(),
            "supported_f1",
            "supported_auroc",
            "supported_precision",
            "supported_recall",
            "supported_balanced_acc",
            "num_present_classes",
        }
        for name in metrics.keys():
            assert isinstance(metrics[name], Metric)

    def test_supported_f1_excludes_absent_classes(self) -> None:
        num_classes = 8
        metrics = supported_classification_metrics(num_classes)

        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        preds = torch.full((6, num_classes), -10.0)
        for index, target in enumerate(targets):
            preds[index, target] = 10.0

        metrics.update(preds, targets)
        computed = metrics.compute()

        assert computed["num_present_classes"].item() == 3
        assert computed["supported_f1"].item() == pytest.approx(1.0)

        per_class_f1 = metrics["supported_f1"]._base.compute()
        support = metrics["supported_f1"]._target_support()
        supported_only_mean = per_class_f1[support > 0].mean().item()
        naive_all_classes_mean = per_class_f1.mean().item()

        assert computed["supported_f1"].item() == pytest.approx(
            supported_only_mean
        )
        assert naive_all_classes_mean == pytest.approx(0.375)
        assert computed["supported_f1"].item() != pytest.approx(
            naive_all_classes_mean
        )

    def test_supported_auroc_excludes_absent_classes(self) -> None:
        num_classes = 8
        metrics = supported_classification_metrics(num_classes)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        preds = torch.full((len(targets), num_classes), -10.0)
        for index, target in enumerate(targets):
            preds[index, target] = 10.0

        metrics.update(preds, targets)

        assert metrics["supported_auroc"].compute().item() == pytest.approx(1.0)

    def test_absent_class_prediction_penalized(self) -> None:
        num_classes = 8
        targets = torch.tensor([0, 1, 2, 0, 1, 2])

        correct_preds = torch.full((6, num_classes), -10.0)
        for index, target in enumerate(targets):
            correct_preds[index, target] = 10.0

        wrong_preds = correct_preds.clone()
        wrong_preds[0] = -10.0
        wrong_preds[0, 7] = 10.0

        correct_metrics = supported_classification_metrics(num_classes)
        wrong_metrics = supported_classification_metrics(num_classes)
        correct_metrics.update(correct_preds, targets)
        wrong_metrics.update(wrong_preds, targets)

        correct_f1 = correct_metrics["supported_f1"].compute().item()
        wrong_f1 = wrong_metrics["supported_f1"].compute().item()

        assert correct_f1 == pytest.approx(1.0)
        assert wrong_f1 < correct_f1

    def test_num_present_classes_count(self) -> None:
        metrics = supported_classification_metrics(num_classes=8)
        targets = torch.tensor([0, 1, 2, 3, 4, 5])
        preds = torch.randn(6, 8)

        metrics.update(preds, targets)
        assert metrics["num_present_classes"].compute().item() == 6

    def test_supported_metrics_clone_prefix(self) -> None:
        metrics = supported_classification_metrics(num_classes=8)
        cloned = metrics.clone(prefix="val/")

        assert set(cloned.keys()) == {f"val/{key}" for key in metrics.keys()}

    def test_supported_metrics_reset(self) -> None:
        metrics = supported_classification_metrics(num_classes=8)
        preds = torch.randn(4, 8)
        targets = torch.tensor([0, 1, 2, 3])

        metrics.update(preds, targets)
        first_compute = metrics.compute()
        assert first_compute["supported_f1"].item() >= 0.0

        metrics.reset()
        for metric in metrics.values():
            metric.update(
                torch.full((2, 8), -10.0),
                torch.tensor([0, 1]),
            )
        second_compute = metrics.compute()
        assert second_compute["num_present_classes"].item() == 2

    def test_supported_metrics_update_after_sanity_validation(self) -> None:
        """All supported metrics must update after an initial compute/reset.

        Lightning computes validation metrics during sanity validation before
        resetting them for epoch zero.  With automatic MetricCollection
        compute groups, the nested base states of the supported metric
        wrappers were not shared, leaving F1/precision/recall frozen at their
        sanity-validation value while AUROC continued to update.
        """
        num_classes = 8
        metrics = supported_classification_metrics(num_classes=num_classes)
        targets = torch.arange(num_classes)

        wrong_preds = torch.full((num_classes, num_classes), -10.0)
        wrong_preds[
            torch.arange(num_classes), (targets + 1) % num_classes
        ] = 10.0
        metrics.update(wrong_preds, targets)
        metrics.compute()  # Simulate sanity-validation epoch-end logging.
        metrics.reset()

        correct_preds = torch.full((num_classes, num_classes), -10.0)
        correct_preds[torch.arange(num_classes), targets] = 10.0
        metrics.update(correct_preds, targets)
        computed = metrics.compute()

        for name in (
            "supported_f1",
            "supported_precision",
            "supported_recall",
            "supported_balanced_acc",
        ):
            assert computed[name].item() == pytest.approx(1.0)


class TestComputeCallback:
    def test_init_validates_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            ComputeTrackingCallback(
                monitor="val/loss",
                mode="invalid",
                sequence_length=0.5,
            )

    def test_state_dict_round_trip(self) -> None:
        callback = ComputeTrackingCallback(
            monitor="val/metric",
            mode="max",
            sequence_length=0.5,
            flops_per_window=123,
        )
        callback._processed_windows = 42
        callback._restored_wall_time_s = 12.5
        callback._best_monitor_value = 0.75
        callback._best_step = 7
        callback._best_examples = 42
        callback._best_windows = 42
        callback._best_flops = 5166
        callback._best_wall_time_s = 12.5

        state = callback.state_dict()
        restored = ComputeTrackingCallback(
            monitor="val/metric",
            mode="max",
            sequence_length=0.5,
            flops_per_window=123,
        )
        restored.load_state_dict(state)

        assert restored._processed_windows == 42
        assert restored._restored_wall_time_s == pytest.approx(12.5)
        assert restored._best_monitor_value == pytest.approx(0.75)
        assert restored._best_step == 7
        assert restored._best_examples == 42
        assert restored._best_windows == 42
        assert restored._best_flops == 5166
        assert restored._best_wall_time_s == pytest.approx(12.5)

    def test_count_batch_windows(self) -> None:
        batch = {"task_index": torch.zeros(4, dtype=torch.long)}
        assert ComputeTrackingCallback._count_batch_windows(batch) == 4

    def test_gpu_model_name_on_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert ComputeTrackingCallback._gpu_model_name() == "cpu"

    def test_peak_memory_on_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert ComputeTrackingCallback._peak_memory_gb() == (0.0, 0.0)

    def test_require_flops_rejects_missing_metadata(self) -> None:
        with pytest.raises(ValueError, match="requires validated"):
            ComputeTrackingCallback(
                monitor="val/metric",
                mode="max",
                sequence_length=0.5,
                require_flops=True,
            )

    def test_compute_metrics_are_scalar_and_fit_end_bypasses_module_log(
        self,
    ) -> None:
        class RejectingModule(torch.nn.Linear):
            def log_dict(self, *args, **kwargs) -> None:
                raise AssertionError(
                    "on_fit_end must not call pl_module.log_dict"
                )

        class RecordingLogger:
            def __init__(self) -> None:
                self.metrics: list[tuple[dict, int]] = []

            def log_hyperparams(self, params: dict) -> None:
                pass

            def log_metrics(self, metrics: dict, step: int) -> None:
                self.metrics.append((metrics, step))

        callback = ComputeTrackingCallback(
            monitor="val/metric",
            mode="max",
            sequence_length=0.5,
            flops_per_window=123,
            flop_method="torch-profiler-v1",
            require_flops=True,
        )
        logger = RecordingLogger()
        trainer = SimpleNamespace(
            world_size=1,
            accumulate_grad_batches=1,
            global_step=3,
            current_epoch=0,
            precision="bf16-mixed",
            datamodule=SimpleNamespace(batch_size=2),
            logger=logger,
            callbacks=[],
            callback_metrics={},
            logged_metrics={},
        )
        module = RejectingModule(2, 2)
        callback._processed_windows = 6

        metrics = callback._build_compute_metrics(trainer, module)
        assert all(
            isinstance(value, (float, int)) for value in metrics.values()
        )

        callback.on_fit_end(trainer, module)
        assert logger.metrics
        assert logger.metrics[0][0]["compute/processed_windows"] == 6


class TestFractionAuditVerification:
    def test_split_hash_matches_phase0_raw_interval_algorithm(self) -> None:
        intervals = SimpleNamespace(
            start=torch.tensor([0.0, 1.5]).numpy(),
            end=torch.tensor([1.0, 2.5]).numpy(),
            behavior_labels=torch.tensor([4, 9]).numpy(),
            __len__=lambda: 2,
        )

        # SimpleNamespace cannot customize len(), so use a tiny interval type.
        class Intervals:
            start = intervals.start
            end = intervals.end
            behavior_labels = intervals.behavior_labels

            def __len__(self) -> int:
                return 2

        expected = _canonical_hash(
            [
                {
                    "recording_id": "recording",
                    "index": 0,
                    "start": float(0.0).hex(),
                    "end": float(1.0).hex(),
                    "label": "4",
                },
                {
                    "recording_id": "recording",
                    "index": 1,
                    "start": float(1.5).hex(),
                    "end": float(2.5).hex(),
                    "label": "9",
                },
            ]
        )
        actual = NeuralDataModule._compute_split_hash("recording", Intervals())
        assert actual == expected

    def test_split_hash_mismatch_fails_before_training(self) -> None:
        with pytest.raises(RuntimeError, match="Runtime split hashes differ"):
            NeuralDataModule._verify_audit_split_hashes(
                "recording",
                {"train": "actual", "valid": "same", "test": "same"},
                {
                    "split_hashes": {
                        "train": "expected",
                        "valid": "same",
                        "test": "same",
                    }
                },
            )


class TestAnalysisProvenance:
    def test_recording_lookup_and_data_to_80_are_species_qualified(
        self,
    ) -> None:
        analysis = load_phase1_analysis_module()
        audit = {
            "protocol": {"seeds": [42]},
            "recordings": [
                {
                    "recording_id": "shared",
                    "species": "minipigs",
                    "subject": "sub-01",
                    "eligible": True,
                    "fraction_availability": {"1.0": {"available": True}},
                },
                {
                    "recording_id": "shared",
                    "species": "monkeys",
                    "subject": "sub-02",
                    "eligible": True,
                    "fraction_availability": {"1.0": {"available": True}},
                },
            ],
        }
        _, _, lookup = analysis.build_audit_tables(audit)
        assert lookup[("minipigs", "shared")]["subject"] == "sub-01"
        assert lookup[("monkeys", "shared")]["subject"] == "sub-02"

        runs = pd.DataFrame(
            [
                {
                    "species": "minipigs",
                    "recording_id": "shared",
                    "subject": "sub-01",
                    "fraction": 1.0,
                    "finished": True,
                    "test_f1": 0.8,
                    "num_present_classes": 6,
                },
                {
                    "species": "monkeys",
                    "recording_id": "shared",
                    "subject": "sub-02",
                    "fraction": 1.0,
                    "finished": True,
                    "test_f1": 0.6,
                    "num_present_classes": 6,
                },
            ]
        )
        result = analysis.compute_data_to_80(runs)
        assert set(result["species"]) == {"minipigs", "monkeys"}

    def test_subject_balanced_summary_keeps_each_fraction(self) -> None:
        analysis = load_phase1_analysis_module()
        runs = pd.DataFrame(
            [
                {
                    "species": "minipigs",
                    "subject": "sub-01",
                    "recording_id": "recording-a",
                    "fraction": fraction,
                    "finished": True,
                    "test_f1": score,
                }
                for fraction, score in ((0.05, 0.2), (1.0, 0.8))
                for _seed in (42, 43, 44)
            ]
        )
        balanced, unweighted = analysis.subject_balanced_summary(runs)

        assert balanced["fraction"].tolist() == [0.05, 1.0]
        assert unweighted["fraction"].tolist() == [0.05, 1.0]


class TestConfigComposition:
    @pytest.fixture(scope="class", autouse=True)
    def _register_resolvers(self) -> None:
        register_resolvers()

    def test_minipig_config_cell_count(self) -> None:
        choices = _neurosoft_supported_cell_sweep_choices(
            AUDIT_PATH, "minipigs"
        )
        assert len(parse_sweep_cells(choices)) == 193

    def test_monkey_config_cell_count(self) -> None:
        choices = _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
        assert len(parse_sweep_cells(choices)) == 62

    def test_total_cells_and_jobs(self) -> None:
        minipig_cells = len(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
            )
        )
        monkey_cells = len(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
            )
        )

        assert minipig_cells + monkey_cells == 255
        assert (minipig_cells + monkey_cells) * PHASE1_SEEDS == 765

    def test_audit_unavailable_cells_not_in_sweep(self, audit: dict) -> None:
        unavailable = load_unavailable_cells(FRACTION_VALIDATION_PATH, audit)
        sweep_cells = set(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "minipigs")
            )
        )
        sweep_cells.update(
            parse_sweep_cells(
                _neurosoft_supported_cell_sweep_choices(AUDIT_PATH, "monkeys")
            )
        )

        assert unavailable.isdisjoint(sweep_cells)

"""Tests for analysis/_wandb_utils.py — no network calls."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import pandas as pd
import pytest

from analysis._wandb_utils import (
    MetricNotFoundError,
    RunNotFoundError,
    default_entity,
    fetch_metric_history,
    fetch_run_summary,
    figures_dir,
    get_run,
    make_run_path,
    unwrap_summary_value,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeRun:
    name: str = "fake-run"
    id: str = "abc123"
    state: str = "finished"
    created_at: str = "2025-01-01"
    summary: dict = field(default_factory=dict)
    _history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def history(self, *, keys=None, samples=10_000, pandas=True):
        df = self._history
        if keys is not None:
            present = [k for k in keys if k in df.columns]
            df = df[present]
        return df


class FakeApi:
    """Minimal stand-in for ``wandb.Api`` that maps paths to ``FakeRun``s."""

    def __init__(self, runs: dict[str, FakeRun] | None = None):
        self._runs = runs or {}

    def run(self, path: str) -> FakeRun:
        if path not in self._runs:
            raise ValueError(f"Run not found: {path}")
        return self._runs[path]


# ---------------------------------------------------------------------------
# make_run_path
# ---------------------------------------------------------------------------


class TestMakeRunPath:
    def test_with_entity(self):
        assert make_run_path("r1", "proj", "team") == "team/proj/r1"

    def test_without_entity(self):
        assert make_run_path("r1", "proj") == "proj/r1"

    def test_none_entity(self):
        assert make_run_path("r1", "proj", None) == "proj/r1"

    def test_empty_string_entity(self):
        assert make_run_path("r1", "proj", "") == "proj/r1"


# ---------------------------------------------------------------------------
# unwrap_summary_value
# ---------------------------------------------------------------------------


class TestUnwrapSummaryValue:
    def test_dict_min(self):
        assert unwrap_summary_value({"min": 0.5, "max": 1.0}, "min") == 0.5

    def test_dict_max(self):
        assert unwrap_summary_value({"min": 0.5, "max": 1.0}, "max") == 1.0

    def test_plain_float(self):
        assert unwrap_summary_value(3.14) == 3.14

    def test_plain_int(self):
        assert unwrap_summary_value(42) == 42.0

    def test_string_numeric(self):
        assert unwrap_summary_value("3.14") == 3.14

    def test_none_returns_none(self):
        assert unwrap_summary_value(None) is None

    def test_missing_key_returns_dict_unchanged(self):
        val = {"max": 1.0}
        assert unwrap_summary_value(val, "min") is val

    def test_non_convertible_returns_unchanged(self):
        assert unwrap_summary_value("not-a-number") == "not-a-number"

    def test_default_key_is_min(self):
        assert unwrap_summary_value({"min": 0.1, "max": 0.9}) == 0.1


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_found(self):
        fake = FakeRun(name="my-run")
        api = FakeApi({"proj/r1": fake})
        run = get_run("r1", "proj", api=api)
        assert run.name == "my-run"

    def test_not_found_raises(self):
        api = FakeApi({})
        with pytest.raises(RunNotFoundError, match="r1"):
            get_run("r1", "proj", api=api)

    def test_with_entity(self):
        fake = FakeRun()
        api = FakeApi({"team/proj/r1": fake})
        run = get_run("r1", "proj", "team", api=api)
        assert run is fake


# ---------------------------------------------------------------------------
# fetch_metric_history
# ---------------------------------------------------------------------------


class TestFetchMetricHistory:
    def _api_with_history(self, path: str, df: pd.DataFrame) -> FakeApi:
        return FakeApi({path: FakeRun(_history=df)})

    def test_single_metric(self):
        df = pd.DataFrame({"_step": [1, 2, 3], "train/loss": [0.9, 0.7, 0.5]})
        api = self._api_with_history("proj/r1", df)
        result = fetch_metric_history("r1", "train/loss", "proj", api=api)
        assert list(result.columns) == ["_step", "train/loss"]
        assert len(result) == 3

    def test_multiple_metrics(self):
        df = pd.DataFrame(
            {
                "_step": [1, 2],
                "train/loss": [0.9, 0.7],
                "val/loss": [0.8, 0.6],
            }
        )
        api = self._api_with_history("proj/r1", df)
        result = fetch_metric_history(
            "r1", ["train/loss", "val/loss"], "proj", api=api
        )
        assert "train/loss" in result.columns
        assert "val/loss" in result.columns

    def test_drops_nan_rows(self):
        df = pd.DataFrame(
            {
                "_step": [1, 2, 3],
                "train/loss": [0.9, float("nan"), 0.5],
            }
        )
        api = self._api_with_history("proj/r1", df)
        result = fetch_metric_history("r1", "train/loss", "proj", api=api)
        assert len(result) == 2

    def test_missing_metric_raises(self):
        df = pd.DataFrame({"_step": [1, 2], "train/loss": [0.9, 0.7]})
        api = self._api_with_history("proj/r1", df)
        with pytest.raises(MetricNotFoundError, match="val/loss"):
            fetch_metric_history("r1", "val/loss", "proj", api=api)

    def test_custom_x_axis(self):
        df = pd.DataFrame(
            {
                "trainer/global_step": [10, 20],
                "train/loss": [0.9, 0.7],
            }
        )
        api = self._api_with_history("proj/r1", df)
        result = fetch_metric_history(
            "r1",
            "train/loss",
            "proj",
            x_axis="trainer/global_step",
            api=api,
        )
        assert "trainer/global_step" in result.columns

    def test_run_not_found(self):
        api = FakeApi({})
        with pytest.raises(RunNotFoundError):
            fetch_metric_history("r1", "train/loss", "proj", api=api)

    def test_reset_index(self):
        df = pd.DataFrame(
            {
                "_step": [1, 2, 3],
                "train/loss": [0.9, float("nan"), 0.5],
            }
        )
        api = self._api_with_history("proj/r1", df)
        result = fetch_metric_history("r1", "train/loss", "proj", api=api)
        assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# fetch_run_summary
# ---------------------------------------------------------------------------


class TestFetchRunSummary:
    def test_basic(self):
        fake = FakeRun(
            state="finished",
            summary={
                "val/loss": {"min": 0.12, "max": 0.45},
                "epoch": 10,
            },
        )
        api = FakeApi({"proj/r1": fake})
        result = fetch_run_summary(
            "r1",
            "proj",
            {"best_val_loss": ("val/loss", "min"), "epoch": ("epoch", "max")},
            api=api,
        )
        assert result["state"] == "finished"
        assert result["best_val_loss"] == pytest.approx(0.12)
        assert result["epoch"] == 10.0

    def test_exclude_state(self):
        fake = FakeRun(state="running", summary={"val/loss": 0.5})
        api = FakeApi({"proj/r1": fake})
        result = fetch_run_summary(
            "r1",
            "proj",
            {"best_val_loss": ("val/loss", "min")},
            api=api,
            include_state=False,
        )
        assert "state" not in result

    def test_missing_summary_key_returns_none(self):
        fake = FakeRun(summary={})
        api = FakeApi({"proj/r1": fake})
        result = fetch_run_summary(
            "r1",
            "proj",
            {"best_val_loss": ("val/loss", "min")},
            api=api,
        )
        assert result["best_val_loss"] is None

    def test_run_not_found(self):
        api = FakeApi({})
        with pytest.raises(RunNotFoundError):
            fetch_run_summary("r1", "proj", {}, api=api)


# ---------------------------------------------------------------------------
# default_entity
# ---------------------------------------------------------------------------


class TestDefaultEntity:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("WANDB_ENTITY", "my-team")
        assert default_entity() == "my-team"

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        assert default_entity() is None

    def test_empty_returns_none(self, monkeypatch):
        monkeypatch.setenv("WANDB_ENTITY", "")
        assert default_entity() is None


# ---------------------------------------------------------------------------
# figures_dir
# ---------------------------------------------------------------------------


class TestFiguresDir:
    def test_creates_dir(self, tmp_path):
        script = tmp_path / "my_script.py"
        script.touch()
        d = figures_dir(str(script))
        assert d.exists()
        assert d.name == "figures"
        assert d.parent == tmp_path


# ---------------------------------------------------------------------------
# Smoke imports — verify every analysis script can be imported.
# ---------------------------------------------------------------------------

ANALYSIS_SCRIPTS = [
    "analysis.001_overfit_single_batch",
    "analysis.002_overfit_single_session",
    "analysis.003_masking_difficulty_hierarchy",
    "analysis.004_channel_identity_decoder",
    "analysis.005_tokenizer_comparison",
    "analysis.006_kemp_sleep_tokenizer_baseline",
    "analysis.007_pretraining_loss_vs_downstream",
]


@pytest.mark.parametrize("module_name", ANALYSIS_SCRIPTS)
def test_smoke_import(module_name):
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), (
        f"{module_name} should expose a main() function"
    )

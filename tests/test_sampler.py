"""Tests for FastRandomFixedWindowSampler.

Covers correctness, determinism, edge cases, no global mutation, and
performance relative to the upstream RandomFixedWindowSampler.
"""

import time

import pytest
import torch
from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex
from torch_brain.samplers import RandomFixedWindowSampler

from foundry.data.samplers import FastRandomFixedWindowSampler


def _collect(sampler):
    """Drain a sampler and return a list of (recording_id, start, end) tuples."""
    return [(idx.recording_id, idx.start, idx.end) for idx in sampler]


def _make_intervals(*session_specs):
    """Build sampling_intervals from (name, [(start, end), ...]) specs."""
    intervals = {}
    for name, pairs in session_specs:
        starts, ends = zip(*pairs) if pairs else ([], [])
        intervals[name] = Interval(list(starts), list(ends))
    return intervals


# ---------------------------------------------------------------------------
# Correctness: fast sampler produces the same windows as upstream
# ---------------------------------------------------------------------------


class TestEquivalenceWithUpstream:
    """FastRandomFixedWindowSampler yields the same windows as upstream."""

    @pytest.fixture(params=[1.0, 2.5, 10.0])
    def window_length(self, request):
        return request.param

    @pytest.fixture
    def intervals(self):
        return _make_intervals(
            ("sess_a", [(0.0, 30.0), (50.0, 80.0)]),
            ("sess_b", [(0.0, 100.0)]),
            ("sess_c", [(10.0, 15.0), (20.0, 60.0)]),
        )

    def test_same_output(self, intervals, window_length):
        seed = 12345
        gen_up = torch.Generator().manual_seed(seed)
        gen_fast = torch.Generator().manual_seed(seed)

        upstream = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=window_length,
            generator=gen_up,
            drop_short=True,
        )
        fast = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=window_length,
            generator=gen_fast,
            drop_short=True,
        )

        up_windows = _collect(upstream)
        fast_windows = _collect(fast)

        assert len(fast_windows) == len(up_windows)
        for (s1, t1_s, t1_e), (s2, t2_s, t2_e) in zip(up_windows, fast_windows):
            assert s1 == s2
            assert abs(t1_s - t2_s) < 1e-9
            assert abs(t1_e - t2_e) < 1e-9


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self):
        intervals = _make_intervals(("s", [(0.0, 50.0)]))
        results = []
        for _ in range(3):
            gen = torch.Generator().manual_seed(42)
            sampler = FastRandomFixedWindowSampler(
                sampling_intervals=intervals,
                window_length=5.0,
                generator=gen,
                drop_short=True,
            )
            results.append(_collect(sampler))

        assert results[0] == results[1] == results[2]

    def test_different_seeds_differ(self):
        intervals = _make_intervals(("s", [(0.0, 50.0)]))
        outputs = []
        for seed in [42, 43]:
            gen = torch.Generator().manual_seed(seed)
            sampler = FastRandomFixedWindowSampler(
                sampling_intervals=intervals,
                window_length=5.0,
                generator=gen,
                drop_short=True,
            )
            outputs.append(_collect(sampler))

        assert outputs[0] != outputs[1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_exact_length_interval(self):
        """Interval exactly equals window_length → exactly one window."""
        intervals = _make_intervals(("s", [(10.0, 20.0)]))
        gen = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=10.0,
            generator=gen,
            drop_short=True,
        )
        windows = _collect(sampler)
        assert len(windows) >= 1
        for sess, s, e in windows:
            assert sess == "s"
            assert s >= 10.0 - 1e-9
            assert e <= 20.0 + 1e-9
            assert abs((e - s) - 10.0) < 1e-9

    def test_short_interval_dropped(self):
        """Short intervals are silently dropped with drop_short=True."""
        intervals = _make_intervals(
            ("short", [(0.0, 1.0)]),
            ("ok", [(0.0, 20.0)]),
        )
        gen = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=5.0,
            generator=gen,
            drop_short=True,
        )
        windows = _collect(sampler)
        sessions = {w[0] for w in windows}
        assert "short" not in sessions
        assert "ok" in sessions

    def test_short_interval_raises(self):
        """Short intervals raise with drop_short=False."""
        intervals = _make_intervals(("short", [(0.0, 1.0)]))
        gen = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=5.0,
            generator=gen,
            drop_short=False,
        )
        with pytest.raises(ValueError, match="too short"):
            _collect(sampler)

    def test_all_short_raises(self):
        """All intervals too short → ValueError on iteration."""
        intervals = _make_intervals(("a", [(0.0, 1.0)]), ("b", [(5.0, 6.0)]))
        gen = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=10.0,
            generator=gen,
            drop_short=True,
        )
        with pytest.raises(ValueError, match="too short"):
            _collect(sampler)

    def test_multiple_sessions(self):
        """Windows from multiple sessions are interleaved by shuffle."""
        intervals = _make_intervals(
            ("a", [(0.0, 20.0)]),
            ("b", [(0.0, 20.0)]),
            ("c", [(0.0, 20.0)]),
        )
        gen = torch.Generator().manual_seed(7)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=5.0,
            generator=gen,
            drop_short=True,
        )
        windows = _collect(sampler)
        sessions = [w[0] for w in windows]
        assert set(sessions) == {"a", "b", "c"}

    def test_window_boundaries(self):
        """Every yielded window has exactly window_length duration and stays in bounds."""
        intervals = _make_intervals(
            ("s", [(0.0, 37.3)]),
            ("t", [(100.0, 155.7)]),
        )
        gen = torch.Generator().manual_seed(99)
        wl = 7.0
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=wl,
            generator=gen,
            drop_short=True,
        )
        for idx in sampler:
            s, e = idx.start, idx.end
            assert abs((e - s) - wl) < 1e-9, f"bad duration: {e - s}"
            if idx.recording_id == "s":
                assert s >= 0.0 - 1e-9 and e <= 37.3 + 1e-9
            else:
                assert s >= 100.0 - 1e-9 and e <= 155.7 + 1e-9

    def test_yields_dataset_index(self):
        """Sampler yields DatasetIndex instances."""
        intervals = _make_intervals(("s", [(0.0, 10.0)]))
        gen = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=5.0,
            generator=gen,
            drop_short=True,
        )
        for item in sampler:
            assert isinstance(item, DatasetIndex)


# ---------------------------------------------------------------------------
# No global mutation
# ---------------------------------------------------------------------------


class TestNoGlobalMutation:
    def test_import_does_not_mutate_upstream(self):
        """Importing the sampler module does not alter RandomFixedWindowSampler."""
        original_iter = RandomFixedWindowSampler.__iter__

        import importlib

        import foundry.data.samplers

        importlib.reload(foundry.data.samplers)

        assert RandomFixedWindowSampler.__iter__ is original_iter

    def test_import_datamodule_does_not_mutate_upstream(self):
        """Importing NeuralDataModule does not alter RandomFixedWindowSampler."""
        original_iter = RandomFixedWindowSampler.__iter__

        import importlib

        import foundry.data.datamodules.base

        importlib.reload(foundry.data.datamodules.base)

        assert RandomFixedWindowSampler.__iter__ is original_iter

    def test_fast_sampler_module_removed(self):
        """The old monkey-patch module no longer exists."""
        with pytest.raises(ImportError):
            import foundry.data.fast_sampler  # noqa: F401


# ---------------------------------------------------------------------------
# Remainder / left-right offset logic
# ---------------------------------------------------------------------------


class TestRemainderHandling:
    """The remainder window is added when left_offset + right_offset >= wl."""

    def test_remainder_window_count_matches_upstream(self):
        intervals = _make_intervals(("s", [(0.0, 23.7)]))
        seed = 777
        wl = 5.0

        gen_up = torch.Generator().manual_seed(seed)
        gen_fast = torch.Generator().manual_seed(seed)

        up = RandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=wl,
            generator=gen_up,
            drop_short=True,
        )
        fast = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=wl,
            generator=gen_fast,
            drop_short=True,
        )

        assert len(_collect(up)) == len(_collect(fast))


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    @pytest.mark.slow
    def test_many_sessions_faster_than_baseline(self):
        """With many sessions, the fast sampler should be substantially faster."""
        n_sessions = 500
        intervals = _make_intervals(
            *[(f"s{i}", [(0.0, 60.0)]) for i in range(n_sessions)]
        )
        wl = 1.0

        gen_fast = torch.Generator().manual_seed(0)
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=intervals,
            window_length=wl,
            generator=gen_fast,
            drop_short=True,
        )

        t0 = time.perf_counter()
        windows = _collect(sampler)
        elapsed = time.perf_counter() - t0

        assert len(windows) > 0
        assert elapsed < 10.0, (
            f"Fast sampler took {elapsed:.2f}s for {n_sessions} sessions — "
            f"expected < 10s"
        )

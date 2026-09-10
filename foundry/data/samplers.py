"""Fast samplers for neural data windowed loading.

Contains :class:`FastRandomFixedWindowSampler` (vectorized drop-in for the
upstream ``RandomFixedWindowSampler``) and :class:`VariableLengthBatchSampler`
(per-batch random window-length selection for multi-length pretraining).
"""

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch_brain.datasets import DatasetIndex
from torch_brain.samplers import RandomFixedWindowSampler


class FastRandomFixedWindowSampler(RandomFixedWindowSampler):
    """Drop-in replacement for ``RandomFixedWindowSampler`` with vectorized iteration.

    Inherits the upstream constructor, ``__len__``, ``drop_short``, and generator
    handling.  Only ``__iter__`` is replaced with a NumPy-accelerated version
    that avoids per-element ``.item()`` calls while preserving the same window
    set, jitter semantics, and shuffle order for a given generator state.
    """

    def __iter__(self):
        if len(self) == 0:
            raise ValueError("All intervals are too short to sample from.")

        tuples: list[tuple[str, float, float]] = []
        wl = self.window_length

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in sampling_intervals:
                interval_length = end - start
                if interval_length < wl:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample "
                            f"from. Minimum length is {wl}."
                        )

                left_offset = (
                    torch.rand(1, generator=self.generator).item() * wl
                )

                starts = np.arange(
                    start + left_offset, end, wl, dtype=np.float64
                )
                valid = starts + wl <= end
                starts = starts[valid]

                for s in starts:
                    tuples.append((session_name, float(s), float(s) + wl))

                if len(starts) > 0:
                    right_offset = end - (starts[-1] + wl)
                else:
                    right_offset = end - start - left_offset

                if right_offset + left_offset >= wl:
                    if right_offset > left_offset:
                        tuples.append(
                            (session_name, float(end - wl), float(end))
                        )
                    else:
                        tuples.append(
                            (session_name, float(start), float(start + wl))
                        )

        perm = torch.randperm(len(tuples), generator=self.generator).tolist()
        for idx in perm:
            name, s, e = tuples[idx]
            yield DatasetIndex(name, s, e)


class NeurosoftFirstFixedWindowSampler(RandomFixedWindowSampler):
    """Emit one onset-anchored window per sampleable NeuroSoft stimulus.

    NeuroSoft source recordings contain a small number of annotation durations
    that are just below 0.5 s because of floating-point timestamp round-off,
    plus some genuinely short 0.1 s stimuli.  A supervised fixed-window task
    needs a deterministic policy for both: accept only intervals at least one
    window long within a narrow timestamp tolerance, and take the *first*
    window from a longer interval.  In particular, a 0.75 s stimulus produces
    ``[start, start + 0.5)`` rather than a jittered late window.

    This is deliberately NeuroSoft-specific.  It must not replace the generic
    sampler, whose multi-window/jitter behavior is used by other datasets.
    """

    timestamp_tolerance_seconds = 1e-9

    @classmethod
    def sampleable_mask(cls, starts, ends, window_length: float) -> np.ndarray:
        """Return the shared NeuroSoft eligibility rule for raw intervals."""
        return np.asarray(ends) - np.asarray(starts) >= (
            window_length - cls.timestamp_tolerance_seconds
        )

    def _is_sampleable(self, start: float, end: float) -> bool:
        return bool(self.sampleable_mask([start], [end], self.window_length)[0])

    def __len__(self) -> int:
        return sum(
            self._is_sampleable(float(start), float(end))
            for intervals in self.sampling_intervals.values()
            for start, end in intervals
        )

    def __iter__(self):
        tuples: list[tuple[str, float, float]] = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in sampling_intervals:
                start = float(start)
                end = float(end)
                if not self._is_sampleable(start, end):
                    if self.drop_short:
                        continue
                    raise ValueError(
                        f"Interval {(start, end)} is too short to sample "
                        f"from. Minimum length is {self.window_length}."
                    )
                tuples.append((session_name, start, start + self.window_length))

        perm = torch.randperm(len(tuples), generator=self.generator).tolist()
        for index in perm:
            name, start, end = tuples[index]
            yield DatasetIndex(name, start, end)


class VariableLengthBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that randomly selects a window length per batch.

    All samples within a batch share the same duration. Lengths are
    drawn uniformly from the provided list each time a batch is formed.

    Per epoch the sampler generates *all* possible windows for *every*
    length, shuffles them globally, and then yields complete batches
    whose members all share the same window duration.  Incomplete
    trailing batches for a given length are dropped.

    Yields lists of :class:`DatasetIndex` (one list = one batch).
    """

    def __init__(
        self,
        sampling_intervals: dict[str, Any],
        window_lengths: list[float],
        batch_size: int,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_lengths = sorted(window_lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def _generate_windows(self, wl: float) -> list[tuple[str, float, float]]:
        """Generate all non-overlapping windows of length *wl*."""
        tuples: list[tuple[str, float, float]] = []
        for session_name, intervals in self.sampling_intervals.items():
            for start, end in intervals:
                if end - start < wl:
                    continue

                left_offset = (
                    torch.rand(1, generator=self.generator).item() * wl
                )
                starts = np.arange(
                    start + left_offset, end, wl, dtype=np.float64
                )
                valid = starts + wl <= end
                starts = starts[valid]

                for s in starts:
                    tuples.append((session_name, float(s), float(s) + wl))

                if len(starts) > 0:
                    right_offset = end - (starts[-1] + wl)
                else:
                    right_offset = end - start - left_offset

                if right_offset + left_offset >= wl:
                    if right_offset > left_offset:
                        tuples.append(
                            (session_name, float(end - wl), float(end))
                        )
                    else:
                        tuples.append(
                            (session_name, float(start), float(start + wl))
                        )
        return tuples

    def __len__(self) -> int:
        total = 0
        for wl in self.window_lengths:
            n_windows = 0
            for intervals in self.sampling_intervals.values():
                for start, end in intervals:
                    if end - start >= wl:
                        n_windows += math.floor((end - start) / wl)
            n_batches = n_windows // self.batch_size
            total += n_batches * self.batch_size
        return total

    def __iter__(self) -> Iterator[list[DatasetIndex]]:
        pools: dict[float, list[tuple[str, float, float]]] = {}
        for wl in self.window_lengths:
            windows = self._generate_windows(wl)
            perm = torch.randperm(
                len(windows), generator=self.generator
            ).tolist()
            pools[wl] = [windows[i] for i in perm]

        all_batches: list[list[DatasetIndex]] = []
        for wl, windows in pools.items():
            bs = self.batch_size
            n_full = len(windows) // bs
            for b in range(n_full):
                batch = [
                    DatasetIndex(name, s, e)
                    for name, s, e in windows[b * bs : (b + 1) * bs]
                ]
                all_batches.append(batch)

            if not self.drop_last and n_full * bs < len(windows):
                batch = [
                    DatasetIndex(name, s, e)
                    for name, s, e in windows[n_full * bs :]
                ]
                all_batches.append(batch)

        batch_perm = torch.randperm(
            len(all_batches), generator=self.generator
        ).tolist()
        for idx in batch_perm:
            yield all_batches[idx]

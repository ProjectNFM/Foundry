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

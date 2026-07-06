"""Fast replacement for RandomFixedWindowSampler.__iter__.

The upstream torch_brain sampler uses torch.arange + .item() per element,
which is extremely slow for datasets with many sessions (O(millions) of
.item() calls). This monkey-patch uses numpy for the arithmetic and defers
DatasetIndex creation to yield-time, providing ~50-100x speedup on datasets
with 1000+ sessions.
"""

import numpy as np
import torch
from torch_brain.datasets import DatasetIndex
from torch_brain.samplers import RandomFixedWindowSampler


def _fast_iter(self):
    """Vectorized __iter__ using numpy; defers DatasetIndex creation."""
    if len(self) == 0:
        raise ValueError("All intervals are too short to sample from.")

    tuples = []
    wl = self.window_length

    for session_name, sampling_intervals in self.sampling_intervals.items():
        for start, end in sampling_intervals:
            interval_length = end - start
            if interval_length < wl:
                if self.drop_short:
                    continue
                else:
                    raise ValueError(
                        f"Interval {(start, end)} is too short to sample from. "
                        f"Minimum length is {wl}."
                    )

            left_offset = torch.rand(1, generator=self.generator).item() * wl

            starts = np.arange(start + left_offset, end, wl, dtype=np.float64)
            valid = starts + wl <= end
            starts = starts[valid]

            for s in starts:
                tuples.append((session_name, s, s + wl))

            if len(starts) > 0:
                right_offset = end - (starts[-1] + wl)
            else:
                right_offset = end - start - left_offset

            if right_offset + left_offset >= wl:
                if right_offset > left_offset:
                    tuples.append((session_name, end - wl, end))
                else:
                    tuples.append((session_name, start, start + wl))

    perm = torch.randperm(len(tuples), generator=self.generator)
    for idx in perm:
        name, s, e = tuples[idx.item()]
        yield DatasetIndex(name, float(s), float(e))


def patch_sampler():
    """Apply the fast __iter__ patch to RandomFixedWindowSampler."""
    RandomFixedWindowSampler.__iter__ = _fast_iter

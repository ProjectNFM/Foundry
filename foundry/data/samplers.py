"""Fast sampler that replaces per-element .item() calls with bulk NumPy generation.

The upstream :class:`~torch_brain.samplers.RandomFixedWindowSampler` creates
``DatasetIndex`` objects eagerly with ``torch.arange`` + ``.item()`` per
element, which is O(millions) of Python↔Torch sync points for datasets with
many sessions.  This subclass performs the arithmetic in NumPy, defers
``DatasetIndex`` creation to yield-time, and converts the final shuffle
permutation once with ``.tolist()``.
"""

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

"""Fast samplers for neural data windowed loading.

Contains :class:`FastRandomFixedWindowSampler` (vectorized drop-in for the
upstream ``RandomFixedWindowSampler``) and :class:`VariableLengthBatchSampler`
(per-batch random window-length selection for multi-length pretraining), plus
the rank-sharding wrapper used for both sampler forms.
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


def _reset_sampler_generator(
    sampler: object, seed: int, seen_objects: set[int]
) -> None:
    """Reset generators in a sampler and any sampler it wraps."""
    if sampler is None or id(sampler) in seen_objects:
        return
    seen_objects.add(id(sampler))

    generator = getattr(sampler, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(seed)

    _reset_sampler_generator(
        getattr(sampler, "sampler", None), seed, seen_objects
    )
    _reset_sampler_generator(
        getattr(getattr(sampler, "dataset", None), "_sampler", None),
        seed,
        seen_objects,
    )


class DeterministicSamplerWrapper(torch.utils.data.Sampler):
    """Reset a wrapped sampler immediately before each iteration.

    Resetting here, rather than in a Lightning epoch callback, is safe with
    worker prefetching because it happens before the DataLoader can request
    the first index.
    """

    def __init__(self, sampler: torch.utils.data.Sampler, seed: int) -> None:
        self.sampler = sampler
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sampler)

    @property
    def num_batches(self) -> int:
        return self.sampler.num_batches

    def __iter__(self):
        _reset_sampler_generator(self.sampler, self.seed, set())
        yield from self.sampler

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes without changing the fixed iteration seed."""
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    """Deterministically shard any sampler's yielded items across ranks.

    Unlike PyTorch's :class:`~torch.utils.data.DistributedSampler`, this wraps
    the output of another sampler, so it supports non-integer ``DatasetIndex``
    values and complete batches from :class:`VariableLengthBatchSampler`.
    Padding follows PyTorch's distributed-sampler behavior and gives every
    rank the same number of items.
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be greater than zero")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas - 1}], got {rank}"
            )
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

    def __len__(self) -> int:
        item_count = getattr(self.sampler, "num_batches", len(self.sampler))
        if self.drop_last:
            return item_count // self.num_replicas
        return math.ceil(item_count / self.num_replicas)

    def __iter__(self):
        items = list(self.sampler)
        per_rank = len(self)
        total_size = per_rank * self.num_replicas

        if self.drop_last:
            items = items[:total_size]
        elif items and len(items) < total_size:
            padding_size = total_size - len(items)
            repeats = math.ceil(padding_size / len(items))
            items += (items * repeats)[:padding_size]

        yield from items[self.rank : total_size : self.num_replicas]

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch changes when the wrapped training sampler uses them."""
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


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

    @property
    def num_batches(self) -> int:
        """Number of complete or partial batches yielded per iteration."""
        total = 0
        for wl in self.window_lengths:
            n_windows = 0
            for intervals in self.sampling_intervals.values():
                for start, end in intervals:
                    if end - start >= wl:
                        n_windows += math.floor((end - start) / wl)
            if self.drop_last:
                total += n_windows // self.batch_size
            else:
                total += math.ceil(n_windows / self.batch_size)
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

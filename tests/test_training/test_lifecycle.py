"""Tests for deterministic validation sampler lifecycle behavior."""

from types import SimpleNamespace

import torch

from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.samplers import (
    DeterministicSamplerWrapper,
    DistributedSamplerWrapper,
    FastRandomFixedWindowSampler,
    VariableLengthBatchSampler,
)
from tests.test_sampler import _collect, _collect_batches, _make_intervals


class _SamplingDataset:
    def __init__(self, root, transform=None, **kwargs):
        self.transform = transform

    def get_sampling_intervals(self, split):
        return _make_intervals(("a", [(0.0, 160.0)]))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def test_fixed_length_validation_repeats_exact_windows():
    sampler = DeterministicSamplerWrapper(
        FastRandomFixedWindowSampler(
            sampling_intervals=_make_intervals(
                ("a", [(0.0, 40.0)]),
                ("b", [(10.0, 70.0)]),
            ),
            window_length=5.0,
            generator=torch.Generator().manual_seed(999),
            drop_short=True,
        ),
        seed=42,
    )

    first = _collect(sampler)
    second = _collect(sampler)

    assert first == second


def test_variable_length_validation_repeats_batches_and_offsets():
    batch_sampler = DeterministicSamplerWrapper(
        VariableLengthBatchSampler(
            sampling_intervals=_make_intervals(
                ("a", [(0.0, 40.0)]),
                ("b", [(10.0, 70.0)]),
            ),
            window_lengths=[2.0, 5.0, 10.0],
            batch_size=4,
            drop_last=False,
            generator=torch.Generator().manual_seed(999),
        ),
        seed=42,
    )

    first = _collect_batches(batch_sampler)
    second = _collect_batches(batch_sampler)

    assert first == second


def test_training_sampler_state_continues_to_advance():
    sampler = FastRandomFixedWindowSampler(
        sampling_intervals=_make_intervals(("a", [(0.0, 100.0)])),
        window_length=5.0,
        generator=torch.Generator().manual_seed(41),
        drop_short=True,
    )
    first = _collect(sampler)
    second = _collect(sampler)

    assert first != second


def test_window_sampler_distributed_shards_repeat_without_overlap():
    samplers = []
    for rank in range(2):
        sampler = DeterministicSamplerWrapper(
            FastRandomFixedWindowSampler(
                sampling_intervals=_make_intervals(("a", [(0.0, 100.0)])),
                window_length=5.0,
                generator=torch.Generator().manual_seed(999),
                drop_short=True,
            ),
            seed=42,
        )
        samplers.append(DistributedSamplerWrapper(sampler, 2, rank))

    def collect_rank_windows():
        return [_collect(sampler) for sampler in samplers]

    first = collect_rank_windows()
    second = collect_rank_windows()

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    unsharded = FastRandomFixedWindowSampler(
        sampling_intervals=_make_intervals(("a", [(0.0, 100.0)])),
        window_length=5.0,
        generator=torch.Generator().manual_seed(42),
        drop_short=True,
    )
    assert sorted(first[0] + first[1]) == sorted(_collect(unsharded))


def test_variable_batches_are_sharded_as_complete_batches():
    samplers = []
    for rank in range(2):
        batch_sampler = DeterministicSamplerWrapper(
            VariableLengthBatchSampler(
                sampling_intervals=_make_intervals(("a", [(0.0, 160.0)])),
                window_lengths=[2.0, 5.0],
                batch_size=4,
                drop_last=True,
                generator=torch.Generator().manual_seed(999),
            ),
            seed=42,
        )
        samplers.append(DistributedSamplerWrapper(batch_sampler, 2, rank))

    def collect_rank_batches():
        return [_collect_batches(sampler) for sampler in samplers]

    first = collect_rank_batches()
    second = collect_rank_batches()

    assert first == second
    assert set(map(str, first[0])).isdisjoint(map(str, first[1]))
    for rank_batches in first:
        for batch in rank_batches:
            assert len(batch) == 4
            assert len({round(end - start, 9) for _, start, end in batch}) == 1


def test_datamodule_wires_determinism_before_distributed_fixed_sharding():
    datamodule = NeuralDataModule(
        dataset_class=_SamplingDataset,
        root="unused",
        batch_size=4,
        sequence_length=5.0,
        seed=41,
    )
    datamodule.setup()
    datamodule._trainer = SimpleNamespace(world_size=2, global_rank=1)

    sampler = datamodule.val_dataloader().sampler

    assert type(sampler).__name__ == "DistributedSamplerWrapper"
    assert type(sampler.sampler).__name__ == "DeterministicSamplerWrapper"
    assert _collect(sampler) == _collect(sampler)


def test_datamodule_wires_determinism_before_variable_batch_sharding():
    datamodule = NeuralDataModule(
        dataset_class=_SamplingDataset,
        root="unused",
        batch_size=4,
        window_lengths=[2.0, 5.0],
        seed=41,
    )
    datamodule.setup()
    datamodule._trainer = SimpleNamespace(world_size=2, global_rank=1)

    batch_sampler = datamodule.val_dataloader().batch_sampler

    assert type(batch_sampler).__name__ == "DistributedSamplerWrapper"
    assert type(batch_sampler.sampler).__name__ == "DeterministicSamplerWrapper"
    assert _collect_batches(batch_sampler) == _collect_batches(batch_sampler)

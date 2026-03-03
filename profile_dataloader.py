"""Profile AJILE12 data loading and tokenization across models.

Measures wall-clock time and memory for each stage of the pipeline:
  1. Dataset setup (HDF5 loading)
  2. Raw sample fetching (no tokenization)
  3. Tokenization per model
  4. Full DataLoader iteration (fetch + tokenize + collate + batch)

Usage:
    uv run python profile_dataloader.py [--num_workers 0] [--batch_size 64]
                                        [--num_batches 50] [--window_length 1.0]
"""

import argparse
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.transforms import Compose

import foundry.data.datasets.modalities  # noqa: F401 – registers modalities
from brainsets.datasets import PetersonBruntonPoseTrajectory2022
from foundry.data.transforms import Patching
from foundry.models import EEGNetEncoder, ShallowConvNet, POYOEEGModel
from foundry.models import TemporalConvAvgPoolClassifier, LinearEmbedding

console = Console()

DATA_ROOT = "./data/processed/"
RECORDING_IDS = ["AJILE12_P01_20000102_ses3_pose_trajectories"]
SPLIT_TYPE = "intrasession"
TASK_TYPE = "active_vs_inactive"
FOLD_NUM = 0

READOUT_SPEC = "ajile_inactive_active"
NUM_CHANNELS = 94
NUM_SAMPLES = 500  # window_length=1.0s @ 500Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    name: str
    times: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.times)) if self.times else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def total(self) -> float:
        return float(np.sum(self.times))


@contextmanager
def track_time(result: TimingResult):
    start = time.perf_counter()
    yield
    result.times.append(time.perf_counter() - start)


def measure_memory(fn):
    """Run *fn*, return (result, peak_memory_MB)."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024


def build_models():
    """Instantiate each model architecture (eval mode, no grad needed)."""
    models = {}

    models["EEGNet"] = EEGNetEncoder(
        readout_specs=[READOUT_SPEC],
        num_channels=NUM_CHANNELS,
        num_samples=NUM_SAMPLES,
    )

    models["ShallowConvNet"] = ShallowConvNet(
        readout_specs=[READOUT_SPEC],
        num_channels=NUM_CHANNELS,
        num_samples=NUM_SAMPLES,
    )

    models["TemporalConv"] = TemporalConvAvgPoolClassifier(
        readout_specs=[READOUT_SPEC],
        num_channels=NUM_CHANNELS,
    )

    poyo = POYOEEGModel(
        input_embedding=LinearEmbedding(embed_dim=128),
        readout_specs=[READOUT_SPEC],
        embed_dim=128,
        sequence_length=1.0,
        latent_step=0.1,
        num_latents_per_step=1,
        depth=2,
    )
    models["POYO-EEG"] = poyo

    for m in models.values():
        m.eval()

    return models


def build_dataset(transform=None):
    return PetersonBruntonPoseTrajectory2022(
        root=DATA_ROOT,
        transform=transform,
        recording_ids=RECORDING_IDS,
        split_type=SPLIT_TYPE,
        task_type=TASK_TYPE,
        fold_num=FOLD_NUM,
    )


def build_dataloader(dataset, window_length, batch_size, num_workers):
    intervals = dataset.get_sampling_intervals(split="train")
    sampler = RandomFixedWindowSampler(
        sampling_intervals=intervals,
        window_length=window_length,
        drop_short=True,
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# Profiling stages
# ---------------------------------------------------------------------------


def profile_dataset_setup():
    console.rule("[bold]1. Dataset Setup")
    result = TimingResult("dataset_setup")
    with track_time(result):
        ds, mem = measure_memory(build_dataset)
    result.peak_memory_mb = mem
    console.print(f"  Time: {result.mean:.3f}s | Peak memory: {mem:.1f} MB")
    return ds, result


def profile_raw_fetch(dataset, window_length, num_samples_to_fetch):
    console.rule("[bold]2. Raw Sample Fetch (no tokenization)")
    intervals = dataset.get_sampling_intervals(split="train")
    sampler = RandomFixedWindowSampler(
        sampling_intervals=intervals,
        window_length=window_length,
        drop_short=True,
        generator=torch.Generator().manual_seed(42),
    )
    result = TimingResult("raw_fetch")
    indices = list(sampler)[:num_samples_to_fetch]
    for idx in indices:
        with track_time(result):
            _ = dataset[idx]

    console.print(
        f"  {len(result.times)} samples | "
        f"Mean: {result.mean * 1000:.2f}ms | "
        f"Std: {result.std * 1000:.2f}ms | "
        f"Total: {result.total:.3f}s"
    )
    return result


def profile_tokenization(dataset, window_length, models, num_samples_to_fetch):
    console.rule("[bold]3. Tokenization (per model)")
    intervals = dataset.get_sampling_intervals(split="train")
    sampler = RandomFixedWindowSampler(
        sampling_intervals=intervals,
        window_length=window_length,
        drop_short=True,
        generator=torch.Generator().manual_seed(42),
    )
    indices = list(sampler)[:num_samples_to_fetch]

    raw_samples = [dataset[idx] for idx in indices]

    patching = Patching(patch_duration=0.1, stride=0.1)

    skipped_models = set()
    results = {}
    for name, model in models.items():
        needs_patching = name == "POYO-EEG"

        if needs_patching and hasattr(model, "initialize_vocabs"):
            sample = raw_samples[0]
            channel_ids = list(sample.channels.ids.astype(str))
            recording_ids = list(dataset.recording_ids)
            model.initialize_vocabs(
                {"session_ids": recording_ids, "channel_ids": channel_ids}
            )

        result = TimingResult(f"tokenize_{name}")
        try:
            for sample in raw_samples:
                with track_time(result):
                    data = patching(sample) if needs_patching else sample
                    _ = model.tokenize(data)
        except Exception as e:
            console.print(f"  [cyan]{name:20s}[/] | [red]SKIPPED[/] — {e}")
            skipped_models.add(name)
            continue

        results[name] = result
        console.print(
            f"  [cyan]{name:20s}[/] | "
            f"Mean: {result.mean * 1000:.2f}ms | "
            f"Std: {result.std * 1000:.2f}ms | "
            f"Total: {result.total:.3f}s"
        )

    return results, skipped_models


def profile_dataloader_iteration(
    models, window_length, batch_size, num_workers, num_batches
):
    console.rule("[bold]4. Full DataLoader Iteration")
    patching = Patching(patch_duration=0.1, stride=0.1)

    results = {}
    for name, model in models.items():
        needs_patching = name == "POYO-EEG"
        transforms = [patching] if needs_patching else []

        ds = build_dataset(transform=Compose(transforms + [model.tokenize]))

        if needs_patching and hasattr(model, "has_lazy_vocabs"):
            if model.has_lazy_vocabs():
                probe = ds[
                    list(
                        RandomFixedWindowSampler(
                            sampling_intervals=ds.get_sampling_intervals(
                                split="train"
                            ),
                            window_length=window_length,
                            drop_short=True,
                            generator=torch.Generator().manual_seed(42),
                        )
                    )[0]
                ]
                channel_ids = list(probe.channels.ids.astype(str))
                recording_ids = list(ds.recording_ids)
                model.initialize_vocabs(
                    {
                        "session_ids": recording_ids,
                        "channel_ids": channel_ids,
                    }
                )

        loader = build_dataloader(ds, window_length, batch_size, num_workers)

        result = TimingResult(f"dataloader_{name}")
        batch_count = 0
        for batch in loader:
            with track_time(result):
                pass  # batch already materialized by iterator
            batch_count += 1
            if batch_count >= num_batches:
                break

        # Re-measure properly: time the full iteration including fetch
        ds2 = build_dataset(transform=Compose(transforms + [model.tokenize]))
        loader2 = build_dataloader(ds2, window_length, batch_size, num_workers)
        result = TimingResult(f"dataloader_{name}")
        batch_count = 0
        t_start = time.perf_counter()
        for batch in loader2:
            result.times.append(time.perf_counter() - t_start)
            batch_count += 1
            if batch_count >= num_batches:
                break
            t_start = time.perf_counter()

        results[name] = result
        throughput = batch_size * batch_count / result.total
        console.print(
            f"  [cyan]{name:20s}[/] | "
            f"Mean: {result.mean * 1000:.2f}ms/batch | "
            f"Std: {result.std * 1000:.2f}ms | "
            f"Total: {result.total:.3f}s | "
            f"Throughput: {throughput:.0f} samples/s"
        )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(all_results, batch_size, num_workers):
    console.rule("[bold green]Summary")
    console.print(f"  batch_size={batch_size}, num_workers={num_workers}\n")

    table = Table(title="Profiling Results")
    table.add_column("Stage", style="bold")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Std (ms)", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")

    for r in all_results:
        table.add_row(
            r.name,
            f"{r.mean * 1000:.2f}",
            f"{r.std * 1000:.2f}",
            f"{r.median * 1000:.2f}",
            f"{r.total:.3f}",
            f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb else "-",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Profile AJILE12 data loading pipeline"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--window_length", type=float, default=1.0)
    args = parser.parse_args()

    console.rule("[bold magenta]AJILE12 DataLoader Profiler")
    console.print(f"  batch_size={args.batch_size}")
    console.print(f"  num_workers={args.num_workers}")
    console.print(f"  num_batches={args.num_batches}")
    console.print(f"  num_samples={args.num_samples}")
    console.print(f"  window_length={args.window_length}s\n")

    models = build_models()

    ds, r_setup = profile_dataset_setup()
    r_fetch = profile_raw_fetch(ds, args.window_length, args.num_samples)
    tok_results = profile_tokenization(
        ds, args.window_length, models, args.num_samples
    )
    dl_results = profile_dataloader_iteration(
        models,
        args.window_length,
        args.batch_size,
        args.num_workers,
        args.num_batches,
    )

    all_results = [r_setup, r_fetch]
    all_results.extend(tok_results.values())
    all_results.extend(dl_results.values())
    print_summary(all_results, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()

"""Diagnose DataLoader worker overhead.

Tests different worker counts and keep_files_open settings to isolate
the HDF5 fork-safety / I/O contention issue.

Usage:
    uv run python profile_dataloader.py 'experiment=pretrain/openneuro_eeg'
"""

import logging
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_BATCHES = 10
BATCH_SIZE = 128


def time_dataloader(dataloader, num_batches, label):
    """Time the dataloader, per-batch."""
    times = []
    t_total = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        t_now = time.perf_counter()
        t_prev = t_now
        if i > 0:
            times.append(t_now - t_prev)

    elapsed = time.perf_counter() - t_total
    if times:
        avg_ms = 1000 * sum(times) / len(times)
        min_ms = 1000 * min(times)
        max_ms = 1000 * max(times)
    else:
        avg_ms = min_ms = max_ms = 0

    print(
        f"  {label:<50} total={elapsed:.2f}s  "
        f"avg={avg_ms:.0f}ms  min={min_ms:.0f}ms  max={max_ms:.0f}ms  "
        f"({num_batches} batches)"
    )
    return elapsed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    from foundry.data.utils import (
        get_max_channels,
        get_sampling_rate,
        get_session_configs,
    )

    torch.set_float32_matmul_precision("high")

    num_channels = OmegaConf.select(
        cfg, "hyperparameters.num_channels", default=None
    )
    sampling_rate = OmegaConf.select(
        cfg, "hyperparameters.sampling_rate", default=None
    )
    session_configs = OmegaConf.select(
        cfg, "hyperparameters.session_configs", default=None
    )

    dm_probe = instantiate(cfg.data, tokenizer=None)
    dm_probe.setup("fit")

    if session_configs is None:
        OmegaConf.update(
            cfg,
            "hyperparameters.session_configs",
            get_session_configs(dm_probe.dataset),
            force_add=True,
        )
    if num_channels is None:
        OmegaConf.update(
            cfg,
            "hyperparameters.num_channels",
            get_max_channels(dm_probe.dataset),
            force_add=True,
        )
    if sampling_rate is None:
        OmegaConf.update(
            cfg,
            "hyperparameters.sampling_rate",
            get_sampling_rate(dm_probe.dataset),
            force_add=True,
        )

    model = instantiate(cfg.model)
    tokenizer_fn = model.tokenize

    def build_datamodule(keep_files_open):
        """Build a datamodule with specific keep_files_open setting."""
        data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
        data_cfg["keep_files_open"] = keep_files_open
        data_cfg = OmegaConf.create(data_cfg)
        dm = instantiate(data_cfg, tokenizer=tokenizer_fn)
        dm.setup("fit")
        vocab_info = {}
        dataset = dm.dataset
        for method_name, key in [
            ("get_recording_ids", "session_ids"),
            ("get_channel_ids", "channel_ids"),
        ]:
            if hasattr(dm, method_name):
                vocab_info[key] = getattr(dm, method_name)()
            elif dataset is not None and hasattr(dataset, method_name):
                vocab_info[key] = getattr(dataset, method_name)()
        model.initialize_vocabs(vocab_info)
        return dm

    def build_dataloader(dm, num_workers, pin_memory=False):
        train_intervals = dm.dataset.get_sampling_intervals(split="train")
        sampler = RandomFixedWindowSampler(
            sampling_intervals=train_intervals,
            window_length=cfg.hyperparameters.sequence_length,
            drop_short=True,
            generator=torch.Generator().manual_seed(42),
        )
        return DataLoader(
            dm.dataset,
            sampler=sampler,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )

    print(f"\n{'=' * 70}")
    print("DATALOADER WORKER OVERHEAD DIAGNOSIS")
    print(f"{'=' * 70}")
    print(f"Batch size: {BATCH_SIZE}, Batches: {NUM_BATCHES}")

    # Test 1: keep_files_open=True (current default)
    print("\n--- keep_files_open=True ---")
    dm_open = build_datamodule(keep_files_open=True)
    for nw in [0, 1, 2, 4, 8]:
        dl = build_dataloader(dm_open, num_workers=nw)
        time_dataloader(dl, NUM_BATCHES, f"num_workers={nw}")
        del dl

    # Test 2: keep_files_open=False
    print("\n--- keep_files_open=False ---")
    dm_closed = build_datamodule(keep_files_open=False)
    for nw in [0, 1, 2, 4, 8]:
        dl = build_dataloader(dm_closed, num_workers=nw)
        time_dataloader(dl, NUM_BATCHES, f"num_workers={nw}")
        del dl

    # Test 3: keep_files_open=True + pin_memory
    print("\n--- keep_files_open=True + pin_memory=True ---")
    for nw in [0, 4]:
        dl = build_dataloader(dm_open, num_workers=nw, pin_memory=True)
        time_dataloader(dl, NUM_BATCHES, f"num_workers={nw}, pin_memory=True")
        del dl


if __name__ == "__main__":
    register_resolvers()
    main()

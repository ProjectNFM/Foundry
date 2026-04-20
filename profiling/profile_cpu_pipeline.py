"""Fine-grained CPU profiling of the data loading pipeline.

Measures per-function wall-clock time for every stage a single sample
passes through, from dataset.__getitem__ to collate output.

Usage:
    uv run python profile_cpu_pipeline.py \
        'experiment=pretrain/openneuro_eeg'
"""

import logging
import time
from collections import defaultdict

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_SAMPLES = 50


class TimingContext:
    """Accumulates wall-clock timings for named code sections."""

    def __init__(self):
        self.timings = defaultdict(list)
        self._stack = []

    def __call__(self, name):
        return _TimingBlock(self, name)

    def summary(self):
        print(f"\n{'=' * 70}")
        print("CPU PIPELINE PER-SAMPLE BREAKDOWN")
        print(f"{'=' * 70}")
        total = 0
        rows = []
        for name, vals in self.timings.items():
            mean_ms = 1000 * sum(vals) / len(vals)
            std_ms = (
                1000
                * (
                    sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                    / max(len(vals) - 1, 1)
                )
                ** 0.5
            )
            rows.append((name, mean_ms, std_ms, len(vals)))
            if not name.startswith("  "):
                total += mean_ms

        print(
            f"\n{'Phase':<45} {'Mean (ms)':>10} {'Std (ms)':>10} {'Calls':>8}"
        )
        print("-" * 78)
        for name, mean_ms, std_ms, count in rows:
            print(f"  {name:<43} {mean_ms:>8.3f}ms {std_ms:>8.3f}ms {count:>8}")
        print("-" * 78)
        print(f"  {'TOTAL (top-level sum)':<43} {total:>8.3f}ms")


class _TimingBlock:
    def __init__(self, ctx, name):
        self.ctx = ctx
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.t0
        self.ctx.timings[self.name].append(elapsed)


def profile_single_sample(dataset, index, tokenize_fn, timer):
    """Profile a single sample through the full CPU pipeline."""

    with timer("1. get_recording (HDF5 load + deepcopy)"):
        data = dataset.get_recording(index.recording_id, index._namespace)

    with timer("2. data.slice (temporal slicing)"):
        sample = data.slice(index.start, index.end)

    with timer("3. tokenize (total)"):
        result = tokenize_fn(sample)

    return result


def profile_tokenize_detailed(model, data_sample, timer):
    """Break down tokenize into sub-steps with timing."""

    with timer("  3a. resolve_signal_source"):
        signal_source, default_type = model._resolve_signal_source(data_sample)

    with timer("  3b. channel filtering (numpy)"):
        modality_field = (
            data_sample.channels.type.astype(str)
            if hasattr(data_sample.channels, "type")
            else np.array([default_type] * len(data_sample.channels)).astype(
                str
            )
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(model.SUPPORTED_MODALITIES)
        )
        channel_ids = data_sample.channels.id[modality_mask].astype(str)

    with timer("  3c. channel_emb.tokenizer (vocab lookup)"):
        channel_tokens = np.asarray(model.channel_emb.tokenizer(channel_ids))

    with timer("  3d. sampling_rate inference"):
        sample_deltas = np.diff(signal_source.timestamps)
        sampling_rate = 1.0 / float(sample_deltas[0])

    with timer("  3e. pretokenize (total)"):
        pretokenized = model.tokenizer.pretokenize(
            signal=signal_source.signal[:, modality_mask],
            channel_tokens=channel_tokens,
            sampling_rate=sampling_rate,
            sequence_length=model.sequence_length,
        )

    with timer("  3f. session_emb.tokenizer (vocab lookup)"):
        input_session_index = model.session_emb.tokenizer(
            data_sample.session.id
        )

    with timer("  3g. output query assembly + pad8/chain"):
        from torch_brain.data import chain, pad8

        pretokenized["input_session_ids"] = str(data_sample.session.id)
        input_timestamps = pretokenized.pop("input_timestamps")
        masking_mask = pretokenized.pop("masking_mask", None)
        reconstruction_targets = pretokenized.pop(
            "reconstruction_targets", None
        )
        masked_timestamps = pretokenized.pop("masked_timestamps", None)

        output_timestamps = torch.tensor([], dtype=torch.float32)
        output_values = {}
        output_task_index = torch.tensor([], dtype=torch.long)
        output_weights = {}
        output_eval_mask = {}
        output_session_index = np.array([], dtype=np.int64)

        if (
            model.reconstruction_head is not None
            and masked_timestamps is not None
        ):
            n_masked = masked_timestamps.shape[0]
            recon_timestamps = torch.zeros(n_masked, dtype=torch.float32)
            from foundry.models.poyo_eeg import RECON_DECODER_ID

            recon_decoder_index = torch.full(
                (n_masked,), RECON_DECODER_ID, dtype=torch.long
            )
            recon_session_index = np.full(n_masked, input_session_index)

            output_timestamps = torch.cat([output_timestamps, recon_timestamps])
            output_task_index = torch.cat(
                [output_task_index, recon_decoder_index]
            )
            output_session_index = np.concatenate(
                [output_session_index, recon_session_index]
            )

        result = {
            **pretokenized,
            "input_timestamps": input_timestamps,
            "input_session_index": input_session_index,
            "latent_index": model._latent_index,
            "latent_timestamps": model._latent_timestamps,
            "output_session_index": pad8(output_session_index),
            "output_timestamps": pad8(output_timestamps),
            "output_decoder_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data_sample.session.id,
            "absolute_start": data_sample.absolute_start,
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }
        if masking_mask is not None:
            result["masking_mask"] = masking_mask
        if reconstruction_targets is not None:
            result["reconstruction_targets"] = reconstruction_targets

    return result


def profile_pretokenize_detailed(
    tokenizer, signal, channel_tokens, sampling_rate, sequence_length, timer
):
    """Break down pretokenize into sub-steps."""

    with timer("    3e-i. channel_strategy.prepare_pretokenize"):
        result = tokenizer.channel_strategy.prepare_pretokenize(
            signal,
            channel_tokens,
            sampling_rate,
        )

    with timer("    3e-ii. patch timestamp computation"):
        padded_signal = result["input_values"]
        if tokenizer._do_patching:
            num_samples = signal.shape[0]
            patch_samples = max(
                1, round(tokenizer.patch_duration * sampling_rate)
            )
            stride_samples = max(1, round(tokenizer.stride * sampling_rate))
            num_patches = (
                max(1, (num_samples - patch_samples) // stride_samples + 1)
                if num_samples > patch_samples
                else 1
            )

            from foundry.models.embeddings.patching import (
                compute_patch_timestamps,
            )

            patch_timestamps = compute_patch_timestamps(
                start_time=0.0,
                num_patches=num_patches,
                patch_duration=tokenizer.patch_duration,
                stride=tokenizer.stride,
            )
            result["input_timestamps"] = patch_timestamps

    with timer("    3e-iii. masking + target extraction"):
        if tokenizer.masking is not None:
            tokenizer._add_masking_fields(
                result,
                padded_signal,
                signal,
                sampling_rate,
                sequence_length,
            )

    return result


def profile_collate(samples, timer):
    """Profile the collate function."""
    from torch_brain.data import collate

    with timer("4. collate"):
        batch = collate(samples)
    return batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
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
        session_configs = get_session_configs(dm_probe.dataset)
        OmegaConf.update(
            cfg,
            "hyperparameters.session_configs",
            session_configs,
            force_add=True,
        )
    if num_channels is None:
        num_channels = get_max_channels(dm_probe.dataset)
        OmegaConf.update(
            cfg, "hyperparameters.num_channels", num_channels, force_add=True
        )
    if sampling_rate is None:
        sampling_rate = get_sampling_rate(dm_probe.dataset)
        OmegaConf.update(
            cfg, "hyperparameters.sampling_rate", sampling_rate, force_add=True
        )

    print(
        f"Dataset: num_channels={num_channels}, sampling_rate={sampling_rate}"
    )
    print(f"Batch size: {cfg.hyperparameters.batch_size}")
    print(f"Sequence length: {cfg.hyperparameters.sequence_length}s")
    print(f"Patch duration: {cfg.hyperparameters.patch_duration}s")

    model = instantiate(cfg.model)
    tokenizer_fn = model.tokenize
    datamodule = instantiate(cfg.data, tokenizer=tokenizer_fn)
    datamodule.setup("fit")

    vocab_info = {}
    dataset = datamodule.dataset
    for method_name, key in [
        ("get_recording_ids", "session_ids"),
        ("get_channel_ids", "channel_ids"),
    ]:
        if hasattr(datamodule, method_name):
            vocab_info[key] = getattr(datamodule, method_name)()
        elif dataset is not None and hasattr(dataset, method_name):
            vocab_info[key] = getattr(dataset, method_name)()
    model.initialize_vocabs(vocab_info)

    # Get sampling intervals and create sample indices
    from torch_brain.data.sampler import RandomFixedWindowSampler

    train_intervals = dataset.get_sampling_intervals(split="train")
    sampler = RandomFixedWindowSampler(
        sampling_intervals=train_intervals,
        window_length=cfg.hyperparameters.sequence_length,
        drop_short=True,
        generator=torch.Generator().manual_seed(42),
    )

    indices = []
    for i, idx in enumerate(sampler):
        if i >= NUM_SAMPLES:
            break
        indices.append(idx)

    print(f"\nProfiling {len(indices)} samples through the CPU pipeline...")

    # ---- Phase 1: Coarse-grained (end-to-end per sample) ----
    timer_coarse = TimingContext()

    for idx in indices[:5]:
        profile_single_sample(dataset, idx, tokenizer_fn, timer_coarse)

    timer_coarse.summary()

    # ---- Phase 2: Fine-grained tokenize breakdown ----
    timer_fine = TimingContext()

    for idx in indices:
        data = dataset.get_recording(idx.recording_id, idx._namespace)
        sample = data.slice(idx.start, idx.end)

        with timer_fine("1. get_recording"):
            data2 = dataset.get_recording(idx.recording_id, idx._namespace)
        with timer_fine("2. data.slice"):
            data2.slice(idx.start, idx.end)

        profile_tokenize_detailed(model, sample, timer_fine)

    timer_fine.summary()

    # ---- Phase 3: Pretokenize sub-breakdown ----
    timer_pretok = TimingContext()

    for idx in indices:
        data = dataset.get_recording(idx.recording_id, idx._namespace)
        sample = data.slice(idx.start, idx.end)

        signal_source, default_type = model._resolve_signal_source(sample)
        modality_field = (
            sample.channels.type.astype(str)
            if hasattr(sample.channels, "type")
            else np.array([default_type] * len(sample.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(model.SUPPORTED_MODALITIES)
        )
        channel_ids = sample.channels.id[modality_mask].astype(str)
        channel_tokens = np.asarray(model.channel_emb.tokenizer(channel_ids))
        sample_deltas = np.diff(signal_source.timestamps)
        sr = 1.0 / float(sample_deltas[0])

        profile_pretokenize_detailed(
            model.tokenizer,
            signal_source.signal[:, modality_mask],
            channel_tokens,
            sr,
            model.sequence_length,
            timer_pretok,
        )

    timer_pretok.summary()

    # ---- Phase 4: Collate profiling ----
    timer_collate = TimingContext()

    samples_for_collate = []
    for idx in indices[: cfg.hyperparameters.batch_size]:
        data = dataset.get_recording(idx.recording_id, idx._namespace)
        sample = data.slice(idx.start, idx.end)
        samples_for_collate.append(tokenizer_fn(sample))

    for _ in range(5):
        profile_collate(samples_for_collate, timer_collate)

    timer_collate.summary()

    # ---- Phase 5: Full batch assembly timing ----
    print(f"\n{'=' * 70}")
    print("FULL BATCH ASSEMBLY TIMING")
    print(f"{'=' * 70}")

    batch_sizes_to_test = [32, 64, 128, 256]
    for bs in batch_sizes_to_test:
        if bs > len(indices):
            break
        t0 = time.perf_counter()
        batch_samples = []
        for idx in indices[:bs]:
            data = dataset.get_recording(idx.recording_id, idx._namespace)
            sample = data.slice(idx.start, idx.end)
            batch_samples.append(tokenizer_fn(sample))
        t_transform = time.perf_counter() - t0

        from torch_brain.data import collate

        t0 = time.perf_counter()
        collate(batch_samples)
        t_collate = time.perf_counter() - t0

        total = t_transform + t_collate
        print(
            f"  batch_size={bs:>4}: transform={1000 * t_transform:.1f}ms  "
            f"collate={1000 * t_collate:.1f}ms  total={1000 * total:.1f}ms  "
            f"per_sample={1000 * t_transform / bs:.2f}ms"
        )


if __name__ == "__main__":
    import time

    register_resolvers()
    main()

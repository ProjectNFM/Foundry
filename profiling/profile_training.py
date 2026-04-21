"""Profile the pretraining pipeline to identify bottlenecks.

Measures wall-clock time for:
  - Data loading (CPU: dataset fetch + tokenize + collate)
  - Batch transfer to device
  - Forward pass (tokenizer + backbone + reconstruction head)
  - Loss computation
  - Backward pass
  - Optimizer step

Also runs PyTorch's autograd profiler for GPU kernel-level detail.

Usage:
    uv run python profile_training.py \
        +experiment=pretrain/openneuro_eeg
"""

import logging
import os
import time
from collections import defaultdict

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_WARMUP_BATCHES = 3
NUM_PROFILE_BATCHES = 20


def _time_dataloader(dataloader, num_batches: int) -> tuple[list[dict], float]:
    """Time the data loading pipeline (CPU-side tokenize + collate)."""
    batches = []
    t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batches.append(batch)
    elapsed = time.perf_counter() - t0
    return batches, elapsed


def _profile_step(batch, lightning_module, device, optimizer):
    """Run one full train step and return per-phase timings."""
    timings = {}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = lightning_module.transfer_batch_to_device(batch, device, 0)
    torch.cuda.synchronize()
    timings["transfer"] = time.perf_counter() - t0

    recon_targets = batch.pop("reconstruction_targets")
    batch.pop("target_values", None)
    batch.pop("target_weights", None)
    batch.pop("session_id", None)
    batch.pop("absolute_start", None)
    batch.pop("eval_mask", None)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = lightning_module.model(**batch, unpack_output=False)
    torch.cuda.synchronize()
    timings["forward"] = time.perf_counter() - t0

    predictions = outputs["reconstruction"]
    masking_mask = batch.get("masking_mask")
    if masking_mask is not None:
        targets = recon_targets[masking_mask]
    else:
        from foundry.models.poyo_eeg import RECON_DECODER_ID

        decoder_idx = batch["output_decoder_index"]
        per_batch = []
        for bi in range(decoder_idx.shape[0]):
            n = (decoder_idx[bi] == RECON_DECODER_ID).sum().item()
            per_batch.append(recon_targets[bi, :n])
        targets = torch.cat(per_batch, dim=0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = lightning_module.loss_fn(predictions, targets)
    torch.cuda.synchronize()
    timings["loss"] = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    timings["backward"] = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    timings["optimizer_step"] = time.perf_counter() - t0

    return timings, loss.item()


def _print_summary(
    all_timings: dict[str, list[float]], data_time_per_batch: float
):
    """Print a formatted profiling summary."""
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    phases = [
        "data_loading",
        "transfer",
        "forward",
        "loss",
        "backward",
        "optimizer_step",
    ]
    phase_labels = {
        "data_loading": "Data Loading (CPU)",
        "transfer": "Batch Transfer to GPU",
        "forward": "Forward Pass",
        "loss": "Loss Computation",
        "backward": "Backward Pass",
        "optimizer_step": "Optimizer Step",
    }

    total_time = sum(
        sum(v) / len(v) for k, v in all_timings.items() if k in phases
    )

    print(f"\n{'Phase':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'% Total':>10}")
    print("-" * 70)

    for phase in phases:
        vals = all_timings.get(phase, [])
        if not vals:
            continue
        mean_ms = 1000 * sum(vals) / len(vals)
        std_ms = (
            1000
            * (
                sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                / max(len(vals) - 1, 1)
            )
            ** 0.5
        )
        mean_s = sum(vals) / len(vals)
        pct = 100 * mean_s / total_time if total_time > 0 else 0
        print(
            f"  {phase_labels.get(phase, phase):<28} {mean_ms:>10.2f}ms {std_ms:>10.2f}ms {pct:>8.1f}%"
        )

    total_ms = 1000 * total_time
    print("-" * 70)
    print(f"  {'TOTAL per step':<28} {total_ms:>10.2f}ms")
    throughput = 1.0 / total_time if total_time > 0 else 0
    print(f"\n  Throughput: {throughput:.2f} steps/sec")

    print("\n" + "=" * 70)
    print("MEMORY USAGE")
    print("=" * 70)
    if torch.cuda.is_available():
        print(
            f"  Peak allocated:  {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        print(
            f"  Peak reserved:   {torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
        )
        print(
            f"  Current alloc:   {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )


def _run_torch_profiler(batch, lightning_module, device, optimizer):
    """Run PyTorch profiler for kernel-level GPU analysis."""
    from torch.profiler import profile, ProfilerActivity, schedule

    def trace_handler(p):
        output = p.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        print("\n" + "=" * 70)
        print("PYTORCH PROFILER - Top 30 CUDA Operations")
        print("=" * 70)
        print(output)

        output_cpu = p.key_averages().table(
            sort_by="cpu_time_total", row_limit=20
        )
        print("\n" + "=" * 70)
        print("PYTORCH PROFILER - Top 20 CPU Operations")
        print("=" * 70)
        print(output_cpu)

        trace_path = "./profiling_output/trace.json"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        p.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to {trace_path}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _step in range(6):
            batch_copy = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_copy = lightning_module.transfer_batch_to_device(
                batch_copy, device, 0
            )

            recon_targets = batch_copy.pop("reconstruction_targets")
            batch_copy.pop("target_values", None)
            batch_copy.pop("target_weights", None)
            batch_copy.pop("session_id", None)
            batch_copy.pop("absolute_start", None)
            batch_copy.pop("eval_mask", None)

            outputs = lightning_module.model(**batch_copy, unpack_output=False)
            predictions = outputs["reconstruction"]
            masking_mask = batch_copy.get("masking_mask")
            if masking_mask is not None:
                targets = recon_targets[masking_mask]
            else:
                targets = recon_targets.reshape(-1, recon_targets.shape[-1])

            loss = lightning_module.loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            prof.step()


def _print_batch_shapes(batch):
    """Print tensor shapes from a batch for understanding dimensions."""
    print("\n" + "=" * 70)
    print("BATCH TENSOR SHAPES")
    print("=" * 70)
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k:<35} {str(v.shape):<25} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:<35} dict with {len(v)} keys")
        else:
            print(f"  {k:<35} {type(v).__name__}")


def _print_model_summary(model):
    """Print model parameter counts per component."""
    print("\n" + "=" * 70)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 70)

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    total = count_params(model)
    print(f"  {'Total':<40} {total:>12,} params")
    print()

    for name, child in model.named_children():
        n = count_params(child)
        pct = 100 * n / total if total > 0 else 0
        print(f"  {name:<40} {n:>12,} ({pct:.1f}%)")


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    from foundry.data.utils import (
        get_max_channels,
        get_sampling_rate,
        get_session_configs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    torch.set_float32_matmul_precision(
        str(
            OmegaConf.select(
                cfg, "run.float32_matmul_precision", default="high"
            )
        )
    )

    # Auto-populate data-driven hyperparams
    session_configs = OmegaConf.select(
        cfg, "hyperparameters.session_configs", default=None
    )
    num_channels = OmegaConf.select(
        cfg, "hyperparameters.num_channels", default=None
    )
    sampling_rate = OmegaConf.select(
        cfg, "hyperparameters.sampling_rate", default=None
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
        f"\nDataset: num_channels={num_channels}, sampling_rate={sampling_rate}"
    )
    print(f"Batch size: {cfg.hyperparameters.batch_size}")
    print(f"Sequence length: {cfg.hyperparameters.sequence_length}s")
    print(f"Patch duration: {cfg.hyperparameters.patch_duration}s")
    print(f"Num workers: {cfg.hyperparameters.num_workers}")

    model = instantiate(cfg.model)
    tokenizer = model.tokenize
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
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

    _print_model_summary(model)

    lightning_module = instantiate(cfg.module, model=model)
    model = model.to(device)
    lightning_module = lightning_module.to(device)
    lightning_module.train()

    optimizer = torch.optim.AdamW(
        lightning_module.parameters(),
        lr=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay,
    )

    # --- Phase 1: Data loading profiling ---
    print("\n" + "=" * 70)
    print("PHASE 1: DATA LOADING PROFILING")
    print("=" * 70)

    train_dl = datamodule.train_dataloader()

    total_needed = NUM_WARMUP_BATCHES + NUM_PROFILE_BATCHES + 1
    print(f"Loading {total_needed} batches for profiling...")
    all_batches, total_data_time = _time_dataloader(train_dl, total_needed)
    print(f"  Loaded {len(all_batches)} batches in {total_data_time:.2f}s")
    if len(all_batches) > 0:
        data_time_per_batch = total_data_time / len(all_batches)
        print(f"  Avg time per batch: {1000 * data_time_per_batch:.2f}ms")
    else:
        data_time_per_batch = 0
        print("  WARNING: No batches loaded!")
        return

    _print_batch_shapes(all_batches[0])

    # --- Phase 2: GPU step profiling ---
    print("\n" + "=" * 70)
    print("PHASE 2: GPU STEP PROFILING")
    print("=" * 70)

    print(f"Warming up with {NUM_WARMUP_BATCHES} batches...")
    for i in range(NUM_WARMUP_BATCHES):
        batch = all_batches[i]
        _profile_step(batch, lightning_module, device, optimizer)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_timings = defaultdict(list)
    losses = []

    print(f"Profiling {NUM_PROFILE_BATCHES} batches...")
    for i in range(NUM_PROFILE_BATCHES):
        batch_idx = NUM_WARMUP_BATCHES + i
        if batch_idx >= len(all_batches):
            break
        batch = all_batches[batch_idx]
        timings, loss_val = _profile_step(
            batch, lightning_module, device, optimizer
        )
        for phase, t in timings.items():
            all_timings[phase].append(t)
        all_timings["data_loading"].append(data_time_per_batch)
        losses.append(loss_val)

    _print_summary(all_timings, data_time_per_batch)

    print(f"\n  Avg loss: {sum(losses) / len(losses):.6f}")

    # --- Phase 3: PyTorch profiler ---
    print("\n" + "=" * 70)
    print("PHASE 3: PYTORCH AUTOGRAD PROFILER")
    print("=" * 70)

    profiler_batch = all_batches[0]
    _run_torch_profiler(profiler_batch, lightning_module, device, optimizer)

    # --- Phase 4: Optimization opportunities ---
    print("\n" + "=" * 70)
    print("CONFIGURATION ANALYSIS")
    print("=" * 70)

    precision = OmegaConf.select(cfg, "trainer.precision", default="32-true")
    print(f"  Precision: {precision}")
    if "32" in str(precision):
        print(
            "  -> SUGGESTION: Use mixed precision (16-mixed or bf16-mixed) for ~2x speedup on L40S"
        )

    pin_memory = OmegaConf.select(cfg, "data.pin_memory", default=False)
    print(f"  Pin memory: {pin_memory}")
    if not pin_memory:
        print(
            "  -> SUGGESTION: Enable pin_memory=true for faster CPU->GPU transfers"
        )

    nw = cfg.hyperparameters.num_workers
    print(f"  Num workers: {nw}")
    cpu_count = os.cpu_count() or 1
    print(f"  CPU count: {cpu_count}")
    if nw < min(8, cpu_count):
        print(
            f"  -> SUGGESTION: Increase num_workers (try {min(8, cpu_count)})"
        )

    compile_enabled = OmegaConf.select(cfg, "run.compile", default=False)
    print(f"  torch.compile: {compile_enabled}")
    if not compile_enabled:
        print("  -> SUGGESTION: Try torch.compile(model) for kernel fusion")

    bs = cfg.hyperparameters.batch_size
    print(f"  Batch size: {bs}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak_used = torch.cuda.max_memory_allocated() / 1e9
        utilization = peak_used / gpu_mem * 100
        print(
            f"  GPU memory utilization: {utilization:.1f}% ({peak_used:.1f}/{gpu_mem:.1f} GB)"
        )
        if utilization < 50:
            print(
                "  -> SUGGESTION: Increase batch_size (GPU memory underutilized)"
            )

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    register_resolvers()
    main()

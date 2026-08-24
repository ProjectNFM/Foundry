"""Synthetic HERO forward/backward profiling; never launches training.

The default matrix has two controlled sweeps: geometrically increasing
durations at fixed channel count and 2/16/64 channels at fixed duration.
Results are emitted as JSON Lines so they can be checked into a validation
report without making timing assertions part of the unit-test suite.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from foundry.models.hero import HEROModel


def _csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return values


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _token_accounting(seconds: int) -> dict[str, int]:
    fine = 128 * seconds
    mid = fine // 4
    coarse = mid // 4
    return {
        "fine_tokens": fine,
        "mid_tokens": mid,
        "coarse_tokens": coarse,
        "temporal_tokens": fine + mid + coarse,
    }


def _hero_flop_accounting(
    *,
    seconds: int,
    channels: int,
    embed_dim: int,
    channel_encoder_layers: int = 3,
    channel_kernel: int = 7,
    spatial_slots: int = 8,
    temporal_slots: int = 4,
    temporal_neighborhood: int = 8,
    lowpass_taps: int = 33,
    local_blocks: int = 2,
    local_window: int = 32,
) -> dict[str, int | float]:
    """Count dense multiply-add FLOPs for the documented reference path.

    LayerNorm, GELU, sigmoid, mask/index operations, timestamp bias, and
    overlap-search bookkeeping are intentionally excluded. A multiply-add is
    two FLOPs. The split makes explicit that only the pre-fusion term depends
    on channel count.
    """
    tokens = _token_accounting(seconds)
    fine = tokens["fine_tokens"]
    mid = tokens["mid_tokens"]
    coarse = tokens["coarse_tokens"]
    d = embed_dim

    first_conv = 2 * channels * fine * d * channel_kernel
    later_convs = (
        2
        * channels
        * fine
        * d
        * d
        * channel_kernel
        * (channel_encoder_layers - 1)
    )
    spatial_kv = 4 * fine * channels * d * d
    spatial_attention = 4 * fine * spatial_slots * channels * d
    spatial_projection = 4 * fine * spatial_slots * d * d
    prefusion = first_conv + later_convs + spatial_kv + spatial_attention
    prefusion += spatial_projection

    local = 0
    local_attention_pairs = 0
    for length in (fine, mid, coarse):
        local += local_blocks * (
            24 * length * d * d + 4 * length * min(local_window, length) * d
        )
        local_attention_pairs += (
            local_blocks * length * min(local_window, length)
        )

    reductions = 0
    temporal_slot_pairs = 0
    for input_length, output_length in ((fine, mid), (mid, coarse)):
        reductions += 2 * input_length * d * lowpass_taps
        reductions += 4 * output_length * temporal_neighborhood * d * d
        reductions += (
            4 * output_length * temporal_slots * temporal_neighborhood * d
        )
        reductions += 4 * output_length * temporal_slots * d * d
        temporal_slot_pairs += (
            output_length * temporal_slots * temporal_neighborhood
        )

    # Gate and value projections for coarse->mid and mid->fine. The bounded
    # overlap-weighted gathers are omitted from FLOPs but counted as pairs.
    alignment_projections = 4 * (fine + mid) * d * d
    temporal = local + reductions + alignment_projections
    temporal_tokens = tokens["temporal_tokens"]
    return {
        "estimated_prefusion_flops": prefusion,
        "estimated_temporal_flops": temporal,
        "estimated_total_flops": prefusion + temporal,
        "estimated_temporal_flops_per_temporal_token": temporal
        / temporal_tokens,
        "hero_local_attention_pairs": local_attention_pairs,
        "hero_temporal_slot_pairs": temporal_slot_pairs,
    }


def _flat_poyo_accounting(
    *,
    seconds: int,
    channels: int,
    token_rate: int,
    latent_step: float,
    latents_per_step: int,
    depth: int,
) -> dict[str, int | float]:
    input_tokens = seconds * token_rate * channels
    latent_tokens = round(seconds / latent_step) * latents_per_step
    return {
        "flat_poyo_input_tokens": input_tokens,
        "flat_poyo_latent_tokens": latent_tokens,
        "flat_poyo_cross_attention_pairs": input_tokens * latent_tokens,
        "flat_poyo_self_attention_pairs": depth * latent_tokens**2,
        "flat_poyo_input_to_latent_ratio": input_tokens / latent_tokens,
    }


def _log_log_slope(records: list[dict], x_key: str, y_key: str) -> float:
    x = [math.log(float(record[x_key])) for record in records]
    y = [math.log(float(record[y_key])) for record in records]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        )
        / denominator
    )


def _write_jsonl(records: list[dict], output: Path | None) -> None:
    text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    print(text, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds", type=_csv_ints, default=_csv_ints("1,2,4,8,16,30")
    )
    parser.add_argument(
        "--channels", type=_csv_ints, default=_csv_ints("2,16,64")
    )
    parser.add_argument("--fixed-channels", type=_positive_int, default=16)
    parser.add_argument("--fixed-seconds", type=_positive_int, default=4)
    parser.add_argument("--embed-dim", type=_positive_int, default=256)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flat-token-rate", type=_positive_int, default=100)
    parser.add_argument("--flat-latent-step", type=float, default=0.1)
    parser.add_argument(
        "--flat-latents-per-step", type=_positive_int, default=16
    )
    parser.add_argument("--flat-depth", type=_positive_int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required for GPU profiling.")
    if args.embed_dim % 8:
        raise SystemExit("--embed-dim must be divisible by 8.")

    torch.manual_seed(17)
    device = torch.device("cuda")
    model = HEROModel(
        task_configs={},
        num_channels=max(max(args.channels), args.fixed_channels),
        embed_dim=args.embed_dim,
        num_attn_heads=8,
        num_spatial_slots=8,
        num_temporal_slots=4,
        num_local_attn_blocks=2,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    cases: dict[tuple[int, int], set[str]] = {}
    for seconds in args.seconds:
        cases.setdefault((seconds, args.fixed_channels), set()).add("duration")
    for channels in args.channels:
        cases.setdefault((args.fixed_seconds, channels), set()).add("channels")

    records: list[dict] = [
        {
            "record_type": "metadata",
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "embed_dim": args.embed_dim,
            "parameters": parameter_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "flop_scope": (
                "dense multiply-adds; excludes normalization, activation, "
                "mask/index, and overlap-search operations"
            ),
        }
    ]

    for seconds, channels in sorted(cases):
        signal = torch.randn(
            1, channels, seconds * 128, device=device, requires_grad=False
        )
        for _ in range(args.warmup):
            model.zero_grad(set_to_none=True)
            warmup_representation = model.encode(
                signal=signal, sampling_rate=128
            )
            warmup_representation.content.square().mean().backward()
        torch.cuda.synchronize()

        elapsed_samples = []
        memory_samples = []
        representation = None
        for _ in range(args.repeats):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            representation = model.encode(signal=signal, sampling_rate=128)
            representation.content.square().mean().backward()
            torch.cuda.synchronize()
            elapsed_samples.append(time.perf_counter() - start)
            memory_samples.append(torch.cuda.max_memory_allocated() / 2**20)

        assert representation is not None
        record = {
            "record_type": "measurement",
            "sweeps": sorted(cases[(seconds, channels)]),
            "seconds": seconds,
            "channels": channels,
            "fine_tokens": representation.content.shape[1],
            "mid_tokens": representation.coverage.mid_valid.shape[1],
            "coarse_tokens": representation.coverage.coarse_valid.shape[1],
            "elapsed_seconds_median": statistics.median(elapsed_samples),
            "elapsed_seconds_min": min(elapsed_samples),
            "peak_memory_mib_max": max(memory_samples),
        }
        record.update(
            _hero_flop_accounting(
                seconds=seconds,
                channels=channels,
                embed_dim=args.embed_dim,
            )
        )
        record.update(
            _flat_poyo_accounting(
                seconds=seconds,
                channels=channels,
                token_rate=args.flat_token_rate,
                latent_step=args.flat_latent_step,
                latents_per_step=args.flat_latents_per_step,
                depth=args.flat_depth,
            )
        )
        records.append(record)

    duration_records = [
        record
        for record in records
        if record.get("record_type") == "measurement"
        and "duration" in record["sweeps"]
    ]
    records.append(
        {
            "record_type": "duration_scaling",
            "fixed_channels": args.fixed_channels,
            "elapsed_log_log_slope": _log_log_slope(
                duration_records, "fine_tokens", "elapsed_seconds_median"
            ),
            "memory_log_log_slope": _log_log_slope(
                duration_records, "fine_tokens", "peak_memory_mib_max"
            ),
            "interpretation": (
                "1 is linear and 2 is quadratic; timing is descriptive, not "
                "a test assertion"
            ),
        }
    )
    _write_jsonl(records, args.output)


if __name__ == "__main__":
    main()

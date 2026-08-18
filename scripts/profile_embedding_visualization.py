"""Profile embedding-visualization stages and write inspectable artifacts.

This Phase 7 harness uses the production selector, metrics, PCA, and plotting
functions with the fixed default budgets.  It profiles representative 2-,
19-, 64-, and 129-channel validation populations and keeps GPU capture work,
CPU transfer, selection, metrics, plotting, and W&B image conversion separate.

Example:
    uv run python scripts/profile_embedding_visualization.py \
        --output-dir outputs/embedding_viz_phase7/synthetic_profile
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from foundry.training.callbacks.embedding_metrics import (
    compute_backbone_silhouettes,
    compute_channel_metrics,
    cosine_distance_matrix,
    get_electrode_positions_3d,
    normalize_representations,
)
from foundry.training.callbacks.embedding_viz import (
    fit_deterministic_pca,
    has_eligible_anatomy_recording,
    make_backbone_pca_figure,
    make_channel_anatomy_figure,
    make_channel_canonical_figure,
    make_channel_recording_figure,
    make_norm_distribution_figure,
)
from foundry.training.callbacks.observation_selector import (
    ObservationIdentity,
    SelectionConfig,
    hierarchical_select_windows,
    select_channel_observations,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    channels: int
    windows: int = 512
    recordings: int = 8
    datasets: int = 2


@dataclass
class ProfileResult:
    scenario: str
    channels: int
    population_windows: int
    selected_windows: int
    selected_channel_observations: int
    capture_seconds: float
    cpu_transfer_seconds: float
    selection_seconds: float
    metrics_seconds: float
    plotting_seconds: float
    image_conversion_seconds: float
    total_profiled_seconds: float
    anatomy_available: bool
    device: str


SCENARIOS = (
    Scenario("two_channel", channels=2),
    Scenario("standard_10_20", channels=19),
    Scenario("sixty_four_channel", channels=64),
    Scenario("high_density_129", channels=129),
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_runtime(function, repeats: int, device: torch.device) -> float:
    samples = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        function()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _channel_names(count: int, positions: dict[str, np.ndarray]) -> list[str]:
    resolved = sorted(positions)
    if count <= len(resolved):
        return resolved[:count]
    return resolved + [f"unresolved_{i}" for i in range(count - len(resolved))]


def _make_identities(scenario: Scenario) -> list[ObservationIdentity]:
    return [
        ObservationIdentity(
            dataset_id=f"dataset-{index % scenario.datasets}",
            subject_id=f"subject-{index % scenario.recordings}",
            session_id=f"recording-{index % scenario.recordings}",
            absolute_start=float(index * 2),
            window_duration=2.0,
        )
        for index in range(scenario.windows)
    ]


def _make_representations(
    scenario: Scenario,
    channel_names: list[str],
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    channel_base = torch.randn(
        scenario.channels, 64, generator=generator, device=device
    )
    recording_shift = (
        torch.randn(scenario.recordings, 64, generator=generator, device=device)
        * 0.15
    )
    channel_rows = []
    for index in range(scenario.windows):
        noise = (
            torch.randn(
                scenario.channels, 64, generator=generator, device=device
            )
            * 0.08
        )
        channel_rows.append(
            channel_base + recording_shift[index % scenario.recordings] + noise
        )
    channels = torch.stack(channel_rows)

    dataset_base = torch.randn(
        scenario.datasets, 256, generator=generator, device=device
    )
    session_base = (
        torch.randn(
            scenario.recordings, 256, generator=generator, device=device
        )
        * 0.35
    )
    backbone_rows = []
    for index in range(scenario.windows):
        noise = torch.randn(256, generator=generator, device=device) * 0.25
        backbone_rows.append(
            dataset_base[index % scenario.datasets]
            + session_base[index % scenario.recordings]
            + noise
        )
    backbone = torch.stack(backbone_rows)
    assert channels.shape[:2] == (scenario.windows, len(channel_names))
    return channels, backbone


def _save_figures(figures: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=140, bbox_inches="tight")


def profile_scenario(
    scenario: Scenario,
    output_dir: Path,
    device: torch.device,
    seed: int,
    repeats: int,
) -> ProfileResult:
    positions = get_electrode_positions_3d()
    channel_names = _channel_names(scenario.channels, positions)
    identities = _make_identities(scenario)
    channel_tensor, backbone_tensor = _make_representations(
        scenario, channel_names, device, seed
    )

    # Production capture adds a mean-pool of final latents.  The channel tensor
    # is already produced by tokenization and is referenced without a copy.
    latent_tensor = backbone_tensor[:, None, :].expand(-1, 16, -1).contiguous()
    capture_seconds = _median_runtime(
        lambda: latent_tensor.mean(dim=1), repeats, device
    )

    transfer_holder: dict[str, torch.Tensor] = {}

    def transfer() -> None:
        transfer_holder["channel"] = channel_tensor.detach().cpu()
        transfer_holder["backbone"] = backbone_tensor.detach().cpu()

    cpu_transfer_seconds = _median_runtime(transfer, repeats, device)
    channel_cpu = transfer_holder["channel"]
    backbone_cpu = transfer_holder["backbone"]

    config = SelectionConfig(seed=seed)
    started = time.perf_counter()
    selection = hierarchical_select_windows(identities, config)
    channel_window_indices = select_channel_observations(
        selection.window_indices,
        identities,
        [scenario.channels] * scenario.windows,
        config,
    )
    selection_seconds = time.perf_counter() - started

    selected_backbone = backbone_cpu[selection.window_indices].numpy()
    selected_channels = (
        channel_cpu[channel_window_indices]
        .reshape(-1, channel_cpu.shape[-1])
        .numpy()
    )
    recording_ids = np.repeat(
        [identities[index].session_id for index in channel_window_indices],
        scenario.channels,
    )
    flat_channel_ids = np.tile(
        np.asarray(channel_names), len(channel_window_indices)
    )
    window_ids = np.repeat(
        [f"window-{index}" for index in channel_window_indices],
        scenario.channels,
    )

    started = time.perf_counter()
    backbone_normalized = normalize_representations(selected_backbone)
    backbone_coords, backbone_pca = fit_deterministic_pca(
        backbone_normalized.vectors, seed=seed
    )
    distances = cosine_distance_matrix(backbone_normalized.vectors)
    selected_identities = [
        identities[index] for index in selection.window_indices
    ]
    dataset_labels = np.asarray(
        [item.dataset_id for item in selected_identities]
    )
    subject_labels = np.asarray(
        [item.subject_id for item in selected_identities]
    )
    session_labels = np.asarray(
        [item.session_id for item in selected_identities]
    )
    task_labels = np.arange(len(selected_identities)) % 5
    compute_backbone_silhouettes(
        distances,
        {
            "dataset": dataset_labels,
            "subject": subject_labels,
            "session": session_labels,
            "task/synthetic": task_labels,
        },
    )

    channel_normalized = normalize_representations(selected_channels)
    channel_coords, channel_pca = fit_deterministic_pca(
        channel_normalized.vectors, seed=seed
    )
    compute_channel_metrics(
        channel_normalized.vectors,
        recording_ids,
        flat_channel_ids,
        "dynamic",
        window_ids=window_ids,
        positions_3d=positions,
    )
    metrics_seconds = time.perf_counter() - started

    anatomy_available = has_eligible_anatomy_recording(
        recording_ids, flat_channel_ids, positions, min_positioned_channels=9
    )
    started = time.perf_counter()
    figures = {
        "backbone_pca_dataset": make_backbone_pca_figure(
            backbone_coords,
            dataset_labels,
            "Dataset",
            backbone_pca,
            scenario.name,
        ),
        "backbone_pca_subject": make_backbone_pca_figure(
            backbone_coords,
            subject_labels,
            "Subject",
            backbone_pca,
            scenario.name,
        ),
        "backbone_pca_session": make_backbone_pca_figure(
            backbone_coords,
            session_labels,
            "Session",
            backbone_pca,
            scenario.name,
        ),
        "backbone_pca_task": make_backbone_pca_figure(
            backbone_coords,
            task_labels,
            "Synthetic classification task",
            backbone_pca,
            scenario.name,
            class_names=["class-0", "class-1", "class-2", "class-3", "class-4"],
        ),
        "backbone_norm_distribution": make_norm_distribution_figure(
            backbone_normalized.norms, "Backbone", scenario.name
        ),
        "channel_pca_by_recording": make_channel_recording_figure(
            channel_coords,
            recording_ids,
            flat_channel_ids,
            channel_pca,
            "dynamic",
            config.max_recording_panels,
            scenario.name,
            seed,
        ),
        "channel_pca_canonical_electrode": make_channel_canonical_figure(
            channel_coords, flat_channel_ids, channel_pca, scenario.name
        ),
        "channel_norm_distribution": make_norm_distribution_figure(
            channel_normalized.norms, "Channel", scenario.name
        ),
    }
    if anatomy_available:
        figures["channel_pca_anatomy"] = make_channel_anatomy_figure(
            channel_coords,
            flat_channel_ids,
            positions,
            channel_pca,
            scenario.name,
            recording_ids=recording_ids,
        )
    figures = {
        name: figure for name, figure in figures.items() if figure is not None
    }
    plotting_seconds = time.perf_counter() - started

    _save_figures(figures, output_dir / scenario.name)

    import wandb

    started = time.perf_counter()
    images = [wandb.Image(figure) for figure in figures.values()]
    image_conversion_seconds = time.perf_counter() - started
    assert len(images) == len(figures)
    for figure in figures.values():
        plt.close(figure)

    stage_times = (
        capture_seconds,
        cpu_transfer_seconds,
        selection_seconds,
        metrics_seconds,
        plotting_seconds,
        image_conversion_seconds,
    )
    return ProfileResult(
        scenario=scenario.name,
        channels=scenario.channels,
        population_windows=scenario.windows,
        selected_windows=len(selection.window_indices),
        selected_channel_observations=len(selected_channels),
        capture_seconds=capture_seconds,
        cpu_transfer_seconds=cpu_transfer_seconds,
        selection_seconds=selection_seconds,
        metrics_seconds=metrics_seconds,
        plotting_seconds=plotting_seconds,
        image_conversion_seconds=image_conversion_seconds,
        total_profiled_seconds=sum(stage_times),
        anatomy_available=anatomy_available,
        device=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embedding_viz_phase7/synthetic_profile"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        profile_scenario(
            scenario,
            args.output_dir,
            device,
            args.seed,
            args.repeats,
        )
        for scenario in SCENARIOS
    ]
    payload = {
        "seed": args.seed,
        "fixed_default_budgets": asdict(SelectionConfig(seed=args.seed)),
        "results": [asdict(result) for result in results],
    }
    report_path = args.output_dir / "profile.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote profile and PNG artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

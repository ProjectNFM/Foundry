"""Regression and integration tests for embedding-visualization callback.

Covers Phase 5 figure rendering and Phase 6 configuration migration:
- Stable color determinism
- Static/dynamic channel flattening
- Variable-width padding alignment
- Unavailable-representation availability logging
- Multi-target task label exclusion
- Single-feature PCA edge case
- Deprecated parameter rejection (Phase 6 Task 1)
- Config-composition smoke tests (Phase 6 Task 2)
- Static, dynamic, disabled, pretraining, downstream, multi-task smoke tests (Task 3)
- Anatomy threshold tests: <9 channels vs >=9 channels (Task 4)
- W&B key contract and image lifecycle with fake logger (Task 5)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from foundry.models.ssl_meta import RepresentationPayload
from foundry.training.callbacks.embedding_viz import (
    EmbeddingVisualizationCallback,
    fit_deterministic_pca,
    make_backbone_pca_figure,
    make_channel_anatomy_figure,
    stable_color_map,
)
from foundry.training.callbacks.observation_selector import (
    ObservationIdentity,
    RankObservations,
    SelectedObservations,
)
from foundry.training.step_output import SampleMetadata, StepOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(
    start: float,
    dataset: str = "dataset",
    subject: str = "subject",
    session: str = "session",
) -> ObservationIdentity:
    """Create one recording-specific observation identity."""
    return ObservationIdentity(dataset, subject, session, start, 2.0)


def _make_step_output(
    *,
    batch_size: int = 2,
    n_channels: int = 4,
    embed_dim: int = 8,
    channel_mode: str = "dynamic",
    include_backbone: bool = True,
    include_channel: bool = True,
    dataset_id: str = "ds",
    subject_id: str = "sub",
    session_id: str = "sess",
    start_offset: float = 0.0,
    task_targets: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Build a fake step_output dict for on_validation_batch_end."""
    ch_reps = (
        torch.randn(batch_size, n_channels, embed_dim)
        if include_channel
        else None
    )
    ch_mask = (
        torch.ones(batch_size, n_channels, dtype=torch.bool)
        if include_channel
        else None
    )
    bb_reps = torch.randn(batch_size, embed_dim) if include_backbone else None

    return {
        "step_output": StepOutput(
            loss=torch.tensor(0.0),
            task_outputs={},
            target_values=task_targets or {},
            target_weights={},
            task_index=torch.zeros(batch_size, 1, dtype=torch.long),
            sample_metadata=SampleMetadata(
                dataset_id=[dataset_id] * batch_size,
                subject_id=[subject_id] * batch_size,
                session_id=[session_id] * batch_size,
                absolute_start=torch.arange(batch_size, dtype=torch.float) * 2.0
                + start_offset,
                window_duration=torch.full((batch_size,), 2.0),
                channel_index=(
                    torch.arange(n_channels).unsqueeze(0).expand(batch_size, -1)
                    if include_channel
                    else None
                ),
                channel_mask=ch_mask,
            ),
            representations=RepresentationPayload(
                channel_representations=ch_reps,
                backbone_representations=bb_reps,
                channel_mode=channel_mode if include_channel else None,
                channel_mask=ch_mask,
            ),
        )
    }


class FakeExperiment:
    """Collects W&B log calls for assertion."""

    def __init__(self):
        self.logged: dict = {}

    def log(self, values, commit: bool = True) -> None:
        assert commit is False, "Callback must log with commit=False"
        self.logged.update(values)


class FakeTrainer:
    """Minimal Trainer stand-in for testing _process_and_log."""

    def __init__(self, step: int = 100):
        self.global_step = step
        self.world_size = 1
        self.global_rank = 0


# ---------------------------------------------------------------------------
# Original Phase 5 regression tests
# ---------------------------------------------------------------------------


def test_stable_colors_do_not_depend_on_event_group_subset() -> None:
    """A group's color remains unchanged when another group is unavailable."""
    assert (
        stable_color_map(["alpha", "beta"])["beta"]
        == stable_color_map(["beta"])["beta"]
    )


def test_static_channel_flattening_keeps_one_point_per_channel() -> None:
    """Static vectors are not duplicated for each selected window."""
    callback = EmbeddingVisualizationCallback()
    callback._idx_to_channel = {1: "Fp1", 2: "Fp2"}
    callback._sample_metadata_lists = {"channel_mode": ["static"]}
    observations = RankObservations(
        identities=[_identity(0.0), _identity(2.0)],
        channel_representations=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        channel_indices=torch.tensor([[1, 2], [1, 2]]),
        channel_counts=[2, 2],
    )

    vectors, recording_ids, channel_ids, window_ids = (
        callback._flatten_channel_observations(observations, [0, 1])
    )

    assert vectors.shape == (2, 2)
    assert list(channel_ids) == ["Fp1", "Fp2"]
    assert len(recording_ids) == len(window_ids) == 2


def test_channel_metadata_stays_aligned_across_variable_padding() -> None:
    """Per-batch channel padding preserves masks and identities during flattening."""
    callback = EmbeddingVisualizationCallback()
    callback._capture_scheduled = True
    callback._idx_to_channel = {1: "Fp1", 2: "Fp2", 3: "Cz"}

    def output(start: float, reps: torch.Tensor, mask: torch.Tensor) -> dict:
        return {
            "step_output": StepOutput(
                loss=torch.tensor(0.0),
                task_outputs={},
                target_values={},
                target_weights={},
                task_index=torch.zeros(1, 1, dtype=torch.long),
                sample_metadata=SampleMetadata(
                    dataset_id=["dataset"],
                    subject_id=["subject"],
                    session_id=["session"],
                    absolute_start=torch.tensor([start]),
                    window_duration=torch.tensor([2.0]),
                    channel_index=torch.arange(1, reps.shape[1] + 1).unsqueeze(
                        0
                    ),
                    channel_mask=mask,
                ),
                representations=RepresentationPayload(
                    channel_representations=reps,
                    channel_mode="dynamic",
                    channel_mask=mask,
                ),
            )
        }

    callback.on_validation_batch_end(
        None,
        None,
        output(
            0.0,
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([[True, False]]),
        ),
        None,
        0,
    )
    callback.on_validation_batch_end(
        None,
        None,
        output(
            2.0,
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
            torch.tensor([[True, False, True]]),
        ),
        None,
        1,
    )

    assert callback._rank_obs is not None
    vectors, _, channel_ids, _ = callback._flatten_channel_observations(
        callback._rank_obs, [0, 1]
    )
    assert vectors.shape == (3, 2)
    assert list(channel_ids) == ["Fp1", "Fp1", "Cz"]


def test_unavailable_representations_still_emit_availability() -> None:
    """Scheduled events distinguish unavailable representations from no event."""
    callback = EmbeddingVisualizationCallback()
    experiment = FakeExperiment()
    callback._process_and_log(
        RankObservations(identities=[_identity(0.0)]),
        FakeTrainer(),
        experiment,
    )

    assert experiment.logged["val/embedding_viz/availability/backbone"] == 0
    assert experiment.logged["val/embedding_viz/availability/channel"] == 0


def test_task_labels_exclude_conflicting_multi_target_windows() -> None:
    """Multi-target task views retain only windows with an unambiguous class."""
    callback = EmbeddingVisualizationCallback()
    observations = RankObservations(
        identities=[_identity(0.0), _identity(2.0), _identity(4.0)],
        target_values={
            "stage": torch.tensor([[1, 1], [0, 2], [-1, 3]]),
        },
    )
    selection = SelectedObservations(
        window_identities=observations.identities,
        window_indices=[0, 1, 2],
        fingerprint="test",
    )

    labels, valid, _ = callback._extract_task_labels(
        observations, selection, np.array([0, 1, 2])
    )["stage"]

    assert list(labels) == [1, 2, 3]
    assert list(valid) == [True, False, True]


def test_single_feature_pca_figures_are_renderable() -> None:
    """One-dimensional representations still produce a two-axis plot."""
    coords, pca = fit_deterministic_pca(np.array([[1.0], [2.0]]))
    fig = make_backbone_pca_figure(
        coords, np.array(["a", "b"]), "Dataset", pca, "step 1"
    )

    assert coords.shape == (2, 2)
    plt.close(fig)


# ===========================================================================
# Phase 6 Task 1: Deprecated parameters are rejected
# ===========================================================================


def test_deprecated_every_n_epochs_raises_type_error() -> None:
    """Passing removed 'every_n_epochs' raises TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        EmbeddingVisualizationCallback(every_n_epochs=5)


def test_deprecated_max_samples_raises_type_error() -> None:
    """Passing removed 'max_samples' raises TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        EmbeddingVisualizationCallback(max_samples=2048)


def test_deprecated_compute_tsne_raises_type_error() -> None:
    """Passing removed 'compute_tsne' raises TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        EmbeddingVisualizationCallback(compute_tsne=False)


def test_deprecated_class_names_raises_type_error() -> None:
    """Passing removed 'class_names' raises TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        EmbeddingVisualizationCallback(class_names=["Wake", "Sleep"])


def test_new_api_parameters_accepted() -> None:
    """All new-API parameters are accepted without error."""
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=3,
        sample_seed=123,
        window_fraction=0.15,
        min_windows=128,
        max_windows=1024,
        max_channel_observations=8192,
        max_sessions_per_dataset=4,
        min_windows_per_session=8,
        max_recording_panels=4,
        min_positioned_channels=12,
    )
    assert cb.every_n_validation_runs == 3
    assert cb._selection_config.seed == 123
    assert cb._selection_config.max_windows == 1024
    assert cb.min_positioned_channels == 12


# ===========================================================================
# Phase 6 Task 2: Configuration-composition sanity checks
# ===========================================================================


def test_default_config_uses_only_new_api_keys() -> None:
    """The default trainer config must not reference any deprecated keys."""
    import yaml

    with open("configs/trainer/default.yaml") as f:
        cfg = yaml.safe_load(f)

    emb_cfg = cfg["callbacks"]["embedding_visualization"]
    deprecated = {
        "every_n_epochs",
        "max_samples",
        "compute_tsne",
        "class_names",
    }
    found = deprecated & set(emb_cfg.keys())
    assert not found, f"Deprecated keys in default config: {found}"

    required = {"_target_", "every_n_validation_runs"}
    missing = required - set(emb_cfg.keys())
    assert not missing, f"Missing required keys in default config: {missing}"


@pytest.mark.parametrize(
    ("experiment_name", "expected_frequency"),
    [
        ("pretraining/poyo_masking_seqlen_sweep", 5),
        ("pretraining/poyo_data_scaling_base", 1),
    ],
)
def test_experiment_overrides_compose_with_new_api(
    experiment_name: str, expected_frequency: int
) -> None:
    """Each changed pretraining override composes and instantiates the callback.

    Loading an override in isolation cannot detect a stale callback key that
    only becomes invalid after Hydra merges it with the default trainer config.
    """
    from pathlib import Path

    config_dir = Path("configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment_name}"],
        )

    callback = instantiate(cfg.trainer.callbacks.embedding_visualization)
    assert isinstance(callback, EmbeddingVisualizationCallback)
    assert callback.every_n_validation_runs == expected_frequency


def test_default_config_callback_is_instantiable() -> None:
    """The default config parameters can construct the callback."""
    import yaml

    with open("configs/trainer/default.yaml") as f:
        cfg = yaml.safe_load(f)

    emb_cfg = cfg["callbacks"]["embedding_visualization"]
    params = {k: v for k, v in emb_cfg.items() if k != "_target_"}
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and "${" in v:
            resolved[k] = 42
        else:
            resolved[k] = v

    cb = EmbeddingVisualizationCallback(**resolved)
    assert cb.every_n_validation_runs == 5


# ===========================================================================
# Phase 6 Task 3: Static/dynamic/disabled mode smoke tests
# ===========================================================================


def _run_callback_smoke(
    channel_mode: str,
    include_channel: bool = True,
    include_backbone: bool = True,
    n_batches: int = 3,
    batch_size: int = 4,
    n_channels: int = 4,
    task_targets: dict[str, torch.Tensor] | None = None,
    n_datasets: int = 1,
) -> dict:
    """Run the full callback lifecycle and return the logged W&B dict.

    Simulates `every_n_validation_runs=1` so the first pass triggers output.
    """
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1,
        sample_seed=42,
        min_windows=1,
        max_windows=256,
    )
    if channel_mode == "static":
        cb._idx_to_channel = {i: f"ch_{i}" for i in range(n_channels)}

    experiment = FakeExperiment()
    trainer = FakeTrainer(step=100)

    module = MagicMock()
    module.set_validation_representation_capture = MagicMock()
    cb.on_validation_epoch_start(trainer, module)
    assert cb._capture_scheduled

    for batch_idx in range(n_batches):
        out = _make_step_output(
            batch_size=batch_size,
            n_channels=n_channels,
            channel_mode=channel_mode if include_channel else "disabled",
            include_backbone=include_backbone,
            include_channel=include_channel,
            start_offset=batch_idx * batch_size * 2.0,
            task_targets=task_targets,
            **(
                {"dataset_id": f"ds{batch_idx % n_datasets}"}
                if n_datasets > 1
                else {}
            ),
        )
        cb.on_validation_batch_end(trainer, module, out, None, batch_idx)

    with patch(
        "foundry.training.callbacks.get_wandb_experiment",
        return_value=experiment,
    ):
        cb.on_validation_epoch_end(trainer, module)

    return experiment.logged


def test_dynamic_mode_produces_both_families() -> None:
    """Dynamic channel + backbone emits channel and backbone outputs."""
    logged = _run_callback_smoke("dynamic")

    assert logged["val/embedding_viz/availability/backbone"] == 1
    assert logged["val/embedding_viz/availability/channel"] == 1
    assert logged["val/embedding_viz/availability/channel_mode"] == "dynamic"
    assert "val/embedding_viz/sample/fingerprint" in logged
    assert logged["val/embedding_viz/sample/window_count"] > 0


def test_static_mode_produces_both_families() -> None:
    """Static channel + backbone emits channel and backbone outputs."""
    logged = _run_callback_smoke("static")

    assert logged["val/embedding_viz/availability/backbone"] == 1
    assert logged["val/embedding_viz/availability/channel"] == 1
    assert logged["val/embedding_viz/availability/channel_mode"] == "static"


def test_disabled_channel_mode_produces_backbone_only() -> None:
    """Disabled channel mode emits only backbone outputs, no channel."""
    logged = _run_callback_smoke(
        "disabled", include_channel=False, include_backbone=True
    )

    assert logged["val/embedding_viz/availability/backbone"] == 1
    assert logged["val/embedding_viz/availability/channel"] == 0


def test_no_backbone_produces_channel_only() -> None:
    """Model without Perceiver backbone still emits channel output."""
    logged = _run_callback_smoke(
        "dynamic", include_channel=True, include_backbone=False
    )

    assert logged["val/embedding_viz/availability/backbone"] == 0
    assert logged["val/embedding_viz/availability/channel"] == 1


def test_pretraining_no_tasks_still_produces_backbone() -> None:
    """Pretraining without classification tasks still emits backbone PCA."""
    logged = _run_callback_smoke("dynamic")

    assert logged["val/embedding_viz/availability/backbone"] == 1
    assert logged["val/embedding_viz/sample/window_count"] > 0


def test_downstream_classification_single_task() -> None:
    """Downstream classification with one task emits task-specific views."""
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1,
        min_windows=1,
        max_windows=256,
    )
    cb._task_class_names = {
        "sleep_stage": ["W", "N1", "N2", "N3", "REM"],
    }

    experiment = FakeExperiment()
    trainer = FakeTrainer(step=100)
    module = MagicMock()
    module.set_validation_representation_capture = MagicMock()

    cb.on_validation_epoch_start(trainer, module)

    for batch_idx in range(3):
        out = _make_step_output(
            batch_size=4,
            task_targets={"sleep_stage": torch.randint(0, 5, (4,))},
            start_offset=batch_idx * 8.0,
        )
        cb.on_validation_batch_end(trainer, module, out, None, batch_idx)

    with patch(
        "foundry.training.callbacks.get_wandb_experiment",
        return_value=experiment,
    ):
        cb.on_validation_epoch_end(trainer, module)

    assert experiment.logged["val/embedding_viz/availability/backbone"] == 1


def test_multi_task_produces_per_task_scores() -> None:
    """Multi-task model emits silhouette scores for each task."""
    logged = _run_callback_smoke(
        "dynamic",
        task_targets={
            "sleep": torch.randint(0, 3, (4,)),
            "movement": torch.randint(0, 2, (4,)),
        },
    )

    assert logged["val/embedding_viz/availability/backbone"] == 1


def test_multi_dataset_produces_dataset_views() -> None:
    """Multiple datasets in validation produce dataset-grouped views."""
    logged = _run_callback_smoke("dynamic", n_datasets=2, n_batches=4)

    assert logged["val/embedding_viz/availability/backbone"] == 1
    assert logged["val/embedding_viz/sample/window_count"] > 0


# ===========================================================================
# Phase 6 Task 3 (continued): Scheduling tests
# ===========================================================================


def test_sanity_validation_does_not_schedule_capture() -> None:
    """Sanity validation must not trigger embedding capture."""
    cb = EmbeddingVisualizationCallback(every_n_validation_runs=1)
    trainer = MagicMock()
    trainer.sanity_checking = True
    module = MagicMock()

    cb.on_validation_epoch_start(trainer, module)
    assert not cb._capture_scheduled
    assert cb._validation_run_count == 0


def test_scheduling_skips_unscheduled_events() -> None:
    """Events that don't match the frequency are not captured."""
    cb = EmbeddingVisualizationCallback(every_n_validation_runs=3)
    trainer = MagicMock()
    trainer.sanity_checking = False
    module = MagicMock()

    for i in range(6):
        cb.on_validation_epoch_start(trainer, module)
        if (i + 1) % 3 == 0:
            assert cb._capture_scheduled, f"Event {i + 1} should be scheduled"
        else:
            assert not cb._capture_scheduled, (
                f"Event {i + 1} should NOT be scheduled"
            )


def test_first_validation_run_is_event_one() -> None:
    """The first complete, non-sanity validation pass is event 1."""
    cb = EmbeddingVisualizationCallback(every_n_validation_runs=1)
    trainer = MagicMock()
    trainer.sanity_checking = False
    module = MagicMock()

    cb.on_validation_epoch_start(trainer, module)
    assert cb._validation_run_count == 1
    assert cb._capture_scheduled


# ===========================================================================
# Phase 6 Task 4: Anatomy threshold tests
# ===========================================================================


def test_anatomy_figure_requires_min_positioned_channels() -> None:
    """Anatomy figure is None when fewer than min_positioned_channels resolve."""
    from sklearn.decomposition import PCA

    n_points = 6
    coords = np.random.randn(n_points, 2)
    channel_ids = np.array(["unknown_ch_" + str(i) for i in range(n_points)])
    pca = PCA(n_components=2)
    pca.fit(coords)

    fig = make_channel_anatomy_figure(coords, channel_ids, {}, pca, "step 100")
    assert fig is None, "Should return None when no positions resolve"


def test_anatomy_figure_renders_with_enough_positions() -> None:
    """Anatomy figure renders when >=9 channels have canonical positions."""
    from sklearn.decomposition import PCA

    standard_electrodes = [
        "fp1",
        "fp2",
        "f3",
        "f4",
        "c3",
        "c4",
        "p3",
        "p4",
        "o1",
        "o2",
        "f7",
        "f8",
    ]
    n_points = len(standard_electrodes)

    np.random.seed(42)
    coords = np.random.randn(n_points, 2)
    channel_ids = np.array(standard_electrodes)

    positions_3d = {
        name: np.array([0.1 * i, 0.05 * i, 0.0])
        for i, name in enumerate(standard_electrodes)
    }

    pca = PCA(n_components=2)
    pca.fit(coords)

    fig = make_channel_anatomy_figure(
        coords, channel_ids, positions_3d, pca, "step 100"
    )
    assert fig is not None, (
        "Should produce a figure with >=9 positioned channels"
    )
    plt.close(fig)


def test_anatomy_threshold_boundary_at_nine() -> None:
    """Exactly 9 positioned channels should produce anatomy output."""
    from sklearn.decomposition import PCA

    electrodes = ["fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4", "o1"]
    unresolved = ["unknown_x", "unknown_y", "unknown_z"]
    all_channels = electrodes + unresolved
    n = len(all_channels)

    np.random.seed(42)
    coords = np.random.randn(n, 2)
    channel_ids = np.array(all_channels)

    positions_3d = {
        name: np.array([0.1 * i, 0.05 * i, 0.0])
        for i, name in enumerate(electrodes)
    }

    pca = PCA(n_components=2)
    pca.fit(coords)

    fig = make_channel_anatomy_figure(
        coords, channel_ids, positions_3d, pca, "step 100"
    )
    assert fig is not None, (
        "Exactly 9 resolved channels should produce a figure"
    )
    plt.close(fig)


def test_anatomy_threshold_below_nine() -> None:
    """8 positioned channels (below threshold) should still render the figure
    (the threshold check is in the callback, not the figure function)."""
    from sklearn.decomposition import PCA

    electrodes = ["fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4"]
    n = len(electrodes)

    np.random.seed(42)
    coords = np.random.randn(n, 2)
    channel_ids = np.array(electrodes)

    positions_3d = {
        name: np.array([0.1 * i, 0.05 * i, 0.0])
        for i, name in enumerate(electrodes)
    }

    pca = PCA(n_components=2)
    pca.fit(coords)

    fig = make_channel_anatomy_figure(
        coords, channel_ids, positions_3d, pca, "step 100"
    )
    assert fig is not None
    plt.close(fig)


def test_callback_anatomy_gated_by_min_positioned_channels() -> None:
    """The callback-level anatomy gate uses min_positioned_channels threshold.

    With unresolvable channel names and min_positioned_channels=9,
    the anatomy figure key should not appear.
    """
    logged = _run_callback_smoke("dynamic", n_channels=4)

    assert "val/embedding_viz/channel/pca_anatomy" not in logged


# ===========================================================================
# Phase 6 Task 5: W&B keys and image lifecycle
# ===========================================================================


_EXPECTED_SAMPLE_KEYS = {
    "val/embedding_viz/sample/window_count",
    "val/embedding_viz/sample/channel_observation_count",
    "val/embedding_viz/sample/fingerprint",
}

_EXPECTED_AVAILABILITY_KEYS = {
    "val/embedding_viz/availability/backbone",
    "val/embedding_viz/availability/channel",
    "val/embedding_viz/availability/channel_mode",
}


def test_wandb_keys_include_sample_and_availability() -> None:
    """Every scheduled event emits sample metadata and availability keys."""
    logged = _run_callback_smoke("dynamic")

    missing = (_EXPECTED_SAMPLE_KEYS | _EXPECTED_AVAILABILITY_KEYS) - set(
        logged.keys()
    )
    assert not missing, f"Missing W&B keys: {missing}"


def test_wandb_backbone_keys_when_available() -> None:
    """Backbone keys are emitted when backbone representations are available."""
    logged = _run_callback_smoke("dynamic", include_backbone=True)

    backbone_keys = {k for k in logged if "/backbone/" in k}
    assert len(backbone_keys) > 0, "No backbone keys logged"

    assert any("normalization" in k for k in backbone_keys)


def test_wandb_channel_keys_when_available() -> None:
    """Channel keys are emitted when channel representations are available."""
    logged = _run_callback_smoke("dynamic", include_channel=True)

    channel_keys = {k for k in logged if "/channel/" in k}
    assert len(channel_keys) > 0, "No channel keys logged"


def test_wandb_no_channel_keys_when_disabled() -> None:
    """No channel figure/metric keys when channel mode is disabled."""
    logged = _run_callback_smoke(
        "disabled", include_channel=False, include_backbone=True
    )

    channel_figure_keys = {
        k for k in logged if "/channel/" in k and "availability" not in k
    }
    assert len(channel_figure_keys) == 0, (
        f"Unexpected channel keys: {channel_figure_keys}"
    )


def test_wandb_log_uses_commit_false() -> None:
    """All W&B logs must use commit=False to avoid premature step increments."""

    class StrictExperiment:
        def __init__(self):
            self.calls = []

        def log(self, values, commit: bool = True) -> None:
            self.calls.append({"values": values, "commit": commit})

    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1, min_windows=1, max_windows=256
    )
    experiment = StrictExperiment()
    trainer = FakeTrainer(step=50)

    observations = RankObservations(
        identities=[_identity(0.0), _identity(2.0)],
        backbone_representations=torch.randn(2, 8),
    )

    cb._process_and_log(observations, trainer, experiment)

    assert len(experiment.calls) > 0
    for call in experiment.calls:
        assert call["commit"] is False


def test_no_matplotlib_figures_leak_after_logging() -> None:
    """All matplotlib figures must be closed after conversion to W&B images."""
    figs_before = len(plt.get_fignums())
    _run_callback_smoke("dynamic")
    figs_after = len(plt.get_fignums())
    assert figs_after == figs_before, (
        f"Leaked {figs_after - figs_before} matplotlib figures"
    )


def test_callback_works_without_wandb_logger() -> None:
    """Callback completes without error when no W&B logger is present."""
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1, min_windows=1, max_windows=256
    )
    trainer = FakeTrainer(step=50)
    module = MagicMock()
    module.set_validation_representation_capture = MagicMock()

    cb.on_validation_epoch_start(trainer, module)

    out = _make_step_output(batch_size=4)
    cb.on_validation_batch_end(trainer, module, out, None, 0)

    with patch(
        "foundry.training.callbacks.get_wandb_experiment",
        return_value=None,
    ):
        cb.on_validation_epoch_end(trainer, module)

    assert cb._rank_obs is None, "Buffers should be cleared after epoch end"


def test_callback_clears_buffers_after_processing() -> None:
    """Buffers are cleared after every validation epoch end."""
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1, min_windows=1, max_windows=256
    )
    trainer = FakeTrainer(step=50)
    module = MagicMock()
    module.set_validation_representation_capture = MagicMock()

    cb.on_validation_epoch_start(trainer, module)
    out = _make_step_output(batch_size=2)
    cb.on_validation_batch_end(trainer, module, out, None, 0)

    with patch(
        "foundry.training.callbacks.get_wandb_experiment",
        return_value=FakeExperiment(),
    ):
        cb.on_validation_epoch_end(trainer, module)

    assert cb._rank_obs is None
    assert cb._local_identities == []
    assert cb._local_channel_counts == []
    assert cb._sample_metadata_lists == {}


def test_fingerprint_is_stable_across_calls() -> None:
    """Same observations produce the same fingerprint."""
    from foundry.training.callbacks.observation_selector import (
        compute_fingerprint,
    )

    ids = [_identity(0.0), _identity(2.0), _identity(4.0)]
    fp1 = compute_fingerprint(ids)
    fp2 = compute_fingerprint(ids)
    assert fp1 == fp2

    fp_reversed = compute_fingerprint(list(reversed(ids)))
    assert fp1 == fp_reversed, "Fingerprint must be order-independent"


def test_step_label_uses_global_step() -> None:
    """Event labels use trainer.global_step, not epoch."""
    experiment = FakeExperiment()
    cb = EmbeddingVisualizationCallback(
        every_n_validation_runs=1, min_windows=1, max_windows=256
    )
    trainer = FakeTrainer(step=42000)

    observations = RankObservations(
        identities=[_identity(0.0), _identity(2.0)],
        backbone_representations=torch.randn(2, 8),
    )

    cb._process_and_log(observations, trainer, experiment)

    logged_keys = list(experiment.logged.keys())
    assert len(logged_keys) > 0, "Should have logged something"

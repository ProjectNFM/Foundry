"""Regression tests for embedding-visualization figures and orchestration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from foundry.training.callbacks.embedding_viz import (
    EmbeddingVisualizationCallback,
    fit_deterministic_pca,
    make_backbone_pca_figure,
    stable_color_map,
)
from foundry.training.callbacks.observation_selector import (
    ObservationIdentity,
    RankObservations,
    SelectedObservations,
)
from foundry.training.step_output import SampleMetadata, StepOutput
from foundry.models.ssl_meta import RepresentationPayload


def _identity(start: float) -> ObservationIdentity:
    """Create one recording-specific observation identity."""
    return ObservationIdentity("dataset", "subject", "session", start, 2.0)


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
    logged = {}

    class Experiment:
        def log(self, values, commit: bool) -> None:
            assert commit is False
            logged.update(values)

    class Trainer:
        global_step = 10

    callback._process_and_log(
        RankObservations(identities=[_identity(0.0)]), Trainer(), Experiment()
    )

    assert logged["val/embedding_viz/availability/backbone"] == 0
    assert logged["val/embedding_viz/availability/channel"] == 0


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

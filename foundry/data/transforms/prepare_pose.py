from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from torch_brain.data import Data, IrregularTimeSeries

logger = logging.getLogger(__name__)


DEFAULT_AJILE_KEYPOINTS: tuple[str, ...] = (
    "l_ear",
    "l_elbow",
    "l_shoulder",
    "l_wrist",
    "nose",
    "r_ear",
    "r_elbow",
    "r_shoulder",
    "r_wrist",
)


class PreparePoseTrajectories:
    """Build a single pose trajectory readout from Ajile keypoints.

    Args:
        readout_id: Name of the readout modality to register.
        keypoints: Ordered keypoint attribute names on ``data.pose``.
    """

    def __init__(
        self,
        readout_id: str = "ajile_pose_estimation",
        keypoints: Sequence[str] = DEFAULT_AJILE_KEYPOINTS,
    ):
        self.readout_id = readout_id
        self.keypoints = tuple(keypoints)

    def __call__(self, data: Data) -> Data:
        pose = getattr(data, "pose", None)
        if pose is None:
            raise ValueError("Data must include a 'pose' timeseries")

        missing_keypoints = [
            keypoint
            for keypoint in self.keypoints
            if not hasattr(pose, keypoint)
        ]
        if missing_keypoints:
            raise ValueError(
                "Missing pose keypoints: " + ", ".join(missing_keypoints)
            )

        pose_components = [
            np.asarray(getattr(pose, keypoint)) for keypoint in self.keypoints
        ]
        pose_values = np.concatenate(pose_components, axis=-1)
        if pose_values.dtype == np.float64:
            pose_values = pose_values.astype(np.float32)

        non_finite_mask = ~np.isfinite(pose_values)
        if non_finite_mask.any():
            n_bad = int(non_finite_mask.sum())
            n_total = pose_values.size
            bad_keypoints = set()
            dim_per_kp = pose_values.shape[-1] // len(self.keypoints)
            for i, kp in enumerate(self.keypoints):
                kp_slice = non_finite_mask[
                    ..., i * dim_per_kp : (i + 1) * dim_per_kp
                ]
                if kp_slice.any():
                    bad_keypoints.add(kp)
            logger.warning(
                "Pose data contains %d / %d non-finite values in keypoints: %s. "
                "Replacing with zeros.",
                n_bad,
                n_total,
                ", ".join(sorted(bad_keypoints)),
            )
            pose_values = np.where(non_finite_mask, 0.0, pose_values)

        data.pose_trajectories = IrregularTimeSeries(
            timestamps=np.asarray(pose.timestamps),
            values=pose_values,
            domain=pose.domain,
        )
        return data

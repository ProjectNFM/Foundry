from __future__ import annotations

from typing import Sequence

import numpy as np
from temporaldata import Data, IrregularTimeSeries


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
    """Build a single pose trajectory readout from Ajile keypoints."""

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

        data.pose_trajectories = IrregularTimeSeries(
            timestamps=np.asarray(pose.timestamps),
            values=pose_values,
            domain=pose.domain,
        )

        if not hasattr(data, "config") or data.config is None:
            data.config = {}

        existing_readouts = data.config.get("multitask_readout")
        if existing_readouts is None:
            data.config["multitask_readout"] = [{"readout_id": self.readout_id}]
            return data

        if not any(
            readout.get("readout_id") == self.readout_id
            for readout in existing_readouts
            if isinstance(readout, dict)
        ):
            existing_readouts.append({"readout_id": self.readout_id})

        return data

# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "moabb~=1.4.3",
#   "scikit-learn~=1.7.2",
# ]
# ///

"""Physionet Motor Imagery pipeline.

Downloads and preprocesses the Physionet EEG Motor Movement/Imagery dataset
(Schalk, Wolpaw et al. 2009) via MOABB, producing per-run HDF5 files with
EEG signals, channel metadata, motor-imagery trial intervals, and
cross-validation splits.

Run with:
    brainsets prepare ./brainsets_pipelines/schalk_wolpaw_physionet_2009 --local --use-active-env
"""

from __future__ import annotations

import logging

from torch_brain.data import BrainsetDescription

from foundry.pipelines.moabb_base import MOABBPipeline

logging.basicConfig(level=logging.INFO)

_EVENT_ID_MAPPING = {
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "hands": "hands",
    "feet": "feet",
    "rest": "rest",
}


class Pipeline(MOABBPipeline):
    brainset_id = "schalk_wolpaw_physionet_2009"

    trial_attr_name = "motor_imagery_trials"
    _event_id_mapping = _EVENT_ID_MAPPING
    max_trial_duration = 4.2

    @property
    def brainset_description(self) -> BrainsetDescription:
        return BrainsetDescription(
            id=self.brainset_id,
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://physionet.org/content/eegmmidb/1.0.0/",
            description=(
                "EEG Motor Movement/Imagery Dataset. 109 subjects performing "
                "motor execution and motor imagery tasks with 64-channel EEG."
            ),
        )

    @classmethod
    def moabb_dataset_cls(cls):
        from moabb.datasets import PhysionetMI

        return PhysionetMI()

    @classmethod
    def moabb_paradigm_cls(cls):
        from moabb.paradigms import MotorImagery

        return MotorImagery()

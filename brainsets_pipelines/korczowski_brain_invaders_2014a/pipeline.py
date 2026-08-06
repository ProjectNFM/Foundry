# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "moabb~=1.4.3",
#   "scikit-learn~=1.7.2",
# ]
# ///

"""Brain Invaders P300 pipeline.

Downloads and preprocesses the Brain Invaders 2014a P300 dataset
(Korczowski et al. 2014) via MOABB, producing per-run HDF5 files with
EEG signals, channel metadata, P300 trial intervals, and
cross-validation splits.

Run with:
    brainsets prepare ./brainsets_pipelines/korczowski_brain_invaders_2014a --local --use-active-env
"""

from __future__ import annotations

import logging

from torch_brain.data import BrainsetDescription

from foundry.pipelines.moabb_base import MOABBPipeline

logging.basicConfig(level=logging.INFO)

_EVENT_ID_MAPPING = {
    "Target": "Target",
    "NonTarget": "NonTarget",
}


class Pipeline(MOABBPipeline):
    brainset_id = "korczowski_brain_invaders_2014a"

    trial_attr_name = "p300_trials"
    _event_id_mapping = _EVENT_ID_MAPPING
    epoch_duration = 1.0

    @property
    def brainset_description(self) -> BrainsetDescription:
        return BrainsetDescription(
            id=self.brainset_id,
            origin_version="1.0.0",
            derived_version="2.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.BI2014a.html#moabb.datasets.BI2014a",
            description=(
                "Brain Invaders 2014a P300 dataset. 64 subjects performing a "
                "P300-based BCI task with 16-channel EEG."
            ),
        )

    @classmethod
    def moabb_dataset_cls(cls):
        from moabb.datasets import BI2014a

        return BI2014a()

    @classmethod
    def moabb_paradigm_cls(cls):
        from moabb.paradigms import P300

        return P300()

# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

"""Getzmann Resting-State EEG pipeline (OpenNeuro ds005385).

Downloads and preprocesses the Getzmann et al. (2024) resting-state EEG
dataset (Dortmund Vital Study) from OpenNeuro, producing per-recording
HDF5 files with EEG signals, channel metadata, and domain information.

Dataset: 608 subjects (baseline), 208 with follow-up. 64 EEG channels,
1000 Hz, EDF format. Four conditions per session: eyes-closed/eyes-open
crossed with pre/post cognitive battery.

Reference:
    Getzmann et al. (2024). Scientific Data.
    DOI: 10.1038/s41597-024-03797-w

Run with:
    brainsets prepare ./brainsets_pipelines/getzmann_resting_ds005385 --local --use-active-env
"""

from torch_brain.pipeline.openneuro import OpenNeuroPipeline


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"
    dataset_id = "ds005385"
    brainset_id = "getzmann_resting_ds005385"
    description = (
        "Getzmann et al. (2024) resting-state EEG dataset from the Dortmund "
        "Vital Study. 608 subjects at baseline (208 with 5-year follow-up), "
        "64-channel EEG at 1000 Hz. Eyes-closed and eyes-open recordings "
        "before and after a cognitive test battery."
    )
    origin_version = "1.0.3"
    derived_version = "1.0.0"

    IGNORE_CHANNELS = ["Status"]

# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

"""Pavlov Verbal Working Memory pipeline (OpenNeuro ds003655).

Downloads and preprocesses the Pavlov & Kotchoubey (2021) verbal working
memory EEG dataset from OpenNeuro, producing per-recording HDF5 files with
EEG signals, channel metadata, and domain information.

Dataset: 156 subjects, 19 EEG + 2 EOG channels, 500 Hz, EEGLAB format.
Paradigm: Modified Sternberg verbal working memory task (retention vs
manipulation, 5/6/7 letter load).

Reference:
    Pavlov & Kotchoubey (2021). Scientific Reports.
    DOI: 10.1038/s41598-020-72940-5

Run with:
    brainsets prepare ./brainsets_pipelines/pavlov_verbal_wm_ds003655 --local --use-active-env
"""

from torch_brain.pipeline.openneuro import OpenNeuroPipeline


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"
    dataset_id = "ds003655"
    brainset_id = "pavlov_verbal_wm_ds003655"
    description = (
        "Pavlov & Kotchoubey (2021) verbal working memory EEG dataset. "
        "156 subjects performing a modified Sternberg task (retention vs "
        "manipulation, 5/6/7 letter load) with 19-channel EEG + 2 EOG."
    )
    origin_version = "1.0.2"
    derived_version = "1.0.0"

    IGNORE_CHANNELS = ["EOGv", "EOGh"]

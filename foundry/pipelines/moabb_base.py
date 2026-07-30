"""Shared base class for MOABB-based brainset pipelines.

Provides common logic for downloading MOABB datasets, extracting EEG signals
and event-based trials, and writing standardised HDF5 files that torch_brain
can consume.

Subclasses set ``moabb_dataset_cls``, ``moabb_paradigm_cls``, ``brainset_id``,
``trial_attr_name``, and ``_event_id_mapping`` then inherit the full download →
process → split pipeline.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from torch_brain.data import (
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.mne import (
    extract_channels,
    extract_measurement_date,
    extract_signal,
)
from torch_brain.utils.split import (
    generate_stratified_folds,
    generate_string_kfold_assignment,
)

logging.basicConfig(level=logging.INFO)

_parser = ArgumentParser()
_parser.add_argument("--redownload", action="store_true")
_parser.add_argument("--reprocess", action="store_true")


class MOABBPipeline(BrainsetPipeline):
    """Base pipeline for MOABB-sourced EEG datasets.

    Subclasses must define:
        - ``brainset_id``              – unique brainset identifier string
        - ``moabb_dataset_cls()``      – @classmethod returning a MOABB dataset *instance*
        - ``moabb_paradigm_cls()``     – @classmethod returning a MOABB paradigm *instance*
        - ``trial_attr_name``          – name for trial intervals on ``Data`` (e.g. ``"motor_imagery_trials"``)
        - ``_event_id_mapping``        – dict mapping raw MNE event descriptions to canonical label strings
        - ``brainset_description``     – ``BrainsetDescription`` instance
        - ``max_trial_duration``       – optional cap on trial length (seconds)
    """

    parser = _parser

    # Subclasses must define moabb_dataset_cls() and moabb_paradigm_cls()
    # as @classmethods returning an *instance* of the MOABB dataset / paradigm.
    trial_attr_name: str
    _event_id_mapping: dict[str, str]
    max_trial_duration: float | None = None

    @property
    @abstractmethod
    def brainset_description(self) -> BrainsetDescription: ...

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @classmethod
    def get_manifest(
        cls, raw_dir: Path, args: Namespace | None
    ) -> pd.DataFrame:
        dataset = cls.moabb_dataset_cls()
        rows = []
        for subject_id in dataset.subject_list:
            rows.append(
                {
                    "subject_id": str(subject_id),
                }
            )
        return pd.DataFrame(rows).set_index("subject_id")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, manifest_item) -> dict[str, Any]:
        self.update_status("DOWNLOADING")
        subject_id = int(manifest_item.Index)
        dataset = self.moabb_dataset_cls()
        paradigm = self.moabb_paradigm_cls()

        raw_dict = dataset.get_data(subjects=[subject_id])
        return {
            "subject_id": subject_id,
            "raw_dict": raw_dict,
            "paradigm": paradigm,
        }

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, download_output: dict[str, Any]) -> None:
        self.update_status("PROCESSING")
        subject_id = download_output["subject_id"]
        raw_dict = download_output["raw_dict"]

        subject_data = raw_dict[subject_id]

        for session_name, session_runs in subject_data.items():
            for run_name, raw in session_runs.items():
                session_id = f"sub{subject_id:03d}_{session_name}_{run_name}"
                output_path = self.processed_dir / f"{session_id}.h5"

                if output_path.exists() and not self.args.reprocess:
                    logging.info(f"Skipping {session_id}, file exists.")
                    continue

                self._process_single_run(
                    raw, subject_id, session_id, output_path
                )

    def _process_single_run(
        self, raw, subject_id: int, session_id: str, output_path: Path
    ) -> None:
        raw.load_data()

        self.update_status("Extracting metadata")
        recording_date = extract_measurement_date(raw)

        subject = SubjectDescription(
            id=f"sub{subject_id:03d}",
            species="HOMO_SAPIENS",
        )
        session = SessionDescription(
            id=session_id, recording_date=recording_date
        )
        device = DeviceDescription(id=session_id, recording_tech="EEG")

        self.update_status("Extracting signals")
        eeg = extract_signal(raw)
        channels = extract_channels(raw)

        self.update_status("Extracting trials")
        trials = self._extract_trials(raw, eeg)

        self.update_status("Creating splits")
        splits = self._create_splits(
            trials, subject.id, session.id, n_folds=3, seed=42
        )

        data = Data(
            brainset=self.brainset_description,
            subject=subject,
            session=session,
            device=device,
            eeg=eeg,
            channels=channels,
            splits=splits,
            domain=eeg.domain,
            **{self.trial_attr_name: trials},
        )

        self.update_status("Storing")
        with h5py.File(output_path, "w") as f:
            data.to_hdf5(f)

        logging.info(f"Saved {session_id} to {output_path}")

    # ------------------------------------------------------------------
    # Trial extraction
    # ------------------------------------------------------------------

    def _extract_trials(self, raw, eeg) -> Interval:
        """Extract event-based trials from MNE Raw annotations.

        Uses the ``_event_id_mapping`` to filter and label events. Trial end
        times are estimated from inter-event gaps but capped at
        ``max_trial_duration`` to avoid abnormally long final trials.
        """
        annotations = raw.annotations
        recording_duration = raw.n_times / raw.info["sfreq"]

        starts = []
        ends = []
        labels = []
        label_ids = []

        label_to_id = {
            label: idx
            for idx, label in enumerate(
                sorted(set(self._event_id_mapping.values()))
            )
        }

        event_onsets = []
        event_labels_raw = []
        for onset, _, desc in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            if desc in self._event_id_mapping:
                event_onsets.append(onset)
                event_labels_raw.append(desc)

        for i, (onset, desc) in enumerate(zip(event_onsets, event_labels_raw)):
            label = self._event_id_mapping[desc]

            start = onset
            if i + 1 < len(event_onsets):
                end = event_onsets[i + 1]
            else:
                end = recording_duration

            if self.max_trial_duration is not None:
                end = min(end, start + self.max_trial_duration)

            starts.append(start)
            ends.append(end)
            labels.append(label)
            label_ids.append(label_to_id[label])

        if not starts:
            raise ValueError("No matching trial events found in annotations.")

        trial_field_name = self._get_trial_value_field()
        return Interval(
            start=np.array(starts),
            end=np.array(ends),
            id=np.array(label_ids, dtype=np.int64),
            **{trial_field_name: np.array(labels)},
        )

    def _get_trial_value_field(self) -> str:
        """Return the name of the value field stored on trial intervals.

        By default derived from ``trial_attr_name``:
        ``motor_imagery_trials`` → ``movements``, ``p300_trials`` → ``targets``.
        Subclasses may override.
        """
        mapping = {
            "motor_imagery_trials": "movements",
            "p300_trials": "targets",
        }
        return mapping.get(self.trial_attr_name, "labels")

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        trials: Interval,
        subject_id: str,
        session_id: str,
        n_folds: int = 3,
        seed: int = 42,
    ) -> Data:
        """Generate intrasession + intersubject splits, following the Kemp pattern."""
        folds = generate_stratified_folds(
            trials,
            stratify_by="id",
            n_folds=n_folds,
            val_ratio=0.2,
            seed=seed,
        )

        folds_dict = {f"fold_{i}": fold for i, fold in enumerate(folds)}
        splits = Data(**folds_dict, domain=trials)

        subject_assignments = generate_string_kfold_assignment(
            string_id=subject_id,
            n_folds=n_folds,
            val_ratio=0.2,
            seed=seed,
        )
        session_assignments = generate_string_kfold_assignment(
            string_id=f"{subject_id}_{session_id}",
            n_folds=n_folds,
            val_ratio=0.2,
            seed=seed,
        )

        for fold_idx, assignment in enumerate(subject_assignments):
            setattr(
                splits, f"intersubject_fold_{fold_idx}_assignment", assignment
            )
        for fold_idx, assignment in enumerate(session_assignments):
            setattr(
                splits, f"intersession_fold_{fold_idx}_assignment", assignment
            )

        return splits

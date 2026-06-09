import numpy as np
from temporaldata import Data

class StandardizeSignal:

    def __init__(self, field: str = "ieeg"):
        self.field = field
        self.mean = None
        self.std = None

    def fit(self, dataset, split="train"):

        intervals = dataset.get_sampling_intervals(split=split)

        samples = []
        for recording_id, intervals in intervals.items():
            recording = dataset.get_recording(recording_id)

            signal = getattr(recording, self.field).signal
            samples.append(signal)
        
        all_recordings = np.concatenate(samples, axis=0)
        self.mean = all_recordings.mean(axis=0)
        self.std = all_recordings.std(axis=0)

    def __call__(self, data: Data) -> Data:
        signal_data = getattr(data, self.field)
        signal_data.signal = (signal_data.signal - self.mean) / self.std
        return data


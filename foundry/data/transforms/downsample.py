import numpy as np
from math import gcd
from scipy.signal import resample_poly
from temporaldata import Data

class DownsampleSignal:

    def __init__(self, field: str, target_sfreq: float, original_sfreq: float = 2000.0):
        self.field = field
        self.target_sfreq = target_sfreq
        self.original_sfreq = original_sfreq

    def __call__(self, data: Data) -> Data:
        signal_data = getattr(data, self.field)

        if self.original_sfreq == self.target_sfreq:
            return data

        if self.target_sfreq > self.original_sfreq:
            raise ValueError(f"target_sfreq {self.target_sfreq} Hz is greater than the signals original frequency ({self.original_sfreq} Hz).")

        g = gcd(int(self.target_sfreq), int(self.original_sfreq))

        up = int(self.target_sfreq) // g
        down = int(self.original_sfreq) // g

        new_signal = resample_poly(signal_data.signal, up, down, axis=0)

        # Timestamp adjustment

        n_new = new_signal.shape[0]

        new_timestamps = (
            signal_data.timestamps[0]
            + np.arange(n_new, dtype=np.float64) / self.target_sfreq
        )

        new_signal_data = type(signal_data)(
            domain=signal_data.domain,
            signal=new_signal,
            timestamps=new_timestamps
        )

        setattr(data, self.field, new_signal_data)
        return data

        
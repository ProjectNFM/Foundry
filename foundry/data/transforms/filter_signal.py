import numpy as np
from scipy import signal as sp_signal
from temporaldata import Data
from typing import Optional, Union

FreqSpec = Union[float, tuple[float, float]]

class FilterSignal:
    """
    Applies notch and/or bandpass filters to a neural signal field.
 
    Filters are specified as a list of ``FreqSpec`` entries and a ``filter_types``
    list of equal length that tags each entry as ``"notch"``, ``"bandstop"``, or
    ``"bandpass"``:
 
    - ``"notch"``    — narrow stopband at a single frequency (e.g. 50 Hz line noise).
                       FreqSpec must be a single float.
    - ``"bandstop"`` — wider stopband over a frequency range (e.g. (48.0, 52.0)).
                       FreqSpec must be a (low, high) tuple.
    - ``"bandpass"`` — retain only the specified band (e.g. (1.0, 200.0)).
                       FreqSpec must be a (low, high) tuple.
 
    Filters are applied in the order given.
 
    Args:
        field:                Signal attribute on the Data object (default ``"ecog"``).
        freqs:                List of FreqSpec entries — float for notch, (low, high) tuple
                              for bandstop/bandpass.
        filter_types:         List of filter type strings, one per entry in ``freqs``.
        notch_quality_factor: Q factor for notch filters; higher = narrower notch
                              (default 30.0, ~1.7 Hz bandwidth at 50 Hz).
        bandstop_order:       Butterworth order for bandstop filters (default 4).
        bandpass_order:       Butterworth order for bandpass filters (default 4).
        sfreq:                Sampling frequency in Hz. If ``None``, inferred from the
                              median inter-sample interval of the signal timestamps.
 
    Example — remove 50 Hz + harmonics, then keep 1-200 Hz:
 
        FilterSignal(
            freqs=[50.0, 100.0, 150.0, (1.0, 200.0)],
            filter_types=["notch", "notch", "notch", "bandpass"],
        )
    """


    def __init__(
            self,
            field: str = "ecog",
            freqs: Optional[list[FreqSpec]] = None,
            filter_types: Optional[list[str]] = None,
            notch_quality_factor: float = 30.0,
            bandstop_order: int = 4,
            bandpass_order: int = 4,
            sfreq: Optional[float] = None,
        ):

        if len(freqs) != len(filter_types):
            raise ValueError(
                f"Length of freqs and filter_types must be the same length, instead got {len(freqs)} and {len(filter_types)}"
            )
        
        valid_types = {"notch", "bandstop", "bandpass"}

        for i, (f, ft) in enumerate(zip(freqs, filter_types)):
            if ft not in valid_types:
                raise ValueError("filter_types[{i}]={ft!r} is not valid. "
                                    f"Choose from {valid_types}.")
            
            if ft == "notch" and not isinstance(f, (int, float)):
                raise ValueError(f"filter_types[{i}]='notch' requires a single float freq, got {f!r}.")
            
            if ft in ("bandstop", "bandpass") and (not isinstance(f, (list, tuple)) or len(f) != 2):
                raise ValueError(f"filter_types[{i}]={ft!r} requires a (low, high) tuple, got {f!r}.")

        self.field = field
        self.freqs = freqs
        self.filter_types = filter_types
        self.notch_quality_factor = notch_quality_factor
        self.bandstop_order = bandstop_order
        self.bandpass_order = bandpass_order
        self.sfreq = sfreq

        self._sos_cache: dict = {}

    @staticmethod
    def _infer_sfreq(timestamps: np.ndarray) -> float:
        diffs = np.diff(timestamps)
        return float(1.0 / np.median(diffs))

    def _get_sos(self, freq: FreqSpec, filter_type: str, sfreq: float) -> np.ndarray:
        key = (freq if isinstance(freq, float) else tuple(freq), filter_type, sfreq)
        if key in self._sos_cache:
            return self._sos_cache[key]
        
        nyq = sfreq / 2.0

        if filter_type == "notch":
            b, a = sp_signal.iirnotch(freq, self.notch_quality_factor, sfreq)
            sos = sp_signal.tf2sos(b, a)

        elif filter_type == "bandstop":
            low, high = freq
            self._check_band(low, high, nyq, filter_type)
            sos = sp_signal(self.bandpass_order, [low / nyq, high / nyq], btype="bandpass", output="sos")
        
        else: # Bandpass
            low, high = freq
            self._check_band(low, high, nyq, filter_type)
            sos = sp_signal(self.bandpass_order, [low / nyq, high / nyq], btype="bandapss", output="sos")

        self._sos_cache[key] = sos
        return sos
            
    @staticmethod
    def _check_band(low: float, high: float, nyq: float, filter_type: str) -> None:
        if low <= 0 or high <= 0:
            raise ValueError(
                f"{filter_type} must have positive frequencies, instead given band ({low}, {high})"
            )
        
        if low > high:
            raise ValueError(
                f"{filter_type} must have low < high, instead given band ({low}, {high})"
            )
        
        if high >= nyq:
            raise ValueError(
                f"{filter_type} must have band upper edge ({high} Hz) below the Nyquist frequency {nyq:.1f} Hz."
            )
        
    def __call__(self, data: Data) -> Data:
        if not self.freqs:
            return data
        
        signal_data = getattr(data, self.field)
        signal = signal_data.signal.astype(np.float64)

        sfreq = self.sfreq if self.sfreq is not None else self._infer_sfreq(signal_data.timestamps)

        min_samples = 3 * (2 * max(self.bandpass_order, self.bandstop_order)) + 1
        if signal.shape[0] < min_samples:
            return data
        
        for freq, filter_type in zip(self.freqs, self.filter_types):

            if filter_type == "notch" and freq >= sfreq / 2.0:
                continue
            sos = self._get_sos(freq, filter_type, sfreq)
            signal = sp_signal.sosfiltfilt(sos, signal, axis=0)

            signal_data.signal = signal.astype(signal_data.signal.dtype)
            return data
import numpy as np
from temporaldata import Data

class BaselineSignal:
    """
    Baseline the ECoG signal for each trial using the pre-stimulus (off) window
    immediately preceding trial onset.

    For each trial in `trials_field`, computes the mean signal over
    [trial.start - baseline_duration, trial.start) and subtracts it
    from the signal within [trial.start, trial.end).

    Args:
        field: Name of the signal attribute on the Data object (e.g. "ecog").
        trials_field: Name of the trials Interval attribute (e.g. "acoustic_stim_trials").
        baseline_duration: Length in seconds of the pre-trial baseline window.
    """
    def __init__(
            self,
            field: str = "ecog",
            trials_field: str = "acoustic_stim_trials",
            baseline_duration: float = 0.5,
        ):

        self.field = field
        self.trails_field = trials_field
        self.baseline_duration = baseline_duration

    def __call__(self, data: Data) -> Data:
        signal_data = getattr(data, self.field)
        trials = getattr(data, self.trails_field)

        timestamps = signal_data.timestamps
        signal = signal_data.signal

        starts = np.asarray(trials.start)
        ends = np.asarray(trials.end)

        for t_start, t_end in zip(starts, ends):
            t_baseline_start = t_start - self.baseline_duration

            baseline_mask = (timestamps >= t_baseline_start) & (timestamps < t_start)
            trial_mask = (timestamps >= t_start) & (timestamps < t_end)

            if baseline_mask.sum() == 0 or trial_mask == 0:
                continue

            baseline_mean = signal[baseline_mask].mean(axis=0)
            signal[trial_mask] -= baseline_mean

        return data
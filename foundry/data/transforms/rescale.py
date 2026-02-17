from temporaldata import Data


class RescaleEEG:
    """Rescale EEG transform that multiplies data by a scaling factor.

    This transform takes a temporalData object and rescales the EEG data by multiplying
    with the specified scaling factor. Assumes that the EEG data is in the 'eeg' field.

    Args:
        factor (float): Scaling factor to multiply the EEG data by.
    """

    def __init__(self, factor: float = 1e5):
        self.factor = factor

    def __call__(self, data: Data) -> Data:
        """Apply rescaling EEG transform to the data.

        Args:
            data: The temporalData object to rescale.

        Returns:
            A new Data object with rescaled EEG time series fields.
        """
        if not hasattr(data, "eeg") or data.eeg is None:
            raise ValueError("Data must have an 'eeg' field")

        data.eeg.signal *= self.factor

        return data

from temporaldata import Data


class RescaleSignal:
    """Rescale temporal signal by a scaling factor.

    This transform is modality-agnostic and works with any neural data field
    (EEG, iEEG, fMRI, PET, etc.) by specifying the field name.

    Args:
        factor: Scaling factor to multiply the signal by.
        field: Name of the data field to rescale (default: "eeg").
    """

    def __init__(self, factor: float = 1e5, field: str = "eeg"):
        self.factor = factor
        self.field = field

    def __call__(self, data: Data) -> Data:
        """Apply rescaling transform to the data.

        Args:
            data: The temporalData object to rescale.

        Returns:
            A new Data object with rescaled time series fields.
        
        Raises:
            ValueError: If the specified field does not exist or is None.
        """
        if not hasattr(data, self.field) or getattr(data, self.field) is None:
            raise ValueError(f"Data must have a '{self.field}' field")

        signal_data = getattr(data, self.field)
        signal_data.signal *= self.factor

        return data
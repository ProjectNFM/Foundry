import numpy as np
from scipy.stats import kurtosis
from temporaldata import Data

class DetectBadChannels:

	def __init__(self,
			  field: str = "ecog",
			  flat_threshold: float = 1e-10,
			  noise_z_threshold: float = 3.0,
			  kurtosis_threshold: float = 5.0,
	):
		self.field = field
		self.flat_threshold = flat_threshold
		self.noise_z_threshold = noise_z_threshold
		self.kurtosis_threshold = kurtosis_threshold
		self.bad_channels_per_session = {}

	def _detect(self, signal: np.ndarray) -> np.ndarray:
		channel_var		= signal.var(axis=0)
		channel_kurt	= kurtosis(signal, axis=0)

		flat = channel_var < self.flat_threshold

		var_z = (channel_var - channel_var.mean()) / channel_var.std()
		noisy = var_z > self.noise_z_threshold

		epileptic = channel_kurt > self.kurtosis_threshold

		return np.where(flat | noisy | epileptic)[0]

	def fit(self, dataset, split="train"):
		intervals = dataset.get_sampling_intervals(split=split)
		for recording_id, intervals in intervals.items(): 
			recording = dataset.get_recording(recording_id)
			signal = getattr(recording, self.field).signal
			bad = self._detect(signal)
			self.bad_channels_per_session[recording_id] = bad
	
	def __call__(self, data: Data) -> Data:
		signal_data = getattr(data, self.field)

		# TODO: Change from hardcoded value
		recording_id = getattr(data, "acoustic_stim_trials").recording_id

		# Sets bad channels signal to 0, change here if instead we want to tag bad channels
		bad = self.bad_channels_per_session.get(str(recording_id), np.array([]))
		if len(bad) > 0:
			signal_data.signal[:, bad] = 0.0
		return data
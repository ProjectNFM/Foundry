import numpy as np
from temporaldata import Data

class CARSignal:

	def __init__(self, field: str = "ieeg"):
		self.field = field

	def __call__(self, data: Data) -> Data:
		signal_data = getattr(data, self.field)

		signal_data.signal = signal_data.signal - signal_data.signal.mean(axis=1, keepdims=True)
		return data

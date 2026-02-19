import numpy as np
import pytest
from temporaldata import Data, Interval, RegularTimeSeries

from foundry.data.transforms import RescaleSignal


class TestRescaleSignal:
    def test_initialization_default(self):
        transform = RescaleSignal()
        assert transform.factor == 1e5

    def test_initialization_custom_factor(self):
        transform = RescaleSignal(factor=2.0)
        assert transform.factor == 2.0

    def test_rescale_with_default_factor(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = data_array.copy() * 1e5

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal()
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_multiply_by_two(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = data_array.copy() * 2.0

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=2.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_multiply_by_half(self):
        data_array = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        expected = data_array.copy() * 0.5

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=0.5)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_multiply_by_one(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=1.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, data_array)

    def test_rescale_negative_values(self):
        data_array = np.array([[-5.0, 10.0], [-15.0, 20.0]])
        expected = data_array.copy() * 2.0

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=2.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_zero_values(self):
        data_array = np.zeros((5, 3))
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=10.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, 0.0)

    def test_rescale_modifies_in_place(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = data_array.copy() * 2.0

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        original_id = id(data)
        original_signal_id = id(data.eeg.signal)

        transform = RescaleSignal(factor=2.0)
        result = transform(data)

        assert id(result) == original_id
        assert id(result.eeg.signal) == original_signal_id
        assert np.allclose(data.eeg.signal, expected)

    def test_rescale_preserves_domain(self):
        data_array = np.random.randn(100, 5)
        domain = Interval(2.5, 7.5)
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=200.0,
                domain=domain,
            ),
            domain=domain,
        )

        transform = RescaleSignal(factor=3.0)
        result = transform(data)

        assert result.eeg.domain.start == domain.start
        assert result.eeg.domain.end == domain.end
        assert result.domain.start == domain.start
        assert result.domain.end == domain.end

    def test_rescale_preserves_sampling_rate(self):
        data_array = np.random.randn(100, 5)
        sampling_rate = 250.0
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=sampling_rate,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=0.1)
        result = transform(data)

        assert result.eeg.sampling_rate == sampling_rate

    def test_rescale_shape_preserved(self):
        data_array = np.random.randn(50, 10, 20)
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=5.0)
        result = transform(data)

        assert result.eeg.signal.shape == data_array.shape

    def test_rescale_large_factor(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = data_array.copy() * 1e6

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=1e6)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_small_factor(self):
        data_array = np.array([[100.0, 200.0], [300.0, 400.0]])
        expected = data_array.copy() * 0.001

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=0.001)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_missing_eeg_field(self):
        data = Data(
            ecg=RegularTimeSeries(
                signal=np.array([[1.0, 2.0]]),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=2.0)

        with pytest.raises(ValueError, match="Data must have an 'eeg' field"):
            transform(data)

    def test_rescale_none_eeg_field(self):
        data = Data(domain=Interval(0.0, 1.0))
        data.eeg = None

        transform = RescaleSignal(factor=2.0)

        with pytest.raises(ValueError, match="Data must have an 'eeg' field"):
            transform(data)

    def test_rescale_negative_factor(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = data_array.copy() * -2.0

        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=-2.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, expected)

    def test_rescale_zero_factor(self):
        data_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = Data(
            eeg=RegularTimeSeries(
                signal=data_array,
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=0.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, 0.0)

    def test_rescale_only_affects_eeg(self):
        eeg_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        ecg_array = np.array([[10.0, 20.0], [30.0, 40.0]])

        data = Data(
            eeg=RegularTimeSeries(
                signal=eeg_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            ecg=RegularTimeSeries(
                signal=ecg_array.copy(),
                sampling_rate=100.0,
                domain=Interval(0.0, 1.0),
            ),
            domain=Interval(0.0, 1.0),
        )

        transform = RescaleSignal(factor=3.0)
        result = transform(data)

        assert np.allclose(result.eeg.signal, eeg_array * 3.0)
        assert np.allclose(result.ecg.signal, ecg_array)

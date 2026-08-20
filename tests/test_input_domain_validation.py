import numpy as np
import pytest

from rtichoke import prepare_performance_data, prepare_performance_data_times


def test_binary_probabilities_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="probabilities must be between 0 and 1"):
        prepare_performance_data(
            probs={"model": np.array([0.1, 1.1])},
            reals=np.array([0, 1]),
            by=0.5,
        )


def test_binary_outcomes_must_be_zero_or_one():
    with pytest.raises(ValueError, match="binary outcomes must contain only 0 and 1"):
        prepare_performance_data(
            probs={"model": np.array([0.1, 0.9])},
            reals=np.array([0, 2]),
            by=0.5,
        )


def test_time_probabilities_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="probabilities must be between 0 and 1"):
        prepare_performance_data_times(
            probs={"model": np.array([-0.1, 0.9])},
            reals=np.array([0, 1]),
            times=np.array([1.0, 2.0]),
            fixed_time_horizons=[1.5],
            by=0.5,
        )


def test_time_outcomes_must_use_supported_event_codes():
    with pytest.raises(ValueError, match="time-dependent outcomes must contain only 0, 1, and 2"):
        prepare_performance_data_times(
            probs={"model": np.array([0.1, 0.9])},
            reals=np.array([0, 3]),
            times=np.array([1.0, 2.0]),
            fixed_time_horizons=[1.5],
            by=0.5,
        )

import numpy as np
import pytest

from rtichoke import prepare_performance_data, prepare_performance_data_times
from rtichoke.processing.combinations import create_breaks_values


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        (0.3, [0.0, 0.3, 0.6, 0.9, 1.0]),
        (0.4, [0.0, 0.4, 0.8, 1.0]),
    ],
)
def test_probability_threshold_breaks_end_at_one_without_overshoot(by, expected):
    breaks = create_breaks_values(None, "probability_threshold", by)

    assert breaks.tolist() == expected
    assert breaks[-1] == 1.0
    assert np.max(breaks) <= 1.0


@pytest.mark.parametrize("by", [0.3, 0.4])
def test_binary_performance_data_never_returns_cutoff_above_one(by):
    probs = {"model": np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])}
    reals = np.array([0, 0, 1, 0, 1, 1])

    result = prepare_performance_data(probs, reals, by=by)
    cutoffs = result.get_column("chosen_cutoff")

    assert cutoffs.max() == 1.0
    assert cutoffs.min() == 0.0
    assert cutoffs.max() <= 1.0


@pytest.mark.parametrize("by", [0.3, 0.4])
def test_time_performance_data_never_returns_cutoff_above_one(by):
    probs = {"model": np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])}
    reals = np.array([0, 0, 1, 0, 1, 1])
    times = np.array([2.0, 4.0, 3.0, 8.0, 5.0, 6.0])

    result = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[6.0],
        by=by,
    )
    cutoffs = result.get_column("chosen_cutoff")

    assert cutoffs.max() == 1.0
    assert cutoffs.min() == 0.0
    assert cutoffs.max() <= 1.0

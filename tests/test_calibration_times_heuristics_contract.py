import numpy as np
import pytest

from rtichoke import (
    create_calibration_curve_times as create_calibration_curve_times_top_level,
)
from rtichoke.calibration import create_calibration_curve_times
from rtichoke.calibration.calibration import (
    create_calibration_curve_times as create_calibration_curve_times_direct,
)


ENTRY_POINTS = [
    create_calibration_curve_times_top_level,
    create_calibration_curve_times,
    create_calibration_curve_times_direct,
]


def _inputs():
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([0, 1, 1, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])
    return probs, reals, times


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_time_calibration_defaults_to_adjusted_heuristics(entry_point):
    probs, reals, times = _inputs()

    default_fig = entry_point(
        probs,
        reals,
        times,
        fixed_time_horizons=[2.0],
    )
    explicit_fig = entry_point(
        probs,
        reals,
        times,
        fixed_time_horizons=[2.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
    )

    np.testing.assert_allclose(default_fig.data[1].x, explicit_fig.data[1].x)
    np.testing.assert_allclose(default_fig.data[1].y, explicit_fig.data[1].y)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_time_calibration_rejects_multiple_heuristic_sets(entry_point):
    probs, reals, times = _inputs()

    with pytest.raises(ValueError, match="exactly one heuristic set"):
        entry_point(
            probs,
            reals,
            times,
            fixed_time_horizons=[2.0],
            heuristics_sets=[
                {
                    "censoring_heuristic": "adjusted",
                    "competing_heuristic": "adjusted_as_negative",
                },
                {
                    "censoring_heuristic": "excluded",
                    "competing_heuristic": "excluded",
                },
            ],
        )

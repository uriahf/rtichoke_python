import numpy as np
import pytest

from rtichoke import create_calibration_curve_times as create_calibration_curve_times_top_level
from rtichoke.calibration import create_calibration_curve_times
from rtichoke.calibration.calibration import (
    create_calibration_curve_times as create_calibration_curve_times_direct,
)


@pytest.mark.parametrize(
    "entry_point",
    [
        create_calibration_curve_times_top_level,
        create_calibration_curve_times,
        create_calibration_curve_times_direct,
    ],
)
def test_create_calibration_curve_times_rejects_multiple_heuristic_sets(entry_point):
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([0, 1, 0, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    heuristics_sets = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "excluded",
        },
    ]

    with pytest.raises(ValueError, match="exactly one heuristics set"):
        entry_point(
            probs,
            reals,
            times,
            fixed_time_horizons=[2.0],
            heuristics_sets=heuristics_sets,
        )


def test_create_calibration_curve_times_allows_one_heuristic_set():
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([0, 1, 0, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    fig = create_calibration_curve_times(
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

    assert fig is not None

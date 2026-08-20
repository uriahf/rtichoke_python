import numpy as np
import pytest

from rtichoke import (
    create_calibration_curve as create_calibration_curve_top_level,
    create_calibration_curve_times as create_calibration_curve_times_top_level,
)
from rtichoke.calibration import (
    create_calibration_curve,
    create_calibration_curve_times,
)
from rtichoke.calibration.calibration import (
    create_calibration_curve as create_calibration_curve_direct,
    create_calibration_curve_times as create_calibration_curve_times_direct,
)


BINARY_ENTRY_POINTS = [
    create_calibration_curve_top_level,
    create_calibration_curve,
    create_calibration_curve_direct,
]
TIME_ENTRY_POINTS = [
    create_calibration_curve_times_top_level,
    create_calibration_curve_times,
    create_calibration_curve_times_direct,
]


@pytest.mark.parametrize("entry_point", BINARY_ENTRY_POINTS)
def test_binary_calibration_rejects_probabilities_outside_unit_interval(entry_point):
    with pytest.raises(ValueError, match="between 0 and 1"):
        entry_point(
            probs={"model": np.array([0.1, 1.1])},
            reals=np.array([0, 1]),
        )


@pytest.mark.parametrize("entry_point", BINARY_ENTRY_POINTS)
def test_binary_calibration_rejects_nonbinary_outcomes(entry_point):
    with pytest.raises(ValueError, match="only 0 and 1"):
        entry_point(
            probs={"model": np.array([0.1, 0.9])},
            reals=np.array([0, 2]),
        )


@pytest.mark.parametrize("entry_point", TIME_ENTRY_POINTS)
def test_time_calibration_rejects_probabilities_outside_unit_interval(entry_point):
    with pytest.raises(ValueError, match="between 0 and 1"):
        entry_point(
            probs={"model": np.array([-0.1, 0.9])},
            reals=np.array([0, 1]),
            times=np.array([1.0, 2.0]),
            fixed_time_horizons=[1.5],
        )


@pytest.mark.parametrize("entry_point", TIME_ENTRY_POINTS)
def test_time_calibration_rejects_unsupported_event_codes(entry_point):
    with pytest.raises(ValueError, match="only 0, 1, and 2"):
        entry_point(
            probs={"model": np.array([0.1, 0.9])},
            reals=np.array([0, 3]),
            times=np.array([1.0, 2.0]),
            fixed_time_horizons=[1.5],
        )


def test_calibration_validation_preserves_supported_multiple_population_shape():
    fig = create_calibration_curve(
        probs={
            "population_a": np.array([0.1, 0.8]),
            "population_b": np.array([0.2, 0.9]),
        },
        reals={
            "population_a": np.array([0, 1]),
            "population_b": np.array([0, 1]),
        },
    )

    assert fig is not None

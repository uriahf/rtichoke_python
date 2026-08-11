import numpy as np
from rtichoke.calibration import create_calibration_curve_times


def test_create_calibration_curve_times():
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
    reals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    heuristics_sets = [
        {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"}
    ]

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        heuristics_sets=heuristics_sets,
    )

    assert fig is not None
    assert len(fig.data) > 0
    assert len(fig.layout.sliders) > 0


def test_create_calibration_curve_times_unequal_size_populations():
    probs = {
        "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "Test": np.array([0.2, 0.8, 0.3, 0.7]),
    }
    reals = {
        "Train": np.array([0, 1, 0, 1, 0, 1]),
        "Test": np.array([0, 1, 0, 0]),
    }
    times = {
        "Train": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "Test": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    heuristics_sets = [
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "adjusted_as_negative",
        }
    ]

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[3.0, 6.0],
        heuristics_sets=heuristics_sets,
    )

    assert {trace.name for trace in fig.data if trace.name} >= {"Train", "Test"}
    assert len(fig.layout.sliders) == 1

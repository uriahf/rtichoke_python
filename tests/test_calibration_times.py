import numpy as np
from rtichoke.calibration import create_calibration_curve_times


def test_create_calibration_curve_times():
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
    reals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    fixed_time_horizons = [5, 10]
    heuristics_sets = [
        {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"}
    ]

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
    )

    assert fig is not None
    assert len(fig.data) > 0
    assert len(fig.layout.sliders) > 0

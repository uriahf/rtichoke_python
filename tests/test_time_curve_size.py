import numpy as np
import pytest

from rtichoke import (
    create_decision_curve_times,
    create_lift_curve_times,
    create_precision_recall_curve_times,
    create_roc_curve_times,
)


@pytest.mark.parametrize(
    "curve_function",
    [
        create_roc_curve_times,
        create_precision_recall_curve_times,
        create_lift_curve_times,
        create_decision_curve_times,
    ],
)
def test_time_curve_size_is_forwarded_to_plot_layout(curve_function):
    probs = {"model": np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])}
    reals = np.array([1, 0, 1, 0, 1, 0])
    times = np.array([1.0, 10.0, 2.0, 11.0, 3.0, 12.0])

    fig = curve_function(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0],
        size=420,
        by=0.1,
    )

    assert fig.layout.width == 420
    assert fig.layout.height == 520

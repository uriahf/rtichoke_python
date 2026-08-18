import numpy as np

from rtichoke.calibration import create_calibration_curve, create_calibration_curve_times


def _assert_square_main_panel(fig):
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
    assert fig.layout.yaxis.constrain == "domain"
    # The histogram has its own y axis and must remain unconstrained.
    assert fig.layout.yaxis2.scaleanchor is None


def test_interactive_calibration_main_panel_is_square():
    probs = {"model": np.linspace(0.05, 0.95, 20)}
    reals = np.array([0, 1] * 10)

    for calibration_type in ("discrete", "smooth"):
        fig = create_calibration_curve(
            probs, reals, calibration_type=calibration_type
        )
        _assert_square_main_panel(fig)
        assert list(fig.layout.xaxis.range) == list(fig.layout.yaxis.range)


def test_interactive_calibration_times_main_panel_is_square():
    probs = {"model": np.linspace(0.05, 0.95, 20)}
    reals = np.array([0, 1] * 10)
    times = np.arange(1.0, 21.0)

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[10.0, 15.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "excluded",
                "competing_heuristic": "excluded",
            }
        ],
    )

    _assert_square_main_panel(fig)
    assert list(fig.layout.xaxis.range) == list(fig.layout.yaxis.range)

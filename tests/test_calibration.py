import numpy as np
from rtichoke.calibration.calibration import create_calibration_curve


def test_create_calibration_curve_smooth():
    probs = {"model_1": np.linspace(0, 1, 100)}
    reals = np.random.randint(0, 2, 100)
    fig = create_calibration_curve(probs, reals, calibration_type="smooth")

    # Check if the figure has the correct number of traces (smooth curve, histogram, and reference line)
    assert len(fig.data) == 3

    # Check reference line data
    reference_line = fig.data[0]
    assert reference_line.name == "Perfectly Calibrated"


def test_create_calibration_curve_smooth_single_point():
    probs = {"model_1": np.array([0.5] * 100)}
    reals = np.random.randint(0, 2, 100)
    fig = create_calibration_curve(probs, reals, calibration_type="smooth")

    # Check that the plot mode is "lines+markers"
    assert fig.data[1].mode == "lines+markers"

    # Check histogram data
    histogram = fig.data[2]
    assert histogram.type == "bar"


def test_create_calibration_curve_multiple_populations_unequal_sizes():
    probs = {
        "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "Test": np.array([0.2, 0.8, 0.3, 0.7]),
    }
    reals = {
        "Train": np.array([0, 1, 0, 1, 0, 1]),
        "Test": np.array([0, 1, 0, 0]),
    }

    for calibration_type in ("discrete", "smooth"):
        fig = create_calibration_curve(
            probs, reals, calibration_type=calibration_type
        )
        assert {trace.name for trace in fig.data if trace.name} >= {"Train", "Test"}

import numpy as np
import pytest

from rtichoke import create_calibration_curve


def test_create_calibration_curve_smooth():
    probs = {"model_1": np.linspace(0, 1, 100)}
    reals = np.random.default_rng(1).integers(0, 2, 100)
    fig = create_calibration_curve(probs, reals, calibration_type="smooth")

    assert len(fig.data) == 3
    assert fig.data[0].name == "Perfectly Calibrated"


def test_create_calibration_curve_smooth_single_point():
    probs = {"model_1": np.array([0.5] * 100)}
    reals = np.random.default_rng(2).integers(0, 2, 100)
    fig = create_calibration_curve(probs, reals, calibration_type="smooth")

    assert fig.data[1].mode == "lines+markers"
    assert fig.data[2].type == "bar"


def test_create_calibration_curve_equal_size_populations():
    probs = {
        "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "Test": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
    }
    reals = {
        "Train": np.array([0, 1, 0, 1, 0, 1]),
        "Test": np.array([0, 1, 0, 1, 0, 0]),
    }

    fig = create_calibration_curve(probs, reals)

    assert {trace.name for trace in fig.data if trace.name} >= {"Train", "Test"}


def test_create_calibration_curve_unequal_size_populations_discrete():
    probs = {
        "Train": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        "Test": np.array([0.2, 0.8, 0.3, 0.7]),
    }
    reals = {
        "Train": np.array([0, 1, 0, 1, 0, 1]),
        "Test": np.array([0, 1, 0, 0]),
    }

    fig = create_calibration_curve(probs, reals, calibration_type="discrete")

    assert {trace.name for trace in fig.data if trace.name} >= {"Train", "Test"}


def test_create_calibration_curve_unequal_size_populations_smooth():
    probs = {
        "Train": np.linspace(0.05, 0.95, 20),
        "Test": np.linspace(0.1, 0.9, 12),
    }
    reals = {
        "Train": np.array([0, 1] * 10),
        "Test": np.array([0, 1] * 6),
    }

    fig = create_calibration_curve(probs, reals, calibration_type="smooth")

    assert {trace.name for trace in fig.data if trace.name} >= {"Train", "Test"}


def test_create_calibration_curve_rejects_mismatched_population_keys():
    probs = {"Train": np.array([0.1, 0.9]), "Validation": np.array([0.2, 0.8])}
    reals = {"Train": np.array([0, 1]), "Test": np.array([0, 1])}

    with pytest.raises(ValueError, match="matching population keys"):
        create_calibration_curve(probs, reals)


def test_create_calibration_curve_rejects_within_population_length_mismatch():
    probs = {
        "Train": np.array([0.1, 0.9, 0.2]),
        "Test": np.array([0.2, 0.8]),
    }
    reals = {
        "Train": np.array([0, 1]),
        "Test": np.array([0, 1]),
    }

    with pytest.raises(ValueError, match="population 'Train'"):
        create_calibration_curve(probs, reals)

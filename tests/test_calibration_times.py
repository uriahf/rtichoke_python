import numpy as np
import pytest
import polars as pl
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from rtichoke import (
    create_calibration_curve_times as create_calibration_curve_times_top_level,
)
from rtichoke.calibration import create_calibration_curve_times
from rtichoke.calibration.calibration import (
    _calculate_adjusted_pseudostates,
    _prepare_adjusted_event_data,
    create_calibration_curve_times as create_calibration_curve_times_direct,
)


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


@pytest.mark.parametrize(
    "entry_point",
    [create_calibration_curve_times_top_level, create_calibration_curve_times_direct],
)
def test_create_calibration_curve_times_allows_adjusted_without_censoring(entry_point):
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([1, 1, 1, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    fig = entry_point(
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


@pytest.mark.parametrize(
    "entry_point",
    [create_calibration_curve_times_top_level, create_calibration_curve_times_direct],
)
def test_create_calibration_curve_times_adjusts_independent_censoring(entry_point):
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([0, 1, 1, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    fig = entry_point(
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
    assert len(fig.data) == 3


def test_adjusted_discrete_calibration_uses_kaplan_meier_risk():
    probs = {"model_1": np.repeat(0.5, 4)}
    reals = np.array([1, 0, 1, 0])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[3.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
    )

    calibration_trace = fig.data[1]
    reference = 1 - KaplanMeierFitter().fit(times, reals).predict(3.0)
    assert calibration_trace.y[0] == pytest.approx(reference)


def test_adjusted_discrete_calibration_matches_aalen_johansen():
    probs = {"model_1": np.repeat(0.4, 5)}
    reals = np.array([1, 2, 0, 2, 0])
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[3.5],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
    )

    reference = (
        AalenJohansenFitter().fit(times, reals, event_of_interest=1).predict(3.5)
    )
    assert fig.data[1].y[0] == pytest.approx(reference)


def test_excluded_competing_events_are_horizon_specific():
    data = pl.DataFrame(
        {
            "reference_group": ["model_1"] * 5,
            "prob": [0.4] * 5,
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "real": [1, 2, 0, 2, 0],
        }
    )

    adjusted = _prepare_adjusted_event_data(data, 3.5, "excluded")

    assert adjusted["time"].to_list() == [1.0, 3.0, 4.0, 5.0]
    assert adjusted.filter(pl.col("time") == 4.0)["real"].item() == 2


def test_adjusted_pseudostates_match_leave_one_out_aalen_johansen():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    reals = np.array([1, 0, 2, 0])
    horizon = 3.5
    data = pl.DataFrame(
        {
            "reference_group": ["model_1"] * 4,
            "prob": [0.1, 0.3, 0.6, 0.8],
            "time": times,
            "real": reals,
        }
    )

    actual = _calculate_adjusted_pseudostates(data, horizon)["model_1"]
    full = AalenJohansenFitter().fit(times, reals, event_of_interest=1).predict(horizon)
    expected = []
    for index in range(len(times)):
        keep = np.arange(len(times)) != index
        leave_one_out = (
            AalenJohansenFitter()
            .fit(times[keep], reals[keep], event_of_interest=1)
            .predict(horizon)
        )
        expected.append(len(times) * full - (len(times) - 1) * leave_one_out)

    np.testing.assert_allclose(actual, expected)


def test_adjusted_smooth_calibration_uses_pseudo_observations():
    probs = {"model_1": np.array([0.1, 0.3, 0.6, 0.8])}
    reals = np.array([1, 0, 1, 0])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[3.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        calibration_type="smooth",
    )

    calibration_trace = fig.data[1]
    assert len(calibration_trace.x) == 101
    assert np.isfinite(np.asarray(calibration_trace.y)).all()


@pytest.mark.parametrize(
    "entry_point",
    [create_calibration_curve_times_top_level, create_calibration_curve_times_direct],
)
def test_create_calibration_curve_times_rejects_competing_as_censored(entry_point):
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([1, 1, 1, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Unsupported calibration heuristics"):
        entry_point(
            probs,
            reals,
            times,
            fixed_time_horizons=[2.0],
            heuristics_sets=[
                {
                    "censoring_heuristic": "excluded",
                    "competing_heuristic": "adjusted_as_censored",
                }
            ],
        )


@pytest.mark.parametrize("smooth_method", ["local_aj", "secondary_cox", "pseudo_values"])
def test_create_calibration_curve_times_smooth_methods(smooth_method):
    np.random.seed(42)
    probs = {"model_1": np.linspace(0.1, 0.9, 20)}
    reals = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    times = np.linspace(1.0, 10.0, 20)

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        calibration_type="smooth",
        smooth_method=smooth_method,
    )

    assert fig is not None
    # Check that trace contains smooth points
    smooth_trace = fig.data[1]
    assert len(smooth_trace.x) == 101
    assert np.isfinite(np.asarray(smooth_trace.y)).all()


def test_create_calibration_curve_times_invalid_smooth_method():
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4])}
    reals = np.array([1, 0, 1, 0])
    times = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Unsupported smooth_method"):
        create_calibration_curve_times(
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
            calibration_type="smooth",
            smooth_method="unknown_method",
        )


@pytest.mark.parametrize("smooth_method", ["local_aj", "secondary_cox", "pseudo_values"])
def test_create_calibration_curve_times_competing_risks(smooth_method):
    probs = {"model_1": np.linspace(0.05, 0.95, 30)}
    # 0 = censored, 1 = event of interest, 2 = competing event
    reals = np.array([0, 1, 2] * 10)
    times = np.linspace(1.0, 15.0, 30)

    fig = create_calibration_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[8.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        calibration_type="smooth",
        smooth_method=smooth_method,
    )

    assert fig is not None
    smooth_trace = fig.data[1]
    assert len(smooth_trace.x) == 101
    assert np.all(np.asarray(smooth_trace.y) >= 0.0)
    assert np.all(np.asarray(smooth_trace.y) <= 1.0)

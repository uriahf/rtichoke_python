from pathlib import Path

import matplotlib.figure
import numpy as np
import plotly.graph_objects as go

from rtichoke import create_gains_curve_times
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._viz_spec_v2 import _gains_times_v2_spec_from_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)

HORIZONS = [5.0, 10.0]
HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]


def _shared_inputs():
    return (
        {
            "Model A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
            "Model B": np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90]),
        },
        np.array([1, 0, 1, 0, 1, 0]),
        np.array([3.0, 12.0, 8.0, 13.0, 14.0, 15.0]),
    )


def _spec(probs, reals, times):
    performance = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )
    return _gains_times_v2_spec_from_performance_data(
        performance, _build_evaluation_metadata(probs, reals, times)
    )


def test_time_gains_uses_one_evaluation_and_series_per_model_horizon():
    probs, reals, times = _shared_inputs()
    spec = _spec(probs, reals, times)

    assert len(spec["evaluations"]) == 2
    assert {evaluation["population"] for evaluation in spec["evaluations"]} == {
        _SHARED_POPULATION
    }
    assert len(spec["series"]) == 4
    assert {
        (series["display"]["group"], series["horizon"]) for series in spec["series"]
    } == {
        ("Model A", 5.0),
        ("Model A", 10.0),
        ("Model B", 5.0),
        ("Model B", 10.0),
    }

    perfect = spec["references"][1:]
    assert len(perfect) == 2
    assert {
        (reference["population"], reference["horizon"]) for reference in perfect
    } == {
        (_SHARED_POPULATION, 5.0),
        (_SHARED_POPULATION, 10.0),
    }


def test_equal_risk_population_horizons_remain_distinct_reference_owners():
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([1, 0, 0, 0]),
        "Population B": np.array([1, 0, 0, 0]),
    }
    times = {
        "Population A": np.array([3.0, 12.0, 13.0, 14.0]),
        "Population B": np.array([3.0, 12.0, 13.0, 14.0]),
    }
    perfect = _spec(probs, reals, times)["references"][1:]

    assert len(perfect) == 4
    by_owner = {
        (reference["population"], reference["horizon"]): reference["points"]
        for reference in perfect
    }
    assert len(by_owner) == 4
    assert by_owner[("Population A", 5.0)] == by_owner[("Population B", 5.0)]


def test_censoring_and_competing_risk_reference_comes_from_performance_layer():
    probs = {"Model A": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])}
    reals = np.array([1, 0, 2, 1, 0, 2])
    times = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 14.0])
    performance = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )
    spec = _gains_times_v2_spec_from_performance_data(
        performance, _build_evaluation_metadata(probs, reals, times)
    )
    calculated = {
        float(row["fixed_time_horizon"]): float(row["real_positives"] / row["n"])
        for row in performance.filter(performance["chosen_cutoff"] == 0)
        .select("fixed_time_horizon", "real_positives", "n")
        .unique()
        .to_dicts()
    }

    assert {
        reference["horizon"]: reference["points"][1]["x"]
        for reference in spec["references"][1:]
    } == calculated


def test_time_gains_renderers_preserve_plotly_default_and_horizons(tmp_path: Path):
    probs, reals, times = _shared_inputs()
    default = create_gains_curve_times(
        probs, reals, times, HORIZONS, heuristics_sets=HEURISTICS, by=0.25
    )
    assert isinstance(default, go.Figure)

    browser = create_gains_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="browser",
    )
    assert isinstance(browser, RtichokeBrowserChart)
    assert {series["horizon"] for series in browser.spec["series"]} == set(HORIZONS)
    assert browser.write_html(tmp_path / "time-gains.html").is_file()

    matplotlib_result = create_gains_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="matplotlib",
    )
    assert isinstance(matplotlib_result, matplotlib.figure.Figure)
    assert [axis.get_title() for axis in matplotlib_result.axes] == [
        "Fixed Time Horizon: 5",
        "Fixed Time Horizon: 10",
    ]


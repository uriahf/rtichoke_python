import numpy as np
import pytest
from great_tables import GT
from reactable import Reactable

from rtichoke.performance_table_reactable import _bar_style, _net_benefit_style

from rtichoke import (
    create_performance_table,
    create_performance_table_times,
    prepare_performance_data,
    prepare_performance_data_times,
    render_performance_table,
)


def _example():
    probs = {"Model 1": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])}
    reals = np.array([0, 0, 1, 0, 1, 1])
    return probs, reals


def _time_example():
    probs = {"Model 1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
    reals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return probs, reals, times


def test_create_performance_table_defaults_to_great_tables():
    probs, reals = _example()
    assert isinstance(create_performance_table(probs, reals, by=0.1), GT)


def test_create_performance_table_supports_reactable():
    probs, reals = _example()
    assert isinstance(
        create_performance_table(probs, reals, by=0.1, renderer="reactable"), Reactable
    )


def test_render_performance_table_accepts_prepared_polars_data():
    probs, reals = _example()
    data = prepare_performance_data(probs, reals, by=0.1)
    assert isinstance(render_performance_table(data), GT)
    assert isinstance(render_performance_table(data, renderer="reactable"), Reactable)


def test_render_performance_table_rejects_empty_data():
    probs, reals = _example()
    data = prepare_performance_data(probs, reals, by=0.1).clear()
    with pytest.raises(ValueError, match="at least one row"):
        render_performance_table(data)


def test_create_performance_table_supports_ppcr_stratification():
    probs, reals = _example()
    assert isinstance(
        create_performance_table(probs, reals, by=0.1, stratified_by=("ppcr",)), GT
    )


def test_create_performance_table_times_defaults_to_great_tables():
    probs, reals, times = _time_example()
    assert isinstance(
        create_performance_table_times(
            probs, reals, times, fixed_time_horizons=[5, 10], by=0.1
        ),
        GT,
    )


def test_create_performance_table_times_supports_reactable():
    probs, reals, times = _time_example()
    assert isinstance(
        create_performance_table_times(
            probs, reals, times, fixed_time_horizons=[5], by=0.1, renderer="reactable"
        ),
        Reactable,
    )


def test_render_performance_table_preserves_multiple_time_horizons():
    probs, reals, times = _time_example()
    data = prepare_performance_data_times(
        probs, reals, times.astype(float), fixed_time_horizons=[5, 10], by=0.1
    )
    assert sorted(data.get_column("fixed_time_horizon").unique().to_list()) == [
        5.0,
        10.0,
    ]
    assert isinstance(render_performance_table(data), GT)


def test_create_performance_table_times_supports_multiple_heuristic_sets():
    probs, reals, times = _time_example()
    heuristics_sets = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
        {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"},
    ]
    data = prepare_performance_data_times(
        probs,
        reals,
        times.astype(float),
        fixed_time_horizons=[5],
        heuristics_sets=heuristics_sets,
        by=0.1,
    )
    assert data.get_column("censoring_heuristic").n_unique() == 2
    assert data.get_column("competing_heuristic").n_unique() == 2
    assert isinstance(render_performance_table(data), GT)


def test_invalid_renderer_is_rejected():
    probs, reals = _example()
    data = prepare_performance_data(probs, reals, by=0.1)
    with pytest.raises(ValueError, match="renderer"):
        render_performance_table(data, renderer="unknown")


def test_reactable_metric_bar_matches_r_colors_and_geometry():
    style = _bar_style(0.25)
    assert "lightgreen 25.0%" in style["background"]
    assert style["backgroundSize"] == "98% 88%"
    assert style["backgroundRepeat"] == "no-repeat"
    assert style["backgroundPosition"] == "center"


@pytest.mark.parametrize(
    ("value", "color", "extent"),
    [(0.5, "lightgreen", "75.0%"), (-0.5, "pink", "25.0%")],
)
def test_reactable_net_benefit_bar_matches_r_diverging_scale(value, color, extent):
    style = _net_benefit_style(value, 1.0)
    assert color in style["background"]
    assert extent in style["background"]
    assert "50%" in style["background"]

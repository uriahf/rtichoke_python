import numpy as np
import polars as pl
import pytest

import rtichoke.performance_table as performance_table_module
from rtichoke import (
    create_calibration_curve_times,
    create_decision_curve_times,
    create_gains_curve_times,
    create_lift_curve_times,
    create_precision_recall_curve_times,
    create_roc_curve_times,
    prepare_performance_data_times,
)
from rtichoke.processing.plotly_helper_functions import (
    _check_if_multiple_populations_are_being_validated_times,
)
from rtichoke.processing.time_reference_lines import _replace_reference_data_times

HORIZONS = [5.0, 10.0]
HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]
COLORS = ["#111111", "#222222"]

PROBS_A = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
PROBS_B = np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90])
REALS_SHARED = np.array([1, 0, 1, 0, 1, 0])
TIMES_SHARED = np.array([3.0, 12.0, 8.0, 13.0, 14.0, 15.0])


def _performance(probs, reals, times):
    return prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )


def _series_keys(performance_data):
    return set(
        performance_data.select("reference_group", "fixed_time_horizon")
        .unique()
        .iter_rows()
    )


def _references(performance_data, curve):
    curve_list = {
        "fixed_time_horizons": HORIZONS,
        "reference_data": pl.DataFrame(),
    }
    return _replace_reference_data_times(
        curve_list,
        performance_data,
        curve=curve,
        min_p_threshold=0.1,
        max_p_threshold=0.9,
    )["reference_data"]


def _reference_names_by_horizon(reference_data):
    return {
        horizon: set(
            reference_data.filter(pl.col("fixed_time_horizon") == horizon)[
                "reference_group"
            ].unique()
        )
        for horizon in HORIZONS
    }


def _shared_reference_names():
    return {
        "roc": {"random_guess"},
        "precision recall": {"random_guess"},
        "gains": {"random_guess", "perfect_model"},
        "lift": {"random_guess", "perfect_model"},
        "decision": {"treat_none", "treat_all"},
        "interventions avoided": {"treat_all", "treat_none"},
    }


def _population_reference_names(populations):
    return {
        "roc": {"random_guess"},
        "precision recall": {f"random_guess_{population}" for population in populations},
        "gains": {"random_guess"}
        | {f"perfect_model_{population}" for population in populations},
        "lift": {"random_guess"}
        | {f"perfect_model_{population}" for population in populations},
        "decision": {"treat_none"}
        | {f"treat_all_{population}" for population in populations},
        "interventions avoided": {"treat_all"}
        | {f"treat_none_{population}" for population in populations},
    }


def test_one_model_one_population_is_one_series_per_horizon():
    performance_data = _performance({"model": PROBS_A}, REALS_SHARED, TIMES_SHARED)

    assert _series_keys(performance_data) == {
        ("model", 5.0),
        ("model", 10.0),
    }


@pytest.mark.parametrize(
    "creator",
    [
        create_roc_curve_times,
        create_precision_recall_curve_times,
        create_gains_curve_times,
        create_lift_curve_times,
        create_decision_curve_times,
    ],
)
def test_multiple_models_keep_series_labels_colors_and_legends_across_horizons(
    creator,
):
    fig = creator(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_SHARED,
        times=TIMES_SHARED,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        color_values=COLORS,
    )

    for model, color in zip(("Model A", "Model B"), COLORS):
        series = [
            trace
            for trace in fig.data
            if trace.name == model and trace.mode == "markers+lines"
        ]
        assert len(series) == len(HORIZONS)
        assert {trace.line.color for trace in series} == {color}
        assert all(trace.showlegend is True for trace in series)
        assert sum(trace.visible is True for trace in series) == 1

        cutoff_markers = [trace for trace in fig.data if trace.name == f"{model} @ cutoff"]
        assert len(cutoff_markers) == len(HORIZONS)
        assert all(trace.showlegend is False for trace in cutoff_markers)


def test_interventions_avoided_keeps_series_semantics_across_horizons():
    fig = create_decision_curve_times(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_SHARED,
        times=TIMES_SHARED,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        decision_type="interventions avoided",
        by=0.25,
        color_values=COLORS,
    )

    for model, color in zip(("Model A", "Model B"), COLORS):
        series = [
            trace
            for trace in fig.data
            if trace.name == model and trace.mode == "markers+lines"
        ]
        assert len(series) == len(HORIZONS)
        assert {trace.line.color for trace in series} == {color}


def test_multiple_models_share_reference_context_at_each_horizon():
    performance_data = _performance(
        {"Model A": PROBS_A, "Model B": PROBS_B},
        REALS_SHARED,
        TIMES_SHARED,
    )

    aj = (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select("reference_group", "fixed_time_horizon", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
    )
    assert not _check_if_multiple_populations_are_being_validated_times(aj)

    for curve, names in _shared_reference_names().items():
        assert _reference_names_by_horizon(_references(performance_data, curve)) == {
            5.0: names,
            10.0: names,
        }


def test_different_prevalence_populations_own_references_at_each_horizon():
    populations = ("Population low", "Population high")
    probs = {populations[0]: PROBS_A, populations[1]: PROBS_B}
    reals = {
        populations[0]: np.array([1, 0, 0, 0, 0, 0]),
        populations[1]: np.array([1, 1, 1, 0, 0, 0]),
    }
    times = {
        populations[0]: np.array([3.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        populations[1]: np.array([2.0, 4.0, 9.0, 12.0, 13.0, 14.0]),
    }
    performance_data = _performance(probs, reals, times)

    assert _series_keys(performance_data) == {
        (populations[0], 5.0),
        (populations[1], 5.0),
        (populations[0], 10.0),
        (populations[1], 10.0),
    }

    for curve, names in _population_reference_names(populations).items():
        assert _reference_names_by_horizon(_references(performance_data, curve)) == {
            5.0: names,
            10.0: names,
        }


def test_equal_prevalence_populations_remain_series_but_references_collapse():
    probs = {"Population A": PROBS_A, "Population B": PROBS_B}
    reals = {
        "Population A": np.array([1, 1, 0, 0, 0, 0]),
        "Population B": np.array([1, 1, 0, 0, 0, 0]),
    }
    times = {
        "Population A": np.array([3.0, 8.0, 12.0, 13.0, 14.0, 15.0]),
        "Population B": np.array([4.0, 9.0, 12.0, 13.0, 14.0, 15.0]),
    }
    performance_data = _performance(probs, reals, times)

    assert _series_keys(performance_data) == {
        ("Population A", 5.0),
        ("Population B", 5.0),
        ("Population A", 10.0),
        ("Population B", 10.0),
    }

    for curve, names in _shared_reference_names().items():
        assert _reference_names_by_horizon(_references(performance_data, curve)) == {
            5.0: names,
            10.0: names,
        }


def test_reference_scope_can_switch_when_prevalence_diverges_by_horizon():
    populations = ("Population low", "Population high")
    performance_data = _performance(
        {populations[0]: PROBS_A, populations[1]: PROBS_B},
        {
            populations[0]: np.array([1, 0, 0, 0, 0, 0]),
            populations[1]: np.array([1, 1, 1, 0, 0, 0]),
        },
        {
            populations[0]: np.array([3.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            populations[1]: np.array([2.0, 7.0, 9.0, 12.0, 13.0, 14.0]),
        },
    )

    references = _reference_names_by_horizon(
        _references(performance_data, "precision recall")
    )
    assert references[5.0] == {"random_guess"}
    assert references[10.0] == {
        "random_guess_Population low",
        "random_guess_Population high",
    }


def test_paired_model_population_inputs_are_generic_series_per_horizon():
    pair_names = ["Model A @ Population A", "Model B @ Population B"]
    probs = dict(zip(pair_names, (PROBS_A, PROBS_B)))
    reals = {
        pair_names[0]: np.array([1, 0, 0, 0, 0, 0]),
        pair_names[1]: np.array([1, 1, 1, 0, 0, 0]),
    }
    times = {
        pair_names[0]: np.array([3.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        pair_names[1]: np.array([2.0, 4.0, 9.0, 12.0, 13.0, 14.0]),
    }
    performance_data = _performance(probs, reals, times)

    assert _series_keys(performance_data) == {
        (pair_names[0], 5.0),
        (pair_names[1], 5.0),
        (pair_names[0], 10.0),
        (pair_names[1], 10.0),
    }
    assert "model" not in performance_data.columns
    assert "population" not in performance_data.columns


def test_time_calibration_has_one_identity_line_per_horizon_and_group_series():
    fig = create_calibration_curve_times(
        probs={"Population A": PROBS_A, "Population B": PROBS_B},
        reals={
            "Population A": np.array([1, 1, 0, 0, 0, 0]),
            "Population B": np.array([1, 1, 0, 0, 0, 0]),
        },
        times={
            "Population A": np.array([3.0, 8.0, 12.0, 13.0, 14.0, 15.0]),
            "Population B": np.array([4.0, 9.0, 12.0, 13.0, 14.0, 15.0]),
        },
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        calibration_type="discrete",
        color_values=COLORS,
    )

    identity = [trace for trace in fig.data if trace.name == "Perfectly Calibrated"]
    assert len(identity) == len(HORIZONS)
    assert sum(trace.visible is True for trace in identity) == 1

    for group, color in zip(("Population A", "Population B"), COLORS):
        traces = [trace for trace in fig.data if trace.name == group]
        curve_traces = [trace for trace in traces if trace.type == "scatter"]
        histogram_traces = [trace for trace in traces if trace.type == "bar"]
        assert len(curve_traces) == len(HORIZONS)
        assert len(histogram_traces) == len(HORIZONS)
        assert {trace.marker.color for trace in curve_traces} == {color}


def test_time_performance_data_preserves_group_and_horizon_dimensions():
    performance_data = _performance(
        {"Model A": PROBS_A, "Model B": PROBS_B},
        REALS_SHARED,
        TIMES_SHARED,
    )

    assert {"reference_group", "fixed_time_horizon"}.issubset(performance_data.columns)
    assert _series_keys(performance_data) == {
        ("Model A", 5.0),
        ("Model B", 5.0),
        ("Model A", 10.0),
        ("Model B", 10.0),
    }


def test_time_performance_table_preserves_group_and_horizon_dimensions(monkeypatch):
    monkeypatch.setattr(
        performance_table_module,
        "render_performance_table",
        lambda performance_data, **kwargs: performance_data,
    )

    table_data = performance_table_module.create_performance_table_times(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_SHARED,
        times=TIMES_SHARED,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )

    assert _series_keys(table_data) == {
        ("Model A", 5.0),
        ("Model B", 5.0),
        ("Model A", 10.0),
        ("Model B", 10.0),
    }

import numpy as np
import polars as pl

from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
    _performance_table_times_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import prepare_performance_data_times
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


def _static(probs, reals, *, by=0.5, stratified_by=("probability_threshold",)):
    data = prepare_performance_data(
        probs, reals, by=by, stratified_by=stratified_by
    )
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    return _performance_table_spec_from_performance_data(data, metadata)


def test_static_one_model_one_population_and_metric_ids():
    probs = {"Model A": np.array([0.1, 0.4, 0.8, 0.9])}
    spec = _static(probs, np.array([0, 0, 1, 1]))

    assert spec["evaluations"] == [
        {
            "id": "evaluation-1",
            "model": "Model A",
            "population": "__shared_population__",
        }
    ]
    assert [metric["id"] for metric in spec["metrics"]] == [
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "sensitivity",
        "specificity",
        "false_positive_rate",
        "ppv",
        "npv",
        "lift",
        "predicted_positives",
        "ppcr",
        "net_benefit",
        "net_benefit_interventions_avoided",
    ]
    assert "seriesId" not in str(spec)


def test_static_multiple_models_share_population_and_ids_are_deterministic():
    probs = {
        "Model A": np.array([0.1, 0.4, 0.8, 0.9]),
        "Model B": np.array([0.2, 0.3, 0.7, 0.95]),
    }
    reals = np.array([0, 0, 1, 1])
    first = _static(probs, reals)
    second = _static(probs, reals)

    assert first == second
    assert [item["id"] for item in first["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert {item["population"] for item in first["evaluations"]} == {
        "__shared_population__"
    }
    assert {item["model"] for item in first["evaluations"]} == {"Model A", "Model B"}


def test_keyed_inputs_are_distinct_populations_with_unknown_model():
    probs = {
        "Population A": np.array([0.1, 0.4, 0.8, 0.9]),
        "Population B": np.array([0.1, 0.4, 0.8, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 0, 1, 1]),
    }
    spec = _static(probs, reals)

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "Population A"},
        {"id": "evaluation-2", "population": "Population B"},
    ]
    assert {row["evaluationId"] for row in spec["rows"]} == {
        "evaluation-1",
        "evaluation-2",
    }


def test_static_probability_threshold_and_ppcr_operating_points():
    probs = {"Model A": np.array([0.1, 0.4, 0.8, 0.9])}
    reals = np.array([0, 0, 1, 1])
    threshold = _static(probs, reals)
    ppcr = _static(probs, reals, stratified_by=("ppcr",))

    assert {row["operatingPoint"]["type"] for row in threshold["rows"]} == {
        "probability_threshold"
    }
    assert {row["operatingPoint"]["type"] for row in ppcr["rows"]} == {"ppcr"}
    assert all(0 <= row["operatingPoint"]["value"] <= 1 for row in ppcr["rows"])


def test_zero_is_preserved_and_missing_metric_is_null():
    data = pl.DataFrame(
        {
            "reference_group": ["Model A"],
            "stratified_by": ["probability_threshold"],
            "chosen_cutoff": [0.5],
            "sensitivity": [0.0],
            "ppv": [None],
        },
        schema_overrides={"ppv": pl.Float64},
    )
    metadata = _build_evaluation_metadata(
        {"Model A": np.array([0.2])}, np.array([0]), np.array([])
    )
    spec = _performance_table_spec_from_performance_data(data, metadata)
    values = {
        value["metricId"]: value["estimate"]
        for value in spec["rows"][0]["values"]
    }

    assert values == {"sensitivity": 0.0, "ppv": None}


def test_time_table_maps_horizon_and_heuristic_context():
    probs = {"Model A": np.array([0.1, 0.3, 0.7, 0.9])}
    reals = np.array([0, 1, 0, 1])
    times = np.array([2.0, 3.0, 7.0, 8.0])
    heuristics = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
        {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"},
    ]
    data = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        heuristics_sets=heuristics,
        by=0.5,
    )
    metadata = _build_evaluation_metadata(probs, reals, times)
    spec = _performance_table_times_spec_from_performance_data(data, metadata)

    assert {row["horizon"] for row in spec["rows"]} == {5.0, 10.0}
    assert {tuple(row["context"].values()) for row in spec["rows"]} == {
        ("adjusted", "adjusted_as_negative"),
        ("excluded", "excluded"),
    }


def test_equal_valued_distinct_evaluations_remain_distinct():
    values = np.array([0.1, 0.4, 0.8, 0.9])
    outcomes = np.array([0, 0, 1, 1])
    probs = {"Population A": values, "Population B": values.copy()}
    reals = {"Population A": outcomes, "Population B": outcomes.copy()}
    spec = _static(probs, reals)

    assert len(spec["evaluations"]) == 2
    assert len({row["evaluationId"] for row in spec["rows"]}) == 2

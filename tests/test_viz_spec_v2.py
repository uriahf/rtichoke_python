import numpy as np

from rtichoke._viz_spec_v2 import (
    _gains_v2_spec_from_performance_data,
    _roc_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)


def _static_metadata(probs, reals):
    return _build_evaluation_metadata(probs, reals, np.array([]))


def test_roc_v2_one_model_one_population():
    probs = {"Model A": np.array([0.05, 0.2, 0.7, 0.95])}
    reals = np.array([0, 0, 1, 1])
    performance_data = prepare_performance_data(probs, reals, by=0.25)

    spec = _roc_v2_spec_from_performance_data(
        performance_data,
        _static_metadata(probs, reals),
    )

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "roc"
    assert spec["evaluations"] == [
        {
            "id": "evaluation-1",
            "model": "Model A",
            "population": _SHARED_POPULATION,
        }
    ]
    assert spec["series"] == [
        {
            "id": "series-1",
            "evaluationId": "evaluation-1",
            "display": {"label": "Model A", "group": "Model A", "role": "model"},
        }
    ]
    assert {row["seriesId"] for row in spec["data"]} == {"series-1"}
    assert spec["references"] == [{"type": "identity", "scope": "global"}]


def test_roc_v2_multiple_models_share_one_population():
    probs = {
        "Model A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Model B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = np.array([0, 0, 1, 1])
    performance_data = prepare_performance_data(probs, reals, by=0.25)

    spec = _roc_v2_spec_from_performance_data(
        performance_data,
        _static_metadata(probs, reals),
    )

    evaluations = spec["evaluations"]
    assert [evaluation["id"] for evaluation in evaluations] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert {evaluation["population"] for evaluation in evaluations} == {
        _SHARED_POPULATION
    }
    assert [evaluation["model"] for evaluation in evaluations] == ["Model A", "Model B"]
    assert [series["id"] for series in spec["series"]] == ["series-1", "series-2"]
    assert {series["display"]["role"] for series in spec["series"]} == {"model"}
    assert {row["seriesId"] for row in spec["data"]} == {"series-1", "series-2"}


def test_roc_v2_keyed_populations_keep_model_identity_unknown():
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 1, 0, 1]),
    }
    performance_data = prepare_performance_data(probs, reals, by=0.25)

    spec = _roc_v2_spec_from_performance_data(
        performance_data,
        _static_metadata(probs, reals),
    )

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "Population A"},
        {"id": "evaluation-2", "population": "Population B"},
    ]
    assert [series["display"] for series in spec["series"]] == [
        {"label": "Population A", "group": "Population A", "role": "population"},
        {"label": "Population B", "group": "Population B", "role": "population"},
    ]
    assert {row["seriesId"] for row in spec["data"]} == {"series-1", "series-2"}


def test_roc_v2_ids_do_not_encode_compatibility_group_labels():
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 1, 0, 1]),
    }
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    spec = _roc_v2_spec_from_performance_data(
        prepare_performance_data(probs, reals, by=0.25),
        _static_metadata(probs, reals),
    )

    renamed_reals = {
        "Cohort X": reals["Population A"],
        "Cohort Y": reals["Population B"],
    }
    renamed_probs = {
        "Cohort X": probs["Population A"],
        "Cohort Y": probs["Population B"],
    }
    renamed_spec = _roc_v2_spec_from_performance_data(
        prepare_performance_data(renamed_probs, renamed_reals, by=0.25),
        _static_metadata(renamed_probs, renamed_reals),
    )

    assert [evaluation["id"] for evaluation in spec["evaluations"]] == [
        evaluation["id"] for evaluation in renamed_spec["evaluations"]
    ]
    assert [series["id"] for series in spec["series"]] == [
        series["id"] for series in renamed_spec["series"]
    ]


def test_gains_v2_uses_production_prevalence_for_perfect_path():
    probs = {"Model A": np.array([0.05, 0.2, 0.7, 0.95])}
    reals = np.array([0, 0, 0, 1])
    performance_data = prepare_performance_data(probs, reals, by=0.25)

    spec = _gains_v2_spec_from_performance_data(
        performance_data, _static_metadata(probs, reals)
    )

    assert spec["type"] == "gains"
    assert spec["x"] == "ppcr"
    assert spec["y"] == "sensitivity"
    assert spec["references"][0] == {
        "type": "identity",
        "scope": "global",
        "label": "Random",
    }
    perfect = spec["references"][1]
    assert perfect["scope"] == "population"
    assert perfect["population"] == _SHARED_POPULATION
    assert perfect["points"] == [
        {"x": 0, "y": 0},
        {"x": 0.25, "y": 1},
        {"x": 1, "y": 1},
    ]


def test_gains_v2_shares_one_perfect_path_across_models():
    probs = {
        "Model A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Model B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = np.array([0, 0, 1, 1])
    spec = _gains_v2_spec_from_performance_data(
        prepare_performance_data(probs, reals, by=0.25),
        _static_metadata(probs, reals),
    )

    assert len(spec["series"]) == 2
    assert len(spec["references"]) == 2
    assert spec["references"][1]["population"] == _SHARED_POPULATION
    assert spec["references"][1]["points"][1]["x"] == 0.5


def test_gains_v2_keeps_equal_prevalence_populations_distinct():
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 1, 0, 1]),
    }
    spec = _gains_v2_spec_from_performance_data(
        prepare_performance_data(probs, reals, by=0.25),
        _static_metadata(probs, reals),
    )

    perfect = spec["references"][1:]
    assert [reference["population"] for reference in perfect] == [
        "Population A",
        "Population B",
    ]
    assert perfect[0]["points"] == perfect[1]["points"]
    assert all("model" not in evaluation for evaluation in spec["evaluations"])

import matplotlib.figure
import numpy as np
import plotly.graph_objects as go
import pytest

from rtichoke import create_lift_curve
from rtichoke._viz_spec_v2 import _lift_v2_spec_from_performance_data
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)


def _shared_model_inputs():
    return (
        {
            "Model A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
            "Model B": np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90]),
        },
        np.array([0, 0, 0, 1, 1, 1]),
    )


def _spec(probs, reals):
    performance = prepare_performance_data(probs, reals, by=0.25)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    return _lift_v2_spec_from_performance_data(performance, metadata)


def test_static_lift_v2_spec_single_model():
    probs = {"Model A": np.array([0.1, 0.4, 0.7, 0.9])}
    reals = np.array([0, 0, 1, 1])
    spec = _spec(probs, reals)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "lift"
    assert spec["x"] == "ppcr"
    assert spec["y"] == "lift"
    assert len(spec["evaluations"]) == 1
    assert len(spec["series"]) == 1
    assert spec["evaluations"][0]["id"] == "evaluation-1"
    assert spec["series"][0]["id"] == "series-1"
    assert spec["series"][0]["evaluationId"] == "evaluation-1"

    # Random Guess & Perfect Prediction
    assert spec["references"][0] == {
        "type": "horizontal",
        "value": 1.0,
        "scope": "global",
        "label": "Random",
    }
    perfect = spec["references"][1]
    assert perfect["type"] == "path"
    assert perfect["scope"] == "population"
    assert perfect["population"] == _SHARED_POPULATION
    assert perfect["points"] == [
        {"x": 0.0, "y": 2.0},
        {"x": 0.5, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]


def test_static_lift_v2_spec_shared_population():
    probs, reals = _shared_model_inputs()
    spec = _spec(probs, reals)

    assert len(spec["evaluations"]) == 2
    assert len(spec["series"]) == 2
    assert spec["series"][0]["id"] == "series-1"
    assert spec["series"][1]["id"] == "series-2"

    # Multiple models on shared population -> ONE Perfect Prediction path
    assert len(spec["references"]) == 2
    assert spec["references"][1]["population"] == _SHARED_POPULATION
    assert spec["references"][1]["points"] == [
        {"x": 0.0, "y": 2.0},
        {"x": 0.5, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]


def test_static_lift_v2_spec_multiple_populations():
    probs = {
        "Population A": np.array([0.1, 0.2, 0.7, 0.9]),
        "Population B": np.array([0.1, 0.3, 0.4, 0.8]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 0, 0, 1]),
    }
    spec = _spec(probs, reals)

    assert len(spec["evaluations"]) == 2
    assert len(spec["series"]) == 2
    assert len(spec["references"]) == 3  # 1 global Random + 2 population Perfect

    pop_a_ref = [
        r for r in spec["references"] if r.get("population") == "Population A"
    ][0]
    pop_b_ref = [
        r for r in spec["references"] if r.get("population") == "Population B"
    ][0]

    # Pop A prevalence = 0.5 -> y = 2.0
    assert pop_a_ref["points"] == [
        {"x": 0.0, "y": 2.0},
        {"x": 0.5, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]
    # Pop B prevalence = 0.25 -> y = 4.0
    assert pop_b_ref["points"] == [
        {"x": 0.0, "y": 4.0},
        {"x": 0.25, "y": 4.0},
        {"x": 1.0, "y": 1.0},
    ]


def test_equal_prevalence_distinct_populations_maintain_separate_references():
    probs = {
        "Population A": np.array([0.1, 0.2, 0.7, 0.9]),
        "Population B": np.array([0.15, 0.25, 0.75, 0.85]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 0, 1, 1]),
    }
    spec = _spec(probs, reals)

    perfect_refs = spec["references"][1:]
    assert len(perfect_refs) == 2
    assert {r["population"] for r in perfect_refs} == {"Population A", "Population B"}
    assert perfect_refs[0]["points"] == perfect_refs[1]["points"]


def test_static_lift_renderers():
    probs, reals = _shared_model_inputs()

    default_fig = create_lift_curve(probs, reals, by=0.25)
    assert isinstance(default_fig, go.Figure)

    mpl_fig = create_lift_curve(probs, reals, by=0.25, renderer="matplotlib")
    assert isinstance(mpl_fig, matplotlib.figure.Figure)

    with pytest.raises(
        ValueError,
        match="Browser rendering for Lift curves requires a newer vendored release of rtichoke_viz",
    ):
        create_lift_curve(probs, reals, by=0.25, renderer="browser")

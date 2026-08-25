from pathlib import Path

import numpy as np
import polars as pl
from plotly.graph_objects import Figure

from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_v2_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke.processing.evaluation_semantics import _EvaluationMetadata
from rtichoke.utility.decision import create_decision_curve, plot_decision_curve


def _performance_data(*, equal_prevalence: bool = False) -> pl.DataFrame:
    p_a = (2, 8)
    p_b = p_a if equal_prevalence else (4, 8)
    rows = []
    for group, (positives, n), values in (
        ("a", p_a, (-12.5, 7.25)),
        ("b", p_b, (3.0, 19.5)),
    ):
        for threshold, value in zip((0.2, 0.5), values):
            rows.append(
                {
                    "reference_group": group,
                    "chosen_cutoff": threshold,
                    "net_benefit_interventions_avoided": value,
                    "real_positives": positives,
                    "n": n,
                }
            )
    return pl.DataFrame(rows)


def test_shared_population_has_two_evaluations_one_treat_none_path_and_global_treat_all():
    data = _performance_data(equal_prevalence=True)
    metadata = {
        "a": _EvaluationMetadata("a", "a", "model-a", "population-1"),
        "b": _EvaluationMetadata("b", "b", "model-b", "population-1"),
    }

    spec = _interventions_avoided_v2_spec_from_performance_data(data, metadata)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "interventions_avoided"
    assert spec["x"] == "threshold"
    assert spec["y"] == "interventionsAvoided"
    assert spec["yAxis"] == {"label": "Interventions Avoided (per 100)"}
    assert [item["id"] for item in spec["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [item["id"] for item in spec["series"]] == ["series-1", "series-2"]
    assert all(series["id"] != series["evaluationId"] for series in spec["series"])
    assert [row["interventionsAvoided"] for row in spec["data"]] == [
        -12.5,
        7.25,
        3.0,
        19.5,
    ]

    treat_all = [r for r in spec["references"] if r["benchmark"] == "treat_all"]
    treat_none = [r for r in spec["references"] if r["benchmark"] == "treat_none"]
    assert treat_all == [
        {
            "type": "horizontal",
            "scope": "global",
            "value": 0.0,
            "label": "Treat All",
            "benchmark": "treat_all",
        }
    ]
    assert len(treat_none) == 1
    assert treat_none[0]["scope"] == "population"
    assert treat_none[0]["population"] == "population-1"
    assert treat_none[0]["points"] == [
        {"x": 0.2, "y": -25.0},
        {"x": 0.5, "y": 50.0},
    ]


def test_distinct_populations_have_population_owned_treat_none_paths():
    data = _performance_data(equal_prevalence=False)
    metadata = {
        "a": _EvaluationMetadata("a", "a", "model-a", "population-a"),
        "b": _EvaluationMetadata("b", "b", "model-b", "population-b"),
    }

    spec = _interventions_avoided_v2_spec_from_performance_data(data, metadata)
    treat_none = [r for r in spec["references"] if r["benchmark"] == "treat_none"]

    assert [r["population"] for r in treat_none] == ["population-a", "population-b"]
    assert treat_none[0]["points"] != treat_none[1]["points"]


def test_distinct_equal_prevalence_populations_remain_distinct_reference_owners():
    data = _performance_data(equal_prevalence=True)
    metadata = {
        "a": _EvaluationMetadata("a", "a", None, "population-a"),
        "b": _EvaluationMetadata("b", "b", None, "population-b"),
    }

    spec = _interventions_avoided_v2_spec_from_performance_data(data, metadata)
    treat_none = [r for r in spec["references"] if r["benchmark"] == "treat_none"]

    assert [r["population"] for r in treat_none] == ["population-a", "population-b"]
    assert treat_none[0]["points"] == treat_none[1]["points"]
    assert all("model" not in evaluation for evaluation in spec["evaluations"])
    assert [series["display"]["role"] for series in spec["series"]] == [
        "population",
        "population",
    ]


def test_model_values_are_copied_not_recomputed():
    data = pl.DataFrame(
        {
            "reference_group": ["a"],
            "chosen_cutoff": [0.37],
            "net_benefit_interventions_avoided": [12.3456789],
            "real_positives": [2],
            "n": [8],
        }
    )
    metadata = {"a": _EvaluationMetadata("a", "a", "model-a", "population-a")}

    spec = _interventions_avoided_v2_spec_from_performance_data(data, metadata)

    assert spec["data"] == [
        {
            "seriesId": "series-1",
            "threshold": 0.37,
            "interventionsAvoided": 12.3456789,
        }
    ]


def test_public_browser_renderer_is_opt_in_and_plotly_remains_default(tmp_path: Path):
    probs = {"model-a": np.array([0.9, 0.7, 0.4, 0.1])}
    reals = np.array([1, 1, 0, 0])

    browser = create_decision_curve(
        probs,
        reals,
        decision_type="interventions avoided",
        by=0.2,
        renderer="browser",
    )
    alias = create_decision_curve(
        probs,
        reals,
        decision_type="interventions avoided",
        by=0.2,
        renderer="rtichoke_viz",
    )
    plotly = create_decision_curve(
        probs,
        reals,
        decision_type="interventions avoided",
        by=0.2,
    )

    assert isinstance(browser, RtichokeBrowserChart)
    assert isinstance(alias, RtichokeBrowserChart)
    assert isinstance(plotly, Figure)
    html_path = browser.write_html(tmp_path / "interventions-avoided.html")
    html = html_path.read_text(encoding="utf-8")
    assert "renderInterventionsAvoidedV2" in html
    assert browser.spec["type"] == "interventions_avoided"


def test_precomputed_browser_input_does_not_fabricate_model_identity():
    browser = plot_decision_curve(
        _performance_data(),
        decision_type="interventions avoided",
        renderer="browser",
    )

    assert isinstance(browser, RtichokeBrowserChart)
    assert all("model" not in item for item in browser.spec["evaluations"])
    assert [item["population"] for item in browser.spec["evaluations"]] == ["a", "b"]


def test_browser_rejects_combined_or_unknown_decision_modes():
    probs = {"model-a": np.array([0.9, 0.7, 0.4, 0.1])}
    reals = np.array([1, 1, 0, 0])

    try:
        create_decision_curve(probs, reals, decision_type="combined", renderer="browser")
    except ValueError as error:
        assert "decision_type='interventions avoided'" in str(error)
    else:
        raise AssertionError("Combined browser Decision Curve mode is out of scope")

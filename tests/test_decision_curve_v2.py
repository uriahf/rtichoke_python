from pathlib import Path

import numpy as np
import polars as pl
from plotly.graph_objects import Figure

from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_v2_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke.processing.evaluation_semantics import _EvaluationMetadata
from rtichoke.utility.decision import create_decision_curve, plot_decision_curve


def _performance_data(*, equal_prevalence: bool = False) -> pl.DataFrame:
    p_a = (2, 8)
    p_b = p_a if equal_prevalence else (4, 8)
    rows = []
    for group, (positives, n), values in (
        ("a", p_a, (0.20, 0.12)),
        ("b", p_b, (0.30, 0.18)),
    ):
        for threshold, net_benefit in zip((0.2, 0.5), values):
            rows.append(
                {
                    "reference_group": group,
                    "chosen_cutoff": threshold,
                    "net_benefit": net_benefit,
                    "real_positives": positives,
                    "n": n,
                }
            )
    return pl.DataFrame(rows)


def test_shared_population_has_two_evaluations_and_one_treat_all_path():
    data = _performance_data(equal_prevalence=True)
    metadata = {
        "a": _EvaluationMetadata("a", "a", "model-a", "population-1"),
        "b": _EvaluationMetadata("b", "b", "model-b", "population-1"),
    }

    spec = _decision_curve_v2_spec_from_performance_data(data, metadata)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "decision_curve"
    assert [item["id"] for item in spec["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [item["id"] for item in spec["series"]] == ["series-1", "series-2"]
    assert all(series["id"] != series["evaluationId"] for series in spec["series"])
    assert [row["netBenefit"] for row in spec["data"]] == [0.20, 0.12, 0.30, 0.18]

    treat_none = [r for r in spec["references"] if r["benchmark"] == "treat_none"]
    treat_all = [r for r in spec["references"] if r["benchmark"] == "treat_all"]
    assert treat_none == [
        {
            "type": "horizontal",
            "scope": "global",
            "value": 0.0,
            "label": "Treat None",
            "benchmark": "treat_none",
        }
    ]
    assert len(treat_all) == 1
    assert treat_all[0]["scope"] == "population"
    assert treat_all[0]["population"] == "population-1"
    first_point = treat_all[0]["points"][0]
    assert first_point["x"] == 0.2
    assert np.isclose(first_point["y"], 0.0625)


def test_distinct_equal_prevalence_populations_remain_distinct_reference_owners():
    data = _performance_data(equal_prevalence=True)
    metadata = {
        "a": _EvaluationMetadata("a", "a", None, "population-a"),
        "b": _EvaluationMetadata("b", "b", None, "population-b"),
    }

    spec = _decision_curve_v2_spec_from_performance_data(data, metadata)
    treat_all = [r for r in spec["references"] if r["benchmark"] == "treat_all"]

    assert [r["population"] for r in treat_all] == ["population-a", "population-b"]
    assert treat_all[0]["points"] == treat_all[1]["points"]
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
            "net_benefit": [0.123456789],
            "real_positives": [2],
            "n": [8],
        }
    )
    metadata = {"a": _EvaluationMetadata("a", "a", "model-a", "population-a")}

    spec = _decision_curve_v2_spec_from_performance_data(data, metadata)

    assert spec["data"] == [
        {"seriesId": "series-1", "threshold": 0.37, "netBenefit": 0.123456789}
    ]


def test_public_browser_renderer_is_opt_in_and_plotly_remains_default(tmp_path: Path):
    probs = {"model-a": np.array([0.9, 0.7, 0.4, 0.1])}
    reals = np.array([1, 1, 0, 0])

    browser = create_decision_curve(probs, reals, by=0.2, renderer="browser")
    alias = create_decision_curve(probs, reals, by=0.2, renderer="rtichoke_viz")
    plotly = create_decision_curve(probs, reals, by=0.2)

    assert isinstance(browser, RtichokeBrowserChart)
    assert isinstance(alias, RtichokeBrowserChart)
    assert isinstance(plotly, Figure)
    html_path = browser.write_html(tmp_path / "decision.html")
    html = html_path.read_text(encoding="utf-8")
    assert "renderDecisionCurveV2" in html
    assert browser.spec["type"] == "decision_curve"


def test_precomputed_browser_input_does_not_fabricate_model_identity():
    browser = plot_decision_curve(_performance_data(), renderer="browser")

    assert isinstance(browser, RtichokeBrowserChart)
    assert all("model" not in item for item in browser.spec["evaluations"])
    assert [item["population"] for item in browser.spec["evaluations"]] == ["a", "b"]


def test_conventional_browser_behavior_remains_unchanged():
    probs = {"model-a": np.array([0.9, 0.7, 0.4, 0.1])}
    reals = np.array([1, 1, 0, 0])

    browser = create_decision_curve(probs, reals, renderer="browser")

    assert isinstance(browser, RtichokeBrowserChart)
    assert browser.spec["type"] == "decision_curve"

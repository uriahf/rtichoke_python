from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from plotly.graph_objects import Figure

from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_times_v2_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke.processing.evaluation_semantics import _EvaluationMetadata
from rtichoke.utility.decision import create_decision_curve_times


def _time_data(*, equal_risk: bool = True) -> pl.DataFrame:
    rows = []
    risks = {"a": (2, 8), "b": ((2, 8) if equal_risk else (4, 8))}
    for group, (positives, n) in risks.items():
        for horizon, offset in ((5.0, 0.0), (10.0, 10.0)):
            for threshold, value in ((0.0, -99.0), (0.25, 1.25 + offset)):
                rows.append(
                    {
                        "reference_group": group,
                        "fixed_time_horizon": horizon,
                        "censoring_heuristic": "excluded",
                        "competing_heuristic": "excluded",
                        "chosen_cutoff": threshold,
                        "net_benefit_interventions_avoided": value,
                        "real_positives": positives,
                        "n": n,
                    }
                )
    return pl.DataFrame(rows)


def test_time_adapter_preserves_identity_values_and_reference_ownership():
    metadata = {
        "a": _EvaluationMetadata("a", "a", "model-a", "population-1"),
        "b": _EvaluationMetadata("b", "b", "model-b", "population-1"),
    }
    spec = cast(
        dict[str, Any],
        _interventions_avoided_times_v2_spec_from_performance_data(
            _time_data(), metadata
        ),
    )

    assert spec["type"] == "interventions_avoided"
    assert [item["id"] for item in spec["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert len(spec["series"]) == 4
    assert len({item["id"] for item in spec["series"]}) == 4
    assert {item["horizon"] for item in spec["series"]} == {5.0, 10.0}
    assert {item["evaluationId"] for item in spec["series"]} == {
        "evaluation-1",
        "evaluation-2",
    }
    assert [row["interventionsAvoided"] for row in spec["data"]] == [
        -99.0,
        1.25,
        -99.0,
        11.25,
        -99.0,
        1.25,
        -99.0,
        11.25,
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
    assert len(treat_none) == 2
    assert {(r["population"], r["horizon"]) for r in treat_none} == {
        ("population-1", 5.0),
        ("population-1", 10.0),
    }
    assert all(r["scope"] == "population_horizon" for r in treat_none)


def test_equal_risk_distinct_populations_remain_distinct_per_horizon():
    metadata = {
        "a": _EvaluationMetadata("a", "a", None, "population-a"),
        "b": _EvaluationMetadata("b", "b", None, "population-b"),
    }
    spec = cast(
        dict[str, Any],
        _interventions_avoided_times_v2_spec_from_performance_data(
            _time_data(), metadata
        ),
    )
    refs = [r for r in spec["references"] if r["benchmark"] == "treat_none"]

    assert len(refs) == 4
    assert {(r["population"], r["horizon"]) for r in refs} == {
        ("population-a", 5.0),
        ("population-a", 10.0),
        ("population-b", 5.0),
        ("population-b", 10.0),
    }
    assert refs[0]["points"] == refs[2]["points"]


def test_public_time_interventions_avoided_browser_is_opt_in(tmp_path: Path):
    probs = {"Model A": np.array([0.2, 0.5, 0.8, 0.9])}
    reals = np.array([0, 0, 1, 1])
    times = np.array([2.0, 7.0, 3.0, 8.0])
    default = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        decision_type="interventions avoided",
        by=0.25,
    )
    browser = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        decision_type="interventions avoided",
        by=0.25,
        renderer="browser",
    )
    alias = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        decision_type="interventions avoided",
        by=0.25,
        renderer="rtichoke_viz",
    )

    assert isinstance(default, Figure)
    assert isinstance(browser, RtichokeBrowserChart)
    assert isinstance(alias, RtichokeBrowserChart)
    assert browser.spec["type"] == "interventions_avoided"
    html = browser.write_html(tmp_path / "time-ia.html").read_text(encoding="utf-8")
    assert "renderInterventionsAvoidedV2" in html
    bundle = (tmp_path / "rtichoke-viz.js").read_text(encoding="utf-8")
    assert "Fixed Time Horizon" in bundle

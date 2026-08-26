import numpy as np
import pytest

from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_times_v2_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata
from rtichoke.utility.decision import create_decision_curve_times


def test_time_decision_curve_v2_spec_structure():
    probs = {
        "Model A": np.array([0.1, 0.4, 0.7, 0.9]),
        "Model B": np.array([0.2, 0.3, 0.6, 0.8]),
    }
    reals = np.array([0, 0, 1, 1])
    times = np.array([5.0, 12.0, 3.0, 8.0])
    fixed_time_horizons = [5.0, 10.0]
    heuristics_sets = [
        {"censoring_heuristic": "excluded", "competing_heuristic": "excluded"}
    ]

    perf_data = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
    )
    metadata = _build_evaluation_metadata(probs, reals, times)

    spec = _decision_curve_times_v2_spec_from_performance_data(perf_data, metadata)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "decision_curve"
    assert spec["x"] == "threshold"
    assert spec["y"] == "netBenefit"

    # Evaluations must have stable IDs across horizons
    eval_ids = [e["id"] for e in spec["evaluations"]]
    assert eval_ids == ["evaluation-1", "evaluation-2"]
    assert spec["evaluations"][0]["model"] == "Model A"
    assert spec["evaluations"][1]["model"] == "Model B"

    # Series must be per evaluation × horizon
    assert len(spec["series"]) == 4  # 2 models × 2 horizons
    series_horizons = [s["horizon"] for s in spec["series"]]
    assert set(series_horizons) == {5.0, 10.0}
    assert all(s["evaluationId"] in eval_ids for s in spec["series"])

    # Global Treat None and population_horizon Treat All references
    references = spec["references"]
    treat_none = [r for r in references if r.get("benchmark") == "treat_none"]
    treat_all = [r for r in references if r.get("benchmark") == "treat_all"]

    assert len(treat_none) == 1
    assert treat_none[0]["scope"] == "global"
    assert treat_none[0]["value"] == 0.0

    assert len(treat_all) == 2  # 1 population × 2 horizons
    for ref in treat_all:
        assert ref["scope"] == "population_horizon"
        assert ref["horizon"] in [5.0, 10.0]
        assert "points" in ref
        assert len(ref["points"]) > 0


def test_create_decision_curve_times_renderer_options():
    probs = {"Model A": np.array([0.2, 0.5, 0.8])}
    reals = np.array([0, 1, 1])
    times = np.array([2.0, 5.0, 8.0])

    browser_chart = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0],
        renderer="browser",
    )
    assert isinstance(browser_chart, RtichokeBrowserChart)
    assert browser_chart.spec["type"] == "decision_curve"

    alias_chart = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0],
        renderer="rtichoke_viz",
    )
    assert isinstance(alias_chart, RtichokeBrowserChart)


def test_create_decision_curve_times_rejects_interventions_avoided_in_browser_mode():
    probs = {"Model A": np.array([0.2, 0.5, 0.8])}
    reals = np.array([0, 1, 1])
    times = np.array([2.0, 5.0, 8.0])

    with pytest.raises(
        ValueError,
        match="Time-dependent Decision Curves support 'plotly', 'browser', and "
        "'rtichoke_viz' renderers for decision_type='conventional'.",
    ):
        create_decision_curve_times(
            probs,
            reals,
            times,
            fixed_time_horizons=[5.0],
            decision_type="interventions avoided",
            renderer="browser",
        )

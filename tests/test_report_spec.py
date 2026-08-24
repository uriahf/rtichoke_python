from __future__ import annotations

from typing import Any

import pytest

from rtichoke._report_spec import _report_spec_from_components
from rtichoke.summary_report import summary_report as legacy_summary_report


def _curve_spec(
    chart_type: str,
    *,
    evaluations: list[dict[str, object]] | None = None,
    horizon: float | None = None,
) -> dict[str, object]:
    evaluations = evaluations or [
        {"id": "evaluation-1", "model": "model-a", "population": "population-a"}
    ]
    series: dict[str, object] = {
        "id": "series-1",
        "evaluationId": "evaluation-1",
        "display": {"label": "Model A", "group": "Model A", "role": "model"},
    }
    if horizon is not None:
        series["horizon"] = horizon
    return {
        "schemaVersion": "2.0",
        "type": chart_type,
        "evaluations": evaluations,
        "series": [series],
        "data": [{"seriesId": "series-1", "cutoff": 0.5}],
        "x": "false_positive_rate",
        "y": "sensitivity",
        "xAxis": {"label": "x", "domain": [0, 1]},
        "yAxis": {"label": "y", "domain": [0, 1]},
        "references": [],
    }


def _performance_table_spec(*, time_dependent: bool = False) -> dict[str, object]:
    row: dict[str, object] = {
        "evaluationId": "evaluation-1",
        "operatingPoint": {"type": "probability_threshold", "value": 0.5},
        "values": [{"metricId": "sensitivity", "estimate": 0.8}],
    }
    if time_dependent:
        row["horizon"] = 365.0
        row["context"] = {
            "censoringHeuristic": "include",
            "competingEventHeuristic": "exclude",
        }
    return {
        "schemaVersion": "2.0",
        "type": "performance_table",
        "evaluations": [
            {"id": "evaluation-1", "model": "model-a", "population": "population-a"}
        ],
        "metrics": [{"id": "sensitivity", "label": "Sensitivity"}],
        "rows": [row],
    }


def test_report_composes_performance_table_roc_and_calibration_in_order() -> None:
    performance_table = _performance_table_spec()
    roc = _curve_spec("roc")
    calibration = _curve_spec("calibration")

    report = _report_spec_from_components(
        [
            {"spec": performance_table, "title": "Performance"},
            {"spec": roc},
            {"spec": calibration, "title": "Calibration"},
        ],
        title="Model report",
    )

    assert report["schemaVersion"] == "1.0"
    assert report["type"] == "report"
    assert report["title"] == "Model report"
    assert [component["id"] for component in report["components"]] == [
        "performance-table",
        "roc",
        "calibration",
    ]
    assert [component["spec"]["type"] for component in report["components"]] == [
        "performance_table",
        "roc",
        "calibration",
    ]


def test_component_ids_are_deterministic_unique_and_separate_from_series_ids() -> None:
    first_roc = _curve_spec("roc")
    second_roc = _curve_spec("roc")

    first = _report_spec_from_components(
        [{"spec": first_roc}, {"spec": second_roc}, {"spec": _curve_spec("lift")}]
    )
    second = _report_spec_from_components(
        [{"spec": first_roc}, {"spec": second_roc}, {"spec": _curve_spec("lift")}]
    )

    ids = [component["id"] for component in first["components"]]
    assert ids == ["roc", "roc-2", "lift"]
    assert ids == [component["id"] for component in second["components"]]
    assert len(ids) == len(set(ids))
    assert "series-1" not in ids


def test_embedded_specs_are_complete_unchanged_and_evaluations_are_component_local() -> None:
    roc = _curve_spec("roc")
    calibration = _curve_spec("calibration")

    report = _report_spec_from_components([{"spec": roc}, {"spec": calibration}])

    assert report["components"][0]["spec"] is roc
    assert report["components"][1]["spec"] is calibration
    assert roc["evaluations"] == calibration["evaluations"]
    assert report["components"][0]["spec"]["evaluations"][0]["id"] == "evaluation-1"
    assert report["components"][1]["spec"]["evaluations"][0]["id"] == "evaluation-1"


def test_report_preserves_model_known_unknown_and_multiple_populations() -> None:
    known = _curve_spec(
        "roc",
        evaluations=[
            {"id": "evaluation-1", "model": "model-a", "population": "population-a"},
            {"id": "evaluation-2", "model": "model-b", "population": "population-b"},
        ],
    )
    unknown = _curve_spec(
        "calibration",
        evaluations=[{"id": "evaluation-1", "population": "population-c"}],
    )

    report = _report_spec_from_components([{"spec": known}, {"spec": unknown}])

    known_evaluations = report["components"][0]["spec"]["evaluations"]
    unknown_evaluation = report["components"][1]["spec"]["evaluations"][0]
    assert known_evaluations[0]["model"] == "model-a"
    assert {item["population"] for item in known_evaluations} == {
        "population-a",
        "population-b",
    }
    assert "model" not in unknown_evaluation
    assert unknown_evaluation["population"] == "population-c"


def test_time_dependent_component_is_embedded_without_recomputation() -> None:
    table = _performance_table_spec(time_dependent=True)
    gains = _curve_spec("gains", horizon=365.0)

    report = _report_spec_from_components([{"spec": table}, {"spec": gains}])

    assert report["components"][0]["spec"] is table
    assert report["components"][1]["spec"] is gains
    assert table["rows"][0]["horizon"] == 365.0
    assert gains["series"][0]["horizon"] == 365.0


def test_all_first_report_component_types_are_supported() -> None:
    specs = [
        _performance_table_spec(),
        _curve_spec("roc"),
        _curve_spec("calibration"),
        _curve_spec("precision_recall"),
        _curve_spec("gains"),
        _curve_spec("lift"),
    ]

    report = _report_spec_from_components([{"spec": spec} for spec in specs])

    assert [component["id"] for component in report["components"]] == [
        "performance-table",
        "roc",
        "calibration",
        "precision-recall",
        "gains",
        "lift",
    ]


def test_report_requires_components_and_rejects_out_of_scope_types() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        _report_spec_from_components([])

    with pytest.raises(ValueError, match="Unsupported ReportSpec component type"):
        _report_spec_from_components([{"spec": {"type": "decision_curve"}}])


def test_existing_public_summary_report_still_uses_r_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _Response:
        def json(self) -> list[dict[str, object]]:
            return [{"report": object()}]

    def fake_send_requests_to_rtichoke_r(**kwargs: Any) -> _Response:
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(
        legacy_summary_report,
        "send_requests_to_rtichoke_r",
        fake_send_requests_to_rtichoke_r,
    )

    legacy_summary_report.create_summary_report(
        {"model-a": [0.1, 0.9]},
        [0, 1],
        url_api="http://example.test/",
    )

    assert calls == [
        {
            "dictionary_to_send": {
                "probs": {"model-a": [0.1, 0.9]},
                "reals": [0, 1],
            },
            "url_api": "http://example.test/",
            "endpoint": "create_summary_report",
        }
    ]

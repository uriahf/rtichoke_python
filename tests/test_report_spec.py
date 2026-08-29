from __future__ import annotations

from typing import Any

import pytest

from rtichoke._report_spec import _report_spec_from_components
from rtichoke.summary_report import summary_report as legacy_summary_report


def _evaluation(
    evaluation_id: str,
    population: str,
    model: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": evaluation_id,
        "population": population,
    }
    if model is not None:
        result["model"] = model
    return result


def _curve_spec(
    chart_type: str,
    *,
    evaluations: list[dict[str, object]] | None = None,
    horizon: float | None = None,
) -> dict[str, Any]:
    evaluations = evaluations or [
        _evaluation("evaluation-1", "population-a", "model-a")
    ]
    display = {
        "label": "Model A",
        "group": "Model A",
        "role": "model",
    }
    series: dict[str, object] = {
        "id": "series-1",
        "evaluationId": "evaluation-1",
        "display": display,
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


def _performance_table_spec(*, time_dependent: bool = False) -> dict[str, Any]:
    row: dict[str, object] = {
        "evaluationId": "evaluation-1",
        "operatingPoint": {
            "type": "probability_threshold",
            "value": 0.5,
        },
        "values": [
            {
                "metricId": "sensitivity",
                "estimate": 0.8,
            }
        ],
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
        "evaluations": [_evaluation("evaluation-1", "population-a", "model-a")],
        "metrics": [
            {
                "id": "sensitivity",
                "label": "Sensitivity",
            }
        ],
        "rows": [row],
    }


def test_core_composition_and_order() -> None:
    performance_table = _performance_table_spec()
    roc = _curve_spec("roc")
    calibration = _curve_spec("calibration")

    report = _report_spec_from_components(
        [
            {
                "spec": performance_table,
                "title": "Performance",
            },
            {"spec": roc},
            {
                "spec": calibration,
                "title": "Calibration",
            },
        ],
        title="Model report",
    )

    assert report["schemaVersion"] == "1.0"
    assert report["type"] == "report"
    assert report["title"] == "Model report"
    component_ids = [component["id"] for component in report["components"]]
    component_types = [component["spec"]["type"] for component in report["components"]]
    assert component_ids == [
        "performance-table",
        "roc",
        "calibration",
    ]
    assert component_types == [
        "performance_table",
        "roc",
        "calibration",
    ]


def test_component_ids_are_deterministic_and_unique() -> None:
    first_roc = _curve_spec("roc")
    second_roc = _curve_spec("roc")
    inputs = [
        {"spec": first_roc},
        {"spec": second_roc},
        {"spec": _curve_spec("lift")},
    ]

    first = _report_spec_from_components(inputs)
    second = _report_spec_from_components(inputs)

    ids = [component["id"] for component in first["components"]]
    second_ids = [component["id"] for component in second["components"]]
    assert ids == ["roc", "roc-2", "lift"]
    assert ids == second_ids
    assert len(ids) == len(set(ids))
    assert "series-1" not in ids


def test_specs_remain_complete_and_component_local() -> None:
    roc = _curve_spec("roc")
    calibration = _curve_spec("calibration")

    report = _report_spec_from_components(
        [
            {"spec": roc},
            {"spec": calibration},
        ]
    )

    assert report["components"][0]["spec"] is roc
    assert report["components"][1]["spec"] is calibration
    assert roc["evaluations"] == calibration["evaluations"]
    assert roc["evaluations"][0]["id"] == "evaluation-1"
    assert calibration["evaluations"][0]["id"] == "evaluation-1"


def test_semantics_pass_through_unchanged() -> None:
    known = _curve_spec(
        "roc",
        evaluations=[
            _evaluation("evaluation-1", "population-a", "model-a"),
            _evaluation("evaluation-2", "population-b", "model-b"),
        ],
    )
    unknown = _curve_spec(
        "calibration",
        evaluations=[_evaluation("evaluation-1", "population-c")],
    )

    report = _report_spec_from_components(
        [
            {"spec": known},
            {"spec": unknown},
        ]
    )

    assert report["components"][0]["spec"] is known
    assert report["components"][1]["spec"] is unknown
    assert known["evaluations"][0]["model"] == "model-a"
    populations = {item["population"] for item in known["evaluations"]}
    assert populations == {
        "population-a",
        "population-b",
    }
    assert "model" not in unknown["evaluations"][0]
    assert unknown["evaluations"][0]["population"] == "population-c"


def test_time_dependent_specs_pass_through_unchanged() -> None:
    table = _performance_table_spec(time_dependent=True)
    gains = _curve_spec("gains", horizon=365.0)

    report = _report_spec_from_components(
        [
            {"spec": table},
            {"spec": gains},
        ]
    )

    assert report["components"][0]["spec"] is table
    assert report["components"][1]["spec"] is gains
    assert table["rows"][0]["horizon"] == 365.0
    assert gains["series"][0]["horizon"] == 365.0


def test_first_report_component_types_are_supported() -> None:
    specs = [
        _performance_table_spec(),
        _curve_spec("roc"),
        _curve_spec("calibration"),
        _curve_spec("precision_recall"),
        _curve_spec("gains"),
        _curve_spec("lift"),
    ]

    report = _report_spec_from_components([{"spec": spec} for spec in specs])

    component_ids = [component["id"] for component in report["components"]]
    assert component_ids == [
        "performance-table",
        "roc",
        "calibration",
        "precision-recall",
        "gains",
        "lift",
    ]


def test_invalid_report_components_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        _report_spec_from_components([])

    with pytest.raises(
        ValueError,
        match="Unsupported ReportSpec component type",
    ):
        _report_spec_from_components(
            [
                {
                    "spec": {
                        "type": "decision_curve",
                    }
                }
            ]
        )


def test_existing_summary_report_still_uses_r_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                "probs": {
                    "model-a": [0.1, 0.9],
                },
                "reals": [0, 1],
            },
            "url_api": "http://example.test/",
            "endpoint": "create_summary_report",
        }
    ]

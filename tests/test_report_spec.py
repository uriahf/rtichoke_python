from __future__ import annotations

from typing import Any

import pytest

from rtichoke._report_spec import _build_report_spec_v11
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


def _summary_metrics_spec(
    metric_type: str, schema_version: str = "1.0"
) -> dict[str, Any]:
    metric_item: dict[str, Any] = {
        "metric": metric_type,
        "owner": {"type": "population", "populationId": "population-1"},
        "estimate": 0.25,
    }
    if metric_type == "event_risk":
        metric_item["horizon"] = 5.0

    return {
        "schemaVersion": schema_version,
        "type": "summary_metrics",
        "title": "Summary",
        "evaluations": [],
        "populations": [{"id": "population-1", "label": "Population"}],
        "metrics": [metric_item],
    }


def test_build_report_spec_v11_structure() -> None:
    prev_spec = _summary_metrics_spec("prevalence")
    roc_spec = _curve_spec("roc")

    sections = [
        {
            "id": "prevalence",
            "title": "Prevalence",
            "components": [
                {"id": "prevalence-summary", "title": "Summary", "spec": prev_spec}
            ],
        },
        {
            "id": "discrimination",
            "title": "Discrimination",
            "groups": [
                {
                    "id": "discrimination-probability-threshold",
                    "title": "By Threshold",
                    "components": [{"id": "roc", "title": "ROC", "spec": roc_spec}],
                }
            ],
        },
    ]

    report = _build_report_spec_v11(sections, title="Test Report")
    assert report["schemaVersion"] == "1.1"
    assert report["type"] == "report"
    assert report["title"] == "Test Report"
    assert len(report["sections"]) == 2
    assert report["sections"][0]["id"] == "prevalence"
    assert report["sections"][0]["items"][0]["id"] == "prevalence-summary"
    assert report["sections"][1]["id"] == "discrimination"
    assert (
        report["sections"][1]["items"][0]["id"]
        == "discrimination-probability-threshold"
    )
    assert report["sections"][1]["items"][0]["components"][0]["id"] == "roc"


def test_summary_metrics_schema_versions_accepted() -> None:
    v10_spec = _summary_metrics_spec("prevalence", "1.0")
    v11_spec = _summary_metrics_spec("event_risk", "1.1")

    sections = [
        {
            "id": "event-risk",
            "components": [{"id": "event-risk", "spec": v11_spec}],
        },
        {
            "id": "prevalence",
            "components": [{"id": "prevalence-summary", "spec": v10_spec}],
        },
    ]
    report = _build_report_spec_v11(sections)
    assert report["schemaVersion"] == "1.1"


def test_type_aware_schema_version_validation() -> None:
    bad_v1_spec = _summary_metrics_spec("prevalence")
    bad_v1_spec["schemaVersion"] = "2.0"  # Invalid: summary_metrics must be 1.0 or 1.1

    sections = [
        {
            "id": "prevalence",
            "components": [{"id": "prevalence-summary", "spec": bad_v1_spec}],
        }
    ]
    with pytest.raises(ValueError, match="requires schemaVersion '1.0'"):
        _build_report_spec_v11(sections)

    bad_v2_spec = _curve_spec("roc")
    bad_v2_spec["schemaVersion"] = "1.0"  # Invalid: roc must be 2.0

    sections_v2 = [
        {
            "id": "discrimination",
            "components": [{"id": "roc", "spec": bad_v2_spec}],
        }
    ]
    with pytest.raises(ValueError, match="requires schemaVersion '2.0'"):
        _build_report_spec_v11(sections_v2)


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

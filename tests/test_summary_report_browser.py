import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rtichoke.summary_report import summary_report as summary_report_module
from rtichoke.summary_report.summary_report import create_summary_report


def _inputs():
    probs = {
        "Model A": np.array(
            [
                0.03,
                0.08,
                0.12,
                0.18,
                0.25,
                0.32,
                0.40,
                0.50,
                0.62,
                0.75,
                0.88,
                0.96,
            ]
        )
    }
    reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    return probs, reals


def _embedded_report(html: str) -> dict[str, Any]:
    start = html.index('<script id="rtichoke-report-spec" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return cast(dict[str, Any], json.loads(html[start:end]))


def test_default_summary_report_keeps_historical_r_backend(monkeypatch, capsys):
    probs, reals = _inputs()
    calls = []

    class Response:
        def json(self):
            return [{"historical": True}]

    def fake_send_requests_to_rtichoke_r(**kwargs):
        calls.append(kwargs)
        return Response()

    def fail_browser(*args, **kwargs):
        raise AssertionError("default path must not invoke RtichokeBrowserReport")

    monkeypatch.setattr(
        summary_report_module,
        "send_requests_to_rtichoke_r",
        fake_send_requests_to_rtichoke_r,
    )
    monkeypatch.setattr(
        summary_report_module.RtichokeBrowserReport,
        "write_html",
        fail_browser,
    )

    result = create_summary_report(probs, reals)

    assert result is None
    assert len(calls) == 1
    assert calls[0]["dictionary_to_send"]["probs"] is probs
    assert calls[0]["dictionary_to_send"]["reals"] is reals
    assert calls[0]["url_api"] == "http://localhost:4242/"
    assert calls[0]["endpoint"] == "create_summary_report"
    assert "dict_keys(['historical'])" in capsys.readouterr().out


def test_browser_summary_report_is_opt_in_and_uses_real_canonical_components(
    tmp_path,
):
    probs, reals = _inputs()
    output = tmp_path / "canonical-summary.html"

    result = create_summary_report(
        probs,
        reals,
        renderer="browser",
        output_file=output,
    )

    assert result == output
    assert output.exists()
    assert (tmp_path / "rtichoke-viz.js").exists()
    assert (tmp_path / "rtichoke-viz.css").exists()

    html = output.read_text(encoding="utf-8")
    report = _embedded_report(html)
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
    assert report["components"][1]["spec"]["schemaVersion"] == "2.0"
    assert report["components"][2]["spec"]["schemaVersion"] == "2.0"
    assert 'import { renderReport } from "./rtichoke-viz.js";' in html
    assert "append(renderReport(spec))" in html
    assert "renderPerformanceTable" not in html
    assert "renderRocV2" not in html
    assert "renderCalibrationV2" not in html


def test_browser_summary_report_preserves_component_local_identity(tmp_path):
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 1, 0, 1]),
    }

    output = create_summary_report(
        probs,
        reals,
        renderer="browser",
        output_file=tmp_path / "populations.html",
    )
    assert isinstance(output, Path)
    report = _embedded_report(output.read_text(encoding="utf-8"))

    assert "evaluations" not in report
    assert "models" not in report
    assert "populations" not in report
    assert "horizon" not in report

    table, roc, calibration = [component["spec"] for component in report["components"]]
    assert table["evaluations"][0]["id"] == "evaluation-1"
    assert roc["evaluations"][0]["id"] == "evaluation-1"
    assert calibration["evaluations"][0]["id"] == "evaluation-1"
    assert [item["population"] for item in calibration["evaluations"]] == [
        "Population A",
        "Population B",
    ]
    assert all("model" not in item for item in calibration["evaluations"])
    assert [item["display"]["role"] for item in calibration["series"]] == [
        "population",
        "population",
    ]


def test_browser_summary_report_rejects_unknown_renderer():
    probs, reals = _inputs()

    try:
        create_summary_report(probs, reals, renderer="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert str(exc) == "renderer must be either 'r' or 'browser'"
    else:
        raise AssertionError("unknown renderer should fail")

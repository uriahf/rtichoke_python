import json
import shutil
import subprocess
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterator, cast

import numpy as np
import pytest

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


def _chrome_executable() -> str:
    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        executable = shutil.which(candidate)
        if executable is not None:
            return executable
    pytest.skip("headless Chrome/Chromium is not available")


def _dump_dom(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _chrome_executable(),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--enable-logging=stderr",
            "--log-level=0",
            "--dump-dom",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _rendered_report_html(dom: str) -> str:
    marker = '<div id="rtichoke-report">'
    start = dom.index(marker) + len(marker)
    end = dom.index('<script id="rtichoke-report-spec"', start)
    return dom[start:end]


@contextmanager
def _serve(directory: Path) -> Iterator[str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _assert_report_rendered(browser: subprocess.CompletedProcess[str]) -> None:
    assert browser.returncode == 0, browser.stderr
    assert "INFO:CONSOLE" not in browser.stderr, browser.stderr
    rendered = _rendered_report_html(browser.stdout)
    assert "Prevalence" in rendered, browser.stderr
    assert "Calibration" in rendered
    assert "Discrimination" in rendered
    assert "Utility" in rendered
    assert "Performance Table" in rendered
    assert "<table" in rendered
    assert rendered.count("<svg") >= 2


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
    viz_js_path = tmp_path / "rtichoke-viz.js"
    assert viz_js_path.exists()
    assert (tmp_path / "rtichoke-viz.css").exists()

    html = output.read_text(encoding="utf-8")
    report = _embedded_report(html)
    assert report["schemaVersion"] == "1.1"
    assert report["type"] == "report"
    section_ids = [section["id"] for section in report["sections"]]
    assert section_ids == [
        "prevalence",
        "calibration",
        "discrimination",
        "utility",
        "performance-table",
    ]
    assert 'import { renderReport } from "./rtichoke-viz.js";' not in html
    assert viz_js_path.read_text(encoding="utf-8") in html
    assert 'sectionGroupPresentation: "tabs"' in html


def test_browser_summary_report_executes_when_opened_directly(tmp_path):
    probs, reals = _inputs()
    output = create_summary_report(
        probs,
        reals,
        renderer="browser",
        output_file=tmp_path / "browser_report.html",
    )
    assert isinstance(output, Path)

    browser = _dump_dom(output.resolve().as_uri())

    _assert_report_rendered(browser)


def test_browser_summary_report_executes_over_localhost(tmp_path):
    probs, reals = _inputs()
    output = create_summary_report(
        probs,
        reals,
        renderer="browser",
        output_file=tmp_path / "browser_report.html",
    )
    assert isinstance(output, Path)

    with _serve(output.parent) as base_url:
        browser = _dump_dom(f"{base_url}/{output.name}")

    _assert_report_rendered(browser)


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

    prev_spec = report["sections"][0]["items"][0]["spec"]
    calib_smooth_spec = report["sections"][1]["items"][0]["spec"]
    calib_discrete_spec = report["sections"][1]["items"][1]["spec"]

    assert prev_spec["schemaVersion"] == "1.0"
    assert prev_spec["type"] == "summary_metrics"
    assert len(prev_spec["populations"]) == 2
    assert (
        "evaluations" not in calib_smooth_spec["yAxis"]
    )  # check omit yAxis.domain on smooth
    assert calib_smooth_spec["yAxis"].get("domain") is None
    assert calib_discrete_spec["yAxis"]["domain"] == [0, 1]


def test_browser_summary_report_rejects_unknown_renderer():
    probs, reals = _inputs()

    try:
        create_summary_report(probs, reals, renderer="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert str(exc) == "renderer must be either 'r' or 'browser'"
    else:
        raise AssertionError("unknown renderer should fail")

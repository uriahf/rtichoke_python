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

from importlib.resources import files
from jsonschema import Draft7Validator, Draft202012Validator

import rtichoke
from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._report_browser import RtichokeBrowserReport
from rtichoke._report_spec import _build_report_spec_v11
from rtichoke._viz_spec_v2 import _prediction_distribution_v2_spec
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.probs_distribution import (
    _prepare_probs_distribution_data,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata
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
    assert [f.name for f in sorted(tmp_path.iterdir())] == ["canonical-summary.html"]
    assert not (tmp_path / "rtichoke-viz.js").exists()
    assert not (tmp_path / "rtichoke-viz.css").exists()

    html = output.read_text(encoding="utf-8")
    assert "<title>Summary Report</title>" in html
    report = _embedded_report(html)
    assert report["schemaVersion"] == "1.1"
    assert report["type"] == "report"
    section_ids = [section["id"] for section in report["sections"]]
    assert section_ids == [
        "prevalence",
        "prediction-distribution",
        "calibration",
        "discrimination",
        "utility",
        "performance-table",
    ]
    assert 'import { renderReport } from "./rtichoke-viz.js";' not in html
    vendor = files("rtichoke").joinpath("_vendor", "rtichoke_viz")
    viz_js = vendor.joinpath("rtichoke-viz.js").read_text(encoding="utf-8")
    viz_css = vendor.joinpath("rtichoke-viz.css").read_text(encoding="utf-8")
    assert viz_js in html
    assert viz_css in html
    assert '<link rel="stylesheet" href="./rtichoke-viz.css">' not in html
    assert "<style>" in html
    assert 'sectionGroupPresentation: "tabs"' in html
    assert 'groupPresentation: "tabs"' in html
    assert 'sectionComponentPresentation: "tabs"' in html
    assert 'groupPresentation: "stacked"' not in html
    assert ".rtichoke-report {\n  max-width: 1040px;" in html


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
    calib_smooth_spec = report["sections"][2]["items"][0]["spec"]
    calib_discrete_spec = report["sections"][2]["items"][1]["spec"]

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


def test_static_performance_table_confusion_matrix_disclosure(tmp_path):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("playwright is not available")

    probs, reals = _inputs()
    output_thresh = tmp_path / "report_disclosure.html"
    create_summary_report(probs, reals, renderer="browser", output_file=output_thresh)

    output_ppcr = tmp_path / "ppcr_table_disclosure.html"
    perf_ppcr = prepare_performance_data(probs, reals, stratified_by=("ppcr",))
    metadata = _build_evaluation_metadata(probs, reals, times=None)
    ppcr_spec = _performance_table_spec_from_performance_data(perf_ppcr, metadata)
    ppcr_report_spec = _build_report_spec_v11(
        [
            {
                "id": "performance-table",
                "title": "Performance Table",
                "components": [
                    {
                        "id": "ppcr-table",
                        "spec": ppcr_spec,
                    }
                ],
            }
        ]
    )
    RtichokeBrowserReport(ppcr_report_spec).write_html(output_ppcr)

    with _serve(tmp_path) as base_url, sync_playwright() as p:
        executable = _chrome_executable()
        try:
            browser = p.chromium.launch(headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page()
        errors: list[str] = []
        page.on("console", lambda msg: print("CONSOLE:", msg.type, msg.text))
        page.on("pageerror", lambda err: print("PAGE ERROR:", err))

        # 0. Cheat sheet presence, placement, and content in static summary report
        page.goto(f"{base_url}/{output_thresh.name}")
        page.wait_for_selector(".rtichoke-report")

        cheat_sheets = page.locator(".rtichoke-cheat-sheet")
        assert cheat_sheets.count() == 1

        cs = cheat_sheets.first
        summary_el = cs.locator("summary")
        assert summary_el.inner_text() == "Performance Metrics Cheat Sheet"

        # Verify element order: .rtichoke-report__header -> .rtichoke-cheat-sheet -> .rtichoke-report__nav
        is_correct_order = page.evaluate("""() => {
            const header = document.querySelector('.rtichoke-report__header');
            const cheatSheet = document.querySelector('.rtichoke-cheat-sheet');
            const nav = document.querySelector('.rtichoke-report__nav');
            if (!header || !cheatSheet || !nav) return false;
            return (
                header.compareDocumentPosition(cheatSheet) & Node.DOCUMENT_POSITION_FOLLOWING &&
                cheatSheet.compareDocumentPosition(nav) & Node.DOCUMENT_POSITION_FOLLOWING
            );
        }""")
        assert is_correct_order is True or is_correct_order == 1

        # Check cheat sheet text/formulas
        cs_text = cs.inner_text()
        assert "Confusion Matrix" in cs_text
        assert "Prevalence" in cs_text
        assert "PPCR" in cs_text
        assert "Sensitivity / Recall / TPR" in cs_text
        assert "Specificity / TNR" in cs_text
        assert "PPV / Precision" in cs_text
        assert "NPV" in cs_text
        assert "Lift" in cs_text
        assert "Net Benefit" in cs_text

        # 1. Static probability threshold performance table disclosure
        page.goto(f"{base_url}/{output_thresh.name}")
        page.wait_for_selector(".rtichoke-performance-table__table")

        toggle_btn = page.locator(
            "button[aria-label='Show confusion matrix detail']"
        ).first
        toggle_btn.wait_for()
        toggle_btn.click()

        container = page.locator(
            ".rtichoke-performance-table__confusion-container"
        ).first
        container.wait_for()
        title_el = container.locator(".rtichoke-performance-table__confusion-title")
        assert title_el.inner_text() == "Confusion Matrix"
        assert (
            container.get_attribute("data-operating-point-type")
            == "probability_threshold"
        )
        assert container.get_attribute("data-operating-point-value") is not None
        assert (
            page.locator(".rtichoke-performance-table__confusion-caption").count() == 0
        )

        # 2. Static PPCR performance table disclosure
        page.goto(f"{base_url}/{output_ppcr.name}")
        page.wait_for_selector(".rtichoke-performance-table__table")

        ppcr_toggle_btn = page.locator(
            "button[aria-label='Show confusion matrix detail']"
        ).first
        ppcr_toggle_btn.wait_for()
        ppcr_toggle_btn.click()

        ppcr_container = page.locator(
            ".rtichoke-performance-table__confusion-container"
        ).first
        ppcr_container.wait_for()
        ppcr_title_el = ppcr_container.locator(
            ".rtichoke-performance-table__confusion-title"
        )
        assert ppcr_title_el.inner_text() == "Confusion Matrix"
        assert ppcr_container.get_attribute("data-operating-point-type") == "ppcr"
        assert ppcr_container.get_attribute("data-operating-point-value") is not None

        assert len(errors) == 0
        browser.close()


def test_browser_summary_report_prediction_distribution_components_render(tmp_path):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("playwright is not available")

    probs, reals = _inputs()
    output = tmp_path / "pred_dist_render.html"
    create_summary_report(probs, reals, renderer="browser", output_file=output)

    with _serve(tmp_path) as base_url, sync_playwright() as p:
        executable = _chrome_executable()
        try:
            browser = p.chromium.launch(headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page()
        errors: list[str] = []
        page.on(
            "console",
            lambda msg: errors.append(msg.text)
            if msg.type in ["error", "warning"]
            and "Failed to load resource" not in msg.text
            else None,
        )
        page.on("pageerror", lambda err: errors.append(str(err)))

        page.goto(f"{base_url}/{output.name}")

        # 1. Activate top-level Prediction Distribution section tab
        pred_dist_section_tab = page.locator(
            "button[aria-controls='prediction-distribution']"
        )
        if pred_dist_section_tab.is_visible():
            pred_dist_section_tab.click()
        page.wait_for_selector("#prediction-distribution")

        # 2. Activate probability threshold group tab and verify prediction-distribution
        thresh_tab = page.locator(
            "button[aria-controls='prediction-distribution-probability-threshold']"
        )
        thresh_tab.click()

        comp_thresh = page.locator("[data-component-id='prediction-distribution']")
        comp_thresh.wait_for()
        assert comp_thresh.is_visible()

        svg_thresh = comp_thresh.locator("svg").first
        svg_thresh.wait_for()
        bbox_thresh = svg_thresh.bounding_box()
        assert bbox_thresh is not None
        assert bbox_thresh["width"] > 0
        assert bbox_thresh["height"] > 0

        # 3. Activate PPCR group tab and verify prediction-distribution-2
        ppcr_tab = page.locator("button[aria-controls='prediction-distribution-ppcr']")
        ppcr_tab.click()

        comp_ppcr = page.locator("[data-component-id='prediction-distribution-2']")
        comp_ppcr.wait_for()
        assert comp_ppcr.is_visible()

        svg_ppcr = comp_ppcr.locator("svg").first
        svg_ppcr.wait_for()
        bbox_ppcr = svg_ppcr.bounding_box()
        assert bbox_ppcr is not None
        assert bbox_ppcr["width"] > 0
        assert bbox_ppcr["height"] > 0

        # 4. Component-level tab set verification: Calibration Smooth & Discrete
        calib_smooth_tab = page.locator(
            "button[aria-controls='panel-calibration-calibration-smooth']"
        )
        calib_discrete_tab = page.locator(
            "button[aria-controls='panel-calibration-calibration']"
        )
        assert calib_smooth_tab.is_visible()
        assert calib_discrete_tab.is_visible()

        # Click Smooth tab and verify component & non-zero chart SVG dimensions
        calib_smooth_tab.click()
        smooth_comp = page.locator("[data-component-id='calibration-smooth']")
        smooth_comp.wait_for()
        assert smooth_comp.is_visible()
        smooth_svg = smooth_comp.locator("svg").first
        smooth_svg.wait_for()
        smooth_bbox = smooth_svg.bounding_box()
        assert smooth_bbox is not None
        assert smooth_bbox["width"] > 0
        assert smooth_bbox["height"] > 0

        # Click Discrete tab and verify component & non-zero chart SVG dimensions
        calib_discrete_tab.click()
        discrete_comp = page.locator("[data-component-id='calibration']")
        discrete_comp.wait_for()
        assert discrete_comp.is_visible()
        discrete_svg = discrete_comp.locator("svg").first
        discrete_svg.wait_for()
        discrete_bbox = discrete_svg.bounding_box()
        assert discrete_bbox is not None
        assert discrete_bbox["width"] > 0
        assert discrete_bbox["height"] > 0

        # 5. Component-level tab set verification: Discrimination Curve Tabs (ROC, Lift, PR, Gains)
        disc_ppcr_group_tab = page.locator(
            "button[aria-controls='discrimination-ppcr']"
        )
        disc_ppcr_group_tab.click()

        lift_comp_tab = page.locator(
            "button[aria-controls='panel-discrimination-ppcr-lift-2']"
        )
        assert lift_comp_tab.is_visible()
        lift_comp_tab.click()

        lift_comp = page.locator("[data-component-id='lift-2']")
        lift_comp.wait_for()
        assert lift_comp.is_visible()
        lift_svg = lift_comp.locator("svg").first
        lift_svg.wait_for()
        lift_bbox = lift_svg.bounding_box()
        assert lift_bbox is not None
        assert lift_bbox["width"] > 0
        assert lift_bbox["height"] > 0

        assert len(errors) == 0, f"Console errors found: {errors}"
        browser.close()


def test_browser_summary_report_structure_and_component_counts(tmp_path):
    probs, reals = _inputs()
    output = tmp_path / "structure_test.html"
    create_summary_report(probs, reals, renderer="browser", output_file=output)

    report = _embedded_report(output.read_text(encoding="utf-8"))

    # Exact report title
    assert report["title"] == "Summary Report"

    # Exactly 6 sections
    sections = report["sections"]
    assert len(sections) == 6
    section_ids = [s["id"] for s in sections]
    assert section_ids == [
        "prevalence",
        "prediction-distribution",
        "calibration",
        "discrimination",
        "utility",
        "performance-table",
    ]
    section_titles = [s["title"] for s in sections]
    assert section_titles == [
        "Prevalence",
        "Prediction Distribution",
        "Calibration",
        "Discrimination",
        "Utility",
        "Performance Table",
    ]

    # Section 1: Prediction Distribution section
    pd_sec = sections[1]
    assert pd_sec["id"] == "prediction-distribution"
    assert pd_sec["title"] == "Prediction Distribution"

    pd_items = pd_sec["items"]
    assert len(pd_items) == 2

    # Group 1: By Probability Threshold
    pd_thresh = pd_items[0]
    assert pd_thresh["id"] == "prediction-distribution-probability-threshold"
    assert pd_thresh["title"] == "By Probability Threshold"
    thresh_pd_comps = pd_thresh["components"]
    assert len(thresh_pd_comps) == 1
    assert thresh_pd_comps[0]["id"] == "prediction-distribution"
    assert thresh_pd_comps[0]["title"] == "Prediction Distribution"
    assert thresh_pd_comps[0]["spec"]["type"] == "prediction_distribution"
    assert (
        thresh_pd_comps[0]["spec"]["operatingPoint"]["dimension"]
        == "probability_threshold"
    )

    # Group 2: By PPCR
    pd_ppcr = pd_items[1]
    assert pd_ppcr["id"] == "prediction-distribution-ppcr"
    assert pd_ppcr["title"] == "By Predicted Positives Condition Rate (PPCR)"
    ppcr_pd_comps = pd_ppcr["components"]
    assert len(ppcr_pd_comps) == 1
    assert ppcr_pd_comps[0]["id"] == "prediction-distribution-2"
    assert ppcr_pd_comps[0]["title"] == "Prediction Distribution"
    assert ppcr_pd_comps[0]["spec"]["type"] == "prediction_distribution"
    assert ppcr_pd_comps[0]["spec"]["operatingPoint"]["dimension"] == "ppcr"

    # Section 3: Discrimination section
    disc_sec = sections[3]
    assert disc_sec["id"] == "discrimination"

    disc_items = disc_sec["items"]
    assert len(disc_items) == 3
    assert disc_items[0]["id"] == "auroc"

    # Discrimination Group 1: By Probability Threshold
    disc_grp_thresh = disc_items[1]
    assert disc_grp_thresh["id"] == "discrimination-probability-threshold"
    assert disc_grp_thresh["title"] == "By Probability Threshold"
    disc_thresh_comps = disc_grp_thresh["components"]
    assert [c["id"] for c in disc_thresh_comps] == [
        "roc",
        "lift",
        "precision-recall",
        "gains",
    ]

    # Discrimination Group 2: By PPCR
    disc_grp_ppcr = disc_items[2]
    assert disc_grp_ppcr["id"] == "discrimination-ppcr"
    assert disc_grp_ppcr["title"] == "By Predicted Positives Condition Rate (PPCR)"
    disc_ppcr_comps = disc_grp_ppcr["components"]
    assert [c["id"] for c in disc_ppcr_comps] == [
        "roc-2",
        "lift-2",
        "precision-recall-2",
        "gains-2",
    ]

    # Performance Table section
    perf_sec = sections[5]
    assert perf_sec["id"] == "performance-table"
    perf_items = perf_sec["items"]
    assert len(perf_items) == 2
    assert perf_items[0]["title"] == "By Probability Threshold"
    assert perf_items[1]["title"] == "By Predicted Positives Condition Rate (PPCR)"

    # Collect all component IDs across report
    all_comp_ids = []
    for sec in sections:
        for item in sec["items"]:
            if item["type"] == "component":
                all_comp_ids.append(item["id"])
            elif item["type"] == "group":
                for comp in item["components"]:
                    all_comp_ids.append(comp["id"])

    assert len(all_comp_ids) == 18
    assert len(set(all_comp_ids)) == 18
    assert "prediction-distribution" in all_comp_ids
    assert "prediction-distribution-2" in all_comp_ids
    assert all_comp_ids.count("prediction-distribution") == 1
    assert all_comp_ids.count("prediction-distribution-2") == 1


def test_browser_summary_report_default_grid_and_standalone_identity(tmp_path):
    probs, reals = _inputs()
    output = tmp_path / "identity_test.html"
    create_summary_report(probs, reals, renderer="browser", output_file=output)

    report = _embedded_report(output.read_text(encoding="utf-8"))

    pd_sec = report["sections"][1]
    embedded_thresh_spec = pd_sec["items"][0]["components"][0]["spec"]
    embedded_ppcr_spec = pd_sec["items"][1]["components"][0]["spec"]

    # Grid check 0.00, 0.01, ..., 1.00
    expected_grid = [round(x, 2) for x in np.linspace(0.0, 1.0, 101)]

    thresh_op_values = [
        round(op["value"], 2) for op in embedded_thresh_spec["operatingPoints"]
    ]
    assert thresh_op_values == expected_grid

    ppcr_op_values = [
        round(op["value"], 2) for op in embedded_ppcr_spec["operatingPoints"]
    ]
    assert ppcr_op_values == expected_grid

    # Direct builder comparison
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    perf_thresh = prepare_performance_data(
        probs, reals, stratified_by=("probability_threshold",), by=0.01
    )
    perf_ppcr = prepare_performance_data(probs, reals, stratified_by=("ppcr",), by=0.01)

    thresh_dist_data = _prepare_probs_distribution_data(
        probs, reals, by=0.01, stratified_by=("probability_threshold",)
    )
    ppcr_dist_data = _prepare_probs_distribution_data(
        probs, reals, by=0.01, stratified_by=("ppcr",)
    )

    direct_thresh_spec = _prediction_distribution_v2_spec(
        distribution_data=thresh_dist_data,
        performance_data=perf_thresh,
        evaluation_metadata=metadata,
        stratified_by=("probability_threshold",),
    )
    direct_ppcr_spec = _prediction_distribution_v2_spec(
        distribution_data=ppcr_dist_data,
        performance_data=perf_ppcr,
        evaluation_metadata=metadata,
        stratified_by=("ppcr",),
    )

    assert embedded_thresh_spec == direct_thresh_spec
    assert embedded_ppcr_spec == direct_ppcr_spec


def test_prediction_distribution_frozen_tied_fixture_n9():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    perf_ppcr = prepare_performance_data(probs, reals, stratified_by=("ppcr",), by=0.20)
    dist_ppcr = _prepare_probs_distribution_data(
        probs, reals, by=0.20, stratified_by=("ppcr",)
    )

    spec = _prediction_distribution_v2_spec(
        distribution_data=dist_ppcr,
        performance_data=perf_ppcr,
        evaluation_metadata=metadata,
        stratified_by=("ppcr",),
    )

    # Rank bins assertion
    rank_bins = spec["rankBins"]
    assert len(rank_bins) == 5
    assert [
        (r["rankLower"], r["rankUpper"], r["positiveMass"], r["negativeMass"])
        for r in rank_bins
    ] == [
        (0.0, 0.2, 1, 1),
        (0.2, 0.4, 2, 2),
        (0.4, 0.6, 0, 0),
        (0.6, 0.8, 1, 0),
        (0.8, 1.0, 1, 1),
    ]

    # Operating points assertion
    ops = spec["operatingPoints"]
    assert len(ops) == 6
    op_table = []
    for op in ops:
        metrics = {m["metricId"]: m["estimate"] for m in op["performance"]}
        op_table.append(
            (
                round(op["value"], 2),
                round(op["cutoff"], 2),
                round(op["realizedPpcr"], 6),
                metrics["true_positives"],
                metrics["false_positives"],
                metrics["true_negatives"],
                metrics["false_negatives"],
            )
        )
    assert op_table == [
        (0.0, 1.0, 0.0, 0, 0, 4, 5),
        (0.2, 0.71, round(2 / 9, 6), 1, 1, 3, 4),
        (0.4, 0.5, round(3 / 9, 6), 2, 1, 3, 3),
        (0.6, 0.5, round(3 / 9, 6), 2, 1, 3, 3),
        (0.8, 0.24, round(7 / 9, 6), 4, 3, 1, 1),
        (1.0, 0.0, 1.0, 5, 4, 0, 0),
    ]


def test_browser_summary_report_multiple_evaluations(tmp_path):
    # Multiple models sharing 1 outcome population
    probs_multi_model = {
        "Model A": np.array([0.1, 0.4, 0.6, 0.9]),
        "Model B": np.array([0.2, 0.3, 0.7, 0.8]),
    }
    reals_multi_model = np.array([0, 0, 1, 1])

    out_multi_model = tmp_path / "multi_model.html"
    create_summary_report(
        probs_multi_model,
        reals_multi_model,
        renderer="browser",
        output_file=out_multi_model,
    )
    report_multi_model = _embedded_report(out_multi_model.read_text(encoding="utf-8"))

    thresh_spec = report_multi_model["sections"][1]["items"][0]["components"][0]["spec"]
    evals = thresh_spec["evaluations"]
    assert len(evals) == 2
    assert evals[0]["id"] == "evaluation-1"
    assert evals[0]["model"] == "Model A"
    assert evals[1]["id"] == "evaluation-2"
    assert evals[1]["model"] == "Model B"

    # Multiple keyed populations
    probs_multi_pop = {
        "Pop A": np.array([0.1, 0.5, 0.9]),
        "Pop B": np.array([0.2, 0.4, 0.6, 0.8, 0.9]),
    }
    reals_multi_pop = {
        "Pop A": np.array([0, 1, 1]),
        "Pop B": np.array([0, 0, 1, 1, 1]),
    }

    out_multi_pop = tmp_path / "multi_pop.html"
    create_summary_report(
        probs_multi_pop, reals_multi_pop, renderer="browser", output_file=out_multi_pop
    )
    report_multi_pop = _embedded_report(out_multi_pop.read_text(encoding="utf-8"))

    thresh_spec_pop = report_multi_pop["sections"][1]["items"][0]["components"][0][
        "spec"
    ]
    evals_pop = thresh_spec_pop["evaluations"]
    assert len(evals_pop) == 2
    assert evals_pop[0]["id"] == "evaluation-1"
    assert evals_pop[0]["population"] == "Pop A"
    assert "model" not in evals_pop[0]
    assert evals_pop[1]["id"] == "evaluation-2"
    assert evals_pop[1]["population"] == "Pop B"
    assert "model" not in evals_pop[1]


def test_browser_summary_report_authoritative_schema_validation(tmp_path):
    probs, reals = _inputs()
    output = tmp_path / "schema_val_report.html"
    create_summary_report(probs, reals, renderer="browser", output_file=output)

    report_spec = _embedded_report(output.read_text(encoding="utf-8"))

    report_schema_path = files("rtichoke").joinpath(
        "_vendor", "rtichoke_viz", "rtichoke-viz-report.schema.json"
    )
    report_schema = json.loads(report_schema_path.read_text(encoding="utf-8"))

    v2_schema_path = files("rtichoke").joinpath(
        "_vendor", "rtichoke_viz", "rtichoke-viz-v2.schema.json"
    )
    v2_schema = json.loads(v2_schema_path.read_text(encoding="utf-8"))

    report_validator = Draft7Validator(report_schema)
    v2_validator = Draft202012Validator(v2_schema)

    report_errors = list(report_validator.iter_errors(report_spec))
    assert not report_errors, f"Report schema validation errors: {report_errors}"

    pd_sec = report_spec["sections"][1]
    thresh_pred_dist = pd_sec["items"][0]["components"][0]["spec"]
    thresh_errors = list(v2_validator.iter_errors(thresh_pred_dist))
    assert not thresh_errors, (
        f"Threshold prediction distribution v2 schema errors: {thresh_errors}"
    )

    ppcr_pred_dist = pd_sec["items"][1]["components"][0]["spec"]
    ppcr_errors = list(v2_validator.iter_errors(ppcr_pred_dist))
    assert not ppcr_errors, (
        f"PPCR prediction distribution v2 schema errors: {ppcr_errors}"
    )


def test_browser_summary_report_does_not_duplicate_producer_calls(
    monkeypatch, tmp_path
):
    probs, reals = _inputs()
    output = tmp_path / "producer_counts.html"

    perf_calls = []
    orig_perf = summary_report_module.prepare_performance_data

    def spy_perf(*args, **kwargs):
        perf_calls.append(kwargs.get("stratified_by"))
        return orig_perf(*args, **kwargs)

    dist_calls = []
    orig_dist = summary_report_module._prepare_probs_distribution_data

    def spy_dist(*args, **kwargs):
        dist_calls.append(kwargs.get("stratified_by"))
        return orig_dist(*args, **kwargs)

    def fail_create_probs_histogram(*args, **kwargs):
        raise AssertionError("report must not call public create_probs_histogram")

    monkeypatch.setattr(summary_report_module, "prepare_performance_data", spy_perf)
    monkeypatch.setattr(
        summary_report_module, "_prepare_probs_distribution_data", spy_dist
    )
    monkeypatch.setattr(rtichoke, "create_probs_histogram", fail_create_probs_histogram)

    create_summary_report(probs, reals, renderer="browser", output_file=output)

    assert perf_calls == [("probability_threshold",), ("ppcr",)]
    assert dist_calls == [("probability_threshold",), ("ppcr",)]


def test_browser_summary_report_non_regression():
    probs, reals = _inputs()

    chart = rtichoke.create_probs_histogram(probs, reals)
    assert isinstance(chart, RtichokeBrowserChart)
    assert chart.spec["type"] == "prediction_distribution"
    assert chart.spec["schemaVersion"] == "2.0"

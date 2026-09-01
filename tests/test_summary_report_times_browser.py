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

import rtichoke
from rtichoke._calibration_viz_spec_v2 import _calibration_v2_spec_from_curve_list
from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_times_v2_spec_from_performance_data,
)
from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_times_v2_spec_from_performance_data,
)
from rtichoke._performance_table_spec import (
    _performance_table_times_spec_from_performance_data,
)
from rtichoke._viz_spec_v2 import (
    _gains_times_v2_spec_from_performance_data,
    _lift_times_v2_spec_from_performance_data,
    _precision_recall_times_v2_spec_from_performance_data,
)
from rtichoke.calibration.calibration import _create_calibration_curve_list_times
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata
from rtichoke.summary_report.summary_report import create_summary_report_times


def _inputs():
    probs = {
        "Model A": np.array(
            [0.05, 0.12, 0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
    }
    reals = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    fixed_time_horizons = [4.0, 8.0]
    return probs, reals, times, fixed_time_horizons


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
    assert "Calibration" in rendered
    assert "Discrimination" in rendered
    assert "Utility" in rendered
    assert "Performance Table" in rendered
    assert "By Predicted Positives Condition Rate (PPCR)" in rendered
    assert "<table" in rendered
    assert rendered.count("<svg") >= 2


def test_public_export_create_summary_report_times():
    assert hasattr(rtichoke, "create_summary_report_times")
    assert rtichoke.create_summary_report_times is create_summary_report_times


def test_summary_report_times_spec_structure_and_ordering(tmp_path):
    probs, reals, times, horizons = _inputs()
    output = tmp_path / "summary_report_times.html"

    result = create_summary_report_times(
        probs, reals, times, fixed_time_horizons=horizons, output_file=output
    )

    assert result == output
    assert output.exists()

    report = _embedded_report(output.read_text(encoding="utf-8"))
    assert report["schemaVersion"] == "1.1"
    assert report["type"] == "report"

    # Exact section hierarchy
    sections = report["sections"]
    section_ids = [s["id"] for s in sections]
    assert section_ids == [
        "calibration",
        "discrimination",
        "utility",
        "performance-table",
    ]

    # Section titles
    assert [s["title"] for s in sections] == [
        "Calibration",
        "Discrimination",
        "Utility",
        "Performance Table",
    ]

    # Explicit omissions check
    assert "prevalence" not in section_ids
    assert "auroc" not in json.dumps(report)

    # Calibration section components
    calib = sections[0]
    assert [c["id"] for c in calib["items"]] == ["calibration-smooth", "calibration"]
    assert [c["title"] for c in calib["items"]] == ["Smooth", "Discrete"]

    # Discrimination section groups and components
    disc = sections[1]
    assert len(disc["items"]) == 2
    g1, g2 = disc["items"]

    assert g1["id"] == "discrimination-probability-threshold"
    assert g1["title"] == "By Probability Threshold"
    assert [c["id"] for c in g1["components"]] == ["precision-recall", "gains", "lift"]
    assert [c["title"] for c in g1["components"]] == [
        "Precision-Recall",
        "Gains",
        "Lift",
    ]

    assert g2["id"] == "discrimination-ppcr"
    assert g2["title"] == "By Predicted Positives Condition Rate (PPCR)"
    assert [c["id"] for c in g2["components"]] == [
        "precision-recall-2",
        "gains-2",
        "lift-2",
    ]
    assert [c["title"] for c in g2["components"]] == [
        "Precision-Recall",
        "Gains",
        "Lift",
    ]

    # Utility section components
    util = sections[2]
    assert [c["id"] for c in util["items"]] == [
        "decision-curve",
        "interventions-avoided",
    ]
    assert [c["title"] for c in util["items"]] == [
        "Decision Curve",
        "Interventions Avoided",
    ]

    # Performance Table section groups and components
    table_sec = sections[3]
    assert len(table_sec["items"]) == 2
    tg1, tg2 = table_sec["items"]

    assert tg1["id"] == "performance-table-probability-threshold"
    assert tg1["title"] == "By Probability Threshold"
    assert [c["id"] for c in tg1["components"]] == ["performance-table"]

    assert tg2["id"] == "performance-table-ppcr"
    assert tg2["title"] == "By Predicted Positives Condition Rate (PPCR)"
    assert [c["id"] for c in tg2["components"]] == ["performance-table-2"]


def test_summary_report_times_preserves_standalone_canonical_producers(tmp_path):
    probs, reals, times, horizons = _inputs()
    heuristics_sets = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ]
    output = tmp_path / "standalone_comparison.html"

    create_summary_report_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        output_file=output,
    )

    report = _embedded_report(output.read_text(encoding="utf-8"))
    metadata = _build_evaluation_metadata(probs, reals, times)

    # Re-generate standalone specs directly
    perf_thresh = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=("probability_threshold",),
        by=0.01,
    )
    perf_ppcr = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=("ppcr",),
        by=0.01,
    )

    calib_list_smooth = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        calibration_type="smooth",
    )

    expected_calib_smooth = _calibration_v2_spec_from_curve_list(
        calib_list_smooth, metadata, calibration_type="smooth"
    )
    expected_pr_thresh = _precision_recall_times_v2_spec_from_performance_data(
        perf_thresh, metadata
    )
    expected_gains_ppcr = _gains_times_v2_spec_from_performance_data(
        perf_ppcr, metadata, operating_point_dimension="ppcr"
    )
    expected_lift_thresh = _lift_times_v2_spec_from_performance_data(
        perf_thresh, metadata
    )
    expected_dc = _decision_curve_times_v2_spec_from_performance_data(
        perf_thresh, metadata
    )
    expected_ia = _interventions_avoided_times_v2_spec_from_performance_data(
        perf_thresh, metadata
    )
    expected_table_thresh = _performance_table_times_spec_from_performance_data(
        perf_thresh, metadata
    )

    report_calib_smooth = report["sections"][0]["items"][0]["spec"]
    report_pr_thresh = report["sections"][1]["items"][0]["components"][0]["spec"]
    report_gains_ppcr = report["sections"][1]["items"][1]["components"][1]["spec"]
    report_lift_thresh = report["sections"][1]["items"][0]["components"][2]["spec"]
    report_dc = report["sections"][2]["items"][0]["spec"]
    report_ia = report["sections"][2]["items"][1]["spec"]
    report_table_thresh = report["sections"][3]["items"][0]["components"][0]["spec"]

    assert report_calib_smooth == expected_calib_smooth
    assert report_pr_thresh == expected_pr_thresh
    assert report_gains_ppcr == expected_gains_ppcr
    assert report_lift_thresh == expected_lift_thresh
    assert report_dc == expected_dc
    assert report_ia == expected_ia
    assert report_table_thresh == expected_table_thresh


def test_summary_report_times_multiple_horizons_ordering_and_context(tmp_path):
    probs = {
        "Model 1": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "Model 2": np.array([0.2, 0.4, 0.6, 0.8, 0.95]),
    }
    reals = np.array([0, 1, 0, 1, 1])
    times = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    horizons = [6.0, 2.0, 8.0]  # Non-sorted explicit horizon order

    output = tmp_path / "multiple_horizons.html"
    create_summary_report_times(
        probs, reals, times, fixed_time_horizons=horizons, output_file=output
    )

    report = _embedded_report(output.read_text(encoding="utf-8"))
    pr_spec = report["sections"][1]["items"][0]["components"][0]["spec"]

    # Verify horizon identities in series preserve fixed_time_horizons order
    series_horizons = [s["horizon"] for s in pr_spec["series"]]
    assert 6.0 in series_horizons
    assert 2.0 in series_horizons or 8.0 in series_horizons


def test_summary_report_times_executes_when_opened_directly(tmp_path):
    probs, reals, times, horizons = _inputs()
    output = create_summary_report_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        output_file=tmp_path / "summary_report_times.html",
    )
    assert isinstance(output, Path)

    browser = _dump_dom(output.resolve().as_uri())
    _assert_report_rendered(browser)


def test_summary_report_times_executes_over_localhost(tmp_path):
    probs, reals, times, horizons = _inputs()
    output = create_summary_report_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        output_file=tmp_path / "summary_report_times.html",
    )
    assert isinstance(output, Path)

    with _serve(output.parent) as base_url:
        browser = _dump_dom(f"{base_url}/{output.name}")

    _assert_report_rendered(browser)


def test_summary_report_times_html_contains_no_literal_nan_or_infinity(tmp_path):
    probs, reals, times, horizons = _inputs()
    output = create_summary_report_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        output_file=tmp_path / "no_nan.html",
    )
    html_text = output.read_text(encoding="utf-8")

    # Embedded JSON script block must not contain unquoted NaN or Infinity tokens
    start = html_text.index(
        '<script id="rtichoke-report-spec" type="application/json">'
    )
    end = html_text.index("</script>", start)
    json_block = html_text[start:end]

    assert ":NaN" not in json_block
    assert ":Infinity" not in json_block
    assert ":-Infinity" not in json_block


def test_time_lift_excludes_only_non_finite_values_retains_boundary_points(tmp_path):
    probs, reals, times, horizons = _inputs()
    heuristics_sets = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ]
    metadata = _build_evaluation_metadata(probs, reals, times)

    perf_thresh = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=("probability_threshold",),
        by=0.01,
    )

    lift_spec = _lift_times_v2_spec_from_performance_data(perf_thresh, metadata)

    # 1. Non-finite values (lift is null or non-existent in data points)
    data_points = lift_spec["data"]
    for pt in data_points:
        assert pt["lift"] is not None
        assert np.isfinite(pt["lift"])

    # 2. Valid finite boundary points are retained (e.g. cutoff = 0.0 / low cutoffs)
    cutoffs = [pt["cutoff"] for pt in data_points]
    assert 0.0 in cutoffs or min(cutoffs) <= 0.05

    # 3. Reference geometry remains present and intact
    refs = lift_spec["references"]
    ref_types = [r["type"] for r in refs]
    assert "horizontal" in ref_types
    assert "path" in ref_types
    horizontal_ref = next(r for r in refs if r["type"] == "horizontal")
    assert horizontal_ref["value"] == 1.0

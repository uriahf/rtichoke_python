import re

import numpy as np

from rtichoke.summary_report.summary_report import create_summary_report


def test_create_summary_report_writes_native_html(tmp_path):
    probs = {"model": np.array([0.05, 0.2, 0.4, 0.7, 0.9])}
    reals = np.array([0, 0, 1, 1, 1])
    output = tmp_path / "report.html"

    result = create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")
    assert result == output
    for text in (
        "Summary Report",
        "Performance Metrics Cheat Sheet",
        "Calibration",
        "Smooth",
        "Discrete",
        "Discrimination",
        "By Probability Threshold",
        "By Predicted Positives Condition Rate (PPCR)",
        "ROC",
        "Lift",
        "Precision Recall",
        "Gains",
        "Utility (Decision Curve)",
        "Performance Table",
        "table-threshold",
        "table-ppcr",
        "Confusion Matrix",
    ):
        assert text in html
    assert "perf(R.tables.threshold" in html
    assert "application/vnd.jupyter.widget-state+json" not in html
    assert "application/vnd.jupyter.widget-view+json" not in html
    assert "@jupyter-widgets" not in html
    assert "d3.scaleLinear()" in html
    assert "send_requests_to_rtichoke_r" not in html
    assert "quarto" not in html.lower()

    # The report must remain usable as a single offline HTML file. JavaScript
    # and styling are embedded directly rather than fetched from a CDN or
    # another external runtime at viewing time.
    assert "cdn.jsdelivr.net" not in html
    assert "unpkg.com" not in html
    assert not re.search(r'<script[^>]+src=["\']https?://', html, re.IGNORECASE)
    assert not re.search(r'<link[^>]+href=["\']https?://', html, re.IGNORECASE)
    assert "Minimal D3-compatible runtime used by rtichoke summary reports" in html

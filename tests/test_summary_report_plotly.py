import re

import numpy as np

from rtichoke import create_summary_report


def test_public_summary_report_uses_embedded_plotly_for_charts(tmp_path):
    probs = {
        "Model A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "Model B": np.array([0.10, 0.25, 0.30, 0.60, 0.70, 0.90]),
    }
    reals = np.array([0, 0, 0, 1, 1, 1])
    output = tmp_path / "report.html"

    create_summary_report(probs, reals, output_file=output, by=0.1)

    html = output.read_text(encoding="utf-8")

    # Plotly.js is embedded once in the self-contained report; there is no CDN
    # or external script/style dependency at viewing time.
    assert "plotly.js v" in html.lower()
    assert "Plotly.react(host,fig.data,fig.layout,config)" in html
    assert "Plotly.react(chart,spec.figure.data,spec.figure.layout,config)" in html
    assert "draw('smoothchart',RP.smooth)" in html
    assert "draw('discretechart',RP.discrete)" in html
    assert "draw('decision',RP.decision)" in html
    assert not re.search(r'<script[^>]+src=["\']https?://', html, re.IGNORECASE)
    assert not re.search(r'<link[^>]+href=["\']https?://', html, re.IGNORECASE)

    # The existing lightweight report shell/table renderer remains in place.
    assert "summary-table auc-table" in html
    assert "rt-perf-wrap" in html
    assert "Performance Metrics Cheat Sheet" in html

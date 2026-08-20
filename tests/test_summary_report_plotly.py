import re

import numpy as np

from rtichoke import create_summary_report
from rtichoke.summary_report import summary_report_plotly


def _tiny_payload():
    figure = {"data": [], "layout": {"width": 500, "height": 550}}
    return {
        "smooth": figure,
        "discrete": figure,
        "threshold": [{"label": "ROC", "figure": figure}],
        "ppcr": [{"label": "ROC", "figure": figure}],
        "decision": figure,
    }


def test_public_export_routes_to_plotly_summary_report():
    assert create_summary_report is summary_report_plotly.create_summary_report


def test_plotly_chart_layer_is_self_contained(monkeypatch):
    monkeypatch.setattr(
        summary_report_plotly,
        "get_plotlyjs",
        lambda: "/*! plotly.js vTEST */",
    )
    html = """<!doctype html><html><head></head><body>
<div id="smoothchart"></div><div id="discretechart"></div>
<div id="thrtabs"></div><div id="thrchart"></div>
<div id="pcrtabs"></div><div id="pcrchart"></div>
<div id="decision"></div></body></html>"""

    rendered = summary_report_plotly._inject_plotly_charts(html, _tiny_payload())

    assert "plotly.js vTEST" in rendered
    assert "Plotly.react(host,fig.data,fig.layout,config)" in rendered
    assert "Plotly.react(chart,spec.figure.data,spec.figure.layout,config)" in rendered
    assert "draw('smoothchart',RP.smooth)" in rendered
    assert "draw('discretechart',RP.discrete)" in rendered
    assert "draw('decision',RP.decision)" in rendered
    assert not re.search(r'<script[^>]+src=["\']https?://', rendered, re.IGNORECASE)
    assert not re.search(r'<link[^>]+href=["\']https?://', rendered, re.IGNORECASE)


def test_page_parity_adds_r_math_and_multi_population_prevalence():
    html = """<html><head></head><body>
<details><div class="metric-formulas"><div>old</div></div></details>
<div id="prev"></div></body></html>"""
    probs = {
        "Population A": np.array([0.1, 0.8]),
        "Population B": np.array([0.2, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 1]),
        "Population B": np.array([0, 1]),
    }

    rendered = summary_report_plotly._wire_page_parity(html, probs, reals)

    assert "r-math-blocks" in rendered
    assert "Lift =" in rendered
    assert "frac compound" in rendered
    assert "r-prevalence-multi" in rendered
    assert "<span>population</span><span>Prevalence</span>" in rendered
    assert "model-badge" in rendered

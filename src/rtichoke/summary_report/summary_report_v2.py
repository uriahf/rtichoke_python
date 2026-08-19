"""Readable integration layer for the lightweight summary report renderers.

This module keeps the legacy report generator stable while the large inline
HTML template is being split into maintainable assets. It post-processes the
legacy HTML to route performance and decision curves through the dedicated
Plotly-parity D3 renderer and applies the shared R-report visual polish layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from rtichoke.summary_report.curve_renderer import curve_renderer_source
from rtichoke.summary_report.summary_report import create_summary_report as _legacy_create_summary_report


def _style_source() -> str:
    return Path(__file__).with_name("report_style.css").read_text(encoding="utf-8")


def _wire_curve_renderer(html: str) -> str:
    """Replace the legacy curve drawing calls with the dedicated renderer."""
    renderer = curve_renderer_source()
    marker = "function curveTabs(specs,nav,chart,strat)"
    if marker not in html:
        raise RuntimeError("Could not locate summary-report curve integration point")
    html = html.replace(marker, renderer + "\n" + marker, 1)
    html = html.replace(
        "draw(s,chart,strat)}}));draw(specs[0],chart,strat)",
        "drawRtichokeCurve(s,chart)}}));drawRtichokeCurve(specs[0],chart)",
        1,
    )
    html = html.replace(
        "draw(R.decision,'#decision','probability_threshold');",
        "drawRtichokeCurve(R.decision,'#decision');",
        1,
    )
    return html


def _wire_report_style(html: str) -> str:
    css = _style_source()
    marker = "</head>"
    if marker not in html:
        raise RuntimeError("Could not locate summary-report head element")
    return html.replace(marker, f"<style>\n{css}\n</style>\n{marker}", 1)


def _wire_r_report_chrome(html: str) -> str:
    """Add the lightweight equivalents of the R Markdown TOC and cheat sheet."""
    # The R reference starts with a compact TOC and a collapsed metric cheat
    # sheet. Keep these semantic and dependency-free rather than importing the
    # Bootstrap/Reactable runtime that accounts for much of the R file weight.
    toc = """<nav class=\"rt-toc\" aria-label=\"Report contents\"><ul>
<li><a href=\"#calibration\">Calibration</a></li>
<li><a href=\"#discrimination\">Discrimination</a></li>
<li><a href=\"#utility\">Utility (Decision Curve)</a></li>
<li><a href=\"#performance-table\">Performance Table</a></li>
</ul></nav>
<details class=\"metric-cheat-sheet\"><summary>Performance Metrics Cheat Sheet</summary>
<table><thead><tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr></thead>
<tbody><tr><th>Real Positive</th><td class=\"cm-tp\">TP</td><td class=\"cm-fn\">FN</td></tr>
<tr><th>Real Negative</th><td class=\"cm-fp\">FP</td><td class=\"cm-tn\">TN</td></tr></tbody></table>
</details>"""
    # Insert immediately before the first report section when possible.
    for marker in ('<section id="calibration"', '<div id="calibration"'):
        if marker in html:
            html = html.replace(marker, toc + "\n" + marker, 1)
            break
    # Add stable anchors to the existing section headings without changing
    # their visible text. These replacements are intentionally tolerant of
    # the legacy template's compact markup.
    replacements = {
        ">Calibration<": ' id="calibration">Calibration<',
        ">Discrimination<": ' id="discrimination">Discrimination<',
        ">Utility<": ' id="utility">Utility<',
        ">Performance Table<": ' id="performance-table">Performance Table<',
    }
    for old, new in replacements.items():
        if old in html and f'id="{new.split("id=\"")[1].split("\"")[0]}"' not in html:
            html = html.replace(old, new, 1)
    return html


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    output_file: str | Path = "summary_report.html",
    by: float = 0.01,
) -> Path:
    """Create the lightweight report using the modular parity renderers."""
    out = Path(output_file)
    _legacy_create_summary_report(probs=probs, reals=reals, output_file=out, by=by)
    html = out.read_text(encoding="utf-8")
    html = _wire_curve_renderer(html)
    html = _wire_r_report_chrome(html)
    html = _wire_report_style(html)
    out.write_text(html, encoding="utf-8")
    return out

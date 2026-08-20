"""Readable integration layer for the lightweight summary report renderers.

This module keeps the legacy report generator stable while the large inline
HTML template is being split into maintainable assets. It post-processes the
legacy HTML to route performance and decision curves through the dedicated
Plotly-parity renderer and applies the shared R-report visual polish layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from rtichoke.summary_report.curve_renderer import curve_renderer_source
from rtichoke.summary_report.summary_report import create_summary_report as _legacy_create_summary_report


def _asset_source(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8")


def _style_source() -> str:
    return _asset_source("report_style.css")


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


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    output_file: str | Path = "summary_report.html",
    by: float = 0.01,
) -> Path:
    """Create a lightweight, self-contained report using parity renderers.

    The base renderer already embeds the bundled visualization runtime. This
    layer only wires the modular curve renderer and shared report styling.
    """
    out = Path(output_file)
    _legacy_create_summary_report(probs=probs, reals=reals, output_file=out, by=by)
    html = out.read_text(encoding="utf-8")
    html = _wire_curve_renderer(html)
    html = _wire_report_style(html)
    out.write_text(html, encoding="utf-8")
    return out

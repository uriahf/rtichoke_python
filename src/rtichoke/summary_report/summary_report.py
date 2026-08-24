"""Public summary-report entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Literal, Union, cast

import numpy as np

from rtichoke._calibration_viz_spec_v2 import _calibration_v2_spec_from_curve_list
from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
)
from rtichoke._report_browser import RtichokeBrowserReport
from rtichoke._report_spec import _report_spec_from_components
from rtichoke._viz_spec_v2 import _roc_v2_spec_from_performance_data
from rtichoke.calibration.calibration import _create_calibration_curve_list
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata
from rtichoke.processing.send_post_request_to_r_rtichoke import (
    send_requests_to_rtichoke_r,
)
from rtichoke.processing.transforms import _create_list_data_to_adjust

SummaryReportRenderer = Literal["r", "browser"]


def create_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    url_api: str = "http://localhost:4242/",
    *,
    renderer: SummaryReportRenderer = "r",
    output_file: str | Path = "summary_report.html",
) -> Path | None:
    """Create an rtichoke model-performance summary report.

    The default ``renderer="r"`` preserves the historical public behavior and
    delegates to the R rtichoke backend at ``url_api``. ``renderer="browser"``
    is an explicit opt-in path that uses Python's existing production
    calculations, canonical standalone component builders, canonical ReportSpec
    assembly, and the vendored ``rtichoke_viz`` ``renderReport()`` composer.

    The first browser report contains a canonical PerformanceTable, ROC-v2, and
    calibration-v2 component, in that order. The browser renderer writes an HTML
    file plus the vendored ``rtichoke-viz.js`` and ``rtichoke-viz.css`` assets
    beside it, and returns the written HTML path. The historical R path retains
    its existing return behavior (``None``).

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or population names to predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true binary outcome labels.
    url_api : str, optional
        The API endpoint URL of the historical R rtichoke backend. Used only by
        ``renderer="r"``. Defaults to ``"http://localhost:4242/"``.
    renderer : {"r", "browser"}, optional
        Summary-report backend. Defaults to ``"r"`` for backward compatibility.
    output_file : str or pathlib.Path, optional
        HTML destination for ``renderer="browser"``. Defaults to
        ``"summary_report.html"``.

    Returns
    -------
    pathlib.Path or None
        The generated HTML path for ``renderer="browser"``; ``None`` for the
        historical R backend.
    """
    if renderer == "browser":
        return _create_browser_summary_report(probs, reals, output_file=output_file)
    if renderer != "r":
        raise ValueError("renderer must be either 'r' or 'browser'")

    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={"probs": probs, "reals": reals},
        url_api=url_api,
        endpoint="create_summary_report",
    )
    print(rtichoke_response.json()[0].keys())
    return None


def _create_browser_summary_report(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    *,
    output_file: str | Path,
) -> Path:
    """Build the first public canonical browser report from production outputs."""
    performance_data = prepare_performance_data(probs, reals)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    calibration_curve_list = _create_calibration_curve_list(probs, reals)

    performance_table = _performance_table_spec_from_performance_data(
        performance_data, metadata
    )
    roc = _roc_v2_spec_from_performance_data(performance_data, metadata)
    calibration = _calibration_v2_spec_from_curve_list(
        calibration_curve_list, metadata
    )

    report = _report_spec_from_components(
        [
            {"title": "Performance", "spec": performance_table},
            {"title": "ROC", "spec": roc},
            {"title": "Calibration", "spec": calibration},
        ],
        title="rtichoke summary report",
    )
    return RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        output_file
    )


def render_summary_report():
    """Render the historical rtichoke Summary Report using Quarto.

    This function is unchanged by the canonical browser-report opt-in path. It
    renders ``aj_estimate_summary_report.qmd`` to ``summary_report.html`` using
    the local Quarto executable.
    """
    template_path = "aj_estimate_summary_report.qmd"
    output_path = "summary_report.html"

    command = [
        "quarto",
        "render",
        template_path,
        "--to",
        "html",
        "--output",
        output_path,
    ]
    subprocess.run(command, check=True)


def create_data_for_summary_report(probs, reals, times, fixed_time_horizons):
    stratified_by = ["probability_threshold", "ppcr"]
    by = 0.1

    list_data_to_adjust_polars = _create_list_data_to_adjust(
        probs, reals, times, stratified_by=stratified_by, by=by, times_dict={}
    )

    return list_data_to_adjust_polars

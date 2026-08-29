"""Public summary-report entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Literal, Union, cast

import numpy as np

from rtichoke._calibration_viz_spec_v2 import _calibration_v2_spec_from_curve_list
from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_v2_spec_from_performance_data,
)
from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_v2_spec_from_performance_data,
)
from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
)
from rtichoke._report_browser import RtichokeBrowserReport
from rtichoke._report_spec import _build_report_spec_v11
from rtichoke._summary_metrics_spec import (
    _auroc_summary_metrics_spec,
    _prevalence_summary_metrics_spec,
)
from rtichoke._viz_spec_v2 import (
    _gains_v2_spec_from_performance_data,
    _lift_v2_spec_from_performance_data,
    _precision_recall_v2_spec_from_performance_data,
    _roc_v2_spec_from_performance_data,
)
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
    """Build the canonical static ReportSpec v1.1 browser summary report."""
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))

    # Stratified by probability threshold
    perf_data_thresh = prepare_performance_data(
        probs, reals, stratified_by=("probability_threshold",), by=0.01
    )
    # Stratified by PPCR
    perf_data_ppcr = prepare_performance_data(
        probs, reals, stratified_by=("ppcr",), by=0.01
    )

    calibration_curve_list = _create_calibration_curve_list(probs, reals)

    # Specs
    prev_spec = _prevalence_summary_metrics_spec(perf_data_thresh, metadata)
    auroc_spec = _auroc_summary_metrics_spec(probs, reals, metadata)

    calib_smooth_spec = _calibration_v2_spec_from_curve_list(
        calibration_curve_list, metadata, calibration_type="smooth"
    )
    calib_discrete_spec = _calibration_v2_spec_from_curve_list(
        calibration_curve_list, metadata, calibration_type="discrete"
    )

    roc_thresh_spec = _roc_v2_spec_from_performance_data(perf_data_thresh, metadata)
    pr_thresh_spec = _precision_recall_v2_spec_from_performance_data(
        perf_data_thresh, metadata
    )
    gains_thresh_spec = _gains_v2_spec_from_performance_data(perf_data_thresh, metadata)
    lift_thresh_spec = _lift_v2_spec_from_performance_data(perf_data_thresh, metadata)

    roc_ppcr_spec = _roc_v2_spec_from_performance_data(perf_data_ppcr, metadata)
    pr_ppcr_spec = _precision_recall_v2_spec_from_performance_data(
        perf_data_ppcr, metadata
    )
    gains_ppcr_spec = _gains_v2_spec_from_performance_data(perf_data_ppcr, metadata)
    lift_ppcr_spec = _lift_v2_spec_from_performance_data(perf_data_ppcr, metadata)

    decision_curve_spec = _decision_curve_v2_spec_from_performance_data(
        perf_data_thresh, metadata
    )
    interventions_avoided_spec = _interventions_avoided_v2_spec_from_performance_data(
        perf_data_thresh, metadata
    )

    table_thresh_spec = _performance_table_spec_from_performance_data(
        perf_data_thresh, metadata
    )
    table_ppcr_spec = _performance_table_spec_from_performance_data(
        perf_data_ppcr, metadata
    )

    sections = [
        {
            "id": "prevalence",
            "title": "Prevalence",
            "components": [
                {
                    "id": "prevalence-summary",
                    "title": "Prevalence summary",
                    "spec": prev_spec,
                }
            ],
        },
        {
            "id": "calibration",
            "title": "Calibration",
            "components": [
                {
                    "id": "calibration-smooth",
                    "title": "Smooth",
                    "spec": calib_smooth_spec,
                },
                {
                    "id": "calibration",
                    "title": "Discrete",
                    "spec": calib_discrete_spec,
                },
            ],
        },
        {
            "id": "discrimination",
            "title": "Discrimination",
            "components": [
                {
                    "id": "auroc",
                    "title": "AUROC",
                    "spec": auroc_spec,
                }
            ],
            "groups": [
                {
                    "id": "discrimination-probability-threshold",
                    "title": "By Probability Threshold",
                    "components": [
                        {"id": "roc", "title": "ROC", "spec": roc_thresh_spec},
                        {
                            "id": "precision-recall",
                            "title": "Precision-Recall",
                            "spec": pr_thresh_spec,
                        },
                        {
                            "id": "gains",
                            "title": "Gains",
                            "spec": gains_thresh_spec,
                        },
                        {"id": "lift", "title": "Lift", "spec": lift_thresh_spec},
                    ],
                },
                {
                    "id": "discrimination-ppcr",
                    "title": "By PPCR",
                    "components": [
                        {"id": "roc-2", "title": "ROC", "spec": roc_ppcr_spec},
                        {
                            "id": "precision-recall-2",
                            "title": "Precision-Recall",
                            "spec": pr_ppcr_spec,
                        },
                        {
                            "id": "gains-2",
                            "title": "Gains",
                            "spec": gains_ppcr_spec,
                        },
                        {"id": "lift-2", "title": "Lift", "spec": lift_ppcr_spec},
                    ],
                },
            ],
        },
        {
            "id": "utility",
            "title": "Utility",
            "components": [
                {
                    "id": "decision-curve",
                    "title": "Decision Curve",
                    "spec": decision_curve_spec,
                },
                {
                    "id": "interventions-avoided",
                    "title": "Interventions Avoided",
                    "spec": interventions_avoided_spec,
                },
            ],
        },
        {
            "id": "performance-table",
            "title": "Performance Table",
            "groups": [
                {
                    "id": "performance-table-probability-threshold",
                    "title": "By Probability Threshold",
                    "components": [
                        {
                            "id": "performance-table",
                            "title": "Performance Table",
                            "spec": table_thresh_spec,
                        }
                    ],
                },
                {
                    "id": "performance-table-ppcr",
                    "title": "By PPCR",
                    "components": [
                        {
                            "id": "performance-table-2",
                            "title": "Performance Table",
                            "spec": table_ppcr_spec,
                        }
                    ],
                },
            ],
        },
    ]

    report = _build_report_spec_v11(sections, title="rtichoke summary report")
    return RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(output_file)


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

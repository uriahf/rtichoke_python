"""Summary-report helpers for rtichoke."""

import subprocess

from rtichoke.processing.send_post_request_to_r_rtichoke import (
    send_requests_to_rtichoke_r,
)
from rtichoke.processing.transforms import (
    _create_list_data_to_adjust,
)


def create_summary_report(probs, reals, url_api="http://localhost:4242/"):
    """Request an rtichoke summary report from the rtichoke R API.

    The current implementation sends model predictions and observed outcomes to
    the ``create_summary_report`` endpoint exposed by the rtichoke R service.
    It prints the keys returned by the service and does not currently return a
    Python report object or write a standalone HTML file.

    Parameters
    ----------
    probs : dict
        Predicted probabilities, typically keyed by model or population name.
    reals : dict
        Observed binary outcomes, typically keyed by population name.
    url_api : str, default="http://localhost:4242/"
        Base URL of the running rtichoke R API service.

    Returns
    -------
    None
        The function currently prints information from the API response.

    Notes
    -----
    A running rtichoke R API service is required at ``url_api``. This helper
    represents the current bridge to the R summary-report implementation; a
    self-contained Python-native HTML report is not yet implemented here.
    """
    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={"probs": probs, "reals": reals},
        url_api=url_api,
        endpoint="create_summary_report",
    )
    print(rtichoke_response.json()[0].keys())


def render_summary_report():
    """Render the rtichoke summary-report Quarto template.

    This internal helper renders ``aj_estimate_summary_report.qmd`` to
    ``summary_report.html`` using the Quarto command-line interface.
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
    """Prepare stratified data used by the summary-report implementation."""
    stratified_by = ["probability_threshold", "ppcr"]
    by = 0.1

    list_data_to_adjust_polars = _create_list_data_to_adjust(
        probs, reals, times, stratified_by=stratified_by, by=by, times_dict={}
    )

    return list_data_to_adjust_polars

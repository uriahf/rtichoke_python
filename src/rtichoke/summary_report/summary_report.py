"""
A module for Summary Report
"""

from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r
from rtichoke.helpers.sandbox_observable_helpers import (
    create_list_data_to_adjust,
)
import subprocess


def create_summary_report(probs, reals, url_api="http://localhost:4242/"):
    """Create rtichoke Summary Report

    Args:
        probs (_type_): _description_
        reals (_type_): _description_
        url_api (str, optional): _description_. Defaults to "http://localhost:4242/".
    """
    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={"probs": probs, "reals": reals},
        url_api=url_api,
        endpoint="create_summary_report",
    )
    print(rtichoke_response.json()[0].keys())


def render_summary_report():
    """
    Render the rtichoke Summary Report using Quarto.

    Args:
        probs (list): A list of probabilities.
        reals (list): A list of real values.
        times (list): A list of absolute numbers representing timestamps.

    Example:
        probs = [0.1, 0.4, 0.8]
        reals = [0, 1, 1]
        times = [1, 3, 5]
        render_summary_report(probs, reals, times)

    This will generate a `summary_report.html` file based on the `summary_report_template.qmd`.
    """
    # Define the path to the template and output file
    template_path = "aj_estimate_summary_report.qmd"
    output_path = "summary_report.html"

    # Prepare the command to render the Quarto document
    command = [
        "quarto",
        "render",
        template_path,
        "--to",
        "html",
        "--output",
        output_path,  # ,
        # "--execute-params",
        # f"probs={probs},reals={reals},times={times}",
    ]

    # Execute the command
    subprocess.run(command, check=True)


def create_data_for_summary_report(probs, reals, times, fixed_time_horizons):
    stratified_by = ["probability_threshold", "ppcr"]
    by = 0.1

    list_data_to_adjust_polars = create_list_data_to_adjust(
        probs, reals, times, stratified_by=stratified_by, by=by
    )

    return list_data_to_adjust_polars

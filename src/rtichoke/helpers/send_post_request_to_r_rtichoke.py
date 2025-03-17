"""
A module for sending post requests to rtichoke r api
"""

import requests
import pandas as pd
from rtichoke.helpers.exported_functions import create_plotly_curve


def send_requests_to_rtichoke_r(dictionary_to_send, url_api, endpoint):
    """Send requests to rtichoke r

    Args:
        dictionary_to_send (_type_): _description_
        url_api (_type_): _description_
        endpoint (_type_): _description_

    Returns:
        _type_: _description_
    """
    rtichoke_response = requests.post(f"{url_api}{endpoint}", json=dictionary_to_send)

    return rtichoke_response


def create_rtichoke_curve(
    probs,
    reals,
    by,
    stratified_by,
    size,
    color_values=None,
    url_api="http://localhost:4242/",
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
):
    """Create rtichoke curve

    Args:
        probs (_type_): _description_
        reals (_type_): _description_
        by (_type_): _description_
        stratified_by (_type_): _description_
        size (_type_): _description_
        color_values (_type_, optional): _description_. Defaults to None.
        url_api (str, optional): _description_. Defaults to "http://localhost:4242/".
        curve (str, optional): _description_. Defaults to "roc".
        min_p_threshold (int, optional): _description_. Defaults to 0.
        max_p_threshold (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if color_values is None:
        color_values = [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#07004D",
            "#E6AB02",
            "#FE5F55",
            "#54494B",
            "#006E90",
            "#BC96E6",
            "#52050A",
            "#1F271B",
            "#BE7C4D",
            "#63768D",
            "#08A045",
            "#320A28",
            "#82FF9E",
            "#2176FF",
            "#D1603D",
            "#585123",
        ]

    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={
            "probs": probs,
            "reals": reals,
            "curve": curve,
            "by": by,
            "stratified_by": stratified_by,
            "size": size,
            "color_values": color_values,
            "min_p_threshold": min_p_threshold,
            "max_p_threshold": max_p_threshold,
        },
        url_api=url_api,
        endpoint="create_rtichoke_curve_list",
    )

    rtichoke_curve_list = rtichoke_response.json()

    if rtichoke_curve_list["size"][0] is None:
        rtichoke_curve_list["size"] = [[None]]

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["reference_data"]
    )
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_ready_for_curve"]
    )
    rtichoke_curve_list["performance_data_for_interactive_marker"] = (
        pd.DataFrame.from_dict(
            rtichoke_curve_list["performance_data_for_interactive_marker"]
        )
    )

    fig = create_plotly_curve(rtichoke_curve_list)

    return fig


def plot_rtichoke_curve(
    performance_data,
    size=None,
    color_values=None,
    url_api="http://localhost:4242/",
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
):
    """plot rtichoke curve

    Args:
        performance_data (_type_): _description_
        size (_type_, optional): _description_. Defaults to None.
        color_values (_type_, optional): _description_. Defaults to None.
        url_api (str, optional): _description_. Defaults to "http://localhost:4242/".
        curve (str, optional): _description_. Defaults to "roc".
        min_p_threshold (int, optional): _description_. Defaults to 0.
        max_p_threshold (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if color_values is None:
        color_values = [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#07004D",
            "#E6AB02",
            "#FE5F55",
            "#54494B",
            "#006E90",
            "#BC96E6",
            "#52050A",
            "#1F271B",
            "#BE7C4D",
            "#63768D",
            "#08A045",
            "#320A28",
            "#82FF9E",
            "#2176FF",
            "#D1603D",
            "#585123",
        ]
    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={
            "performance_data": performance_data.to_json(orient="records"),
            "curve": curve,
            "size": size,
            "color_values": color_values,
            "min_p_threshold": min_p_threshold,
            "max_p_threshold": max_p_threshold,
        },
        url_api=url_api,
        endpoint="plot_rtichoke_curve_list",
    )

    rtichoke_curve_list = rtichoke_response.json()

    if rtichoke_curve_list["size"][0] is None:
        rtichoke_curve_list["size"] = [[None]]

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["reference_data"]
    )
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_ready_for_curve"]
    )
    rtichoke_curve_list["performance_data_for_interactive_marker"] = (
        pd.DataFrame.from_dict(
            rtichoke_curve_list["performance_data_for_interactive_marker"]
        )
    )

    fig = create_plotly_curve(rtichoke_curve_list)

    return fig

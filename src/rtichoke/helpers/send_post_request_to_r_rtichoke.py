import requests
import pandas as pd
from rtichoke.helpers.exported_functions import create_plotly_curve


def send_requests_to_rtichoke_r(dictionary_to_send, url_api, endpoint):
    r = requests.post(f"{url_api}{endpoint}", json=dictionary_to_send)

    return r


def create_rtichoke_curve(
    probs,
    reals,
    by=0.01,
    stratified_by="probability_threshold",
    size=None,
    color_values=[
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
    ],
    url_api="http://localhost:4242/",
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
):
    r = send_requests_to_rtichoke_r(
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

    rtichoke_curve_list = r.json()

    if rtichoke_curve_list["size"][0] is None:
        rtichoke_curve_list["size"] = [[None]]

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["reference_data"]
    )
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_ready_for_curve"]
    )
    rtichoke_curve_list[
        "performance_data_for_interactive_marker"
    ] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_for_interactive_marker"]
    )

    fig = create_plotly_curve(rtichoke_curve_list)

    return fig


def plot_rtichoke_curve(
    performance_data,
    size=None,
    color_values=[
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
    ],
    url_api="http://localhost:4242/",
    curve="roc",
    min_p_threshold=0,
    max_p_threshold=1,
):
    r = send_requests_to_rtichoke_r(
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

    rtichoke_curve_list = r.json()

    if rtichoke_curve_list["size"][0] is None:
        rtichoke_curve_list["size"] = [[None]]

    rtichoke_curve_list["reference_data"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["reference_data"]
    )
    rtichoke_curve_list["performance_data_ready_for_curve"] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_ready_for_curve"]
    )
    rtichoke_curve_list[
        "performance_data_for_interactive_marker"
    ] = pd.DataFrame.from_dict(
        rtichoke_curve_list["performance_data_for_interactive_marker"]
    )

    fig = create_plotly_curve(rtichoke_curve_list)

    return fig

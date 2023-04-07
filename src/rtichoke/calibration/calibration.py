from rtichoke.helpers.send_post_request_to_r_rtichoke import (
    send_requests_to_rtichoke_r,
)
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_calibration_curve(
    probs,
    reals,
    type="discrete",
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
):
    r = send_requests_to_rtichoke_r(
        dictionary_to_send={
            "probs": probs,
            "reals": reals,
            "size": size,
            "color_values ": color_values,
        },url_api=url_api,
        endpoint="create_calibration_curve_list",
    )

    calibration_curve_list = r.json()

    calibration_curve_list["deciles_dat"] = pd.DataFrame.from_dict(
        calibration_curve_list["deciles_dat"]
    )
    calibration_curve_list["smooth_dat"] = pd.DataFrame.from_dict(
        calibration_curve_list["smooth_dat"]
    )
    calibration_curve_list["reference_data"] = pd.DataFrame.from_dict(
        calibration_curve_list["reference_data"]
    )
    calibration_curve_list["histogram_for_calibration"] = pd.DataFrame.from_dict(
        calibration_curve_list["histogram_for_calibration"]
    )

    calibration_curve = create_plotly_curve_from_calibration_curve_list(
        calibration_curve_list=calibration_curve_list, type=type
    )

    return calibration_curve


def create_plotly_curve_from_calibration_curve_list(
    calibration_curve_list,
    type="discrete",
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
):

    calibration_curve = make_subplots(
        rows=2, cols=1, shared_xaxes=True, x_title="Predicted", row_heights=[0.8, 0.2]
    )

    calibration_curve.update_layout(
        {
            "xaxis": {"showgrid": False},
            "yaxis": {"showgrid": False},
            "barmode": "overlay",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "legend": {
                "orientation": "h",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "y": 1.3,
                "bgcolor": "rgba(0, 0, 0, 0)",
            },
            "showlegend": calibration_curve_list["performance_type"][0] != "one model",
        }
    )

    calibration_curve.add_trace(
        go.Scatter(
            x=calibration_curve_list["reference_data"]["x"].values.tolist(),
            y=calibration_curve_list["reference_data"]["y"].values.tolist(),
            hovertext=calibration_curve_list["reference_data"]["text"].values.tolist(),
            name="Perfectly Calibrated",
            legendgroup="Perfectly Calibrated",
            hoverinfo="text",
            line=dict(
                width=2,
                dash="dot",
                color=calibration_curve_list["group_colors_vec"]["reference_line"][0],
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if type == "discrete":

        for reference_group in list(calibration_curve_list["group_colors_vec"].keys()):
            if any(
                calibration_curve_list["deciles_dat"]["reference_group"]
                == reference_group
            ):

                calibration_curve.add_trace(
                    go.Scatter(
                        x=calibration_curve_list["deciles_dat"]["x"][
                            calibration_curve_list["deciles_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        y=calibration_curve_list["deciles_dat"]["y"][
                            calibration_curve_list["deciles_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        hovertext=calibration_curve_list["deciles_dat"]["text"][
                            calibration_curve_list["deciles_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        name=reference_group,
                        legendgroup=reference_group,
                        hoverinfo="text",
                        mode="lines+markers",
                        marker=dict(
                            size=10,
                            color=calibration_curve_list["group_colors_vec"][
                                reference_group
                            ][0],
                        ),
                    ),
                    row=1,
                    col=1,
                )

    if type == "smooth":

        for reference_group in list(calibration_curve_list["group_colors_vec"].keys()):
            if any(
                calibration_curve_list["smooth_dat"]["reference_group"]
                == reference_group
            ):

                calibration_curve.add_trace(
                    go.Scatter(
                        x=calibration_curve_list["smooth_dat"]["x"][
                            calibration_curve_list["smooth_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        y=calibration_curve_list["smooth_dat"]["y"][
                            calibration_curve_list["smooth_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        hovertext=calibration_curve_list["smooth_dat"]["text"][
                            calibration_curve_list["smooth_dat"]["reference_group"]
                            == reference_group
                        ].values.tolist(),
                        name=reference_group,
                        legendgroup=reference_group,
                        hoverinfo="text",
                        mode="lines",
                        marker=dict(
                            size=10,
                            color=calibration_curve_list["group_colors_vec"][
                                reference_group
                            ][0],
                        ),
                    ),
                    row=1,
                    col=1,
                )

    for reference_group in list(calibration_curve_list["group_colors_vec"].keys()):
        if any(
            calibration_curve_list["histogram_for_calibration"]["reference_group"]
            == reference_group
        ):

            calibration_curve.add_trace(
                go.Bar(
                    x=calibration_curve_list["histogram_for_calibration"]["mids"][
                        calibration_curve_list["histogram_for_calibration"][
                            "reference_group"
                        ]
                        == reference_group
                    ].values.tolist(),
                    y=calibration_curve_list["histogram_for_calibration"]["counts"][
                        calibration_curve_list["histogram_for_calibration"][
                            "reference_group"
                        ]
                        == reference_group
                    ].values.tolist(),
                    hovertext=calibration_curve_list["histogram_for_calibration"][
                        "text"
                    ][
                        calibration_curve_list["histogram_for_calibration"][
                            "reference_group"
                        ]
                        == reference_group
                    ].values.tolist(),
                    name=reference_group,
                    width=0.01,
                    legendgroup=reference_group,
                    hoverinfo="text",
                    marker_color=calibration_curve_list["group_colors_vec"][
                        reference_group
                    ][0],
                    showlegend=False,
                    opacity=calibration_curve_list["histogram_opacity"][0],
                ),
                row=2,
                col=1,
            )

    print(calibration_curve_list["axes_ranges"]["xaxis"])

    calibration_curve.update_xaxes(
        zeroline=True,
        range=calibration_curve_list["axes_ranges"]["xaxis"],
        zerolinewidth=1,
        zerolinecolor="black",
        fixedrange=False,
    )
    calibration_curve.update_yaxes(
        zeroline=True,
        range=calibration_curve_list["axes_ranges"]["yaxis"],
        zerolinewidth=1,
        zerolinecolor="black",
        fixedrange=False,
        row=1,
        col=1,
    )
    calibration_curve.update_yaxes(title="Observed", row=1, col=1)

    print("size")
    print(calibration_curve_list["size"])
    print(calibration_curve_list["size"][0])

    calibration_curve.update_layout(
        width=calibration_curve_list["size"][0][0],
        height=calibration_curve_list["size"][0][0],
    )

    return calibration_curve
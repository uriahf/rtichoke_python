"""
A module for Creating Plotly Curves from rtichoke curve dictionaries
"""

import plotly.graph_objects as go

from rtichoke.helpers.plotly_helper_functions import (
    create_non_interactive_curve,
    create_interactive_marker,
    create_reference_lines_for_plotly,
)

# TODO: Fix zoom for plotly curves


def create_plotly_curve_polars(rtichoke_curve_dict):
    # non_interactive_curve_list = []
    return None


def create_plotly_curve(rtichoke_curve_dict):
    """

    Parameters
    ----------
    rtichoke_curve_dict :


    Returns
    -------

    """

    # reference_data,
    # performance_data_ready_for_curve,
    # group_colors_vec,
    # axis_ranges

    reference_data_list = []
    non_interactive_curve = []
    interactive_marker = []

    curve_layout = {
        "xaxis": {
            "showgrid": False,
        },
        "yaxis": {"showgrid": False},
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 1.3,
        },
        "height": rtichoke_curve_dict["size"][0][0],
        "width": rtichoke_curve_dict["size"][0][0],
        "updatemenus": [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "visible": False,
                        "args": [None, {"frame": {"duration": 500, "redraw": False}}],
                    }
                ],
            }
        ],
    }

    for reference_group in list(rtichoke_curve_dict["group_colors_vec"].keys()):
        if rtichoke_curve_dict["perf_dat_type"][0] not in [
            "several models",
            "several populations",
        ]:
            interactive_marker_color = "#f6e3be"
        else:
            interactive_marker_color = rtichoke_curve_dict["group_colors_vec"][
                reference_group
            ][0]
        if not rtichoke_curve_dict["reference_data"].empty:
            if any(
                rtichoke_curve_dict["reference_data"]["reference_group"]
                == reference_group
            ):
                reference_data_list.append(
                    create_reference_lines_for_plotly(
                        rtichoke_curve_dict["reference_data"][
                            rtichoke_curve_dict["reference_data"]["reference_group"]
                            == reference_group
                        ],
                        rtichoke_curve_dict["group_colors_vec"][reference_group][0],
                    )
                )
        if any(
            rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"]
            == reference_group
        ):
            non_interactive_curve.append(
                create_non_interactive_curve(
                    rtichoke_curve_dict["performance_data_ready_for_curve"][
                        rtichoke_curve_dict["performance_data_ready_for_curve"][
                            "reference_group"
                        ]
                        == reference_group
                    ],
                    rtichoke_curve_dict["group_colors_vec"][reference_group][0],
                    reference_group,
                )
            )
        if any(
            rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"]
            == reference_group
        ):
            interactive_marker.append(
                create_interactive_marker(
                    rtichoke_curve_dict["performance_data_for_interactive_marker"][
                        rtichoke_curve_dict["performance_data_for_interactive_marker"][
                            "reference_group"
                        ]
                        == reference_group
                    ],
                    interactive_marker_color,
                    0,
                    reference_group,
                )
            )

    frames = []

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": rtichoke_curve_dict["animation_slider_prefix"][0],
            "visible": True,
            "xanchor": "left",
        },
        "transition": {"duration": 300, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for k in range(
        len(
            rtichoke_curve_dict["performance_data_ready_for_curve"][
                "stratified_by"
            ].unique()
        )
    ):
        frame_data = reference_data_list + non_interactive_curve
        for reference_group in list(rtichoke_curve_dict["group_colors_vec"].keys()):
            if rtichoke_curve_dict["perf_dat_type"][0] not in [
                "several models",
                "several populations",
            ]:
                interactive_marker_color = "#f6e3be"
            else:
                interactive_marker_color = rtichoke_curve_dict["group_colors_vec"][
                    reference_group
                ][0]

            if any(
                rtichoke_curve_dict["performance_data_ready_for_curve"][
                    "reference_group"
                ]
                == reference_group
            ):
                frame_data.append(
                    create_interactive_marker(
                        rtichoke_curve_dict["performance_data_for_interactive_marker"][
                            rtichoke_curve_dict[
                                "performance_data_for_interactive_marker"
                            ]["reference_group"]
                            == reference_group
                        ],
                        interactive_marker_color,
                        k,
                        reference_group,
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(k)))
        slider_step = {
            "args": [
                [k],
                {"frame": {"duration": 300, "redraw": False}, "mode": "immediate"},
            ],
            "label": str(
                rtichoke_curve_dict["performance_data_ready_for_curve"][
                    "stratified_by"
                ].unique()[k]
            ),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    curve_layout["sliders"] = [sliders_dict]
    fig = go.Figure(
        data=reference_data_list + non_interactive_curve + interactive_marker,
        layout=curve_layout,
        frames=frames,
    )

    fig.update_layout(
        {
            "legend": {
                "orientation": "h",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "y": 1.3,
                "bgcolor": "rgba(0, 0, 0, 0)",
            },
            "showlegend": rtichoke_curve_dict["perf_dat_type"][0] != "one model",
        }
    )

    fig.update_xaxes(
        zeroline=True,
        range=rtichoke_curve_dict["axes_ranges"]["xaxis"],
        zerolinewidth=1,
        zerolinecolor="black",
        fixedrange=True,
        title={"text": rtichoke_curve_dict["axes_labels"]["xaxis"][0]},
    )
    fig.update_yaxes(
        zeroline=True,
        range=rtichoke_curve_dict["axes_ranges"]["yaxis"],
        zerolinewidth=1,
        zerolinecolor="black",
        fixedrange=True,
        title={"text": rtichoke_curve_dict["axes_labels"]["yaxis"][0]},
    )
    return fig

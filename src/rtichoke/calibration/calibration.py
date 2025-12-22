"""
A module for Calibration Curves
"""

from typing import Any, Dict, List, Optional, Union

# import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
import polars as pl
import numpy as np

# from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r


def create_calibration_curve(
    probs: Dict[str, List[float]],
    reals: Dict[str, List[int]],
    calibration_type: str = "discrete",
    size: Optional[int] = None,
    color_values: Optional[List[str]] = [
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
    url_api: str = "http://localhost:4242/",
) -> Figure:
    """Creates Calibration Curve

    Args:
        probs (Dict[str, List[float]]): _description_
        reals (Dict[str, List[int]]): _description_
        calibration_type (str, optional): _description_. Defaults to "discrete".
        size (Optional[int], optional): _description_. Defaults to None.
        color_values (List[str], optional): _description_. Defaults to None.
        url_api (_type_, optional): _description_. Defaults to "http://localhost:4242/".

    Returns:
        Figure: _description_
    """
    pass

    # rtichoke_response = send_requests_to_rtichoke_r(
    #     dictionary_to_send={
    #         "probs": probs,
    #         "reals": reals,
    #         "size": size,
    #         "color_values ": color_values,
    #     },
    #     url_api=url_api,
    #     endpoint="create_calibration_curve_list",
    # )

    # calibration_curve_list = rtichoke_response.json()

    # calibration_curve_list["deciles_dat"] = pd.DataFrame.from_dict(
    #     calibration_curve_list["deciles_dat"]
    # )
    # calibration_curve_list["smooth_dat"] = pd.DataFrame.from_dict(
    #     calibration_curve_list["smooth_dat"]
    # )
    # calibration_curve_list["reference_data"] = pd.DataFrame.from_dict(
    #     calibration_curve_list["reference_data"]
    # )
    # calibration_curve_list["histogram_for_calibration"] = pd.DataFrame.from_dict(
    #     calibration_curve_list["histogram_for_calibration"]
    # )

    # calibration_curve = create_plotly_curve_from_calibration_curve_list(
    #     calibration_curve_list=calibration_curve_list, calibration_type=calibration_type
    # )

    # return calibration_curve


def _create_plotly_curve_from_calibration_curve_list(
    calibration_curve_list: Dict[str, Any], calibration_type: str = "discrete"
) -> Figure:
    """Create plotly curve from calibration curve list

    Args:
        calibration_curve_list (Dict[str, Any]): _description_
        calibration_type (str, optional): _description_. Defaults to "discrete".

    Returns:
        Figure: _description_
    """
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
            x=calibration_curve_list["reference_data"]["x"],
            y=calibration_curve_list["reference_data"]["y"],
            # hovertext=calibration_curve_list["reference_data"]["text"],
            name="Perfectly Calibrated",
            legendgroup="Perfectly Calibrated",
            # hoverinfo="text",
            line={
                "width": 2,
                "dash": "dot",
                "color": calibration_curve_list["colors_dictionary"]["reference_line"][
                    0
                ],
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if calibration_type == "discrete":
        print(calibration_curve_list["deciles_dat"])

        for reference_group in calibration_curve_list["colors_dictionary"].keys():
            dec_sub = calibration_curve_list["deciles_dat"].filter(
                pl.col("reference_group") == reference_group
            )

            print(dec_sub)

            calibration_curve.add_trace(
                go.Scatter(
                    x=dec_sub.get_column("x").to_list(),
                    y=dec_sub.get_column("y").to_list(),
                    # hovertext=dec_sub.get_column("text").to_list(),
                    name=reference_group,
                    legendgroup=reference_group,
                    # hoverinfo="text",
                    mode="lines+markers",
                    marker={
                        "size": 10,
                        "color": calibration_curve_list["colors_dictionary"][
                            reference_group
                        ][0],
                    },
                ),
                row=1,
                col=1,
            )

    if calibration_type == "smooth":
        for reference_group in list(calibration_curve_list["colors_dictionary"].keys()):
            if any(
                calibration_curve_list["smooth_dat"]["reference_group"]
                == reference_group
            ):
                calibration_curve.add_trace(
                    go.Scatter(
                        x=calibration_curve_list["smooth_dat"]["x"][
                            calibration_curve_list["smooth_dat"]["reference_group"]
                            == reference_group
                        ],
                        y=calibration_curve_list["smooth_dat"]["y"][
                            calibration_curve_list["smooth_dat"]["reference_group"]
                            == reference_group
                        ],
                        # hovertext=calibration_curve_list["smooth_dat"]["text"][
                        #     calibration_curve_list["smooth_dat"]["reference_group"]
                        #     == reference_group
                        # ],
                        name=reference_group,
                        legendgroup=reference_group,
                        # hoverinfo="text",
                        mode="lines",
                        marker={
                            "size": 10,
                            "color": calibration_curve_list["colors_dictionary"][
                                reference_group
                            ][0],
                        },
                    ),
                    row=1,
                    col=1,
                )

    for reference_group in list(calibration_curve_list["colors_dictionary"].keys()):
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
                    ],
                    y=calibration_curve_list["histogram_for_calibration"]["counts"][
                        calibration_curve_list["histogram_for_calibration"][
                            "reference_group"
                        ]
                        == reference_group
                    ],
                    # hovertext=calibration_curve_list["histogram_for_calibration"][
                    #     "text"
                    # ][
                    #     calibration_curve_list["histogram_for_calibration"][
                    #         "reference_group"
                    #     ]
                    #     == reference_group
                    # ],
                    name=reference_group,
                    width=0.01,
                    legendgroup=reference_group,
                    # hoverinfo="text",
                    marker_color=calibration_curve_list["colors_dictionary"][
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


def _make_deciles_dat_binary(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    n_bins: int = 10,
    reference_group_name_if_array: str = "overall",
) -> pl.DataFrame:
    if isinstance(reals, dict):
        reference_groups_keys = list(reals.keys())
        y_list = [
            np.asarray(reals[reference_group]).ravel()
            for reference_group in reference_groups_keys
        ]
        lengths = np.array([len(y) for y in y_list], dtype=np.int64)
        offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths)])
        n_total = int(offsets[-1])

        frames: list[pl.DataFrame] = []
        for model, p_all in probs.items():
            p_all = np.asarray(p_all).ravel()
            if p_all.shape[0] != n_total:
                raise ValueError(
                    f"probs['{model}'] length={p_all.shape[0]} does not match "
                    f"sum of population sizes={n_total}."
                )

            for i, pop in enumerate(reference_groups_keys):
                start = int(offsets[i])
                end = int(offsets[i + 1])

                frames.append(
                    pl.DataFrame(
                        {
                            "reference_group": pop,
                            "model": model,
                            "prob": p_all[start:end].astype(float, copy=False),
                            "real": y_list[i].astype(float, copy=False),
                        }
                    )
                )

        df = pl.concat(frames, how="vertical")

    else:
        y = np.asarray(reals).ravel()
        n = y.shape[0]
        frames = []
        for model, p in probs.items():
            p = np.asarray(p).ravel()
            if p.shape[0] != n:
                raise ValueError(
                    f"probs['{model}'] length={p.shape[0]} does not match reals length={n}."
                )
            frames.append(
                pl.DataFrame(
                    {
                        "reference_group": reference_group_name_if_array,
                        "model": model,
                        "prob": p.astype(float, copy=False),
                        "real": y.astype(float, copy=False),
                    }
                )
            )

        df = pl.concat(frames, how="vertical")

    labels = [str(i) for i in range(1, n_bins + 1)]

    df = df.with_columns(
        [
            pl.col("prob").cast(pl.Float64),
            pl.col("real").cast(pl.Float64),
            pl.col("prob")
            .qcut(n_bins, labels=labels)
            .over(["reference_group", "model"])
            .alias("decile"),
        ]
    ).with_columns(pl.col("decile").cast(pl.Int32))

    deciles_data = (
        df.group_by(["reference_group", "model", "decile"])
        .agg(
            [
                pl.len().alias("n"),
                pl.mean("prob").alias("x"),
                pl.mean("real").alias("y"),
            ]
        )
        .sort(["reference_group", "model", "decile"])
    )

    return deciles_data


def _check_performance_type_by_probs_and_reals(
    probs: Dict[str, np.ndarray], reals: Union[np.ndarray, Dict[str, np.ndarray]]
) -> str:
    if isinstance(reals, dict) and len(reals) > 1:
        return "multiple populations"
    if len(probs) > 1:
        return "multiple models"
    return "one model"


def _create_calibration_curve_list(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    size: int = 600,
    color_values: List[str] = [
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
) -> Dict[str, Any]:
    deciles_data = _make_deciles_dat_binary(probs, reals)

    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)

    reference_data = _create_reference_data_for_calibration_curve()

    reference_groups = deciles_data["reference_group"].unique().to_list()

    colors_dictionary = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )

    calibration_curve_list = {
        "deciles_dat": deciles_data,
        # "smooth_dat": smooth_dat,
        "reference_data": reference_data,
        # "histogram_for_calibration": histogram_for_calibration,
        # "histogram_opacity": [0.4],
        # "axes_ranges": axes_ranges,
        "colors_dictionary": colors_dictionary,
        "performance_type": [performance_type],
        # "size": [(size_value, size_value)],
    }

    return calibration_curve_list


def _create_reference_data_for_calibration_curve() -> pl.DataFrame:
    x_ref = np.linspace(0, 1, 101)
    reference_data = pl.DataFrame({"x": x_ref, "y": x_ref})
    reference_data = reference_data.with_columns(
        pl.concat_str(
            [
                pl.lit("<b>Perfectly Calibrated</b><br>Predicted: "),
                pl.col("x").round(3).cast(pl.Utf8),
                pl.lit("<br>Observed: "),
                pl.col("y").round(3).cast(pl.Utf8),
            ]
        ).alias("text")
    )
    return reference_data


def _create_colors_dictionary_for_calibration(
    reference_groups: List[str],
    color_values: List[str],
    performance_type: str = "one model",
) -> Dict[str, List[str]]:
    if performance_type == "one model":
        colors = ["black"]
    else:
        colors = color_values[: len(reference_groups)]

    return {
        "reference_line": ["#BEBEBE"],
        **{
            group: [colors[i % len(colors)]] for i, group in enumerate(reference_groups)
        },
    }

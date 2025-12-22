"""
A module for Calibration Curves
"""

from typing import Any, Dict, List, Optional

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
from rtichoke.utility.check_performance_type import check_performance_type_by_probs_and_reals
from rtichoke.utility.create_reference_group_color_vector import create_reference_group_color_vector
import statsmodels.api as sm
import numpy as np


def _define_limits_for_calibration_plot(deciles_dat: pl.DataFrame) -> List[float]:
    if deciles_dat.height == 1:
        l, u = 0, 1
    else:
        l = max(0, min(deciles_dat["x"].min(), deciles_dat["y"].min()))
        u = max(deciles_dat["x"].max(), deciles_dat["y"].max())

    return [l - (u - l) * 0.05, u + (u - l) * 0.05]


def _create_calibration_curve_list(
    probs: Dict[str, List[float]],
    reals: Dict[str, List[int]],
    color_values: List[str],
    size: Optional[int],
) -> Dict[str, Any]:
    if not probs:
        return {}

    performance_type = check_performance_type_by_probs_and_reals(probs, reals)
    reference_groups = list(probs.keys())
    group_colors_vec = create_reference_group_color_vector(
        reference_groups, performance_type, color_values
    )

    deciles_dfs = []
    smooth_dfs = []

    if performance_type == "several populations":
        for group in reference_groups:
            deciles_df = _make_deciles_dat(probs[group], reals[group])
            deciles_df = deciles_df.with_columns(
                pl.lit(group).alias("reference_group")
            )
            deciles_dfs.append(deciles_df)

            if len(set(probs[group])) == 1:
                smooth_df = pl.DataFrame(
                    {
                        "x": [probs[group][0]],
                        "y": [np.mean(reals[group])],
                        "reference_group": [group],
                    }
                )
            else:
                lowess = sm.nonparametric.lowess(
                    reals[group], probs[group], it=0
                )
                xout = np.linspace(0, 1, 101)
                smooth_df = pl.DataFrame(
                    {
                        "x": xout,
                        "y": np.interp(xout, lowess[:, 0], lowess[:, 1]),
                        "reference_group": group,
                    }
                )
            smooth_dfs.append(smooth_df)
    else:
        real_values = next(iter(reals.values()))
        for group in reference_groups:
            deciles_df = _make_deciles_dat(probs[group], real_values)
            deciles_df = deciles_df.with_columns(
                pl.lit(group).alias("reference_group")
            )
            deciles_dfs.append(deciles_df)

            if len(set(probs[group])) == 1:
                smooth_df = pl.DataFrame(
                    {
                        "x": [probs[group][0]],
                        "y": [np.mean(real_values)],
                        "reference_group": [group],
                    }
                )
            else:
                lowess = sm.nonparametric.lowess(
                    real_values, probs[group], it=0
                )
                xout = np.linspace(0, 1, 101)
                smooth_df = pl.DataFrame(
                    {
                        "x": xout,
                        "y": np.interp(xout, lowess[:, 0], lowess[:, 1]),
                        "reference_group": group,
                    }
                )
            smooth_dfs.append(smooth_df)

    deciles_dat = pl.concat(deciles_dfs)
    smooth_dat = pl.concat(smooth_dfs).drop_nulls()

    hover_text_discrete = "Predicted: {x:.3f}<br>Observed: {y:.3f} ({sum_reals} / {total_obs})"
    hover_text_smooth = "Predicted: {x:.3f}<br>Observed: {y:.3f}"
    if performance_type != "one model":
        hover_text_discrete = "<b>{reference_group}</b><br>" + hover_text_discrete
        hover_text_smooth = "<b>{reference_group}</b><br>" + hover_text_smooth

    deciles_dat = deciles_dat.with_columns(
        pl.struct(deciles_dat.columns)
        .apply(lambda row: hover_text_discrete.format(**row))
        .alias("text")
    )
    smooth_dat = smooth_dat.with_columns(
        pl.struct(smooth_dat.columns)
        .apply(lambda row: hover_text_smooth.format(**row))
        .alias("text")
    )

    limits = _define_limits_for_calibration_plot(deciles_dat)
    axes_ranges = {"xaxis": limits, "yaxis": limits}

    x_ref = np.linspace(0, 1, 101)
    reference_data = pl.DataFrame({"x": x_ref, "y": x_ref})
    reference_data = reference_data.with_columns(
        pl.lit(
            "<b>Perfectly Calibrated</b><br>Predicted: "
            + reference_data["x"].round(3).cast(str)
            + "<br>Observed: "
            + reference_data["y"].round(3).cast(str)
        ).alias("text")
    )

    hist_dfs = []
    for group, prob_values in probs.items():
        counts, mids = np.histogram(prob_values, bins=np.arange(0, 1.01, 0.01))
        hist_df = pl.DataFrame(
            {"mids": mids[:-1] + 0.005, "counts": counts, "reference_group": group}
        )
        hist_df = hist_df.with_columns(
            (
                pl.col("counts").cast(str)
                + " observations in ["
                + (pl.col("mids") - 0.005).round(3).cast(str)
                + ", "
                + (pl.col("mids") + 0.005).round(3).cast(str)
                + "]"
            ).alias("text")
        )
        hist_dfs.append(hist_df)

    histogram_for_calibration = pl.concat(hist_dfs)

    return {
        "performance_type": [performance_type],
        "size": [[size]],
        "deciles_dat": deciles_dat,
        "smooth_dat": smooth_dat,
        "group_colors_vec": group_colors_vec,
        "axes_ranges": axes_ranges,
        "reference_data": reference_data,
        "histogram_for_calibration": histogram_for_calibration,
        "histogram_opacity": [1 / len(probs)],
    }


def _make_deciles_dat(probs: List[float], reals: List[int]) -> pl.DataFrame:
    """
    Creates a DataFrame with deciles for the calibration curve.
    """
    if len(set(probs)) == 1:
        return pl.DataFrame(
            {
                "quintile": [1],
                "x": [probs[0]],
                "y": [sum(reals) / len(reals)],
                "sum_reals": [sum(reals)],
                "total_obs": [len(reals)],
            }
        )
    else:
        df = pl.DataFrame({"probs": probs, "reals": reals})
        # Replicating dplyr's ntile(10)
        df = df.with_columns(
            (
                (pl.col("probs").rank("ordinal", seed=1) * 10) / (pl.count() + 1)
            ).floor().cast(pl.Int64).alias("quintile")
        )

        quintile_df = (
            df.group_by("quintile")
            .agg(
                (pl.col("reals").sum() / pl.count()).alias("y"),
                pl.col("probs").mean().alias("x"),
                pl.col("reals").sum().alias("sum_reals"),
                pl.count().alias("total_obs"),
            )
            .sort("quintile")
        )
        return quintile_df


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
) -> Figure:
    """Creates Calibration Curve

    Args:
        probs (Dict[str, List[float]]): A dictionary where keys are model names and values are lists of predicted probabilities.
        reals (Dict[str, List[int]]): A dictionary where keys are population names and values are lists of actual outcomes (0 or 1).
        calibration_type (str, optional): The type of calibration curve to create, either "discrete" or "smooth". Defaults to "discrete".
        size (Optional[int], optional): The size of the plot. Defaults to None.
        color_values (List[str], optional): A list of hex color codes for the plot. Defaults to a predefined list.

    Returns:
        Figure: A Plotly Figure object representing the calibration curve.
    """
    calibration_curve_list = _create_calibration_curve_list(
        probs=probs, reals=reals, color_values=color_values, size=size
    )

    calibration_curve_list["deciles_dat"] = calibration_curve_list[
        "deciles_dat"
    ].to_pandas()
    calibration_curve_list["smooth_dat"] = calibration_curve_list[
        "smooth_dat"
    ].to_pandas()
    calibration_curve_list["reference_data"] = calibration_curve_list[
        "reference_data"
    ].to_pandas()
    calibration_curve_list["histogram_for_calibration"] = calibration_curve_list[
        "histogram_for_calibration"
    ].to_pandas()

    calibration_curve = create_plotly_curve_from_calibration_curve_list(
        calibration_curve_list=calibration_curve_list,
        calibration_type=calibration_type,
    )

    return calibration_curve


def create_plotly_curve_from_calibration_curve_list(
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
            x=calibration_curve_list["reference_data"]["x"].values.tolist(),
            y=calibration_curve_list["reference_data"]["y"].values.tolist(),
            hovertext=calibration_curve_list["reference_data"]["text"].values.tolist(),
            name="Perfectly Calibrated",
            legendgroup="Perfectly Calibrated",
            hoverinfo="text",
            line={
                "width": 2,
                "dash": "dot",
                "color": calibration_curve_list["group_colors_vec"]["reference_line"][
                    0
                ],
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if calibration_type == "discrete":
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
                        marker={
                            "size": 10,
                            "color": calibration_curve_list["group_colors_vec"][
                                reference_group
                            ][0],
                        },
                    ),
                    row=1,
                    col=1,
                )

    if calibration_type == "smooth":
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
                        marker={
                            "size": 10,
                            "color": calibration_curve_list["group_colors_vec"][
                                reference_group
                            ][0],
                        },
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

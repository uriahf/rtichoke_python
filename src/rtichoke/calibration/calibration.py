"""
A module for Calibration Curves
"""

from typing import Any, Dict, List, Union, cast

# import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
import polars as pl
import numpy as np

# from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r


def create_calibration_curve(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    calibration_type: str = "discrete",
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

    calibration_curve_list = _create_calibration_curve_list(
        probs, reals, size=size, color_values=color_values
    )

    calibration_curve = _create_plotly_curve_from_calibration_curve_list(
        calibration_curve_list, calibration_type=calibration_type
    )

    return calibration_curve


def create_calibration_curve_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: List[float],
    heuristics_sets: List[Dict[str, str]],
    calibration_type: str = "discrete",
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
) -> Figure:
    """Creates a time-dependent Calibration Curve with a slider for different time horizons."""

    calibration_curve_list_times = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        size=size,
        color_values=color_values,
    )

    fig = _create_plotly_curve_from_calibration_curve_list_times(
        calibration_curve_list_times, calibration_type=calibration_type
    )

    return fig


def _create_plotly_curve_from_calibration_curve_list_times(
    calibration_curve_list: Dict[str, Any], calibration_type: str = "discrete"
) -> Figure:
    """
    Creates a plotly figure for time-dependent calibration curves.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, x_title="Predicted", row_heights=[0.8, 0.2]
    )

    initial_horizon = calibration_curve_list["fixed_time_horizons"][0]

    # Add traces for each horizon, initially visible only for the first horizon
    for horizon in calibration_curve_list["fixed_time_horizons"]:
        visible = horizon == initial_horizon

        # Reference Line
        fig.add_trace(
            go.Scatter(
                x=calibration_curve_list["reference_data"]["x"],
                y=calibration_curve_list["reference_data"]["y"],
                hovertext=calibration_curve_list["reference_data"]["text"],
                name="Perfectly Calibrated",
                legendgroup="Perfectly Calibrated",
                hoverinfo="text",
                line={"width": 2, "dash": "dot", "color": "#BEBEBE"},
                showlegend=False,
                visible=visible,
            ),
            row=1,
            col=1,
        )

        for group in calibration_curve_list["reference_group_keys"]:
            color = calibration_curve_list["colors_dictionary"][group][0]

            # Calibration curve (discrete or smooth)
            if calibration_type == "discrete":
                data_subset = calibration_curve_list["deciles_dat"].filter(
                    (pl.col("reference_group") == group)
                    & (pl.col("fixed_time_horizon") == horizon)
                )
                mode = "lines+markers"
            else:  # smooth
                data_subset = calibration_curve_list["smooth_dat"].filter(
                    (pl.col("reference_group") == group)
                    & (pl.col("fixed_time_horizon") == horizon)
                )
                mode = "lines+markers" if data_subset.height == 1 else "lines"

            fig.add_trace(
                go.Scatter(
                    x=data_subset["x"],
                    y=data_subset["y"],
                    hovertext=data_subset["text"],
                    name=group,
                    legendgroup=group,
                    hoverinfo="text",
                    mode=mode,
                    marker={"size": 10, "color": color},
                    visible=visible,
                ),
                row=1,
                col=1,
            )

            # Histogram
            hist_subset = calibration_curve_list["histogram_for_calibration"].filter(
                (pl.col("reference_group") == group)
                & (pl.col("fixed_time_horizon") == horizon)
            )
            fig.add_trace(
                go.Bar(
                    x=hist_subset["mids"],
                    y=hist_subset["counts"],
                    hovertext=hist_subset["text"],
                    name=group,
                    width=0.01,
                    legendgroup=group,
                    hoverinfo="text",
                    marker_color=color,
                    showlegend=False,
                    opacity=0.4,
                    visible=visible,
                ),
                row=2,
                col=1,
            )

    # Create slider
    steps = []
    num_traces_per_horizon = 1 + 2 * len(calibration_curve_list["reference_group_keys"])

    for i, horizon in enumerate(calibration_curve_list["fixed_time_horizons"]):
        visibility = [False] * (
            num_traces_per_horizon * len(calibration_curve_list["fixed_time_horizons"])
        )
        for j in range(num_traces_per_horizon):
            visibility[i * num_traces_per_horizon + j] = True
        step = dict(
            method="restyle",
            args=[{"visible": visibility}],
            label=str(horizon),
        )
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Time Horizon: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    # Layout
    fig.update_layout(
        sliders=sliders,
        xaxis={
            "showgrid": False,
            "range": calibration_curve_list["axes_ranges"]["xaxis"],
        },
        yaxis={
            "showgrid": False,
            "range": calibration_curve_list["axes_ranges"]["yaxis"],
            "title": "Observed",
        },
        barmode="overlay",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend={
            "orientation": "h",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 1.3,
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
        showlegend=calibration_curve_list["performance_type"][0] != "one model",
        width=calibration_curve_list["size"][0][0],
        height=calibration_curve_list["size"][0][0],
    )

    return fig


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
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
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
            hovertext=calibration_curve_list["reference_data"]["text"],
            name="Perfectly Calibrated",
            legendgroup="Perfectly Calibrated",
            hoverinfo="text",
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
        reference_groups = [
            k
            for k in calibration_curve_list["colors_dictionary"].keys()
            if k != "reference_line"
        ]
        for reference_group in reference_groups:
            dec_sub = calibration_curve_list["deciles_dat"].filter(
                pl.col("reference_group") == reference_group
            )

            calibration_curve.add_trace(
                go.Scatter(
                    x=dec_sub.get_column("x").to_list(),
                    y=dec_sub.get_column("y").to_list(),
                    hovertext=dec_sub.get_column("text").to_list(),
                    name=reference_group,
                    legendgroup=reference_group,
                    hoverinfo="text",
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

        hist = calibration_curve_list["histogram_for_calibration"]

        for reference_group in reference_groups:
            hist_sub = hist.filter(pl.col("reference_group") == reference_group)
            if hist_sub.height == 0:
                continue

            calibration_curve.add_trace(
                go.Bar(
                    x=hist_sub.get_column("mids").to_list(),
                    y=hist_sub.get_column("counts").to_list(),
                    hovertext=hist_sub.get_column("text").to_list(),
                    name=reference_group,
                    width=0.01,
                    legendgroup=reference_group,
                    hoverinfo="text",
                    marker_color=calibration_curve_list["colors_dictionary"][
                        reference_group
                    ][0],
                    showlegend=False,
                    opacity=0.4,
                ),
                row=2,
                col=1,
            )

    if calibration_type == "smooth":
        smooth_dat = calibration_curve_list["smooth_dat"]
        reference_groups = [
            k
            for k in calibration_curve_list["colors_dictionary"].keys()
            if k != "reference_line"
        ]

        for reference_group in reference_groups:
            smooth_sub = smooth_dat.filter(pl.col("reference_group") == reference_group)
            if smooth_sub.height == 0:
                continue

            mode = "lines+markers" if smooth_sub.height == 1 else "lines"

            calibration_curve.add_trace(
                go.Scatter(
                    x=smooth_sub.get_column("x").to_list(),
                    y=smooth_sub.get_column("y").to_list(),
                    hovertext=smooth_sub.get_column("text").to_list(),
                    name=reference_group,
                    legendgroup=reference_group,
                    hoverinfo="text",
                    mode=mode,
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

        hist = calibration_curve_list["histogram_for_calibration"]

        for reference_group in reference_groups:
            hist_sub = hist.filter(pl.col("reference_group") == reference_group)
            if hist_sub.height == 0:
                continue

            calibration_curve.add_trace(
                go.Bar(
                    x=hist_sub.get_column("mids").to_list(),
                    y=hist_sub.get_column("counts").to_list(),
                    hovertext=hist_sub.get_column("text").to_list(),
                    name=reference_group,
                    width=0.01,
                    legendgroup=reference_group,
                    hoverinfo="text",
                    marker_color=calibration_curve_list["colors_dictionary"][
                        reference_group
                    ][0],
                    showlegend=False,
                    opacity=0.4,
                ),
                row=2,
                col=1,
            )

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

    calibration_curve.update_layout(
        width=calibration_curve_list["size"][0][0],
        height=calibration_curve_list["size"][0][0],
    )

    return calibration_curve


def _make_deciles_dat_binary(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    n_bins: int = 10,
) -> pl.DataFrame:
    if isinstance(reals, dict):
        reference_groups_keys = list(reals.keys())
        y_list = [
            np.asarray(reals[str(reference_group)]).ravel()
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
                        "reference_group": model,
                        "model": model,
                        "prob": p.astype(float, copy=False),
                        "real": y.astype(float, copy=False),
                    }
                )
            )

        df = pl.concat(frames, how="vertical")

    df = df.with_columns(
        [
            pl.col("prob").cast(pl.Float64),
            pl.col("real").cast(pl.Float64),
            (
                (pl.col("prob").rank("ordinal").over(["reference_group", "model"]) - 1)
                * n_bins
                // pl.len().over(["reference_group", "model"])
                + 1
            ).alias("decile"),
        ]
    )

    deciles_data = (
        df.group_by(["reference_group", "model", "decile"])
        .agg(
            [
                pl.len().alias("n"),
                pl.mean("prob").alias("x"),
                pl.mean("real").alias("y"),
                pl.sum("real").alias("n_reals"),
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
    smooth_dat = _calculate_smooth_curve(probs, reals, performance_type)

    deciles_data, smooth_dat = _add_hover_text_to_calibration_data(
        deciles_data, smooth_dat, performance_type
    )

    reference_data = _create_reference_data_for_calibration_curve()

    reference_groups = list(probs.keys())

    colors_dictionary = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )

    histogram_for_calibration = _create_histogram_for_calibration(probs)

    limits = _define_limits_for_calibration_plot(deciles_data)
    axes_ranges = {"xaxis": limits, "yaxis": limits}

    smooth_dat = _calculate_smooth_curve(probs, reals, performance_type)

    calibration_curve_list = {
        "deciles_dat": deciles_data,
        "smooth_dat": smooth_dat,
        "reference_data": reference_data,
        "histogram_for_calibration": histogram_for_calibration,
        # "histogram_opacity": [0.4],
        "axes_ranges": axes_ranges,
        "colors_dictionary": colors_dictionary,
        "performance_type": [performance_type],
        "size": [(size, size)],
    }

    return calibration_curve_list


def _create_reference_data_for_calibration_curve() -> pl.DataFrame:
    x_ref = np.linspace(0, 1, 101)
    reference_data = pl.DataFrame({"x": x_ref, "y": x_ref})
    reference_data = reference_data.with_columns(
        pl.concat_str(
            [
                pl.lit("<b>Perfectly Calibrated</b><br>Predicted: "),
                pl.col("x").map_elements(lambda x: f"{x:.3f}", return_dtype=pl.Utf8),
                pl.lit("<br>Observed: "),
                pl.col("y").map_elements(lambda y: f"{y:.3f}", return_dtype=pl.Utf8),
            ]
        ).alias("text")
    )
    return reference_data


def _calculate_smooth_curve(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    performance_type: str,
) -> pl.DataFrame:
    """
    Calculate the smoothed calibration curve using lowess.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smooth_frames = []

    # Helper function to process a single probability and real array
    def process_single_array(p, r, group_name):
        if len(np.unique(p)) == 1:
            return pl.DataFrame(
                {
                    "x": [np.unique(p)[0]],
                    "y": [np.mean(r)],
                    "reference_group": [group_name],
                }
            )
        else:
            # lowess returns a 2D array where the first column is x and the second is y
            smoothed = lowess(r, p, it=0)
            xout = np.linspace(0, 1, 101)
            yout = np.interp(xout, smoothed[:, 0], smoothed[:, 1])
            return pl.DataFrame(
                {"x": xout, "y": yout, "reference_group": [group_name] * len(xout)}
            )

    if isinstance(reals, dict):
        for model_name, prob_array in probs.items():
            # This logic assumes that for multiple populations, one model's probs are evaluated against multiple real outcomes.
            # This might need adjustment based on the exact structure for multiple models and populations.
            if len(probs) == 1 and len(reals) > 1:  # One model, multiple populations
                for pop_name, real_array in reals.items():
                    frame = process_single_array(prob_array, real_array, pop_name)
                    smooth_frames.append(frame)
            else:  # Multiple models, potentially multiple populations
                for group_name in reals.keys():
                    if group_name in probs:
                        frame = process_single_array(
                            probs[str(group_name)],
                            reals[str(group_name)],
                            str(group_name),
                        )
                        smooth_frames.append(frame)

    else:  # reals is a single numpy array
        for group_name, prob_array in probs.items():
            frame = process_single_array(prob_array, reals, group_name)
            smooth_frames.append(frame)

    if not smooth_frames:
        return pl.DataFrame(
            schema={
                "x": pl.Float64,
                "y": pl.Float64,
                "reference_group": pl.Utf8,
                "text": pl.Utf8,
            }
        )

    smooth_dat = pl.concat(smooth_frames)

    if performance_type != "one model":
        smooth_dat = smooth_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("<b>"),
                    pl.col("reference_group"),
                    pl.lit("</b><br>Predicted: "),
                    pl.col("x").map_elements(
                        lambda x: f"{x:.3f}", return_dtype=pl.Utf8
                    ),
                    pl.lit("<br>Observed: "),
                    pl.col("y").map_elements(
                        lambda y: f"{y:.3f}", return_dtype=pl.Utf8
                    ),
                ]
            ).alias("text")
        )
    else:
        smooth_dat = smooth_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("Predicted: "),
                    pl.col("x").map_elements(
                        lambda x: f"{x:.3f}", return_dtype=pl.Utf8
                    ),
                    pl.lit("<br>Observed: "),
                    pl.col("y").map_elements(
                        lambda y: f"{y:.3f}", return_dtype=pl.Utf8
                    ),
                ]
            ).alias("text")
        )
    return smooth_dat


def _add_hover_text_to_calibration_data(
    deciles_dat: pl.DataFrame,
    smooth_dat: pl.DataFrame,
    performance_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Adds hover text to the deciles and smooth dataframes."""
    if performance_type != "one model":
        deciles_dat = deciles_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("<b>"),
                    pl.col("reference_group"),
                    pl.lit("</b><br>Predicted: "),
                    pl.col("x").round(3),
                    pl.lit("<br>Observed: "),
                    pl.col("y").round(3),
                    pl.lit(" ( "),
                    pl.col("n_reals"),
                    pl.lit(" / "),
                    pl.col("n"),
                    pl.lit(" )"),
                ]
            ).alias("text")
        )
        smooth_dat = smooth_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("<b>"),
                    pl.col("reference_group"),
                    pl.lit("</b><br>Predicted: "),
                    pl.col("x").round(3),
                    pl.lit("<br>Observed: "),
                    pl.col("y").round(3),
                ]
            ).alias("text")
        )
    else:
        deciles_dat = deciles_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("Predicted: "),
                    pl.col("x").round(3),
                    pl.lit("<br>Observed: "),
                    pl.col("y").round(3),
                    pl.lit(" ( "),
                    pl.col("n_reals"),
                    pl.lit(" / "),
                    pl.col("n"),
                    pl.lit(" )"),
                ]
            ).alias("text")
        )
        smooth_dat = smooth_dat.with_columns(
            pl.concat_str(
                [
                    pl.lit("Predicted: "),
                    pl.col("x").round(3),
                    pl.lit("<br>Observed: "),
                    pl.col("y").round(3),
                ]
            ).alias("text")
        )
    return deciles_dat, smooth_dat


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


def _create_histogram_for_calibration(probs: Dict[str, np.ndarray]) -> pl.DataFrame:
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

    return histogram_for_calibration


def _define_limits_for_calibration_plot(deciles_dat: pl.DataFrame) -> List[float]:
    if deciles_dat.height == 1:
        lower_bound, upper_bound = 0.0, 1.0
    else:
        lower_bound = float(
            max(
                0,
                min(
                    cast(float, deciles_dat["x"].min()),
                    cast(float, deciles_dat["y"].min()),
                ),
            )
        )
        upper_bound = float(
            max(
                cast(float, deciles_dat["x"].max()),
                cast(float, deciles_dat["y"].max()),
            )
        )

    return [
        lower_bound - (upper_bound - lower_bound) * 0.05,
        upper_bound + (upper_bound - lower_bound) * 0.05,
    ]


def _build_initial_df_for_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
) -> pl.DataFrame:
    """Builds the initial DataFrame for time-dependent calibration curves."""

    # Convert all inputs to dictionaries of arrays to unify processing
    if not isinstance(reals, dict):
        reals = {"single_population": np.asarray(reals)}
    if not isinstance(times, dict):
        times = {"single_population": np.asarray(times)}

    # Verify matching keys and lengths
    if reals.keys() != times.keys():
        raise ValueError("Keys in reals and times dictionaries do not match.")
    for key in reals:
        if len(reals[key]) != len(times[key]):
            raise ValueError(
                f"Length mismatch for population '{key}' in reals and times."
            )

    # Create a base DataFrame with population data
    population_frames = []
    for key in reals:
        population_frames.append(
            pl.DataFrame(
                {
                    "reference_group": key,
                    "real": reals[key],
                    "time": times[key],
                }
            )
        )
    base_df = pl.concat(population_frames)

    # Prepare model predictions
    # Single model case
    if len(probs) == 1:
        model_name, prob_array = next(iter(probs.items()))
        if len(prob_array) != base_df.height:
            raise ValueError(
                f"Length of probabilities for model '{model_name}' does not match total number of observations."
            )
        return base_df.with_columns(
            pl.Series("prob", prob_array), pl.lit(model_name).alias("model")
        )

    # Multiple models
    else:
        # One model per population (keys must match)
        if probs.keys() == reals.keys():
            prob_frames = []
            for model_name, prob_array in probs.items():
                pop_df = base_df.filter(pl.col("reference_group") == model_name)
                if len(prob_array) != pop_df.height:
                    raise ValueError(
                        f"Length of probabilities for model '{model_name}' does not match population size."
                    )
                prob_frames.append(
                    pop_df.with_columns(
                        pl.Series("prob", prob_array), pl.lit(model_name).alias("model")
                    )
                )
            return pl.concat(prob_frames)
        # Multiple models on a single population
        elif len(reals) == 1:
            final_frames = []
            for model_name, prob_array in probs.items():
                if len(prob_array) != base_df.height:
                    raise ValueError(
                        f"Length of probabilities for model '{model_name}' does not match population size."
                    )
                final_frames.append(
                    base_df.with_columns(
                        pl.Series("prob", prob_array),
                        pl.lit(model_name).alias(
                            "reference_group"
                        ),  # Overwrite reference_group with model name
                    )
                )
            return pl.concat(final_frames)

    raise ValueError("Unsupported combination of probs, reals, and times structures.")


def _apply_heuristics_and_censoring(
    df: pl.DataFrame,
    horizon: float,
    censoring_heuristic: str,
    competing_heuristic: str,
) -> pl.DataFrame:
    """
    Applies censoring and competing risk heuristics to the data for a given time horizon.
    """
    # Administrative censoring: outcomes after horizon are negative
    df_adj = df.with_columns(
        pl.when(pl.col("time") > horizon)
        .then(0)
        .otherwise(pl.col("real"))
        .alias("real")
    )

    # Heuristics for events before or at horizon
    if censoring_heuristic == "excluded":
        df_adj = df_adj.filter(~((pl.col("real") == 0) & (pl.col("time") <= horizon)))

    if competing_heuristic == "excluded":
        df_adj = df_adj.filter(~((pl.col("real") == 2) & (pl.col("time") <= horizon)))
    elif competing_heuristic == "adjusted_as_negative":
        df_adj = df_adj.with_columns(
            pl.when((pl.col("real") == 2) & (pl.col("time") <= horizon))
            .then(0)
            .otherwise(pl.col("real"))
            .alias("real")
        )
    elif competing_heuristic == "adjusted_as_composite":
        df_adj = df_adj.with_columns(
            pl.when((pl.col("real") == 2) & (pl.col("time") <= horizon))
            .then(1)
            .otherwise(pl.col("real"))
            .alias("real")
        )

    return df_adj


def _create_calibration_curve_list_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: List[float],
    heuristics_sets: List[Dict[str, str]],
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
    """
    Creates the data structures needed for a time-dependent calibration curve plot.
    """
    # Part 1: Prepare initial dataframe from inputs
    initial_df = _build_initial_df_for_times(probs, reals, times)

    # Part 2: Iterate and generate calibration data for each horizon/heuristic
    all_deciles = []
    all_smooth = []
    all_histograms = []

    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)

    for horizon in fixed_time_horizons:
        for heuristics in heuristics_sets:
            censoring_heuristic = heuristics["censoring_heuristic"]
            competing_heuristic = heuristics["competing_heuristic"]

            if (
                censoring_heuristic == "adjusted"
                or competing_heuristic == "adjusted_as_censored"
            ):
                continue

            df_adj = _apply_heuristics_and_censoring(
                initial_df, horizon, censoring_heuristic, competing_heuristic
            )

            if df_adj.height == 0:
                continue

            # Re-create probs and reals dicts for helpers
            probs_adj = {
                group[0]: group_df["prob"].to_numpy()
                for group, group_df in df_adj.group_by("reference_group")
            }
            reals_adj = {
                group[0]: group_df["real"].to_numpy()
                for group, group_df in df_adj.group_by("reference_group")
            }
            # If single population initially, reals_adj should be an array
            if not isinstance(reals, dict) and len(probs) == 1:
                reals_adj = next(iter(reals_adj.values()))

            # Deciles
            deciles_data = _make_deciles_dat_binary(probs_adj, reals_adj)
            all_deciles.append(
                deciles_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
            )

            # Smooth curve
            smooth_data = _calculate_smooth_curve(
                probs_adj, reals_adj, performance_type
            )
            all_smooth.append(
                smooth_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
            )

            # Histogram
            hist_data = _create_histogram_for_calibration(probs_adj)
            all_histograms.append(
                hist_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
            )

    # Part 3: Combine results and create final dictionary
    if not all_deciles:
        raise ValueError(
            "No data remaining after applying heuristics and time horizons."
        )
    deciles_dat_final = pl.concat(all_deciles)
    smooth_dat_final = pl.concat(all_smooth)
    histogram_final = pl.concat(all_histograms)

    # Add hover text
    deciles_dat_final, smooth_dat_final = _add_hover_text_to_calibration_data(
        deciles_dat_final, smooth_dat_final, performance_type
    )

    reference_data = _create_reference_data_for_calibration_curve()
    reference_groups = list(probs.keys())
    colors_dictionary = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )
    limits = _define_limits_for_calibration_plot(deciles_dat_final)
    axes_ranges = {"xaxis": limits, "yaxis": limits}

    calibration_curve_list = {
        "deciles_dat": deciles_dat_final,
        "smooth_dat": smooth_dat_final,
        "reference_data": reference_data,
        "histogram_for_calibration": histogram_final,
        "axes_ranges": axes_ranges,
        "colors_dictionary": colors_dictionary,
        "performance_type": [performance_type],
        "size": [(size, size)],
        "fixed_time_horizons": fixed_time_horizons,
        "reference_group_keys": reference_groups,
    }

    return calibration_curve_list

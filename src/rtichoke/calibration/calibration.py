"""
A module for Calibration Curves
"""

from typing import Any, Dict, List, Union, cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
import polars as pl
import numpy as np
from polarstate import predict_aj_estimates, prepare_event_table
from smoothstate import smooth_state_lowess
from ._secondary_cox import calculate_secondary_cox_smooth


def _validate_n_bins(n_bins: Any) -> int:
    """Validates that n_bins is a positive integer >= 1."""
    if isinstance(n_bins, bool):
        raise ValueError("n_bins must be a positive integer >= 1.")
    if isinstance(n_bins, (int, np.integer)):
        if n_bins < 1:
            raise ValueError("n_bins must be a positive integer >= 1.")
        return int(n_bins)
    raise ValueError("n_bins must be a positive integer >= 1.")


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
    *,
    n_bins: int = 10,
) -> Figure:
    """Creates a Calibration Curve.

    This function generates a calibration curve, which evaluates how well
    the predicted probabilities from one or more models align with the
    observed binary outcomes. It can plot either discrete binned calibration
    (10 bins by default) or a smoothed calibration curve.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names to 1-D numpy arrays of
        predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true binary labels (0 or 1). Can be a single array or a dictionary
        mapping names to label arrays.
    calibration_type : str, optional
        The type of calibration curve to plot. Options are ``"discrete"`` (binned)
        or ``"smooth"`` (smoothed lowess). Defaults to ``"discrete"``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.
    color_values : List[str], optional
        A list of hex color strings for the plot lines/markers.
    n_bins : int, optional
        Number of bins for discrete calibration curves. Defaults to 10.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the calibration curve.
    """
    n_bins = _validate_n_bins(n_bins)
    calibration_curve_list = _create_calibration_curve_list(
        probs,
        reals,
        calibration_type=calibration_type,
        size=size,
        color_values=color_values,
        n_bins=n_bins,
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
    smooth_method: str = "local_aj",
    bandwidth: Union[float, None] = None,
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
    *,
    n_bins: int = 10,
) -> Figure:
    """Create a time-dependent calibration curve across fixed horizons.

    This function generates time-dependent calibration curves evaluating predicted
    probabilities against observed outcomes over specified prediction horizons.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names to 1-D numpy arrays of
        predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True outcome indicators (0 for censored, 1 for event of interest, 2 for
        competing risk).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        Follow-up times corresponding to `reals`.
    fixed_time_horizons : List[float]
        List of prediction horizons (times) at which to evaluate calibration.
    heuristics_sets : List[Dict[str, str]]
        List of heuristic dictionaries defining censoring and competing risk
        adjustments.
    calibration_type : str, optional
        Type of calibration plot, either ``"discrete"`` (binned) or ``"smooth"``.
        Defaults to ``"discrete"``.
    smooth_method : str, optional
        Smoothing method when `calibration_type="smooth"`. Supported options are
        ``"local_aj"`` (Gerds' local Aalen-Johansen/KM neighborhood estimation),
        ``"secondary_cox"`` (Austin, Harrell & McLernon secondary Cox regression with 3-knot restricted cubic splines on complementary log-log predictions),
        or ``"pseudo_values"`` (jackknife pseudo-values lowess). Defaults to ``"local_aj"``.
    bandwidth : Union[float, None], optional
        Bandwidth fraction for ``"local_aj"`` neighborhood smoothing. Defaults to None.
    size : int, optional
        Width and height of the Plotly figure in pixels. Defaults to 600.
    color_values : List[str], optional
        List of hex color strings for traces.
    n_bins : int, optional
        Number of bins for discrete calibration curves. Defaults to 10.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the time-dependent calibration curve.

    Raises
    ------
    ValueError
        If a heuristic set requests `competing_heuristic='adjusted_as_censored'`.
    """
    n_bins = _validate_n_bins(n_bins)

    unsupported_competing_as_censored = any(
        heuristics.get("competing_heuristic") == "adjusted_as_censored"
        for heuristics in heuristics_sets
    )

    if unsupported_competing_as_censored:
        raise ValueError(
            "Unsupported calibration heuristics: "
            "create_calibration_curve_times() does not support "
            "competing_heuristic='adjusted_as_censored'."
        )

    calibration_curve_list_times = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        calibration_type=calibration_type,
        smooth_method=smooth_method,
        bandwidth=bandwidth,
        size=size,
        color_values=color_values,
        n_bins=n_bins,
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
                data_subset = calibration_curve_list["calibration_bins_dat"].filter(
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
    """Create plotly curve from calibration curve list"""
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
            bin_sub = calibration_curve_list["calibration_bins_dat"].filter(
                pl.col("reference_group") == reference_group
            )

            calibration_curve.add_trace(
                go.Scatter(
                    x=bin_sub.get_column("x").to_list(),
                    y=bin_sub.get_column("y").to_list(),
                    hovertext=bin_sub.get_column("text").to_list(),
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


def _make_calibration_bins_dat_binary(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    n_bins: int = 10,
) -> pl.DataFrame:
    n_bins = _validate_n_bins(n_bins)

    if isinstance(reals, dict):
        frames: list[pl.DataFrame] = []

        if probs.keys() == reals.keys():
            for population in reals:
                p = np.asarray(probs[population]).ravel()
                y = np.asarray(reals[population]).ravel()
                if p.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"Length mismatch for population '{population}': "
                        f"probs has length {p.shape[0]} but reals has length {y.shape[0]}."
                    )
                frames.append(
                    pl.DataFrame(
                        {
                            "reference_group": population,
                            "model": population,
                            "prob": p.astype(float, copy=False),
                            "real": y.astype(float, copy=False),
                        }
                    )
                )
        elif len(probs) == 1:
            reference_groups_keys = list(reals.keys())
            y_list = [
                np.asarray(reals[str(reference_group)]).ravel()
                for reference_group in reference_groups_keys
            ]
            lengths = np.array([len(y) for y in y_list], dtype=np.int64)
            offsets = np.concatenate(
                [np.array([0], dtype=np.int64), np.cumsum(lengths)]
            )
            n_total = int(offsets[-1])
            model, p_all = next(iter(probs.items()))
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
        else:
            raise ValueError(
                "When probs and reals are dictionaries, their population keys must match."
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

    # Apply dplyr::ntile() bucket allocation per (reference_group, model)
    group_cols = ["reference_group", "model"]
    partitioned_frames = []

    for key, sub_df in df.group_by(group_cols, maintain_order=True):
        reference_group, model = key
        N = sub_df.height
        p_vals = sub_df["prob"].to_numpy()

        if N == 0:
            sub_df_with_bin = sub_df.with_columns(
                pl.lit(1, dtype=pl.Int64).alias("bin")
            )
        elif len(np.unique(p_vals)) == 1:
            # All predictions identical -> one aggregate calibration bin = 1
            sub_df_with_bin = sub_df.with_columns(
                pl.lit(1, dtype=pl.Int64).alias("bin")
            )
        elif n_bins > N:
            # B > N -> occupied bin labels are 1..N, one observation each in stable order
            ord_ranks = sub_df["prob"].rank("ordinal").to_numpy().astype(int)
            sub_df_with_bin = sub_df.with_columns(
                pl.Series("bin", ord_ranks, dtype=pl.Int64)
            )
        else:
            # B <= N -> dplyr::ntile() semantics:
            # stable ordinal rank 1..N
            ord_ranks = sub_df["prob"].rank("ordinal").to_numpy().astype(int)
            q = N // n_bins
            rem = N % n_bins
            # first rem bins have size q + 1, remaining B - rem bins have size q
            # build lookup array bin_for_rank mapping rank (1-indexed) -> bin (1-indexed)
            bin_for_rank = np.empty(N + 1, dtype=int)
            curr_rank = 1
            for b in range(1, n_bins + 1):
                bin_size = q + 1 if b <= rem else q
                bin_for_rank[curr_rank : curr_rank + bin_size] = b
                curr_rank += bin_size

            assigned_bins = bin_for_rank[ord_ranks]
            sub_df_with_bin = sub_df.with_columns(
                pl.Series("bin", assigned_bins, dtype=pl.Int64)
            )

        partitioned_frames.append(sub_df_with_bin)

    df = pl.concat(partitioned_frames)

    calibration_bins_data = (
        df.group_by(["reference_group", "model", "bin"])
        .agg(
            [
                pl.len().alias("n"),
                pl.mean("prob").alias("x"),
                pl.mean("real").alias("y"),
                pl.sum("real").alias("n_reals"),
            ]
        )
        .sort(["reference_group", "model", "bin"])
    )

    return calibration_bins_data


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
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    n_bins = _validate_n_bins(n_bins)
    effective_n_bins = n_bins if calibration_type == "discrete" else 10
    calibration_bins_data = _make_calibration_bins_dat_binary(
        probs, reals, n_bins=effective_n_bins
    )
    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)
    smooth_dat = _calculate_smooth_curve(probs, reals, performance_type)

    calibration_bins_data, smooth_dat = _add_hover_text_to_calibration_data(
        calibration_bins_data, smooth_dat, performance_type
    )

    reference_data = _create_reference_data_for_calibration_curve()

    reference_groups = list(probs.keys())

    colors_dictionary = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )

    histogram_for_calibration = _create_histogram_for_calibration(probs)

    limits = _define_limits_for_calibration_plot(calibration_bins_data)
    axes_ranges = {"xaxis": limits, "yaxis": limits}

    calibration_curve_list = {
        "calibration_bins_dat": calibration_bins_data,
        "smooth_dat": smooth_dat,
        "reference_data": reference_data,
        "histogram_for_calibration": histogram_for_calibration,
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
            smoothed = smooth_state_lowess(p, r)
            return smoothed.with_columns(pl.lit(group_name).alias("reference_group"))

    if isinstance(reals, dict):
        for model_name, prob_array in probs.items():
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
    calibration_bins_dat: pl.DataFrame,
    smooth_dat: pl.DataFrame,
    performance_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Adds hover text to the calibration bins and smooth dataframes."""
    if performance_type != "one model":
        calibration_bins_dat = calibration_bins_dat.with_columns(
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
        calibration_bins_dat = calibration_bins_dat.with_columns(
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
    return calibration_bins_dat, smooth_dat


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


def _define_limits_for_calibration_plot(
    calibration_bins_dat: pl.DataFrame,
) -> List[float]:
    if calibration_bins_dat.height == 1:
        lower_bound, upper_bound = 0.0, 1.0
    else:
        lower_bound = float(
            max(
                0,
                min(
                    cast(float, calibration_bins_dat["x"].min()),
                    cast(float, calibration_bins_dat["y"].min()),
                ),
            )
        )
        upper_bound = float(
            max(
                cast(float, calibration_bins_dat["x"].max()),
                cast(float, calibration_bins_dat["y"].max()),
            )
        )

    padding = (upper_bound - lower_bound) * 0.05
    return [
        max(0.0, lower_bound - padding),
        min(1.0, upper_bound + padding),
    ]


def _build_initial_df_for_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
) -> pl.DataFrame:
    """Builds the initial DataFrame for time-dependent calibration curves."""

    # Convert all inputs to dictionaries of arrays to unify processing
    reals_was_dict = isinstance(reals, dict)
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
        expressions = [
            pl.Series("prob", prob_array),
            pl.lit(model_name).alias("model"),
        ]
        if not reals_was_dict:
            expressions.append(pl.lit(model_name).alias("reference_group"))
        return base_df.with_columns(expressions)

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


def _prepare_adjusted_event_data(
    df: pl.DataFrame, horizon: float, competing_heuristic: str
) -> pl.DataFrame:
    """Prepare event histories for Aalen-Johansen estimation."""
    event_data = df
    if competing_heuristic == "excluded":
        event_data = event_data.filter(
            ~((pl.col("real") == 2) & (pl.col("time") <= horizon))
        )
    elif competing_heuristic == "adjusted_as_composite":
        event_data = event_data.with_columns(
            pl.when((pl.col("real") == 2) & (pl.col("time") <= horizon))
            .then(1)
            .otherwise(pl.col("real"))
            .alias("real")
        )
    return event_data


def _aj_risk_at_horizon(df: pl.DataFrame, horizon: float) -> float:
    """Estimate target-event cumulative incidence at one horizon."""
    event_table = prepare_event_table(
        df.select(pl.col("time").alias("times"), pl.col("real").alias("reals"))
    )
    estimate = predict_aj_estimates(event_table, pl.Series([horizon]))
    return float(estimate["state_occupancy_probability_1"][0])


def _make_adjusted_calibration_bins_data(
    df: pl.DataFrame, horizon: float, n_bins: int = 10
) -> pl.DataFrame:
    """Create calibration groups using within-group Aalen-Johansen risks."""
    n_bins = _validate_n_bins(n_bins)
    grouped = df.with_columns(
        (
            (pl.col("prob").rank("average").over("reference_group") - 1)
            * n_bins
            // pl.len().over("reference_group")
            + 1
        ).alias("bin")
    )
    rows = []
    for key, group_df in grouped.group_by(["reference_group", "bin"]):
        reference_group, bin_val = key
        estimate = _aj_risk_at_horizon(group_df, horizon)
        n = group_df.height
        rows.append(
            {
                "reference_group": reference_group,
                "model": reference_group,
                "bin": bin_val,
                "n": n,
                "x": cast(float, group_df["prob"].mean()),
                "y": estimate,
                "n_reals": estimate * n,
            }
        )
    return pl.DataFrame(rows).sort(["reference_group", "bin"])


def _calculate_local_aj_smooth(
    df_adj: pl.DataFrame,
    horizon: float,
    performance_type: str,
    bandwidth: Union[float, None] = None,
) -> pl.DataFrame:
    """Calculate smoothed calibration curve using local Aalen-Johansen estimation (Gerds' method)."""
    smooth_frames = []

    for key, group_df in df_adj.group_by("reference_group", maintain_order=True):
        group_name = str(key[0])
        n = group_df.height
        probs = group_df["prob"].to_numpy()

        if len(np.unique(probs)) == 1:
            y_est = _aj_risk_at_horizon(group_df, horizon)
            xout = np.linspace(0, 1, 101)
            smooth_frames.append(
                pl.DataFrame(
                    {
                        "x": xout,
                        "y": [y_est] * len(xout),
                        "reference_group": [group_name] * len(xout),
                    }
                )
            )
            continue

        xout = np.linspace(0, 1, 101)
        yout = []

        if bandwidth is not None:
            k = max(5, int(bandwidth * n))
        else:
            k = max(10, min(n, int(0.2 * n)))

        for p0 in xout:
            distances = np.abs(probs - p0)
            idx = np.argsort(distances, kind="stable")[:k]
            sub_df = group_df[idx]
            y_est = _aj_risk_at_horizon(sub_df, horizon)
            yout.append(y_est)

        smooth_frames.append(
            pl.DataFrame(
                {
                    "x": xout,
                    "y": np.array(yout),
                    "reference_group": [group_name] * len(xout),
                }
            )
        )

    if not smooth_frames:
        return pl.DataFrame(
            schema={
                "x": pl.Float64,
                "y": pl.Float64,
                "reference_group": pl.Utf8,
            }
        )

    smooth_dat = pl.concat(smooth_frames)
    return smooth_dat


def _calculate_adjusted_pseudostates(
    df: pl.DataFrame, horizon: float
) -> Dict[str, np.ndarray]:
    """Calculate leave-one-out AJ pseudo-observations without a new dependency."""
    pseudo_by_group: Dict[str, np.ndarray] = {}
    for key, group_df in df.group_by("reference_group", maintain_order=True):
        reference_group = str(key[0])
        n = group_df.height
        theta = _aj_risk_at_horizon(group_df, horizon)
        if n == 1:
            pseudo_by_group[reference_group] = np.array([theta])
            continue
        leave_one_out = np.array(
            [
                _aj_risk_at_horizon(
                    group_df.slice(0, i).vstack(group_df.slice(i + 1)), horizon
                )
                for i in range(n)
            ]
        )
        pseudo_by_group[reference_group] = n * theta - (n - 1) * leave_one_out
    return pseudo_by_group


def _create_calibration_curve_list_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: List[float],
    heuristics_sets: List[Dict[str, str]],
    calibration_type: str = "discrete",
    smooth_method: str = "local_aj",
    bandwidth: Union[float, None] = None,
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
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Creates the data structures needed for a time-dependent calibration curve plot.
    """
    n_bins = _validate_n_bins(n_bins)
    effective_n_bins = n_bins if calibration_type == "discrete" else 10

    # Part 1: Prepare initial dataframe from inputs
    initial_df = _build_initial_df_for_times(probs, reals, times)
    # Part 2: Iterate and generate calibration data for each horizon/heuristic
    all_calibration_bins = []
    all_smooth = []
    all_histograms = []

    performance_type = _check_performance_type_by_probs_and_reals(probs, reals)

    for horizon in fixed_time_horizons:
        for heuristics in heuristics_sets:
            censoring_heuristic = heuristics["censoring_heuristic"]
            competing_heuristic = heuristics["competing_heuristic"]

            if competing_heuristic == "adjusted_as_censored":
                continue

            if censoring_heuristic == "adjusted":
                df_adj = _prepare_adjusted_event_data(
                    initial_df, horizon, competing_heuristic
                )
                if df_adj.height == 0:
                    continue

                calibration_bins_data = _make_adjusted_calibration_bins_data(
                    df_adj, horizon, n_bins=effective_n_bins
                )
                probs_adj = {
                    key[0]: group_df["prob"].to_numpy()
                    for key, group_df in df_adj.group_by(
                        "reference_group", maintain_order=True
                    )
                }
                if calibration_type == "smooth":
                    if smooth_method == "local_aj":
                        smooth_data = _calculate_local_aj_smooth(
                            df_adj, horizon, performance_type, bandwidth=bandwidth
                        )
                    elif smooth_method == "secondary_cox":
                        smooth_data = calculate_secondary_cox_smooth(
                            df_adj,
                            horizon,
                            performance_type,
                            aj_risk_at_horizon=_aj_risk_at_horizon,
                        )
                    elif smooth_method == "pseudo_values":
                        pseudo_by_group = _calculate_adjusted_pseudostates(
                            df_adj, horizon
                        )
                        smooth_data = _calculate_smooth_curve(
                            probs_adj, pseudo_by_group, performance_type
                        )
                    else:
                        raise ValueError(
                            f"Unsupported smooth_method: '{smooth_method}'. "
                            "Supported options are 'local_aj', 'secondary_cox', and 'pseudo_values'."
                        )
                else:
                    smooth_data = calibration_bins_data.select(
                        "x", "y", "reference_group"
                    )
                hist_data = _create_histogram_for_calibration(probs_adj)

                all_calibration_bins.append(
                    calibration_bins_data.with_columns(
                        pl.lit(horizon).alias("fixed_time_horizon")
                    )
                )
                all_smooth.append(
                    smooth_data.with_columns(
                        pl.lit(horizon).alias("fixed_time_horizon")
                    )
                )
                all_histograms.append(
                    hist_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
                )
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

            # Calibration bins
            calibration_bins_data = _make_calibration_bins_dat_binary(
                probs_adj, reals_adj, n_bins=effective_n_bins
            )
            all_calibration_bins.append(
                calibration_bins_data.with_columns(
                    pl.lit(horizon).alias("fixed_time_horizon")
                )
            )

            # Smooth curve
            if calibration_type == "smooth":
                if smooth_method == "local_aj":
                    smooth_data = _calculate_local_aj_smooth(
                        df_adj, horizon, performance_type, bandwidth=bandwidth
                    )
                elif smooth_method == "secondary_cox":
                    smooth_data = calculate_secondary_cox_smooth(
                        df_adj,
                        horizon,
                        performance_type,
                        aj_risk_at_horizon=_aj_risk_at_horizon,
                    )
                elif smooth_method == "pseudo_values":
                    smooth_data = _calculate_smooth_curve(
                        probs_adj, reals_adj, performance_type
                    )
                else:
                    raise ValueError(
                        f"Unsupported smooth_method: '{smooth_method}'. "
                        "Supported options are 'local_aj', 'secondary_cox', and 'pseudo_values'."
                    )
            else:
                smooth_data = calibration_bins_data.select("x", "y", "reference_group")
            all_smooth.append(
                smooth_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
            )

            # Histogram
            hist_data = _create_histogram_for_calibration(probs_adj)
            all_histograms.append(
                hist_data.with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
            )

    # Part 3: Combine results and create final dictionary
    if not all_calibration_bins:
        raise ValueError(
            "No data remaining after applying heuristics and time horizons."
        )
    calibration_bins_dat_final = pl.concat(all_calibration_bins)
    smooth_dat_final = pl.concat(all_smooth)
    histogram_final = pl.concat(all_histograms)

    # Add hover text
    calibration_bins_dat_final, smooth_dat_final = _add_hover_text_to_calibration_data(
        calibration_bins_dat_final, smooth_dat_final, performance_type
    )

    reference_data = _create_reference_data_for_calibration_curve()
    reference_groups = list(probs.keys())
    colors_dictionary = _create_colors_dictionary_for_calibration(
        reference_groups, color_values, performance_type
    )
    limits = _define_limits_for_calibration_plot(calibration_bins_dat_final)
    axes_ranges = {"xaxis": limits, "yaxis": limits}

    calibration_curve_list = {
        "calibration_bins_dat": calibration_bins_dat_final,
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

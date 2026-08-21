"""
A module for Gains Curves using Plotly helpers
"""

from typing import Dict, List, Sequence, Union
from plotly.graph_objs._figure import Figure
from rtichoke.processing.binary_color_values import _apply_color_values_binary
from rtichoke.processing.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
    _create_reference_lines_data,
    _check_if_multiple_populations_are_being_validated_times,
)
from rtichoke.processing.time_reference_lines import (
    _create_rtichoke_plotly_curve_times_reference_safe,
)
import numpy as np
import polars as pl


def _get_gains_aj_estimates_times(performance_data: pl.DataFrame) -> pl.DataFrame:
    """Return one horizon-specific event estimate per reference group.

    For a gains reference curve the required event probability is the overall
    event probability at the horizon. At probability threshold 0 everyone is
    classified positive, so ``real_positives / n`` gives that quantity without
    mixing in the cutoff-specific estimate at threshold 1.
    """
    return (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select("reference_group", "fixed_time_horizon", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
        .select("reference_group", "fixed_time_horizon", "aj_estimate")
        .sort(["reference_group", "fixed_time_horizon"])
    )


def _replace_gains_reference_data_times(
    curve_list: dict, performance_data: pl.DataFrame
) -> dict:
    """Replace time-dependent gains references with one AJ estimate per horizon."""
    aj_estimates = _get_gains_aj_estimates_times(performance_data)
    references = []

    for horizon in curve_list["fixed_time_horizons"]:
        aj_horizon = aj_estimates.filter(pl.col("fixed_time_horizon") == horizon)
        multiple_populations = _check_if_multiple_populations_are_being_validated_times(
            aj_horizon
        )
        references.append(
            _create_reference_lines_data(
                curve="gains",
                aj_estimates_from_performance_data=aj_horizon,
                multiple_populations=multiple_populations,
            ).with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
        )

    curve_list["reference_data"] = (
        pl.concat(references, how="vertical") if references else pl.DataFrame()
    )
    return curve_list


def create_gains_curve(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    by: float = 0.01,
    stratified_by: Sequence[str] = ["probability_threshold"],
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
    """Creates a Gains curve.

    A Gains curve is a marketing and business analytics tool that evaluates
    the performance of a predictive model. It shows the percentage of
    positive outcomes (the "gain") that can be captured by targeting a
    certain percentage of the population, sorted by predicted probability.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names to 1-D numpy arrays of
        predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true binary labels (0 or 1).
    by : float, optional
        The step size for the probability thresholds. Defaults to 0.01.
    stratified_by : Sequence[str], optional
        Variables for stratification. Defaults to ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.
    color_values : List[str], optional
        A list of hex color strings for the plot lines.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the Gains curve.
    """
    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="gains",
    )
    return _apply_color_values_binary(fig, color_values)


def plot_gains_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plots a Gains curve from pre-computed performance data.

    This function is useful for plotting a Gains curve directly from a
    DataFrame that already contains the necessary performance metrics.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame with performance metrics. It must include columns
        for the percentage of the population targeted and the corresponding
        gain, along with any stratification variables.
    stratified_by : Sequence[str], optional
        The columns in `performance_data` used for stratification. Defaults to
        ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the Gains curve.
    """
    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve="gains",
    )
    return fig


def create_gains_curve_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict] = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ],
    by: float = 0.01,
    stratified_by: Sequence[str] = ["probability_threshold"],
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
    """Creates a time-dependent Gains curve.

    Generates a Gains curve for time-to-event models, which is evaluated at
    specified time horizons and handles censored data and competing risks.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary of predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses.
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        The event or censoring times.
    fixed_time_horizons : list[float]
        A list of time points for performance evaluation.
    heuristics_sets : list[Dict], optional
        Specifies how to handle censored data and competing events.
    by : float, optional
        The step size for probability thresholds. Defaults to 0.01.
    stratified_by : Sequence[str], optional
        Variables for stratification. Defaults to ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.
    color_values : List[str], optional
        A list of hex color strings for the plot lines.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object for the time-dependent Gains curve.
    """
    return _create_rtichoke_plotly_curve_times_reference_safe(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="gains",
    )

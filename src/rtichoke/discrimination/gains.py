"""
A module for Gains Curves using Plotly helpers
"""

from typing import Dict, List, Sequence, Union
from plotly.graph_objs._figure import Figure
from rtichoke.processing.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_times,
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
)
import numpy as np
import polars as pl


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
    return fig


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

    fig = _create_rtichoke_plotly_curve_times(
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

    return fig

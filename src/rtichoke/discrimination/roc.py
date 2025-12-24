"""
A module for ROC Curves
"""

from typing import Dict, List, Union, Sequence
from plotly.graph_objs._figure import Figure
from rtichoke.processing.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_times,
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
)
import numpy as np
import polars as pl


def create_roc_curve(
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
    """Creates a Receiver Operating Characteristic (ROC) curve.

    This function generates an ROC curve, which visualizes the diagnostic
    ability of a binary classifier system as its discrimination threshold is
    varied. The curve plots the True Positive Rate (TPR) against the
    False Positive Rate (FPR) at various threshold settings.

    It first calculates the performance data using the provided probabilities
    and true labels, and then generates the plot.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names to 1-D numpy arrays of
        predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true binary labels (0 or 1). Can be a single array for all
        probabilities or a dictionary mapping names to label arrays.
    by : float, optional
        The step size for the probability thresholds, controlling the curve's
        granularity. Defaults to 0.01.
    stratified_by : Sequence[str], optional
        Variables for stratification. Defaults to ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.
    color_values : List[str], optional
        A list of hex color strings for the plot lines. A default palette is
        used if not provided.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the ROC curve.
    """
    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="roc",
    )
    return fig


def plot_roc_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plots an ROC curve from pre-computed performance data.

    This function is useful when you have already computed the performance
    metrics (TPR, FPR, etc.) and want to generate an ROC plot directly from
    that data.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame containing the necessary performance metrics. It must
        include columns for the true positive rate (tpr) and false positive
        rate (fpr), along with any stratification variables.
    stratified_by : Sequence[str], optional
        The columns in `performance_data` used for stratification. Defaults to
        ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the ROC curve.
    """
    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve="roc",
    )

    return fig


def create_roc_curve_times(
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
    """Creates a time-dependent Receiver Operating Characteristic (ROC) curve.

    This function generates an ROC curve for time-to-event models. It evaluates
    the model's performance at specified time horizons, handling censored data
    and competing risks according to the chosen heuristics.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary of predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses (e.g., 0=censored, 1=event, 2=competing).
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
        A Plotly ``Figure`` object representing the time-dependent ROC curve.
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
        curve="roc",
    )

    return fig

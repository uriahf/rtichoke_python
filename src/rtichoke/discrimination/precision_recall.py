"""
A module for Precision-Recall Curves using Plotly helpers
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


def create_precision_recall_curve(
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
    """Creates a Precision-Recall curve.

    This function generates a Precision-Recall curve, which is a common
    alternative to the ROC curve, particularly for imbalanced datasets. It
    plots precision (Positive Predictive Value) against recall (True Positive
    Rate) for a binary classifier at different probability thresholds.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names to 1-D numpy arrays of
        predicted probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true binary labels (0 or 1). Can be a single array or a dictionary
        mapping names to label arrays.
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
        A Plotly ``Figure`` object representing the Precision-Recall curve.
    """
    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="precision recall",
    )
    return fig


def plot_precision_recall_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plots a Precision-Recall curve from pre-computed performance data.

    This function is useful when you have already computed the performance
    metrics and want to generate a Precision-Recall plot directly.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame with the necessary performance metrics, including
        precision (ppv) and recall (tpr), along with any stratification variables.
    stratified_by : Sequence[str], optional
        The columns in `performance_data` used for stratification. Defaults to
        ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the Precision-Recall curve.
    """
    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve="precision recall",
    )
    return fig


def create_precision_recall_curve_times(
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
    """Creates a time-dependent Precision-Recall curve.

    Generates a Precision-Recall curve for time-to-event models, evaluating
    performance at specified time horizons. It handles censored data and
    competing risks based on the provided heuristics.

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
        A Plotly ``Figure`` object for the time-dependent Precision-Recall curve.
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
        curve="precision recall",
    )

    return fig

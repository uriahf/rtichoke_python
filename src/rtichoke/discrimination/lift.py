"""
A module for Lift Curves using Plotly helpers
"""

from typing import Dict, List, Sequence, Union
from plotly.graph_objs._figure import Figure
from rtichoke.helpers.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
)
import numpy as np
import polars as pl


def create_lift_curve(
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
    """Create Lift Curve.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Dictionary mapping a label or group name to an array of predicted
        probabilities for the positive class.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        Ground-truth binary labels (0/1) as a single array, or a dictionary
        mapping the same label/group keys used in ``probs`` to arrays of
        ground-truth labels.
    by : float, optional
        Resolution for probability thresholds when computing the curve
        (step size). Default is 0.01.
    stratified_by : Sequence[str], optional
        Sequence of column names to stratify the performance data by.
        Default is ["probability_threshold"].
    size : int, optional
        Plot size in pixels (width and height). Default is 600.
    color_values : List[str], optional
        List of color hex strings to use for the plotted lines. If not
        provided, a default palette is used.

    Returns
    -------
    Figure
        A Plotly ``Figure`` containing the Lift curve(s).

    Notes
    -----
    The function delegates computation and plotting to
    ``_create_rtichoke_plotly_curve_binary`` and returns the resulting
    Plotly figure.
    """
    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="lift",
    )
    return fig


def plot_lift_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plot Lift curve from performance data.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame containing performance metrics for the Lift curve.
        Expected columns include (but may not be limited to)
        ``probability_threshold`` and lift-related metrics, plus any
        stratification columns.
    stratified_by : Sequence[str], optional
        Sequence of column names used for stratification in the
        ``performance_data``. Default is ["probability_threshold"].
    size : int, optional
        Plot size in pixels (width and height). Default is 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` containing the Lift plot.

    Notes
    -----
    This function wraps ``_plot_rtichoke_curve_binary`` to produce a
    ready-to-render Plotly figure from precomputed performance data.
    """
    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve="lift",
    )
    return fig

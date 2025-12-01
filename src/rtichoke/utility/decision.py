"""
A module for Decision Curves using Plotly helpers
"""

from typing import Dict, List, Sequence, Union
from plotly.graph_objs._figure import Figure
from rtichoke.helpers.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
)
import numpy as np
import polars as pl


def create_decision_curve(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    decision_type: str = "conventional",
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
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
    """Create Decision Curve.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Dictionary mapping a label or group name to an array of predicted
        probabilities for the positive class.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        Ground-truth binary labels (0/1) as a single array, or a dictionary
        mapping the same label/group keys used in ``probs`` to arrays of
        ground-truth labels.
    decision_type : str, optional
        Either ``"conventional"`` (decision curve) or another value that
        implies the "interventions avoided" variant. Default is
        ``"conventional"``.
    min_p_threshold : float, optional
        Minimum probability threshold to include in the curve. Default is 0.
    max_p_threshold : float, optional
        Maximum probability threshold to include in the curve. Default is 1.
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
        A Plotly ``Figure`` containing the Decision curve.

    Notes
    -----
    The function selects the appropriate curve name based on
    ``decision_type`` and delegates computation and plotting to
    ``_create_rtichoke_plotly_curve_binary``. Additional keyword arguments
    (like ``min_p_threshold`` and ``max_p_threshold``) are forwarded to
    the helper.
    """
    if decision_type == "conventional":
        curve = "decision"
    else:
        curve = "interventions avoided"

    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    return fig


def plot_decision_curve(
    performance_data: pl.DataFrame,
    decision_type: str = "conventional",
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plot Decision Curve from performance data.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame containing performance metrics for the Decision
        curve. Expected columns include (but may not be limited to)
        ``probability_threshold`` and decision-curve metrics, plus any
        stratification columns.
    decision_type : str
        ``"conventional"`` for decision curves, otherwise the
        "interventions avoided" variant will be used.
    min_p_threshold : float, optional
        Minimum probability threshold to include in the curve. Default is 0.
    max_p_threshold : float, optional
        Maximum probability threshold to include in the curve. Default is 1.
    stratified_by : Sequence[str], optional
        Sequence of column names used for stratification in the
        ``performance_data``. Default is ["probability_threshold"].
    size : int, optional
        Plot size in pixels (width and height). Default is 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` containing the Decision plot.

    Notes
    -----
    This function wraps ``_plot_rtichoke_curve_binary`` to produce a
    ready-to-render Plotly figure from precomputed performance data.
    Additional keyword arguments (``min_p_threshold``, ``max_p_threshold``)
    are forwarded to the helper.
    """
    if decision_type == "conventional":
        curve = "decision"
    else:
        curve = "interventions avoided"

    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    return fig

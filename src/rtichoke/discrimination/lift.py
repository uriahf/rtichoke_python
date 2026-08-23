"""
A module for Lift Curves using Plotly helpers
"""

from typing import Any, Dict, List, Sequence, Union
from plotly.graph_objs._figure import Figure
from rtichoke.processing.binary_color_values import _apply_color_values_binary
from rtichoke.processing.plotly_helper_functions import (
    _create_rtichoke_plotly_curve_binary,
    _plot_rtichoke_curve_binary,
)
from rtichoke.processing.time_reference_lines import (
    _create_rtichoke_plotly_curve_times_reference_safe,
)
import numpy as np
import polars as pl
from rtichoke._renderers import _render_lift_v2, _validate_renderer
from rtichoke._viz_spec_v2 import (
    _lift_times_v2_spec_from_performance_data,
    _lift_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


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
    renderer: str = "plotly",
) -> Any:
    """Creates a Lift curve.

    A Lift curve is a visual tool used to evaluate the performance of a
    classification model. It shows how much better the model is at identifying
    positive outcomes compared to a random guess. The "lift" is the ratio of
    the results obtained with the model to the results from a random selection.

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

    renderer : {"plotly", "matplotlib", "browser", "rtichoke_viz"}, optional
        Rendering backend. The default, ``"plotly"``, preserves the existing
        return value and behavior. ``"matplotlib"`` requires the optional
        Matplotlib dependency. ``"browser"`` and its ``"rtichoke_viz"`` alias
        return an offline browser chart backed by the packaged TypeScript bundle.

    Returns
    -------
    Figure or RtichokeBrowserChart
        A Plotly or Matplotlib figure, or an offline browser chart, depending
        on ``renderer``.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        performance_data = prepare_performance_data(
            probs=probs,
            reals=reals,
            stratified_by=stratified_by,
            by=by,
        )
        evaluation_metadata = _build_evaluation_metadata(probs, reals, np.array([]))
        spec = _lift_v2_spec_from_performance_data(
            performance_data, evaluation_metadata
        )
        return _render_lift_v2(
            spec,
            renderer=selected_renderer,
            size=size,
            color_values=color_values,
        )

    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="lift",
    )
    return _apply_color_values_binary(fig, color_values)


def plot_lift_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
) -> Figure:
    """Plots a Lift curve from pre-computed performance data.

    This function is useful for plotting a Lift curve directly from a
    DataFrame that already contains the necessary performance metrics.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame with performance metrics. It must include columns
        for the lift values and the percentage of the population targeted,
        along with any stratification variables.
    stratified_by : Sequence[str], optional
        The columns in `performance_data` used for stratification. Defaults to
        ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.

    Returns
    -------
    Figure
        A Plotly ``Figure`` object representing the Lift curve.
    """
    fig = _plot_rtichoke_curve_binary(
        performance_data,
        size=size,
        curve="lift",
    )
    return fig


def create_lift_curve_times(
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
    renderer: str = "plotly",
) -> Any:
    """Creates a time-dependent Lift curve.

    Generates a Lift curve for time-to-event models, which is evaluated at
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

    renderer : {"plotly", "matplotlib", "browser", "rtichoke_viz"}, optional
        Rendering backend. Plotly remains the default production behavior.

    Returns
    -------
    Figure or RtichokeBrowserChart
        A Plotly or Matplotlib figure, or an offline browser chart, depending
        on ``renderer``.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        from rtichoke.performance_data.performance_data_times import (
            prepare_performance_data_times,
        )

        performance_data = prepare_performance_data_times(
            probs,
            reals,
            times,
            fixed_time_horizons=fixed_time_horizons,
            heuristics_sets=heuristics_sets,
            by=by,
            stratified_by=stratified_by,
        )
        evaluation_metadata = _build_evaluation_metadata(probs, reals, times)
        spec = _lift_times_v2_spec_from_performance_data(
            performance_data, evaluation_metadata
        )
        return _render_lift_v2(
            spec,
            renderer=selected_renderer,
            size=size,
            color_values=color_values,
        )

    fig = _create_rtichoke_plotly_curve_times_reference_safe(
        probs,
        reals,
        times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="lift",
    )

    return fig

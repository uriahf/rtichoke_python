"""
A module for Precision-Recall Curves using Plotly helpers
"""

from typing import Any, Dict, List, Sequence, Union
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

from rtichoke._renderers import RtichokeBrowserChart, _validate_renderer
from rtichoke._viz_spec_v2 import (
    _precision_recall_times_v2_spec_from_performance_data,
    _precision_recall_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _build_evaluation_metadata,
)


def _derive_op_dim_from_stratified_by(stratified_by: Sequence[str]) -> str:
    if "ppcr" in stratified_by:
        return "ppcr"
    return "probability_threshold"


def _precision_recall_browser_chart(
    performance_data: pl.DataFrame,
    evaluation_metadata: dict[str, _EvaluationMetadata],
    *,
    size: int,
    operating_point_dimension: str = "probability_threshold",
) -> RtichokeBrowserChart:
    spec = _precision_recall_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        operating_point_dimension=operating_point_dimension,
    )
    return RtichokeBrowserChart(spec=spec, size=size)


def _performance_data_evaluation_metadata(
    performance_data: pl.DataFrame,
) -> dict[str, _EvaluationMetadata]:
    """Treat pre-computed groups as populations when model identity is unknown."""
    groups = list(
        dict.fromkeys(
            str(value) for value in performance_data["reference_group"].to_list()
        )
    )
    return {
        group: _EvaluationMetadata(
            reference_group=group,
            evaluation=group,
            model=None,
            population=group,
        )
        for group in groups
    }


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
    renderer: str = "plotly",
) -> Any:
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
        A list of hex color strings for the Plotly lines.
    renderer : {"plotly", "browser", "rtichoke_viz"}, optional
        Rendering backend. ``"plotly"`` remains the default. ``"browser"`` and
        its ``"rtichoke_viz"`` alias return a canonical offline browser chart.

    Returns
    -------
    Figure or RtichokeBrowserChart
        A Plotly ``Figure`` or canonical offline browser chart.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        if selected_renderer == "matplotlib":
            raise ValueError(
                "Precision-Recall supports 'plotly', 'browser', and 'rtichoke_viz' "
                "renderers."
            )
        performance_data = prepare_performance_data(
            probs=probs,
            reals=reals,
            by=by,
            stratified_by=stratified_by,
        )
        evaluation_metadata = _build_evaluation_metadata(probs, reals, np.array([]))
        op_dim = _derive_op_dim_from_stratified_by(stratified_by)
        return _precision_recall_browser_chart(
            performance_data,
            evaluation_metadata,
            size=size,
            operating_point_dimension=op_dim,
        )

    fig = _create_rtichoke_plotly_curve_binary(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        curve="precision recall",
    )
    return _apply_color_values_binary(fig, color_values)


def plot_precision_recall_curve(
    performance_data: pl.DataFrame,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
    renderer: str = "plotly",
) -> Any:
    """Plots a Precision-Recall curve from pre-computed performance data.

    This function is useful when you have already computed the performance
    metrics and want to generate a Precision-Recall plot directly. Pre-computed
    data does not encode separate model identity, so canonical browser rendering
    treats each ``reference_group`` as a population with unknown model identity.

    Parameters
    ----------
    performance_data : pl.DataFrame
        A Polars DataFrame with the necessary performance metrics, including
        precision (ppv) and recall (sensitivity), along with the production
        prevalence quantities ``real_positives`` and ``n``.
    stratified_by : Sequence[str], optional
        The columns in `performance_data` used for stratification. Defaults to
        ``["probability_threshold"]``.
    size : int, optional
        The width and height of the plot in pixels. Defaults to 600.
    renderer : {"plotly", "browser", "rtichoke_viz"}, optional
        Rendering backend. ``"plotly"`` remains the default.

    Returns
    -------
    Figure or RtichokeBrowserChart
        A Plotly ``Figure`` or canonical offline browser chart.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        if selected_renderer == "matplotlib":
            raise ValueError(
                "Precision-Recall supports 'plotly', 'browser', and 'rtichoke_viz' "
                "renderers."
            )
        evaluation_metadata = _performance_data_evaluation_metadata(performance_data)
        op_dim = _derive_op_dim_from_stratified_by(stratified_by)
        return _precision_recall_browser_chart(
            performance_data,
            evaluation_metadata,
            size=size,
            operating_point_dimension=op_dim,
        )

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
    renderer: str = "plotly",
) -> Any:
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
    renderer : {"plotly", "browser", "rtichoke_viz"}, optional
        Rendering backend. ``"plotly"`` remains the default. ``"browser"`` and
        its ``"rtichoke_viz"`` alias return a canonical offline browser chart.

    Returns
    -------
    Figure or RtichokeBrowserChart
        A Plotly ``Figure`` or canonical offline browser chart depending on ``renderer``.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        if selected_renderer == "matplotlib":
            raise ValueError(
                "Precision-Recall supports 'plotly', 'browser', and 'rtichoke_viz' "
                "renderers."
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
        op_dim = _derive_op_dim_from_stratified_by(stratified_by)
        spec = _precision_recall_times_v2_spec_from_performance_data(
            performance_data,
            evaluation_metadata,
            operating_point_dimension=op_dim,
        )
        return RtichokeBrowserChart(spec=spec, size=size)

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
        curve="precision recall",
    )

    return fig

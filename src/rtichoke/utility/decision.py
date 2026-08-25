"""
A module for Decision Curves using Plotly helpers and canonical browser rendering.
"""

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import polars as pl
from plotly.graph_objs._figure import Figure

from rtichoke._decision_curve_viz_spec_v2 import (
    _decision_curve_v2_spec_from_performance_data,
)
from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_v2_spec_from_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart, _validate_renderer
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.binary_color_values import _apply_color_values_binary
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _build_evaluation_metadata,
)
from rtichoke.processing.plotly_helper_functions import _plot_rtichoke_curve_binary
from rtichoke.processing.time_reference_lines import (
    _create_rtichoke_plotly_curve_times_reference_safe,
)


def _decision_curve_browser_chart(
    performance_data: pl.DataFrame,
    evaluation_metadata: dict[str, _EvaluationMetadata],
    *,
    size: int,
    min_p_threshold: float,
    max_p_threshold: float,
) -> RtichokeBrowserChart:
    spec = _decision_curve_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    return RtichokeBrowserChart(spec=spec, size=size)


def _interventions_avoided_browser_chart(
    performance_data: pl.DataFrame,
    evaluation_metadata: dict[str, _EvaluationMetadata],
    *,
    size: int,
    min_p_threshold: float,
    max_p_threshold: float,
) -> RtichokeBrowserChart:
    spec = _interventions_avoided_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
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
    renderer: str = "plotly",
) -> Any:
    """Creates a Decision Curve.

    ``renderer="plotly"`` preserves the historical default. For static
    conventional Decision Curves and static Interventions Avoided curves,
    ``"browser"`` and ``"rtichoke_viz"`` return a canonical
    :class:`RtichokeBrowserChart` built from already-computed production values.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        if selected_renderer == "matplotlib" or decision_type not in {
            "conventional",
            "interventions avoided",
        }:
            raise ValueError(
                "Static Decision Curves support 'plotly', 'browser', and "
                "'rtichoke_viz' renderers for decision_type='conventional' or "
                "decision_type='interventions avoided'."
            )
        performance_data = prepare_performance_data(
            probs=probs,
            reals=reals,
            stratified_by=stratified_by,
            by=by,
        )
        evaluation_metadata = _build_evaluation_metadata(probs, reals, np.array([]))
        if decision_type == "conventional":
            return _decision_curve_browser_chart(
                performance_data,
                evaluation_metadata,
                size=size,
                min_p_threshold=min_p_threshold,
                max_p_threshold=max_p_threshold,
            )
        return _interventions_avoided_browser_chart(
            performance_data,
            evaluation_metadata,
            size=size,
            min_p_threshold=min_p_threshold,
            max_p_threshold=max_p_threshold,
        )

    if decision_type == "conventional":
        curve = "decision"
    else:
        curve = "interventions avoided"

    performance_data = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=stratified_by,
        by=by,
    )
    fig = _plot_rtichoke_curve_binary(
        performance_data=performance_data,
        stratified_by=stratified_by[0],
        curve=curve,
        size=size,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    fig.update_xaxes(range=[min_p_threshold, max_p_threshold])
    return _apply_color_values_binary(fig, color_values)


def plot_decision_curve(
    performance_data: pl.DataFrame,
    decision_type: str = "conventional",
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
    stratified_by: Sequence[str] = ["probability_threshold"],
    size: int = 600,
    renderer: str = "plotly",
) -> Any:
    """Plots a Decision Curve from pre-computed performance data.

    For browser rendering, pre-computed ``reference_group`` values are treated
    as distinct populations because separate model identity is not encoded in
    this input shape.
    """
    selected_renderer = _validate_renderer(renderer)
    if selected_renderer != "plotly":
        if selected_renderer == "matplotlib" or decision_type not in {
            "conventional",
            "interventions avoided",
        }:
            raise ValueError(
                "Static Decision Curves support 'plotly', 'browser', and "
                "'rtichoke_viz' renderers for decision_type='conventional' or "
                "decision_type='interventions avoided'."
            )
        evaluation_metadata = _performance_data_evaluation_metadata(performance_data)
        if decision_type == "conventional":
            return _decision_curve_browser_chart(
                performance_data,
                evaluation_metadata,
                size=size,
                min_p_threshold=min_p_threshold,
                max_p_threshold=max_p_threshold,
            )
        return _interventions_avoided_browser_chart(
            performance_data,
            evaluation_metadata,
            size=size,
            min_p_threshold=min_p_threshold,
            max_p_threshold=max_p_threshold,
        )

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
    fig.update_xaxes(range=[min_p_threshold, max_p_threshold])
    return fig


def create_decision_curve_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    decision_type: str = "conventional",
    heuristics_sets: list[Dict] = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ],
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
    """Creates a time-dependent Decision Curve using the existing Plotly path."""
    if decision_type == "conventional":
        curve = "decision"
    else:
        curve = "interventions avoided"

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
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    return fig

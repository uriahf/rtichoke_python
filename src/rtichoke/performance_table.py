"""Performance-table API with multiple rendering backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Literal, Union

import numpy as np
import polars as pl

from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.performance_table_great_tables import (
    render_performance_table_great_tables,
)
from rtichoke.performance_table_reactable import (
    DEFAULT_COLORS,
    render_performance_table_reactable,
)

PerformanceTableRenderer = Literal["great_tables", "reactable"]

_DEFAULT_HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]


def render_performance_table(
    performance_data: pl.DataFrame,
    color_values: Sequence[str] = DEFAULT_COLORS,
    renderer: PerformanceTableRenderer = "great_tables",
):
    """Render prepared performance data with a selected table backend."""
    if renderer == "great_tables":
        return render_performance_table_great_tables(
            performance_data, color_values=color_values
        )
    if renderer == "reactable":
        return render_performance_table_reactable(
            performance_data, color_values=color_values
        )
    raise ValueError("renderer must be either 'great_tables' or 'reactable'")


def create_performance_table(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    by: float = 0.01,
    stratified_by: Sequence[str] = ("probability_threshold",),
    color_values: Sequence[str] = DEFAULT_COLORS,
    renderer: PerformanceTableRenderer = "great_tables",
):
    """Create an R-style rtichoke performance table."""
    performance_data = prepare_performance_data(
        probs=probs, reals=reals, by=by, stratified_by=stratified_by
    )
    return render_performance_table(
        performance_data, color_values=color_values, renderer=renderer
    )


def create_performance_table_times(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict] = _DEFAULT_HEURISTICS,
    by: float = 0.01,
    stratified_by: Sequence[str] = ("probability_threshold",),
    color_values: Sequence[str] = DEFAULT_COLORS,
    renderer: PerformanceTableRenderer = "great_tables",
):
    """Create a time-dependent rtichoke performance table.

    Numerical results come from ``prepare_performance_data_times()``. The table
    keeps time horizon and censoring/competing-event heuristics visible so that
    multiple requested evaluation scenarios are not collapsed in presentation.
    Observed times are normalized to floating point at this public wrapper
    boundary; fixed-horizon normalization is handled by the shared time-dependent
    performance pipeline.
    """
    if isinstance(times, dict):
        normalized_times = {
            key: np.asarray(value, dtype=float) for key, value in times.items()
        }
    else:
        normalized_times = np.asarray(times, dtype=float)

    performance_data = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=normalized_times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        by=by,
        stratified_by=stratified_by,
    )
    return render_performance_table(
        performance_data, color_values=color_values, renderer=renderer
    )

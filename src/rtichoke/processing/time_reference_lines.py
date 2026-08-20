"""Helpers for horizon-specific time-dependent reference curves."""

from typing import Dict, Sequence, Union

import numpy as np
import polars as pl
from plotly.graph_objs._figure import Figure

from rtichoke.performance_data.performance_data_times import prepare_performance_data_times
from rtichoke.processing.plotly_helper_functions import (
    _check_if_multiple_populations_are_being_validated_times,
    _create_plotly_curve_times,
    _create_reference_lines_data,
    _create_rtichoke_curve_list_times,
)


def _get_reference_aj_estimates_times(performance_data: pl.DataFrame) -> pl.DataFrame:
    """Return one event-risk estimate per reference group and horizon.

    At probability threshold 0 everyone is classified positive, so
    ``real_positives / n`` is the horizon-specific event probability. Using
    only cutoff 0 avoids mixing that estimate with cutoff-specific values from
    the opposite boundary.
    """
    return (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select("reference_group", "fixed_time_horizon", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
        .select("reference_group", "fixed_time_horizon", "aj_estimate")
        .sort(["reference_group", "fixed_time_horizon"])
    )


def _replace_reference_data_times(
    curve_list: dict,
    performance_data: pl.DataFrame,
    curve: str,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> dict:
    """Replace prevalence-dependent references with cutoff-0 estimates."""
    aj_estimates = _get_reference_aj_estimates_times(performance_data)
    references = []

    for horizon in curve_list["fixed_time_horizons"]:
        aj_horizon = aj_estimates.filter(pl.col("fixed_time_horizon") == horizon)
        multiple_populations = _check_if_multiple_populations_are_being_validated_times(
            aj_horizon
        )
        references.append(
            _create_reference_lines_data(
                curve=curve,
                aj_estimates_from_performance_data=aj_horizon,
                multiple_populations=multiple_populations,
                min_p_threshold=min_p_threshold,
                max_p_threshold=max_p_threshold,
            ).with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
        )

    curve_list["reference_data"] = (
        pl.concat(references, how="vertical") if references else pl.DataFrame()
    )
    return curve_list


def _create_rtichoke_plotly_curve_times_reference_safe(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
    fixed_time_horizons: list[float],
    heuristics_sets: list[Dict],
    min_p_threshold: float = 0,
    max_p_threshold: float = 1,
    by: float = 0.01,
    stratified_by: Sequence[str] = ("probability_threshold",),
    size: int = 600,
    color_values=None,
    curve: str = "precision recall",
) -> Figure:
    """Create a time-dependent curve with horizon-specific reference data."""
    performance_data = prepare_performance_data_times(
        probs,
        reals,
        times,
        by=by,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=stratified_by,
    )

    # Preserve the existing plotting behavior here; this helper only corrects
    # construction of prevalence-dependent reference lines.
    curve_list = _create_rtichoke_curve_list_times(
        performance_data,
        stratified_by=stratified_by[0],
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    curve_list = _replace_reference_data_times(
        curve_list,
        performance_data,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )

    return _create_plotly_curve_times(curve_list)

"""Helpers for horizon-specific time-dependent reference curves."""

from typing import Dict, Sequence, Union

import numpy as np
import polars as pl
from plotly.graph_objs._figure import Figure

from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata
from rtichoke.processing.plotly_helper_functions import (
    _check_if_multiple_populations_are_being_validated_times,
    _create_plotly_curve_times,
    _create_reference_lines_data,
    _create_rtichoke_curve_list_times,
)


def _get_reference_aj_estimates_times(performance_data: pl.DataFrame) -> pl.DataFrame:
    """Return the cutoff-0 event risk for each group and horizon."""
    return (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select("reference_group", "fixed_time_horizon", "real_positives", "n")
        .unique()
        .with_columns((pl.col("real_positives") / pl.col("n")).alias("aj_estimate"))
        .select("reference_group", "fixed_time_horizon", "aj_estimate")
        .sort(["reference_group", "fixed_time_horizon"])
    )


def _apply_color_values_times(curve_list: dict, color_values) -> dict:
    """Apply custom colors while retaining R's single-model black styling."""
    if color_values is None or not curve_list["multiple_reference_groups"]:
        return curve_list

    reference_groups = curve_list["reference_group_keys"]
    if len(color_values) < len(reference_groups):
        raise ValueError(
            "color_values must contain at least one color per reference group"
        )

    colors_dictionary = curve_list["colors_dictionary"]
    for index, reference_group in enumerate(reference_groups):
        color = color_values[index]
        for key in (
            reference_group,
            f"random_guess_{reference_group}",
            f"perfect_model_{reference_group}",
            f"treat_none_{reference_group}",
            f"treat_all_{reference_group}",
        ):
            if key in colors_dictionary:
                colors_dictionary[key] = color

    return curve_list


def _replace_reference_data_times(
    curve_list: dict,
    performance_data: pl.DataFrame,
    curve: str,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
    multiple_populations: bool | None = None,
) -> dict:
    """Rebuild prevalence-dependent references from cutoff-0 event risk."""
    aj_estimates = _get_reference_aj_estimates_times(performance_data)
    references = []

    for horizon in curve_list["fixed_time_horizons"]:
        aj_horizon = aj_estimates.filter(pl.col("fixed_time_horizon") == horizon)
        horizon_has_multiple_populations = (
            multiple_populations
            if multiple_populations is not None
            else _check_if_multiple_populations_are_being_validated_times(aj_horizon)
        )
        references.append(
            _create_reference_lines_data(
                curve=curve,
                aj_estimates_from_performance_data=aj_horizon,
                multiple_populations=horizon_has_multiple_populations,
                min_p_threshold=min_p_threshold,
                max_p_threshold=max_p_threshold,
            ).with_columns(pl.lit(horizon).alias("fixed_time_horizon"))
        )

    curve_list["reference_data"] = pl.concat(references, how="vertical")
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
    """Create a time-dependent curve with population-scoped references."""
    performance_data = prepare_performance_data_times(
        probs,
        reals,
        times,
        by=by,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=stratified_by,
    )
    curve_list = _create_rtichoke_curve_list_times(
        performance_data,
        stratified_by=stratified_by[0],
        size=size,
        curve=curve,
        min_p_threshold=min_p_threshold,
        max_p_threshold=max_p_threshold,
    )
    curve_list = _apply_color_values_times(curve_list, color_values)

    evaluation_metadata = _build_evaluation_metadata(probs, reals, times)
    multiple_populations = (
        len({metadata.population for metadata in evaluation_metadata.values()}) > 1
    )

    return _create_plotly_curve_times(
        _replace_reference_data_times(
            curve_list,
            performance_data,
            curve=curve,
            min_p_threshold=min_p_threshold,
            max_p_threshold=max_p_threshold,
            multiple_populations=multiple_populations,
        )
    )

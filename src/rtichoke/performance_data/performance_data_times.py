"""
A module for Performance Data with Time Dimension
"""

from typing import Dict, Union
from collections.abc import Sequence

import numpy as np
import polars as pl

from rtichoke.processing.adjustments import create_adjusted_data
from rtichoke.processing.combinations import (
    create_aj_data_combinations,
    create_breaks_values,
)
from rtichoke.processing.time_input_validation import _validate_time_input_alignment
from rtichoke.processing.transforms import (
    _calculate_cumulative_aj_data,
    _create_list_data_to_adjust,
    _turn_cumulative_aj_to_performance_data,
    cast_and_join_adjusted_data,
)


def prepare_performance_data_times(
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
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> pl.DataFrame:
    """Prepare performance data for models with time-to-event outcomes."""
    final_adjusted_data = prepare_binned_classification_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=fixed_time_horizons,
        heuristics_sets=heuristics_sets,
        stratified_by=stratified_by,
        by=by,
        risk_set_scope=["pooled_by_cutoff"],
    )

    cumulative_aj_data = _calculate_cumulative_aj_data(final_adjusted_data)
    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    group_order = {group: index for index, group in enumerate(probs)}
    horizon_order = {
        float(horizon): index for index, horizon in enumerate(fixed_time_horizons)
    }
    heuristic_order = {
        f'{heuristics["censoring_heuristic"]}\x1f{heuristics["competing_heuristic"]}': index
        for index, heuristics in enumerate(heuristics_sets)
    }

    return (
        performance_data.with_columns(
            pl.col("reference_group")
            .replace_strict(group_order, default=len(group_order))
            .alias("_reference_group_order"),
            pl.col("fixed_time_horizon")
            .replace_strict(horizon_order, default=len(horizon_order))
            .alias("_fixed_time_horizon_order"),
            pl.concat_str(
                ["censoring_heuristic", "competing_heuristic"], separator="\x1f"
            )
            .replace_strict(heuristic_order, default=len(heuristic_order))
            .alias("_heuristic_order"),
        )
        .sort(
            [
                "_fixed_time_horizon_order",
                "_heuristic_order",
                "stratified_by",
                "chosen_cutoff",
                "_reference_group_order",
            ]
        )
        .drop(
            "_reference_group_order",
            "_fixed_time_horizon_order",
            "_heuristic_order",
        )
    )


def prepare_binned_classification_data_times(
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
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
    risk_set_scope: Sequence[str] = ["pooled_by_cutoff", "within_stratum"],
) -> pl.DataFrame:
    """Prepare binned, time-dependent classification data."""
    _validate_time_input_alignment(probs=probs, reals=reals, times=times)
    fixed_time_horizons = [float(horizon) for horizon in fixed_time_horizons]

    breaks = create_breaks_values(None, "probability_threshold", by)

    aj_data_combinations = create_aj_data_combinations(
        list(probs.keys()),
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=fixed_time_horizons,
        stratified_by=stratified_by,
        by=by,
        breaks=breaks,
        risk_set_scope=risk_set_scope,
    )

    list_data_to_adjust = _create_list_data_to_adjust(
        aj_data_combinations,
        probs,
        reals,
        times,
        stratified_by=stratified_by,
        by=by,
    )

    adjusted_data = create_adjusted_data(
        list_data_to_adjust,
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=fixed_time_horizons,
        breaks=breaks,
        stratified_by=stratified_by,
        risk_set_scope=risk_set_scope,
    )

    return cast_and_join_adjusted_data(
        aj_data_combinations,
        adjusted_data,
    ).with_columns(pl.col("reals_estimate").fill_null(0.0))

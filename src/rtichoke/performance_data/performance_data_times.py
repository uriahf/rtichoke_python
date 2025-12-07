"""
A module for Performance Data with Time Dimension
"""

from typing import Dict, Union
import polars as pl
from collections.abc import Sequence
from rtichoke.helpers.sandbox_observable_helpers import (
    create_breaks_values,
    create_aj_data_combinations,
    _create_list_data_to_adjust,
    create_adjusted_data,
    cast_and_join_adjusted_data,
    _calculate_cumulative_aj_data,
    _turn_cumulative_aj_to_performance_data,
)

import numpy as np


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
    """Prepare performance data with a time dimension.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Mapping from dataset name to predicted probabilities (1-D numpy arrays).
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True event labels. Can be a single array aligned to pooled probabilities
        or a dictionary mapping each dataset name to its true-label array. Labels
        are expected to be integers (e.g., 0/1 for binary, or competing event codes).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        Event or censoring times corresponding to `reals`. Either a single array
        or a dictionary keyed like `probs` when multiple datasets are provided.
    fixed_time_horizons : list[float]
        Fixed time horizons (same units as `times`) at which to evaluate performance.
    heuristics_sets : list[Dict], optional
        List of heuristic dictionaries controlling censoring/competing-event handling.
        Default is a single heuristic set:
        ``[{"censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative"}]``
    stratified_by : Sequence[str], optional
        Stratification variables used to create combinations/breaks. Defaults to
        ``("probability_threshold",)``.
    by : float, optional
        Step width for probability-threshold breaks (used to create the grid of
        cutoffs). Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing performance metrics computed across probability
        thresholds and fixed time horizons. Columns include the probability cutoff,
        fixed time horizon, heuristic identifiers, and AJ-derived performance measures.
    """
    # 1. Get the underlying binned time-dependent classification data
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

    # 2. Apply AJ cumulative machinery
    cumulative_aj_data = _calculate_cumulative_aj_data(final_adjusted_data)

    # 3. Turn AJ output into performance metrics
    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    return performance_data


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
    """
    Prepare probability-binned, time-dependent classification data.

    This constructs the underlying, binned data across probability thresholds,
    fixed time horizons, and heuristic sets. It returns the adjusted data
    before the cumulative Aalen–Johansen and performance-transformation steps.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Mapping from dataset name to predicted probabilities (1-D numpy arrays).
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True event labels. Can be a single array aligned to pooled probabilities
        or a dictionary mapping each dataset name to its true-label array. Labels
        are expected to be integers (e.g., 0/1 for binary, or competing event codes).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        Event or censoring times corresponding to `reals`. Either a single array
        or a dictionary keyed like `probs` when multiple datasets are provided.
    fixed_time_horizons : list[float]
        Fixed time horizons (same units as `times`) at which to evaluate performance.
    heuristics_sets : list[Dict], optional
        List of heuristic dictionaries controlling censoring/competing-event handling.
        Default is a single heuristic set:
        ``[{"censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative"}]``
    stratified_by : Sequence[str], optional
        Stratification variables used to create combinations/breaks. Defaults to
        ``("probability_threshold",)``.
    by : float, optional
        Step width for probability-threshold breaks (used to create the grid of
        cutoffs). Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing probability-binned, time-dependent
        classification data (one row per combination of dataset / bin /
        time horizon / heuristic / strata). This is the basis for histograms,
        calibration diagnostics, and time-dependent performance curves.
    """
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

    final_adjusted_data = cast_and_join_adjusted_data(
        aj_data_combinations,
        adjusted_data,
    ).with_columns(pl.col("reals_estimate").fill_null(0.0))

    return final_adjusted_data

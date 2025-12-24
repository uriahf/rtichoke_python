"""
A module for Performance Data with Time Dimension
"""

from typing import Dict, Union
import polars as pl
from collections.abc import Sequence
from rtichoke.processing.adjustments import create_adjusted_data
from rtichoke.processing.combinations import (
    create_aj_data_combinations,
    create_breaks_values,
)
from rtichoke.processing.transforms import (
    _calculate_cumulative_aj_data,
    _create_list_data_to_adjust,
    _turn_cumulative_aj_to_performance_data,
    cast_and_join_adjusted_data,
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
    """Prepare performance data for models with time-to-event outcomes.

    This function calculates a comprehensive set of performance metrics for
    models predicting time-to-event outcomes. It handles censored data and
    competing events by applying specified heuristics at different time
    horizons. The function first bins the data using
    `prepare_binned_classification_data_times` and then computes cumulative,
    Aalen-Johansen-based performance metrics.

    The resulting dataframe is the primary input for time-dependent plotting
    functions.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities of an event occurring by a given time.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses. Can be a single array or a dictionary.
        Labels should be integers indicating the outcome (e.g., 0=censored,
        1=event of interest, 2=competing event).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        The event or censoring times corresponding to the `reals`. Can be a
        single array or a dictionary.
    fixed_time_horizons : list[float]
        A list of time points at which to evaluate the model's performance.
    heuristics_sets : list[Dict], optional
        A list of dictionaries, each specifying how to handle censored data
        and competing events. The default is
        ``[{"censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative"}]``.
    stratified_by : Sequence[str], optional
        Variables by which to stratify the analysis. Defaults to
        ``("probability_threshold",)``.
    by : float, optional
        The step size for probability thresholds. Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with performance metrics computed across probability
        thresholds and time horizons. It includes columns for cutoffs, time
        points, heuristics, and performance measures.
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
    Prepare binned, time-dependent classification data.

    This function constructs the foundational binned data needed for
    time-to-event performance analysis. It bins predictions by probability
    thresholds, applies censoring and competing event heuristics, and stratifies
    the data across specified time horizons. The output is a detailed breakdown
    of outcomes within each bin, which can be used for calibration or passed to
    `prepare_performance_data_times` for full performance metric calculation.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities.
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event statuses (e.g., 0=censored, 1=event, 2=competing).
    times : Union[np.ndarray, Dict[str, np.ndarray]]
        The event or censoring times.
    fixed_time_horizons : list[float]
        A list of time points for performance evaluation.
    heuristics_sets : list[Dict], optional
        Specifies how to handle censored data and competing events.
    stratified_by : Sequence[str], optional
        Variables for stratification. Defaults to ``("probability_threshold",)``.
    by : float, optional
        The step size for probability thresholds. Defaults to ``0.01``.
    risk_set_scope : Sequence[str], optional
        Defines the scope for risk set calculations. Defaults to
        ``["pooled_by_cutoff", "within_stratum"]``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with binned, time-dependent data. Each row
        represents a unique combination of dataset, bin, time horizon,
        heuristic, and other strata.
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

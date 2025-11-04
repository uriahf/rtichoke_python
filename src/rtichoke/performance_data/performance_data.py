"""
A module for Performance Data
"""

from typing import Dict, Union
import polars as pl
from collections.abc import Sequence
from rtichoke.helpers.sandbox_observable_helpers import (
    _create_aj_data_combinations_binary,
    create_breaks_values,
    _create_list_data_to_adjust_binary,
    _create_adjusted_data_binary,
    _cast_and_join_adjusted_data_binary,
    _calculate_cumulative_aj_data_binary,
    _turn_cumulative_aj_to_performance_data,
)
import numpy as np


def prepare_performance_data(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by: Sequence[str] = ["probability_threshold"],
    by: float = 0.01,
) -> pl.DataFrame:
    """Prepare performance data for binary classification.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        Mapping from dataset name to predicted probabilities (1-D numpy arrays).
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        True event labels. Can be a single array aligned to pooled probabilities
        or a dictionary mapping each dataset name to its true-label array. Labels
        are expected to be binary integers (0/1).
    stratified_by : Sequence[str], optional
        Stratification variables used to create combinations/breaks. Defaults to
        ``["probability_threshold"]``.
    by : float, optional
        Step width for probability-threshold breaks (used to create the grid of
        cutoffs). Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing performance metrics computed across probability
        thresholds. Columns include the probability cutoff and performance measures.

    Examples
    --------
    >>> import numpy as np
    >>> probs_dict_test = {
    ...     "small_data_set": np.array(
    ...         [0.9, 0.85, 0.95, 0.88, 0.6, 0.7, 0.51, 0.2, 0.1, 0.33]
    ...     )
    ... }
    >>> reals_dict_test = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]

    >>> prepare_performance_data(
    ...     probs_dict_test,
    ...     reals_dict_test,
    ...     by = 0.1
    ... )
    """

    breaks = create_breaks_values(None, "probability_threshold", by)

    aj_data_combinations = _create_aj_data_combinations_binary(
        list(probs.keys()), stratified_by=stratified_by, by=by, breaks=breaks
    )

    list_data_to_adjust = _create_list_data_to_adjust_binary(
        aj_data_combinations, probs, reals, stratified_by=stratified_by, by=by
    )

    adjusted_data = _create_adjusted_data_binary(
        list_data_to_adjust, breaks=breaks, stratified_by=stratified_by
    )

    final_adjusted_data = _cast_and_join_adjusted_data_binary(
        aj_data_combinations, adjusted_data
    )

    cumulative_aj_data = _calculate_cumulative_aj_data_binary(final_adjusted_data)

    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    return performance_data

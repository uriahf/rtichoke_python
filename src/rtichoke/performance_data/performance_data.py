"""
A module for Performance Data
"""

from typing import Dict, Union
import polars as pl
from collections.abc import Sequence
from rtichoke.processing.adjustments import _create_adjusted_data_binary
from rtichoke.processing.combinations import (
    _create_aj_data_combinations_binary,
    create_breaks_values,
)
from rtichoke.processing.transforms import (
    _calculate_cumulative_aj_data_binary,
    _cast_and_join_adjusted_data_binary,
    _create_list_data_to_adjust_binary,
    _turn_cumulative_aj_to_performance_data,
)
import numpy as np


def prepare_binned_classification_data(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> pl.DataFrame:
    """
    Prepare probability-binned classification data for binary outcomes.

    This function serves as the foundation for many of the performance analysis
    visualizations. It takes predicted probabilities and true binary outcomes,
    bins them by probability thresholds, and calculates the number of true
    positives, false positives, true negatives, and false negatives within each
    bin. This detailed, binned data can then be used to generate calibration
    plots or be aggregated to compute various performance metrics.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities (1-D numpy arrays).
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event labels. This can be a single numpy array that is aligned
        with all pooled probabilities or a dictionary mapping each dataset name
        to its corresponding array of true labels. Labels must be binary (0 or 1).
    stratified_by : Sequence[str], optional
        A sequence of strings specifying the variables by which to stratify the
        data. The default is ``("probability_threshold",)``, which bins the data
        based on predicted probabilities.
    by : float, optional
        The step size to use when creating bins for the probability thresholds.
        This determines the granularity of the analysis. Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the binned classification data. Each row
        represents a unique combination of model/dataset, probability bin, and
        any other stratification variables. It forms the basis for subsequent
        performance calculations.
    """
    breaks = create_breaks_values(None, "probability_threshold", by)

    aj_data_combinations = _create_aj_data_combinations_binary(
        list(probs.keys()),
        stratified_by=stratified_by,
        by=by,
        breaks=breaks,
    )

    list_data_to_adjust = _create_list_data_to_adjust_binary(
        aj_data_combinations,
        probs,
        reals,
        stratified_by=stratified_by,
        by=by,
    )

    adjusted_data = _create_adjusted_data_binary(
        list_data_to_adjust,
        breaks=breaks,
        stratified_by=stratified_by,
    )

    final_adjusted_data = _cast_and_join_adjusted_data_binary(
        aj_data_combinations,
        adjusted_data,
    )

    return final_adjusted_data


def prepare_performance_data(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by: Sequence[str] = ("probability_threshold",),
    by: float = 0.01,
) -> pl.DataFrame:
    """Prepare performance data for binary classification models.

    This function computes a comprehensive set of performance metrics for one
    or more binary classification models across a range of probability
    thresholds. It builds upon the binned data from
    `prepare_binned_classification_data` by cumulatively summing the counts
    and calculating metrics like sensitivity (TPR), specificity, precision
    (PPV), and net benefit.

    This resulting dataframe is the primary input for plotting functions like
    `plot_roc_curve`, `plot_precision_recall_curve`, etc.

    Parameters
    ----------
    probs : Dict[str, np.ndarray]
        A dictionary mapping model or dataset names (str) to their predicted
        probabilities (1-D numpy arrays).
    reals : Union[np.ndarray, Dict[str, np.ndarray]]
        The true event labels. This can be a single numpy array that is aligned
        with all pooled probabilities or a dictionary mapping each dataset name
        to its corresponding array of true labels. Labels must be binary (0 or 1).
    stratified_by : Sequence[str], optional
        A sequence of strings specifying the variables by which to stratify the
        data. The default is ``("probability_threshold",)``.
    by : float, optional
        The step size for probability thresholds, determining the number of
        points at which performance is evaluated. Defaults to ``0.01``.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame where each row corresponds to a probability cutoff
        for a given model/dataset. Columns include the cutoff value and a rich
        set of performance metrics (e.g., `tpr`, `fpr`, `ppv`, `net_benefit`).

    Examples
    --------
    >>> import numpy as np
    >>> probs_dict_test = {
    ...     "small_data_set": np.array(
    ...         [0.9, 0.85, 0.95, 0.88, 0.6, 0.7, 0.51, 0.2, 0.1, 0.33]
    ...     )
    ... }
    >>> reals_dict_test = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    >>> performance_df = prepare_performance_data(
    ...     probs=probs_dict_test,
    ...     reals=reals_dict_test,
    ...     by=0.1
    ... )
    """
    final_adjusted_data = prepare_binned_classification_data(
        probs=probs,
        reals=reals,
        stratified_by=stratified_by,
        by=by,
    )

    cumulative_aj_data = _calculate_cumulative_aj_data_binary(final_adjusted_data)

    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    return performance_data

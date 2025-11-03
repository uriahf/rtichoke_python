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
)
import numpy as np


def prepare_performance_data(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    stratified_by: Sequence[str] = ["probability_threshold"],
    by: float = 0.01,
) -> pl.DataFrame:
    """Prepare Performance Data

    Args:
        probs (Dict[str, List[float]]): _description_
        reals (Dict[str, List[int]]): _description_
        stratified_by (str, optional): _description_. Defaults to "probability_threshold".
        url_api (_type_, optional): _description_. Defaults to "http://localhost:4242/".

    Returns:
        DataFrame: _description_
    """

    breaks = create_breaks_values(None, "probability_threshold", by)

    aj_data_combinations = _create_aj_data_combinations_binary(
        list(probs.keys()), stratified_by=stratified_by, by=by, breaks=breaks
    )

    performance_data = _create_list_data_to_adjust_binary(aj_data_combinations)

    return performance_data

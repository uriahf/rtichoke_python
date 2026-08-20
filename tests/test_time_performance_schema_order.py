import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from rtichoke import prepare_performance_data_times


PROBS = {"model": np.array([0.1, 0.2, 0.5, 0.5, 0.8, 0.9])}
REALS = np.array([0, 0, 1, 0, 1, 1])
TIMES = np.array([8.0, 7.0, 3.0, 6.0, 2.0, 1.0])


def _prepare(stratified_by):
    return prepare_performance_data_times(
        probs=PROBS,
        reals=REALS,
        times=TIMES,
        fixed_time_horizons=[5.0],
        stratified_by=stratified_by,
        by=0.5,
    )


def test_time_performance_schema_order_is_stable_across_stratification():
    threshold_only = _prepare(("probability_threshold",))
    ppcr_only = _prepare(("ppcr",))
    combined = _prepare(("probability_threshold", "ppcr"))

    assert ppcr_only.columns == threshold_only.columns
    assert combined.columns == threshold_only.columns


def test_combined_time_ppcr_values_match_ppcr_only():
    ppcr_only = _prepare(("ppcr",)).sort(
        ["reference_group", "fixed_time_horizon", "chosen_cutoff"]
    )
    combined_ppcr = (
        _prepare(("probability_threshold", "ppcr"))
        .filter(pl.col("stratified_by") == "ppcr")
        .select(ppcr_only.columns)
        .sort(["reference_group", "fixed_time_horizon", "chosen_cutoff"])
    )

    assert_frame_equal(combined_ppcr, ppcr_only)

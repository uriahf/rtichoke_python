import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from rtichoke import prepare_performance_data, prepare_performance_data_times


EXPECTED_BINARY_COUNTS_AT_HALF = {
    "true_positives": 1.0,
    "false_positives": 0.0,
    "true_negatives": 1.0,
    "false_negatives": 1.0,
}


def test_binary_probability_equal_to_cutoff_is_predicted_negative_like_r():
    result = prepare_performance_data(
        probs={"model": np.array([0.4, 0.5, 0.6])},
        reals=np.array([0, 1, 1]),
        by=0.5,
    )

    row = result.filter(pl.col("chosen_cutoff") == 0.5).row(0, named=True)

    for column, expected in EXPECTED_BINARY_COUNTS_AT_HALF.items():
        assert row[column] == expected


def test_time_probability_equal_to_cutoff_remains_positive_for_dcurves_parity():
    result = prepare_performance_data_times(
        probs={"model": np.array([0.4, 0.5, 0.6])},
        reals=np.array([1, 1, 1]),
        times=np.array([3.0, 1.0, 1.0]),
        fixed_time_horizons=[2.0],
        by=0.5,
    )

    row = result.filter(pl.col("chosen_cutoff") == 0.5).row(0, named=True)

    assert row["true_positives"] == 2.0
    assert row["false_positives"] == 0.0
    assert row["true_negatives"] == 1.0
    assert row["false_negatives"] == 0.0


def test_binary_cutoff_zero_still_predicts_everyone_positive():
    result = prepare_performance_data(
        probs={"model": np.array([0.0, 0.5, 1.0])},
        reals=np.array([0, 1, 1]),
        by=0.5,
    )

    row = result.filter(pl.col("chosen_cutoff") == 0.0).row(0, named=True)

    assert row["predicted_positives"] == 3
    assert row["true_negatives"] == 0
    assert row["false_negatives"] == 0


def test_binary_cutoff_one_predicts_everyone_negative():
    result = prepare_performance_data(
        probs={"model": np.array([0.0, 0.5, 1.0])},
        reals=np.array([0, 1, 1]),
        by=0.5,
    )

    row = result.filter(pl.col("chosen_cutoff") == 1.0).row(0, named=True)

    assert row["predicted_positives"] == 0
    assert row["true_positives"] == 0
    assert row["false_positives"] == 0


def test_binary_cutoff_adjustment_does_not_change_ppcr_stratification():
    probs = {"model": np.array([0.1, 0.2, 0.5, 0.5, 0.8, 0.9])}
    reals = np.array([0, 0, 1, 0, 1, 1])

    ppcr_only = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=("ppcr",),
        by=0.5,
    ).sort(["reference_group", "chosen_cutoff"])

    combined_ppcr = (
        prepare_performance_data(
            probs=probs,
            reals=reals,
            stratified_by=("probability_threshold", "ppcr"),
            by=0.5,
        )
        .filter(pl.col("stratified_by") == "ppcr")
        .sort(["reference_group", "chosen_cutoff"])
        .select(ppcr_only.columns)
    )

    assert_frame_equal(combined_ppcr, ppcr_only)

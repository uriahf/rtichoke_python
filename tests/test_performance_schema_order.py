import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from rtichoke import prepare_performance_data


PROBS = {"model": np.array([0.1, 0.2, 0.5, 0.5, 0.8, 0.9])}
REALS = np.array([0, 0, 1, 0, 1, 1])


EXPECTED_COLUMNS = [
    "reference_group",
    "stratified_by",
    "chosen_cutoff",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "predicted_positives",
    "predicted_negatives",
    "real_positives",
    "real_negatives",
    "n",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "false_positive_rate",
    "lift",
    "net_benefit",
    "net_benefit_interventions_avoided",
    "ppcr",
]


def test_binary_performance_schema_order_is_stable_across_stratification():
    for stratified_by in [
        ("probability_threshold",),
        ("ppcr",),
        ("probability_threshold", "ppcr"),
    ]:
        result = prepare_performance_data(
            probs=PROBS,
            reals=REALS,
            stratified_by=stratified_by,
            by=0.5,
        )

        assert result.columns == EXPECTED_COLUMNS


def test_combined_ppcr_values_match_ppcr_only():
    ppcr_only = prepare_performance_data(
        probs=PROBS,
        reals=REALS,
        stratified_by=("ppcr",),
        by=0.5,
    ).sort(["reference_group", "chosen_cutoff"])

    combined_ppcr = (
        prepare_performance_data(
            probs=PROBS,
            reals=REALS,
            stratified_by=("probability_threshold", "ppcr"),
            by=0.5,
        )
        .filter(pl.col("stratified_by") == "ppcr")
        .sort(["reference_group", "chosen_cutoff"])
    )

    assert_frame_equal(combined_ppcr, ppcr_only)

import numpy as np
from polars.testing import assert_frame_equal

from rtichoke import prepare_performance_data, prepare_performance_data_times


def _sort_binary(df):
    return df.sort(["reference_group", "stratified_by", "chosen_cutoff"])


def _sort_times(df):
    return df.sort(
        [
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "stratified_by",
            "chosen_cutoff",
            "reference_group",
        ]
    )


def test_binary_combined_stratification_preserves_each_component():
    probs = {
        "group_a": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "group_b": np.array([0.10, 0.25, 0.40, 0.60, 0.80, 0.90]),
    }
    reals = {
        "group_a": np.array([0, 0, 1, 0, 1, 1]),
        "group_b": np.array([0, 1, 0, 1, 1, 1]),
    }

    combined = prepare_performance_data(
        probs=probs,
        reals=reals,
        stratified_by=["probability_threshold", "ppcr"],
        by=0.25,
    )

    for stratification in ["probability_threshold", "ppcr"]:
        isolated = prepare_performance_data(
            probs=probs,
            reals=reals,
            stratified_by=[stratification],
            by=0.25,
        )
        from_combined = combined.filter(
            combined["stratified_by"] == stratification
        )
        assert_frame_equal(
            _sort_binary(from_combined),
            _sort_binary(isolated),
            check_row_order=True,
            check_column_order=True,
        )


def test_time_combined_stratification_preserves_each_component_across_groups_and_horizons():
    probs = {
        "group_a": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "group_b": np.array([0.10, 0.25, 0.40, 0.60, 0.80, 0.90]),
    }
    reals = {
        "group_a": np.array([0, 1, 0, 1, 2, 1]),
        "group_b": np.array([0, 0, 1, 2, 1, 1]),
    }
    times = {
        "group_a": np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "group_b": np.array([0.4, 1.2, 1.6, 2.2, 2.6, 3.2]),
    }
    horizons = [1.5, 2.5]
    heuristics = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        }
    ]

    combined = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics,
        stratified_by=["probability_threshold", "ppcr"],
        by=0.25,
    )

    for stratification in ["probability_threshold", "ppcr"]:
        isolated = prepare_performance_data_times(
            probs=probs,
            reals=reals,
            times=times,
            fixed_time_horizons=horizons,
            heuristics_sets=heuristics,
            stratified_by=[stratification],
            by=0.25,
        )
        from_combined = combined.filter(
            combined["stratified_by"] == stratification
        )
        assert_frame_equal(
            _sort_times(from_combined),
            _sort_times(isolated),
            check_row_order=True,
            check_column_order=True,
        )

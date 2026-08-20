import numpy as np

from rtichoke import prepare_performance_data_times


def test_prepare_performance_data_times_groups_reference_groups_for_comparison():
    probs = {
        "population_b": np.array([0.8, 0.2, 0.6, 0.4]),
        "population_a": np.array([0.7, 0.1, 0.9, 0.3]),
    }
    reals = {
        "population_b": np.array([1, 0, 1, 0]),
        "population_a": np.array([0, 1, 1, 0]),
    }
    times = {
        "population_b": np.array([2.0, 7.0, 4.0, 9.0]),
        "population_a": np.array([8.0, 3.0, 5.0, 10.0]),
    }
    horizons = [8.0, 5.0]
    heuristics_sets = [
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "excluded",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
    ]

    result = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        by=0.2,
    )

    comparison_columns = [
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "stratified_by",
        "chosen_cutoff",
    ]
    expected_group_order = list(probs)

    for block in result.partition_by(comparison_columns, maintain_order=True):
        assert block["reference_group"].to_list() == expected_group_order

    observed_horizons = (
        result.select("fixed_time_horizon")
        .unique(maintain_order=True)["fixed_time_horizon"]
        .to_list()
    )
    assert observed_horizons == horizons

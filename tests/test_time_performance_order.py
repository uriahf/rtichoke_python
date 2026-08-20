import numpy as np

from rtichoke import prepare_performance_data_times


def test_prepare_performance_data_times_returns_deterministic_order():
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

    result = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[8.0, 5.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "excluded",
                "competing_heuristic": "excluded",
            },
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            },
        ],
        by=0.2,
    )

    sort_columns = [
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "stratified_by",
        "chosen_cutoff",
    ]

    assert result.equals(result.sort(sort_columns))

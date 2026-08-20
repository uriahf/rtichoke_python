import numpy as np

from rtichoke import prepare_performance_data_times


def test_prepare_performance_data_times_preserves_r_style_input_order():
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

    block_keys = []
    for row in result.select(
        [
            "reference_group",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
        ]
    ).unique(maintain_order=True).iter_rows():
        block_keys.append(row)

    expected_block_keys = [
        (
            population,
            horizon,
            heuristics["censoring_heuristic"],
            heuristics["competing_heuristic"],
        )
        for population in probs
        for horizon in horizons
        for heuristics in heuristics_sets
    ]

    assert block_keys == expected_block_keys

    for block in result.partition_by(
        [
            "reference_group",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
        ],
        maintain_order=True,
    ):
        assert block["chosen_cutoff"].to_list() == sorted(block["chosen_cutoff"].to_list())

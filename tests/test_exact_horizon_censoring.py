import numpy as np
import pytest
import polars as pl
from polarstate import prepare_event_table

from rtichoke.processing.adjustments import _extract_excluded_events
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
    prepare_binned_classification_data_times,
)


@pytest.fixture
def fixture_a_data():
    """Exact eight-subject fixture (Fixture A).

    ID | times | reals
    A  | 2     | 1
    B  | 4     | 2
    C  | 5     | 0
    D  | 7     | 0
    E  | 10    | 2
    F  | 10    | 1
    G  | 12    | 0
    H  | 14    | 1

    0 = censored, 1 = primary event, 2 = competing event.
    """
    reals = np.array([1, 2, 0, 0, 2, 1, 0, 1])
    times = np.array([2.0, 4.0, 5.0, 7.0, 10.0, 10.0, 12.0, 14.0])
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])}
    return probs, reals, times


def test_extract_excluded_events_unit_and_edge_cases():
    """Directly test _extract_excluded_events for correct horizon boundary predicates, unsorted horizons, early horizons, and tied event times."""
    # Data with early horizon < min(times), tied times at t=10, and censoring/competing at horizons
    # reals: 0 = censored, 1 = event, 2 = competing
    df = pl.DataFrame(
        {
            "reals": [1, 2, 0, 0, 2, 1],
            "times": [2.0, 4.0, 5.0, 7.0, 10.0, 10.0],
        }
    )
    event_table = prepare_event_table(df)

    # 1. Unsorted horizons including early horizon (1.0 < min(times))
    horizons = [10.0, 1.0, 5.0]

    # Test censoring_heuristic="excluded", competing_heuristic="excluded"
    res = _extract_excluded_events(event_table, horizons, "excluded", "excluded")

    # Order in res matches requested horizons order
    h10 = res.filter(pl.col("fixed_time_horizon") == 10.0).to_dicts()[0]
    h1 = res.filter(pl.col("fixed_time_horizon") == 1.0).to_dicts()[0]
    h5 = res.filter(pl.col("fixed_time_horizon") == 5.0).to_dicts()[0]

    # At horizon 1.0: 0 censored (time < 1), 0 competing (time <= 1)
    assert h1["real_censored_est"] == 0.0
    assert h1["real_competing_est"] == 0.0

    # At horizon 5.0:
    # - Censored at t < 5.0: none (t=5.0 is NOT < 5.0) -> 0.0
    # - Competing at t <= 5.0: t=4.0 -> 1.0
    assert h5["real_censored_est"] == 0.0
    assert h5["real_competing_est"] == 1.0

    # At horizon 10.0:
    # - Censored at t < 10.0: t=5.0 and t=7.0 -> 2.0
    # - Competing at t <= 10.0: t=4.0 and t=10.0 -> 2.0
    assert h10["real_censored_est"] == 2.0
    assert h10["real_competing_est"] == 2.0


def test_fixture_a_numerical_oracle_all_eight_combinations(fixture_a_data):
    """Numerically assert state quantities across all eight heuristic combinations at horizons 5 and 10."""
    probs, reals, times = fixture_a_data

    horizons = [5.0, 10.0]

    censoring_options = ["excluded", "adjusted"]
    competing_options = [
        "excluded",
        "adjusted_as_negative",
        "adjusted_as_composite",
        "adjusted_as_censored",
    ]

    heuristics_sets = [
        {"censoring_heuristic": c, "competing_heuristic": comp}
        for c in censoring_options
        for comp in competing_options
    ]

    perf_df = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        by=0.1,
    )

    binned_df = prepare_binned_classification_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        by=0.1,
    )
    assert binned_df.height > 0

    # Independently validated reference values for cutoff 0.0
    # Expected dict key: (censoring_heuristic, competing_heuristic, horizon) -> (n, real_positives, real_negatives)
    expected_oracle = {
        # censoring = "excluded"
        ("excluded", "excluded", 5.0): (7.0, 1.0, 6.0),
        ("excluded", "excluded", 10.0): (4.0, 2.0, 2.0),
        ("excluded", "adjusted_as_negative", 5.0): (8.0, 1.0, 7.0),
        ("excluded", "adjusted_as_negative", 10.0): (6.0, 2.0, 4.0),
        ("excluded", "adjusted_as_composite", 5.0): (8.0, 2.0, 6.0),
        ("excluded", "adjusted_as_composite", 10.0): (6.0, 4.0, 2.0),
        ("excluded", "adjusted_as_censored", 5.0): (8.0, 1.0, 7.0),
        ("excluded", "adjusted_as_censored", 10.0): (6.0, 2.0, 4.0),
        # censoring = "adjusted"
        ("adjusted", "excluded", 5.0): (13.0, 2.0, 11.0),
        ("adjusted", "excluded", 10.0): (13.0, 5.142857, 7.857143),
        ("adjusted", "adjusted_as_negative", 5.0): (8.0, 1.0, 7.0),
        ("adjusted", "adjusted_as_negative", 10.0): (8.0, 2.5, 5.5),
        ("adjusted", "adjusted_as_composite", 5.0): (8.0, 2.0, 6.0),
        ("adjusted", "adjusted_as_composite", 10.0): (8.0, 5.0, 3.0),
        ("adjusted", "adjusted_as_censored", 5.0): (8.0, 1.0, 7.0),
        ("adjusted", "adjusted_as_censored", 10.0): (8.0, 2.75, 5.25),
    }

    for (c, comp, h), (
        exp_n,
        exp_pos,
        exp_neg,
    ) in expected_oracle.items():
        sub = perf_df.filter(
            (pl.col("fixed_time_horizon") == h)
            & (pl.col("censoring_heuristic") == c)
            & (pl.col("competing_heuristic") == comp)
            & (pl.col("chosen_cutoff") == 0.0)
        )
        assert sub.height == 1, f"Expected 1 row for c={c}, comp={comp}, h={h}"

        n_val = float(sub["n"][0])
        pos_val = float(sub["real_positives"][0])
        neg_val = float(sub["real_negatives"][0])

        assert pytest.approx(n_val, abs=1e-4) == exp_n, (
            f"c={c}, comp={comp}, h={h} N mismatch"
        )
        assert pytest.approx(pos_val, abs=1e-4) == exp_pos, (
            f"c={c}, comp={comp}, h={h} pos mismatch"
        )
        assert pytest.approx(neg_val, abs=1e-4) == exp_neg, (
            f"c={c}, comp={comp}, h={h} neg mismatch"
        )

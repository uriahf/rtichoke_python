import numpy as np
import pytest
import polars as pl

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
    # Deterministic placeholder probabilities
    probs = {"model_1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])}
    return probs, reals, times


def test_fixture_a_exact_horizon_censoring_semantics(fixture_a_data):
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

    # 1. Subject C (censored at time 5) retention/exclusion assertions
    # Under censoring_heuristic="excluded", at horizon 5, Subject C (time 5.0) is retained.
    # At horizon 10, Subject C (time 5.0 < 10.0) is excluded.

    # At horizon 5, censoring="excluded", competing="adjusted_as_negative": 0 subjects censored at time < 5. Retained N = 8.
    h5_exc = perf_df.filter(
        (pl.col("fixed_time_horizon") == 5.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "adjusted_as_negative")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h5_exc["n"][0] == 8.0

    # At horizon 10, censoring="excluded", competing="adjusted_as_negative":
    # Subjects C (time 5) and D (time 7) censored at time < 10 are excluded. Retained N = 6.
    h10_exc = perf_df.filter(
        (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "adjusted_as_negative")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h10_exc["n"][0] == 6.0

    # 2. Verify all eight combinations produced non-empty data at both horizons
    for h in horizons:
        for heur in heuristics_sets:
            c = heur["censoring_heuristic"]
            comp = heur["competing_heuristic"]
            sub = perf_df.filter(
                (pl.col("fixed_time_horizon") == h)
                & (pl.col("censoring_heuristic") == c)
                & (pl.col("competing_heuristic") == comp)
            )
            assert sub.height > 0, f"Missing rows for h={h}, c={c}, comp={comp}"

    # 3. Check specific event counts at horizon 10 for censoring="excluded", competing="excluded"
    # Excluded from cohort:
    # - Censored at t < 10: C (t=5), D (t=7) -> 2
    # - Competing at t <= 10: B (t=4), E (t=10) -> 2
    # Retained N = 4 (A, F, G, H).
    # Real positives = 2 (A, F), Real negatives = 2 (G, H).
    h10_both_exc = perf_df.filter(
        (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "excluded")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h10_both_exc["n"][0] == 4.0
    assert h10_both_exc["real_positives"][0] == 2.0  # A and F
    assert h10_both_exc["real_negatives"][0] == 2.0  # G and H

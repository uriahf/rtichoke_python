import numpy as np
import pytest
import polars as pl
from polarstate import prepare_event_table

from rtichoke.processing.adjustments import _extract_excluded_events
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
    prepare_binned_classification_data_times,
    _compute_population_event_risk_times,
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
    df = pl.DataFrame(
        {
            "reals": [1, 2, 0, 0, 2, 1],
            "times": [2.0, 4.0, 5.0, 7.0, 10.0, 10.0],
        }
    )
    event_table = prepare_event_table(df)

    # Unsorted horizons including early horizon (1.0 < min(times))
    horizons = [10.0, 1.0, 5.0]

    res = _extract_excluded_events(event_table, horizons, "excluded", "excluded")

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


def test_unconditional_population_event_risk_oracle(fixture_a_data):
    """Directly assert unconditional full-population KM/AJ event risk estimates across all combinations and horizons."""
    _, reals, times = fixture_a_data

    # Expected unconditional KM/AJ event risk estimates P(T <= h, event = 1)
    expected_unconditional_risk = {
        # Under censoring="excluded", "excluded":
        # At h=5: 7 retained subjects (A,B,C,D,E,F,G,H except none censored < 5, B excluded <= 5 -> N=7). Risk = 1/7 = 0.142857
        # At h=10: 4 retained subjects (A,F,G,H). Positives at t<=10 are A(t=2), F(t=10) -> 2 events out of 4 = 2/4 = 0.50
        ("excluded", "excluded", 5.0): 1 / 7,  # 0.142857
        ("excluded", "excluded", 10.0): 2 / 4,  # 0.500000
        ("excluded", "adjusted_as_negative", 5.0): 0.125,
        ("excluded", "adjusted_as_negative", 10.0): 2 / 6,  # 0.333333
        ("excluded", "adjusted_as_composite", 5.0): 0.25,
        ("excluded", "adjusted_as_composite", 10.0): 4 / 6,  # 0.666667
        ("excluded", "adjusted_as_censored", 5.0): 0.125,
        ("excluded", "adjusted_as_censored", 10.0): 3
        / 8,  # 0.375000 (KM with competing as censoring)
        ("adjusted", "excluded", 5.0): 1 / 7,  # 0.142857
        ("adjusted", "excluded", 10.0): 4 / 9,  # 0.444444
        ("adjusted", "adjusted_as_negative", 5.0): 0.125,
        ("adjusted", "adjusted_as_negative", 10.0): 5 / 16,  # 0.31250
        ("adjusted", "adjusted_as_composite", 5.0): 0.25,
        ("adjusted", "adjusted_as_composite", 10.0): 5 / 8,  # 0.625
        ("adjusted", "adjusted_as_censored", 5.0): 0.125,
        ("adjusted", "adjusted_as_censored", 10.0): 11 / 32,  # 0.34375
    }

    for (c, comp, h), exp_risk in expected_unconditional_risk.items():
        risk = _compute_population_event_risk_times(
            reals, times, horizon=h, censoring_heuristic=c, competing_heuristic=comp
        )
        assert pytest.approx(risk, abs=1e-5) == exp_risk, (
            f"Unconditional risk mismatch for c={c}, comp={comp}, h={h}"
        )


def test_fixture_a_retained_subjects_and_performance_producer_outputs(fixture_a_data):
    """Assert exact retained subject identities/counts and performance producer outputs for Fixture A."""
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

    # 1. Assert exact retained raw cohort size N under censoring="excluded"
    # At h=5.0: Subject C (censored at t=5.0) is retained because time == 5.0 is NOT < 5.0. N = 8.
    h5_exc_neg = perf_df.filter(
        (pl.col("fixed_time_horizon") == 5.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "adjusted_as_negative")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h5_exc_neg["n"][0] == 8.0

    # At h=10.0: Subjects C (t=5.0) and D (t=7.0) are censored at t < 10.0 and excluded. Retained N = 6.
    h10_exc_neg = perf_df.filter(
        (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "adjusted_as_negative")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h10_exc_neg["n"][0] == 6.0

    # At h=10.0 with both censoring="excluded" and competing="excluded":
    # Excluded: C (t=5, c), D (t=7, c), B (t=4, comp), E (t=10, comp) -> 4 excluded.
    # Retained: A (t=2, pos), F (t=10, pos), G (t=12, neg), H (t=14, neg) -> 4 retained.
    h10_both_exc = perf_df.filter(
        (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("censoring_heuristic") == "excluded")
        & (pl.col("competing_heuristic") == "excluded")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h10_both_exc["n"][0] == 4.0
    assert h10_both_exc["real_positives"][0] == 2.0  # A and F
    assert h10_both_exc["real_negatives"][0] == 2.0  # G and H

    # 2. Performance producer cutoff-aggregated expectations
    # Under censoring_heuristic="adjusted", the performance producer pools within-stratum AJ state risk estimates
    # over probability cutoff bins, yielding total cutoff-binned mass (e.g. n=13 across 10 threshold bins for single cohort).
    h10_adj_excl = perf_df.filter(
        (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("censoring_heuristic") == "adjusted")
        & (pl.col("competing_heuristic") == "excluded")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    assert h10_adj_excl["n"][0] == 13.0
    assert pytest.approx(h10_adj_excl["real_positives"][0], abs=1e-4) == 5.142857
    assert pytest.approx(h10_adj_excl["real_negatives"][0], abs=1e-4) == 7.857143

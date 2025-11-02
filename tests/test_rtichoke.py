"""
A module for tests
"""

from rtichoke.helpers.sandbox_observable_helpers import (
    create_aj_data,
    extract_aj_estimate_for_strata,
    assign_and_explode_polars,
    _aj_adjusted_events,
)

# from rtichoke import rtichoke
import polars as pl
from polars.testing import assert_frame_equal
import pytest

TIMES = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 4.3, 31.5]
REALS = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
TIME_HORIZONS = [10.0, 30.0, 50.0]
BREAKS: list[float] = [0.0, 0.5, 1.0]


def _expected(
    negatives: list[float],
    positives: list[float],
    competing: list[float],
    censored: list[float],
    censoring: str,
    competing_assump: str,
) -> pl.DataFrame:
    estimate_origin_enum = pl.Enum(["fixed_time_horizons", "event_table"])
    return pl.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1.0, 2.0, 3.0],
            "times": [1.0, 2.0, 3.0],
            "real_negatives_est": negatives,
            "real_positives_est": positives,
            "real_competing_est": competing,
            "real_censored_est": censored,
            "censoring_assumption": [censoring] * 3,
            "competing_assumption": [competing_assump] * 3,
            "estimate_origin": pl.Series(
                ["fixed_time_horizons"] * 3, dtype=estimate_origin_enum
            ),
        }
    )


@pytest.mark.parametrize(
    "censoring_assumption, competing_assumption, expected",
    [
        (
            "adjusted",
            "adjusted_as_negative",
            _expected(
                [4.0, 4.0, 8 / 3],
                [0.0, 0.0, 4 / 3],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                "adjusted",
                "adjusted_as_negative",
            ),
        ),
        (
            "excluded",
            "adjusted_as_negative",
            _expected(
                [4.0, 3.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                "excluded",
                "adjusted_as_negative",
            ),
        ),
        (
            "adjusted",
            "adjusted_as_censored",
            _expected(
                [5.0, 5.0, 10 / 3],
                [0.0, 0.0, 5 / 3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                "adjusted",
                "adjusted_as_censored",
            ),
        ),
        (
            "excluded",
            "adjusted_as_censored",
            _expected(
                [5.0, 4.0, 8 / 3],
                [0.0, 0.0, 4 / 3],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                "excluded",
                "adjusted_as_censored",
            ),
        ),
        (
            "adjusted",
            "adjusted_as_composite",
            _expected(
                [4.0, 4.0, 8 / 3],
                [1.0, 1.0, 7 / 3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                "adjusted",
                "adjusted_as_composite",
            ),
        ),
        (
            "excluded",
            "adjusted_as_composite",
            _expected(
                [4.0, 3.0, 2.0],
                [1.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                "excluded",
                "adjusted_as_composite",
            ),
        ),
        (
            "adjusted",
            "excluded",
            _expected(
                [4.0, 4.0, 8 / 3],
                [0.0, 0.0, 4 / 3],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                "adjusted",
                "excluded",
            ),
        ),
        (
            "excluded",
            "excluded",
            _expected(
                [4.0, 3.0, 2.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                "excluded",
                "excluded",
            ),
        ),
    ],
)
def test_create_aj_data(
    censoring_assumption: str,
    competing_assumption: str,
    expected: pl.DataFrame,
) -> None:
    df = pl.DataFrame(
        {
            "strata": ["group1"] * 5,
            "reals": [0, 1, 2, 1, 0],
            "times": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    horizons = [1.0, 2.0, 3.0]
    breaks = [0.0, 0.5, 1.0]

    result = create_aj_data(
        df,
        breaks=breaks,
        censoring_heuristic=censoring_assumption,
        competing_heuristic=competing_assumption,
        fixed_time_horizons=horizons,
    ).sort("fixed_time_horizon")

    assert_frame_equal(result, expected)


def test_extract_aj_estimate_for_strata_basic() -> None:
    df = pl.DataFrame(
        {
            "strata": ["group1"] * 5,
            "reals": [0, 1, 2, 1, 0],
            "times": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    horizons = [1.0, 2.0, 3.0]
    estimate_origin_enum = pl.Enum(["fixed_time_horizons", "event_table"])
    result = extract_aj_estimate_for_strata(df, horizons, full_event_table=False).sort(
        "fixed_time_horizon"
    )

    expected = pl.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "times": [1.0, 2.0, 3.0],
            "fixed_time_horizon": [1.0, 2.0, 3.0],
            "real_negatives_est": [4.0, 4.0, 8 / 3],
            "real_positives_est": [0.0, 0.0, 4 / 3],
            "real_competing_est": [1.0, 1.0, 1.0],
            "estimate_origin": pl.Series(
                ["fixed_time_horizons"] * 3, dtype=estimate_origin_enum
            ),
        }
    )

    assert_frame_equal(result, expected)


AJ_EXPECTED = {
    ("adjusted", "adjusted_as_negative"): [
        (8.88888888888889, 1.1111111111111112, 0.0),
        (5.555555555555555, 3.3333333333333335, 1.1111111111111112),
        (0.0, 7.407407407407407, 2.5925925925925926),
    ],
    ("adjusted", "adjusted_as_censored"): [
        (8.88888888888889, 1.1111111111111112, 0.0),
        (6.349206349206349, 3.6507936507936507, 0.0),
        (0.0, 10.0, 0.0),
    ],
    ("adjusted", "adjusted_as_composite"): [
        (8.88888888888889, 1.1111111111111112, 0.0),
        (5.555555555555555, 4.444444444444445, 0.0),
        (0.0, 10.0, 0.0),
    ],
    ("adjusted", "excluded"): [
        (8.88888888888889, 1.1111111111111112, 0.0),
        (5.625, 3.375, 0.0),
        (0.0, 8.0, 0.0),
    ],
    ("excluded", "adjusted_as_negative"): [
        (8.0, 1.0, 0.0),
        (5.0, 3.0, 1.0),
        (0.0, 6.0, 2.0),
    ],
    ("excluded", "adjusted_as_censored"): [
        (8.0, 1.0, 0.0),
        (5.714285714285714, 3.2857142857142856, 0.0),
        (0.0, 8.0, 0.0),
    ],
    ("excluded", "adjusted_as_composite"): [
        (8.0, 1.0, 0.0),
        (5.0, 4.0, 0.0),
        (0.0, 8.0, 0.0),
    ],
    ("excluded", "excluded"): [
        (8.0, 1.0, 0.0),
        (5.0, 3.0, 0.0),
        (0.0, 6.0, 0.0),
    ],
}

EXCLUDED_EXPECTED = {
    "adjusted": [0.0, 0.0, 0.0],
    "excluded": [1.0, 1.0, 2.0],
}

COMPETING_EXCLUDED = {
    "excluded": [0.0, 1.0, 2.0],
    "adjusted_as_negative": [0.0, 0.0, 0.0],
    "adjusted_as_censored": [0.0, 0.0, 0.0],
    "adjusted_as_composite": [0.0, 0.0, 0.0],
}


def _expected_aj_df(neg, pos, comp, include_comp=True):
    estimate_origin_enum = pl.Enum(["fixed_time_horizons", "event_table"])

    data = {
        "strata": ["group1"] * 3,
        "times": TIME_HORIZONS,
        "fixed_time_horizon": TIME_HORIZONS,
        "real_negatives_est": [neg[0], neg[1], neg[2]],
        "real_positives_est": [pos[0], pos[1], pos[2]],
        "estimate_origin": pl.Series(
            ["fixed_time_horizons"] * 3, dtype=estimate_origin_enum
        ),
    }
    if include_comp:
        data["real_competing_est"] = [comp[0], comp[1], comp[2]]

    cols = [
        "strata",
        "times",
        "fixed_time_horizon",
        "real_negatives_est",
        "real_positives_est",
    ]
    if include_comp:
        cols.append("real_competing_est")
    cols.append("estimate_origin")

    return pl.DataFrame(data)[cols]


def _expected_excluded_df(censoring, competing):
    return pl.DataFrame(
        {
            "strata": ["group1"] * 3,
            "fixed_time_horizon": TIME_HORIZONS,
            "real_censored_est": EXCLUDED_EXPECTED[censoring],
            "real_competing_est": COMPETING_EXCLUDED[competing],
            "times": TIME_HORIZONS,
        }
    )


@pytest.mark.parametrize(
    "censoring, competing",
    [
        (c, cc)
        for c in ["adjusted", "excluded"]
        for cc in [
            "adjusted_as_negative",
            "adjusted_as_censored",
            "adjusted_as_composite",
            "excluded",
        ]
    ],
)
def test_aj_adjusted_events(censoring: str, competing: str) -> None:
    df = pl.DataFrame(
        {"strata": ["group1"] * len(TIMES), "reals": REALS, "times": TIMES}
    )
    exploded = assign_and_explode_polars(df, TIME_HORIZONS)
    result = _aj_adjusted_events(
        df,
        BREAKS,
        exploded,
        censoring,
        competing,
        TIME_HORIZONS,
        full_event_table=False,
    ).sort("fixed_time_horizon")

    neg = [v[0] for v in AJ_EXPECTED[(censoring, competing)]]
    pos = [v[1] for v in AJ_EXPECTED[(censoring, competing)]]
    comp_vals = [v[2] for v in AJ_EXPECTED[(censoring, competing)]]
    include_comp = competing != "excluded"
    expected = _expected_aj_df(neg, pos, comp_vals, include_comp)
    assert_frame_equal(result, expected)

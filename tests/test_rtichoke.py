"""
A module for tests
"""

from rtichoke.helpers.sandbox_observable_helpers import (
    create_aj_data,
    extract_aj_estimate_for_strata,
)

# from rtichoke import rtichoke
import polars as pl
from polars.testing import assert_frame_equal
import pytest


def _expected(
    negatives: list[float],
    positives: list[float],
    competing: list[float],
    censored: list[float],
    censoring: str,
    competing_assump: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1.0, 2.0, 3.0],
            "real_negatives_est": negatives,
            "real_positives_est": positives,
            "real_competing_est": competing,
            "real_censored_est": censored,
            "censoring_assumption": [censoring] * 3,
            "competing_assumption": [competing_assump] * 3,
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

    result = create_aj_data(
        df,
        censoring_assumption=censoring_assumption,
        competing_assumption=competing_assumption,
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

    result = extract_aj_estimate_for_strata(df, horizons).sort("fixed_time_horizon")

    expected = pl.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1.0, 2.0, 3.0],
            "real_negatives_est": [4.0, 4.0, 8 / 3],
            "real_positives_est": [0.0, 0.0, 4 / 3],
            "real_competing_est": [1.0, 1.0, 1.0],
        }
    )

    assert_frame_equal(result, expected)

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


def test_create_aj_data() -> None:
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
        censoring_assumption="adjusted",
        competing_assumption="adjusted_as_negative",
        fixed_time_horizons=horizons,
    ).sort("fixed_time_horizon")

    expected = pl.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1.0, 2.0, 3.0],
            "real_negatives_est": [4.0, 4.0, 8 / 3],
            "real_positives_est": [0.0, 0.0, 4 / 3],
            "real_competing_est": [1.0, 1.0, 1.0],
            "real_censored_est": [0.0, 0.0, 0.0],
            "censoring_assumption": ["adjusted", "adjusted", "adjusted"],
            "competing_assumption": [
                "adjusted_as_negative",
                "adjusted_as_negative",
                "adjusted_as_negative",
            ],
        }
    )

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

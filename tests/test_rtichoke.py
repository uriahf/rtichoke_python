"""
A module for tests
"""

# from rtichoke import rtichoke

import pytest

polars = pytest.importorskip("polars")
lifelines = pytest.importorskip("lifelines")

from polars.testing import assert_frame_equal

from rtichoke.helpers.sandbox_observable_helpers import (
    create_aj_data,
    extract_aj_estimate_for_strata,
)


def test_create_aj_data() -> None:
    df = polars.DataFrame(
        {
            "strata": ["group1"] * 5,
            "reals": [0, 1, 2, 1, 0],
            "times": [5, 3, 1, 4, 2],
        }
    )
    horizons = [1, 2, 3]

    result = create_aj_data(
        df,
        censoring_assumption="adjusted",
        competing_assumption="adjusted_as_negative",
        fixed_time_horizons=horizons,
    ).sort("fixed_time_horizon")

    expected = polars.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1, 2, 3],
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
    df = polars.DataFrame(
        {
            "strata": ["group1"] * 5,
            "reals": [0, 1, 2, 1, 0],
            "times": [5, 3, 1, 4, 2],
        }
    )
    horizons = [1, 2, 3]

    result = extract_aj_estimate_for_strata(df, horizons).sort("fixed_time_horizon")

    expected = polars.DataFrame(
        {
            "strata": ["group1", "group1", "group1"],
            "fixed_time_horizon": [1, 2, 3],
            "real_negatives_est": [4.0, 4.0, 8 / 3],
            "real_positives_est": [0.0, 0.0, 4 / 3],
            "real_competing_est": [1.0, 1.0, 1.0],
        }
    )

    assert_frame_equal(result, expected)

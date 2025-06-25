"""
A module for tests
"""

# from rtichoke import rtichoke


import pytest

polars = pytest.importorskip("polars")
lifelines = pytest.importorskip("lifelines")

from polars.testing import assert_frame_equal

from rtichoke.helpers.sandbox_observable_helpers import create_aj_data


def test_create_aj_data() -> None:
    df = polars.DataFrame(
        {
            "strata": ["group1"] * 4,
            "reals": [0, 1, 0, 2],
            "times": [1, 2, 3, 4],
        }
    )
    horizons = [2, 4]

    result = create_aj_data(
        df,
        censoring_assumption="adjusted",
        competing_assumption="adjusted_as_negative",
        fixed_time_horizons=horizons,
    ).sort("fixed_time_horizon")

    expected = polars.DataFrame(
        {
            "strata": ["group1", "group1"],
            "fixed_time_horizon": [2, 4],
            "real_negatives_est": [8 / 3, 0.0],
            "real_positives_est": [4 / 3, 4 / 3],
            "real_competing_est": [0.0, 8 / 3],
            "real_censored_est": [0.0, 0.0],
            "censoring_assumption": ["adjusted", "adjusted"],
            "competing_assumption": [
                "adjusted_as_negative",
                "adjusted_as_negative",
            ],
        }
    )

    assert_frame_equal(result, expected)
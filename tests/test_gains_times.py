import polars as pl
import pytest

from rtichoke.discrimination.gains import (
    _get_gains_aj_estimates_times,
    _replace_gains_reference_data_times,
)


def test_gains_reference_uses_cutoff_zero_event_probability():
    performance_data = pl.DataFrame(
        {
            "reference_group": ["model", "model"],
            "fixed_time_horizon": [5.0, 5.0],
            "chosen_cutoff": [0.0, 1.0],
            "real_positives": [20.0, 5.0],
            "n": [100.0, 100.0],
        }
    )

    aj = _get_gains_aj_estimates_times(performance_data)

    assert aj.height == 1
    assert aj["aj_estimate"].item() == 0.2

    curve_list = {
        "fixed_time_horizons": [5.0],
        "reference_data": pl.DataFrame(),
    }
    fixed = _replace_gains_reference_data_times(curve_list, performance_data)
    perfect = fixed["reference_data"].filter(
        pl.col("reference_group") == "perfect_model"
    )

    # For p=0.2, the perfect gains curve has y=x/p, so at x=0.1 y=0.5.
    y_at_point_one = perfect.filter(pl.col("x") == 0.1)["y"].item()
    assert y_at_point_one == 0.5


def test_gains_reference_is_population_and_horizon_specific():
    performance_data = pl.DataFrame(
        {
            "reference_group": [
                "population_a",
                "population_a",
                "population_b",
                "population_b",
                "population_a",
                "population_a",
                "population_b",
                "population_b",
            ],
            "fixed_time_horizon": [5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "chosen_cutoff": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "real_positives": [20.0, 5.0, 40.0, 10.0, 30.0, 7.0, 60.0, 15.0],
            "n": [100.0] * 8,
        }
    )

    aj = _get_gains_aj_estimates_times(performance_data)
    assert aj.height == 4

    expected_aj = {
        ("population_a", 5.0): 0.2,
        ("population_b", 5.0): 0.4,
        ("population_a", 10.0): 0.3,
        ("population_b", 10.0): 0.6,
    }
    for (population, horizon), expected in expected_aj.items():
        value = aj.filter(
            (pl.col("reference_group") == population)
            & (pl.col("fixed_time_horizon") == horizon)
        )["aj_estimate"].item()
        assert value == pytest.approx(expected)

    curve_list = {
        "fixed_time_horizons": [5.0, 10.0],
        "reference_data": pl.DataFrame(),
    }
    reference_data = _replace_gains_reference_data_times(curve_list, performance_data)[
        "reference_data"
    ]

    expected_y_at_point_one = {
        ("perfect_model_population_a", 5.0): 0.1 / 0.2,
        ("perfect_model_population_b", 5.0): 0.1 / 0.4,
        ("perfect_model_population_a", 10.0): 0.1 / 0.3,
        ("perfect_model_population_b", 10.0): 0.1 / 0.6,
    }
    for (reference_group, horizon), expected in expected_y_at_point_one.items():
        y = reference_data.filter(
            (pl.col("reference_group") == reference_group)
            & (pl.col("fixed_time_horizon") == horizon)
            & (pl.col("x") == 0.1)
        )["y"].item()
        assert y == pytest.approx(expected)

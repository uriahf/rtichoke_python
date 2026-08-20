import polars as pl
import pytest

from rtichoke.processing.time_reference_lines import (
    _get_reference_aj_estimates_times,
    _replace_reference_data_times,
)


def _performance_data_with_boundary_drift() -> pl.DataFrame:
    return pl.DataFrame(
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


def test_reference_prevalence_uses_cutoff_zero_per_population_and_horizon():
    aj = _get_reference_aj_estimates_times(_performance_data_with_boundary_drift())

    assert aj.height == 4
    expected = {
        ("population_a", 5.0): 0.2,
        ("population_b", 5.0): 0.4,
        ("population_a", 10.0): 0.3,
        ("population_b", 10.0): 0.6,
    }
    for (population, horizon), value in expected.items():
        actual = aj.filter(
            (pl.col("reference_group") == population)
            & (pl.col("fixed_time_horizon") == horizon)
        )["aj_estimate"].item()
        assert actual == pytest.approx(value)


@pytest.mark.parametrize(
    ("curve", "reference_group", "x", "expected"),
    [
        ("precision recall", "random_guess_population_a", 0.1, 0.2),
        ("precision recall", "random_guess_population_b", 0.1, 0.4),
        ("lift", "perfect_model_population_a", 0.1, 5.0),
        ("lift", "perfect_model_population_b", 0.1, 2.5),
        ("decision", "treat_all_population_a", 0.1, 0.11111111111111112),
        ("decision", "treat_all_population_b", 0.1, 0.33333333333333337),
    ],
)
def test_prevalence_dependent_references_are_population_specific_at_horizon_five(
    curve, reference_group, x, expected
):
    curve_list = {
        "fixed_time_horizons": [5.0, 10.0],
        "reference_data": pl.DataFrame(),
    }
    reference_data = _replace_reference_data_times(
        curve_list,
        _performance_data_with_boundary_drift(),
        curve=curve,
    )["reference_data"]

    y = reference_data.filter(
        (pl.col("reference_group") == reference_group)
        & (pl.col("fixed_time_horizon") == 5.0)
        & (pl.col("x") == x)
    )["y"].item()

    assert y == pytest.approx(expected)


def test_precision_recall_reference_changes_with_horizon():
    curve_list = {
        "fixed_time_horizons": [5.0, 10.0],
        "reference_data": pl.DataFrame(),
    }
    reference_data = _replace_reference_data_times(
        curve_list,
        _performance_data_with_boundary_drift(),
        curve="precision recall",
    )["reference_data"]

    p5 = reference_data.filter(
        (pl.col("reference_group") == "random_guess_population_a")
        & (pl.col("fixed_time_horizon") == 5.0)
        & (pl.col("x") == 0.1)
    )["y"].item()
    p10 = reference_data.filter(
        (pl.col("reference_group") == "random_guess_population_a")
        & (pl.col("fixed_time_horizon") == 10.0)
        & (pl.col("x") == 0.1)
    )["y"].item()

    assert p5 == pytest.approx(0.2)
    assert p10 == pytest.approx(0.3)

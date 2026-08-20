import numpy as np
import polars as pl
import pytest

from rtichoke import (
    create_decision_curve_times,
    create_lift_curve_times,
    create_precision_recall_curve_times,
)
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


def _public_curve_inputs():
    probs = {
        "population_a": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "population_b": np.array([0.10, 0.30, 0.50, 0.70]),
    }
    reals = {
        "population_a": np.array([0, 1, 0, 1, 0, 1]),
        "population_b": np.array([0, 1, 1, 1]),
    }
    times = {
        "population_a": np.array([2.0, 3.0, 6.0, 7.0, 11.0, 12.0]),
        "population_b": np.array([1.0, 4.0, 8.0, 9.0]),
    }
    return probs, reals, times


def _trace(fig, name: str, visible: bool):
    matches = [trace for trace in fig.data if trace.name == name and trace.visible is visible]
    assert len(matches) == 1
    return matches[0]


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


def test_public_precision_recall_references_are_population_and_horizon_specific():
    probs, reals, times = _public_curve_inputs()
    fig = create_precision_recall_curve_times(
        probs, reals, times, fixed_time_horizons=[5.0, 10.0], by=0.1
    )

    a5 = _trace(fig, "random_guess_population_a", True)
    b5 = _trace(fig, "random_guess_population_b", True)
    a10 = _trace(fig, "random_guess_population_a", False)
    b10 = _trace(fig, "random_guess_population_b", False)

    assert float(a5.y[0]) == pytest.approx(1 / 3)
    assert float(b5.y[0]) == pytest.approx(1 / 2)
    assert float(a10.y[0]) == pytest.approx(1 / 2)
    assert float(b10.y[0]) == pytest.approx(3 / 4)


def test_public_lift_references_are_population_and_horizon_specific():
    probs, reals, times = _public_curve_inputs()
    fig = create_lift_curve_times(
        probs, reals, times, fixed_time_horizons=[5.0, 10.0], by=0.1
    )

    a5 = _trace(fig, "perfect_model_population_a", True)
    b5 = _trace(fig, "perfect_model_population_b", True)
    a10 = _trace(fig, "perfect_model_population_a", False)
    b10 = _trace(fig, "perfect_model_population_b", False)

    assert float(a5.y[0]) == pytest.approx(3.0)
    assert float(b5.y[0]) == pytest.approx(2.0)
    assert float(a10.y[0]) == pytest.approx(2.0)
    assert float(b10.y[0]) == pytest.approx(4 / 3)


def test_public_decision_references_are_population_and_horizon_specific():
    probs, reals, times = _public_curve_inputs()
    fig = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        by=0.1,
        min_p_threshold=0.1,
        max_p_threshold=0.9,
    )

    a5 = _trace(fig, "treat_all_population_a", True)
    b5 = _trace(fig, "treat_all_population_b", True)
    a10 = _trace(fig, "treat_all_population_a", False)
    b10 = _trace(fig, "treat_all_population_b", False)

    # Reference x-grid starts at 0.1 after threshold filtering.
    assert float(a5.x[0]) == pytest.approx(0.1)
    assert float(b5.x[0]) == pytest.approx(0.1)
    assert float(a10.x[0]) == pytest.approx(0.1)
    assert float(b10.x[0]) == pytest.approx(0.1)

    def treat_all(p):
        x = 0.1
        return p - (1 - p) * x / (1 - x)

    assert float(a5.y[0]) == pytest.approx(treat_all(1 / 3))
    assert float(b5.y[0]) == pytest.approx(treat_all(1 / 2))
    assert float(a10.y[0]) == pytest.approx(treat_all(1 / 2))
    assert float(b10.y[0]) == pytest.approx(treat_all(3 / 4))

import numpy as np
import polars as pl
import pytest

from rtichoke import create_precision_recall_curve_times
from rtichoke.processing.time_reference_lines import _replace_reference_data_times

HORIZONS = [5.0, 10.0]
HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]


def _equal_risk_performance_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reference_group": [
                "Population A",
                "Population B",
                "Population A",
                "Population B",
            ],
            "fixed_time_horizon": [5.0, 5.0, 10.0, 10.0],
            "chosen_cutoff": [0.0, 0.0, 0.0, 0.0],
            "real_positives": [2.0, 2.0, 3.0, 3.0],
            "n": [6.0, 6.0, 6.0, 6.0],
        }
    )


def _nonempty_random_reference_names(fig) -> set[str]:
    return {
        trace.name
        for trace in fig.data
        if "random_guess" in trace.name and len(trace.x) > 0
    }


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        ("roc", {"random_guess"}),
        (
            "precision recall",
            {"random_guess_Population A", "random_guess_Population B"},
        ),
        (
            "gains",
            {
                "random_guess",
                "perfect_model_Population A",
                "perfect_model_Population B",
            },
        ),
        (
            "lift",
            {
                "random_guess",
                "perfect_model_Population A",
                "perfect_model_Population B",
            },
        ),
        (
            "decision",
            {"treat_none", "treat_all_Population A", "treat_all_Population B"},
        ),
        (
            "interventions avoided",
            {"treat_all", "treat_none_Population A", "treat_none_Population B"},
        ),
    ],
)
def test_equal_risk_populations_keep_population_scoped_references(curve, expected):
    curve_list = {"fixed_time_horizons": HORIZONS, "reference_data": pl.DataFrame()}
    reference_data = _replace_reference_data_times(
        curve_list,
        _equal_risk_performance_data(),
        curve=curve,
        multiple_populations=True,
    )["reference_data"]

    for horizon in HORIZONS:
        actual = set(
            reference_data.filter(pl.col("fixed_time_horizon") == horizon)[
                "reference_group"
            ].unique()
        )
        assert actual == expected


def test_public_equal_risk_populations_keep_distinct_precision_recall_references():
    probs = {
        "Population A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "Population B": np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90]),
    }
    reals = {
        "Population A": np.array([1, 1, 0, 0, 0, 0]),
        "Population B": np.array([1, 1, 0, 0, 0, 0]),
    }
    times = {
        "Population A": np.array([3.0, 8.0, 12.0, 13.0, 14.0, 15.0]),
        "Population B": np.array([4.0, 9.0, 12.0, 13.0, 14.0, 15.0]),
    }

    fig = create_precision_recall_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )

    assert _nonempty_random_reference_names(fig) == {
        "random_guess_Population A",
        "random_guess_Population B",
    }


def test_public_multiple_models_still_share_population_reference():
    probs = {
        "Model A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
        "Model B": np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90]),
    }
    reals = np.array([1, 1, 0, 0, 0, 0])
    times = np.array([3.0, 8.0, 12.0, 13.0, 14.0, 15.0])

    fig = create_precision_recall_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )

    assert _nonempty_random_reference_names(fig) == {"random_guess"}

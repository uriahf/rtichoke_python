import numpy as np
import polars as pl
import pytest

from rtichoke import (
    create_decision_curve,
    create_gains_curve,
    create_lift_curve,
    create_precision_recall_curve,
    create_roc_curve,
    prepare_performance_data,
)
from rtichoke.calibration.calibration import _create_calibration_curve_list
from rtichoke.processing.plotly_helper_functions import _create_reference_lines_data

PROBS_A = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95, 0.25, 0.85])
PROBS_B = np.array([0.10, 0.20, 0.40, 0.60, 0.70, 0.90, 0.30, 0.80])
REALS_EQUAL = np.array([0, 0, 0, 0, 1, 1, 1, 1])
REALS_LOW = np.array([0, 0, 0, 0, 0, 0, 1, 1])
REALS_HIGH = np.array([0, 0, 1, 1, 1, 1, 1, 1])
CUSTOM_COLORS = ["#111111", "#222222"]

CURVE_CREATORS = [
    create_roc_curve,
    create_precision_recall_curve,
    create_gains_curve,
    create_lift_curve,
    create_decision_curve,
]


def _reference_groups(curve: str, prevalences: dict[str, float]) -> set[str]:
    aj = pl.DataFrame(
        {
            "reference_group": list(prevalences),
            "aj_estimate": list(prevalences.values()),
        }
    )
    refs = _create_reference_lines_data(
        curve=curve,
        aj_estimates_from_performance_data=aj,
        multiple_populations=len(prevalences) > 1,
    )
    return set(refs["reference_group"].unique().to_list())


def test_one_model_one_population_is_one_evaluation():
    performance_data = prepare_performance_data(
        probs={"model": PROBS_A},
        reals=REALS_EQUAL,
        by=0.25,
    )

    assert performance_data["reference_group"].unique().to_list() == ["model"]

    fig = create_roc_curve(
        probs={"model": PROBS_A},
        reals=REALS_EQUAL,
        by=0.25,
    )
    model_trace = next(trace for trace in fig.data if trace.name == "model")
    assert model_trace.line.color == "#000000"


@pytest.mark.parametrize("creator", CURVE_CREATORS)
def test_multiple_models_share_one_population_context(creator):
    fig = creator(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_EQUAL,
        by=0.25,
        color_values=CUSTOM_COLORS,
    )

    model_traces = [trace for trace in fig.data if trace.showlegend is True]
    assert {trace.name for trace in model_traces} == {"Model A", "Model B"}
    assert [trace.line.color for trace in model_traces] == CUSTOM_COLORS


def test_interventions_avoided_keeps_model_grouping_and_colors():
    fig = create_decision_curve(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_EQUAL,
        decision_type="interventions avoided",
        by=0.25,
        color_values=CUSTOM_COLORS,
    )

    model_traces = [trace for trace in fig.data if trace.showlegend is True]
    assert {trace.name for trace in model_traces} == {"Model A", "Model B"}
    assert [trace.line.color for trace in model_traces] == CUSTOM_COLORS


def test_multiple_models_share_population_scoped_references():
    prevalence = {"population": 0.5}

    assert _reference_groups("roc", prevalence) == {"random_guess"}
    assert _reference_groups("precision recall", prevalence) == {"random_guess"}
    assert _reference_groups("gains", prevalence) == {
        "random_guess",
        "perfect_model",
    }
    assert _reference_groups("lift", prevalence) == {
        "random_guess",
        "perfect_model",
    }
    assert _reference_groups("decision", prevalence) == {
        "treat_none",
        "treat_all",
    }
    assert _reference_groups("interventions avoided", prevalence) == {
        "treat_all",
        "treat_none",
    }


def test_different_prevalence_populations_own_distinct_references():
    performance_data = prepare_performance_data(
        probs={"Population low": PROBS_A, "Population high": PROBS_A},
        reals={"Population low": REALS_LOW, "Population high": REALS_HIGH},
        by=0.25,
    )
    assert set(performance_data["reference_group"].unique().to_list()) == {
        "Population low",
        "Population high",
    }

    prevalence = {"Population low": 0.25, "Population high": 0.75}
    assert _reference_groups("roc", prevalence) == {"random_guess"}
    assert _reference_groups("precision recall", prevalence) == {
        "random_guess_Population low",
        "random_guess_Population high",
    }
    assert _reference_groups("gains", prevalence) == {
        "random_guess",
        "perfect_model_Population low",
        "perfect_model_Population high",
    }
    assert _reference_groups("lift", prevalence) == {
        "random_guess",
        "perfect_model_Population low",
        "perfect_model_Population high",
    }
    assert _reference_groups("decision", prevalence) == {
        "treat_none",
        "treat_all_Population low",
        "treat_all_Population high",
    }
    assert _reference_groups("interventions avoided", prevalence) == {
        "treat_all",
        "treat_none_Population low",
        "treat_none_Population high",
    }


def test_equal_prevalence_populations_remain_distinct_contexts():
    aj = pl.DataFrame(
        {
            "reference_group": ["Population A", "Population B"],
            "aj_estimate": [0.5, 0.5],
        }
    )
    refs = _create_reference_lines_data(
        curve="precision recall",
        aj_estimates_from_performance_data=aj,
        multiple_populations=True,
    )

    assert set(refs["reference_group"].unique().to_list()) == {
        "random_guess_Population A",
        "random_guess_Population B",
    }

    a = refs.filter(pl.col("reference_group") == "random_guess_Population A")["y"]
    b = refs.filter(pl.col("reference_group") == "random_guess_Population B")["y"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_paired_inputs_are_generic_reference_groups():
    pairs = {
        "Model A @ Population A": PROBS_A,
        "Model B @ Population B": PROBS_B,
    }
    reals = {
        "Model A @ Population A": REALS_LOW,
        "Model B @ Population B": REALS_HIGH,
    }
    performance_data = prepare_performance_data(probs=pairs, reals=reals, by=0.25)

    assert set(performance_data["reference_group"].unique().to_list()) == set(pairs)
    assert "model" not in performance_data.columns
    assert "population" not in performance_data.columns


def test_calibration_keeps_grouping_and_one_global_identity_line():
    multi_model = _create_calibration_curve_list(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_EQUAL,
    )
    assert multi_model["performance_type"] == ["multiple models"]
    assert set(
        multi_model["calibration_bins_dat"]["reference_group"].unique().to_list()
    ) == {
        "Model A",
        "Model B",
    }
    assert multi_model["reference_data"].height == 101

    multi_population = _create_calibration_curve_list(
        probs={"Population low": PROBS_A, "Population high": PROBS_B},
        reals={"Population low": REALS_LOW, "Population high": REALS_HIGH},
    )
    assert multi_population["performance_type"] == ["multiple populations"]
    assert set(
        multi_population["calibration_bins_dat"]["reference_group"].unique().to_list()
    ) == {"Population low", "Population high"}
    assert multi_population["reference_data"].height == 101


def test_performance_table_input_keeps_current_grouping_semantics():
    models = prepare_performance_data(
        probs={"Model A": PROBS_A, "Model B": PROBS_B},
        reals=REALS_EQUAL,
        by=0.25,
    )
    populations = prepare_performance_data(
        probs={"Population A": PROBS_A, "Population B": PROBS_B},
        reals={"Population A": REALS_LOW, "Population B": REALS_HIGH},
        by=0.25,
    )

    assert set(models["reference_group"].unique().to_list()) == {
        "Model A",
        "Model B",
    }
    assert set(populations["reference_group"].unique().to_list()) == {
        "Population A",
        "Population B",
    }

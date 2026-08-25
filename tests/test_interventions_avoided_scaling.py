import numpy as np
import polars as pl
import pytest

from rtichoke import (
    create_decision_curve,
    plot_decision_curve,
    prepare_performance_data,
)
from rtichoke.processing.plotly_helper_functions import _create_reference_lines_data


PROBS = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
REALS = np.array([1, 0, 1, 0, 1, 0, 0, 1])
THRESHOLDS = [0.25, 0.5, 0.75]
EXPECTED_IA = [-25.0, 0.0, 25.0]
OLD_BUGGY_IA = [12.125, 24.75, 37.375]
EXPECTED_TN = [1.0, 2.0, 3.0]
EXPECTED_FN = [1.0, 2.0, 3.0]


def _performance_rows() -> pl.DataFrame:
    data = prepare_performance_data(probs=PROBS, reals=REALS, by=0.25)
    data = data.filter(pl.col("chosen_cutoff").is_in(THRESHOLDS))
    return data.sort("chosen_cutoff")


def test_static_interventions_avoided_matches_r_definition_per_100():
    """Expected values characterize the current R static IA definition."""
    rows = _performance_rows()

    np.testing.assert_allclose(rows["true_negatives"].to_numpy(), EXPECTED_TN)
    np.testing.assert_allclose(rows["false_negatives"].to_numpy(), EXPECTED_FN)
    actual = rows["net_benefit_interventions_avoided"].to_numpy()
    np.testing.assert_allclose(actual, EXPECTED_IA)
    assert not np.allclose(actual, OLD_BUGGY_IA)


def test_static_interventions_avoided_count_and_nb_forms_are_equivalent():
    rows = _performance_rows()
    prevalence = float(REALS.mean())

    for row in rows.iter_rows(named=True):
        threshold = float(row["chosen_cutoff"])
        tn = float(row["true_negatives"])
        fn = float(row["false_negatives"])
        n = float(row["n"])
        net_benefit = float(row["net_benefit"])
        net_benefit_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        from_counts = 100 * (tn / n - fn / n * (1 - threshold) / threshold)
        from_nb = 100 * (net_benefit - net_benefit_all) * (1 - threshold) / threshold
        actual = float(row["net_benefit_interventions_avoided"])

        assert actual == pytest.approx(from_counts)
        assert from_counts == pytest.approx(from_nb)


def test_interventions_avoided_model_and_references_use_per_100_units():
    rows = _performance_rows()
    aj = pl.DataFrame({"reference_group": ["population"], "aj_estimate": [0.5]})
    refs = _create_reference_lines_data(
        curve="interventions avoided",
        aj_estimates_from_performance_data=aj,
        multiple_populations=False,
        min_p_threshold=0.25,
        max_p_threshold=0.75,
    )

    treat_all = refs.filter(pl.col("reference_group") == "treat_all")
    assert np.allclose(treat_all["y"].to_numpy(), 0.0)

    is_treat_none = pl.col("reference_group") == "treat_none"
    is_test_threshold = pl.col("x").is_in(THRESHOLDS)
    treat_none = refs.filter(is_treat_none & is_test_threshold).sort("x")
    expected_treat_none = [-100.0, 0.0, 100.0 / 3.0]
    np.testing.assert_allclose(treat_none["y"].to_numpy(), expected_treat_none)
    np.testing.assert_allclose(
        rows["net_benefit_interventions_avoided"].to_numpy(), EXPECTED_IA
    )


def test_static_interventions_avoided_public_apis_are_unchanged():
    data = prepare_performance_data(probs=PROBS, reals=REALS, by=0.25)
    created = create_decision_curve(
        probs=PROBS,
        reals=REALS,
        decision_type="interventions avoided",
        by=0.25,
        min_p_threshold=0.25,
        max_p_threshold=0.75,
    )
    plotted = plot_decision_curve(
        data,
        decision_type="interventions avoided",
        min_p_threshold=0.25,
        max_p_threshold=0.75,
    )

    assert created.__class__.__name__ == "Figure"
    assert plotted.__class__.__name__ == "Figure"

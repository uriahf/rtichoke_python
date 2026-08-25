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
EXPECTED_COUNTS = {
    0.25: (1.0, 1.0),
    0.5: (2.0, 2.0),
    0.75: (3.0, 3.0),
}
# These values match the current R static Interventions Avoided definition.
EXPECTED_IA = {0.25: -25.0, 0.5: 0.0, 0.75: 25.0}
OLD_BUGGY_IA = {0.25: 12.125, 0.5: 24.75, 0.75: 37.375}


def _performance_rows() -> pl.DataFrame:
    performance_data = prepare_performance_data(probs=PROBS, reals=REALS, by=0.25)
    return performance_data.filter(pl.col("chosen_cutoff").is_in(THRESHOLDS)).sort(
        "chosen_cutoff"
    )


def test_static_interventions_avoided_matches_r_definition_per_100():
    rows = _performance_rows()

    for row in rows.iter_rows(named=True):
        threshold = float(row["chosen_cutoff"])
        tn, fn = EXPECTED_COUNTS[threshold]
        assert float(row["true_negatives"]) == pytest.approx(tn)
        assert float(row["false_negatives"]) == pytest.approx(fn)

        expected = EXPECTED_IA[threshold]
        actual = float(row["net_benefit_interventions_avoided"])
        assert actual == pytest.approx(expected)
        assert actual != pytest.approx(OLD_BUGGY_IA[threshold])


def test_static_interventions_avoided_count_and_net_benefit_forms_are_equivalent():
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
        from_net_benefit = (
            100
            * (net_benefit - net_benefit_all)
            * (1 - threshold)
            / threshold
        )

        assert float(row["net_benefit_interventions_avoided"]) == pytest.approx(
            from_counts
        )
        assert from_counts == pytest.approx(from_net_benefit)


def test_interventions_avoided_model_and_references_use_same_per_100_unit():
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

    treat_none = refs.filter(
        (pl.col("reference_group") == "treat_none") & pl.col("x").is_in(THRESHOLDS)
    ).sort("x")
    expected_treat_none = np.array(
        [
            100 * (1 - 0.5 - 0.5 * (1 - threshold) / threshold)
            for threshold in THRESHOLDS
        ]
    )
    np.testing.assert_allclose(treat_none["y"].to_numpy(), expected_treat_none)

    np.testing.assert_allclose(
        rows["net_benefit_interventions_avoided"].to_numpy(),
        np.array([EXPECTED_IA[threshold] for threshold in THRESHOLDS]),
    )


def test_static_interventions_avoided_public_apis_are_unchanged():
    performance_data = prepare_performance_data(probs=PROBS, reals=REALS, by=0.25)

    created = create_decision_curve(
        probs=PROBS,
        reals=REALS,
        decision_type="interventions avoided",
        by=0.25,
        min_p_threshold=0.25,
        max_p_threshold=0.75,
    )
    plotted = plot_decision_curve(
        performance_data,
        decision_type="interventions avoided",
        min_p_threshold=0.25,
        max_p_threshold=0.75,
    )

    assert created.__class__.__name__ == "Figure"
    assert plotted.__class__.__name__ == "Figure"

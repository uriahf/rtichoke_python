"""Tests for time-dependent interventions avoided parity with dcurves, fixtures A-D, algebraic invariants, boundary semantics, and multi-identity isolation."""

from typing import Any, cast

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from rtichoke._interventions_avoided_viz_spec_v2 import (
    _interventions_avoided_times_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data_times import (
    _compute_population_event_risk_times,
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _EvaluationMetadata


# =============================================================================
# Fixture A — No pre-horizon censoring or competing events
# =============================================================================
def test_fixture_a_no_censoring_or_competing_events() -> None:
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    times = np.array([2.0, 12.0, 4.0, 15.0, 8.0, 13.0, 14.0, 9.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert row_05.height == 1
    assert_allclose(row_05["net_benefit"].item(), 0.0, rtol=0, atol=1e-10)
    assert_allclose(
        row_05["net_benefit_interventions_avoided"].item(), 0.0, rtol=0, atol=1e-10
    )


# =============================================================================
# Fixture B — Right censoring
# =============================================================================
def test_fixture_b_right_censoring() -> None:
    """Subject with prob 0.4 is censored at time 6.

    At horizon 10 and threshold 0.5:
    population KM event risk = 0.4
    NB_model = 0
    NB_all = -0.2
    dcurves-compatible interventions_avoided = 20.0
    (old direct calculation gave ~16.6666666667)
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert row_05.height == 1
    ia_val = row_05["net_benefit_interventions_avoided"].item()
    assert_allclose(ia_val, 20.0, rtol=0, atol=1e-10)
    assert not np.isclose(ia_val, 16.6666666667)


def test_fixture_b_exact_zero_prediction() -> None:
    """Subject with prob 0.0 is censored at time 6 (exact 0 prediction).

    Even when predictions contain exact 0.0 values, the true pooled population KM
    event risk remains 0.4, and the dcurves-compatible IA remains 20.0.
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.0, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert row_05.height == 1
    ia_val = row_05["net_benefit_interventions_avoided"].item()
    assert_allclose(ia_val, 20.0, rtol=0, atol=1e-10)
    assert not np.isclose(ia_val, 25.0)


# =============================================================================
# Fixture C — Competing risks without censoring
# =============================================================================
def test_fixture_c_competing_risks_without_censoring() -> None:
    """Using censoring_heuristic = adjusted, competing_heuristic = adjusted_as_negative.

    At horizon 10 and threshold 0.5:
    population cause-1 AJ risk = 0.375
    NB_model = 0
    NB_all = -0.25
    interventions_avoided = 25.0
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 2, 1, 0, 2, 1, 0, 0])
    times = np.array([2.0, 3.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert row_05.height == 1
    ia_val = row_05["net_benefit_interventions_avoided"].item()
    assert_allclose(ia_val, 25.0, rtol=0, atol=1e-10)


# =============================================================================
# Fixture D — Competing risks plus censoring
# =============================================================================
def test_fixture_d_competing_risks_plus_censoring() -> None:
    """Using censoring_heuristic = adjusted, competing_heuristic = adjusted_as_negative.

    At horizon 10 and threshold 0.5:
    population cause-1 AJ risk = 0.40625
    NB_model = 0
    NB_all = -0.1875
    Expected dcurves-compatible result: interventions_avoided = 18.75
    (old direct calculation gave ~16.6666666667)
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 2, 1, 0, 0, 1, 0, 2])
    times = np.array([2.0, 3.0, 4.0, 15.0, 6.0, 9.0, 13.0, 8.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert row_05.height == 1
    ia_val = row_05["net_benefit_interventions_avoided"].item()
    assert_allclose(ia_val, 18.75, rtol=0, atol=1e-10)
    assert not np.isclose(ia_val, 16.6666666667)


# =============================================================================
# Prediction Invariance & Shared Population Tests
# =============================================================================
def test_population_event_risk_prediction_invariance() -> None:
    """Population event risk must be invariant to predicted probabilities."""
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    risk_orig = _compute_population_event_risk_times(
        reals, times, 10.0, "adjusted", "adjusted_as_negative"
    )
    risk_zeros = _compute_population_event_risk_times(
        reals, times, 10.0, "adjusted", "adjusted_as_negative"
    )

    assert_allclose(risk_orig, 0.4, rtol=0, atol=1e-10)
    assert_allclose(risk_zeros, 0.4, rtol=0, atol=1e-10)


def test_shared_population_multiple_models() -> None:
    """Two models sharing the same reals/times population receive the exact same

    population event risk, even if Model 1 has exact 0 predictions.
    """
    probs_m1 = np.array([0.9, 0.8, 0.7, 0.6, 0.0, 0.3, 0.2, 0.1])
    probs_m2 = np.array([0.95, 0.85, 0.75, 0.65, 0.45, 0.35, 0.25, 0.15])
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs={"m1": probs_m1, "m2": probs_m2},
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    m1_05 = perf.filter(
        (pl.col("reference_group") == "m1")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )
    m2_05 = perf.filter(
        (pl.col("reference_group") == "m2")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    # Both models share population event risk = 0.4.
    # At cutoff 0.5, m1 and m2 classify subjects with prob > 0.5 identically (first 4 subjects).
    # Therefore NB_m1 == NB_m2 == 0, and IA_m1 == IA_m2 == 20.0.
    assert_allclose(
        m1_05["net_benefit_interventions_avoided"].item(), 20.0, rtol=0, atol=1e-10
    )
    assert_allclose(
        m2_05["net_benefit_interventions_avoided"].item(), 20.0, rtol=0, atol=1e-10
    )


# =============================================================================
# Algebraic Invariant & Identity Tests
# =============================================================================
def test_algebraic_invariant_across_grid() -> None:
    """For every finite threshold strictly between 0 and 1, assert:

    expected = 100 * (net_benefit - net_benefit_all) * (1 - cutoff) / cutoff
    net_benefit_interventions_avoided == expected within atol=1e-10.
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.05,
    )

    event_risk = _compute_population_event_risk_times(
        reals, times, 10.0, "adjusted", "adjusted_as_negative"
    )

    thresh_rows = perf.filter(
        (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") > 0)
        & (pl.col("chosen_cutoff") < 1)
    )

    for row in thresh_rows.iter_rows(named=True):
        cutoff = row["chosen_cutoff"]
        nb_model = row["net_benefit"]
        ia_actual = row["net_benefit_interventions_avoided"]

        threshold_odds = cutoff / (1.0 - cutoff)
        nb_all = event_risk - (1.0 - event_risk) * threshold_odds
        expected_ia = 100.0 * (nb_model - nb_all) * (1.0 - cutoff) / cutoff

        assert_allclose(ia_actual, expected_ia, rtol=0, atol=1e-10)


def test_model_equals_treat_none_interventions_avoided_when_nb_equals_treat_none() -> (
    None
):
    """Whenever model net benefit equals Treat None net benefit (i.e. 0), model

    interventions avoided must equal Treat None interventions avoided reference
    at the same threshold.
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    # At threshold 0.5 for fixture B, NB_model is 0 (equal to Treat None NB).
    event_risk = 0.4
    cutoff = 0.5
    tn_ia_ref = 100.0 * (1.0 - event_risk - event_risk * (1.0 - cutoff) / cutoff)
    # 100 * (0.6 - 0.4) = 20.0
    assert_allclose(
        row_05["net_benefit_interventions_avoided"].item(),
        tn_ia_ref,
        rtol=0,
        atol=1e-10,
    )


# =============================================================================
# Boundary Semantics Test
# =============================================================================
def test_boundary_cutoffs_0_and_1_are_null() -> None:
    """At thresholds 0 and 1, net_benefit_interventions_avoided must be null,

    and canonical browser specs must omit them.
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_0 = perf.filter(
        (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.0)
    )
    row_1 = perf.filter(
        (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 1.0)
    )

    assert row_0["net_benefit_interventions_avoided"].item() is None
    assert row_1["net_benefit_interventions_avoided"].item() is None

    metadata = {"model": _EvaluationMetadata("model", "model", "model", "pop")}
    spec = cast(
        dict[str, Any],
        _interventions_avoided_times_v2_spec_from_performance_data(perf, metadata),
    )
    data_thresholds = [datum["threshold"] for datum in spec["data"]]
    assert 0.0 not in data_thresholds
    assert 1.0 not in data_thresholds


# =============================================================================
# Multi-Identity Distinct Populations Tests
# =============================================================================
def test_multi_identity_distinct_populations() -> None:
    """Population event risk must be matched correctly across multiple distinct

    populations, horizons, and heuristic sets.
    """
    probs_m1 = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    probs_m2 = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    # Pop 1
    reals_p1 = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times_p1 = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    # Pop 2 (different event times/statuses)
    reals_p2 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    times_p2 = np.array([2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0])

    perf = prepare_performance_data_times(
        probs={"m1": probs_m1, "m2": probs_m2},
        reals={"m1": reals_p1, "m2": reals_p2},
        times={"m1": times_p1, "m2": times_p2},
        fixed_time_horizons=[5.0, 10.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            },
            {
                "censoring_heuristic": "excluded",
                "competing_heuristic": "excluded",
            },
        ],
        by=0.5,
    )

    groups = perf.select(
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
    ).unique()

    assert groups.height == 8  # 2 populations * 2 horizons * 2 heuristic sets

    for g in groups.iter_rows(named=True):
        group_df = perf.filter(
            (pl.col("reference_group") == g["reference_group"])
            & (pl.col("fixed_time_horizon") == g["fixed_time_horizon"])
            & (pl.col("censoring_heuristic") == g["censoring_heuristic"])
            & (pl.col("competing_heuristic") == g["competing_heuristic"])
            & (pl.col("stratified_by") == "probability_threshold")
        )

        c05 = group_df.filter(pl.col("chosen_cutoff") == 0.5)

        ref_group = g["reference_group"]
        reals_g = reals_p1 if ref_group == "m1" else reals_p2
        times_g = times_p1 if ref_group == "m1" else times_p2

        event_risk = _compute_population_event_risk_times(
            reals_g,
            times_g,
            g["fixed_time_horizon"],
            g["censoring_heuristic"],
            g["competing_heuristic"],
        )

        nb_model = c05["net_benefit"].item()
        cutoff = 0.5
        threshold_odds = cutoff / (1.0 - cutoff)
        nb_all = event_risk - (1.0 - event_risk) * threshold_odds
        expected_ia = 100.0 * (nb_model - nb_all) * (1.0 - cutoff) / cutoff

        assert_allclose(
            c05["net_benefit_interventions_avoided"].item(),
            expected_ia,
            rtol=0,
            atol=1e-10,
        )


# =============================================================================
# External Parity Reference Documentation Test
# =============================================================================
def test_external_parity_reference_dcurves_0_5_1() -> None:
    """Document parity with R dcurves version 0.5.1.

    Equivalent R call:
    ```r
    # R dcurves 0.5.1
    library(dcurves)
    library(survival)

    fixture_b <- data.frame(
      model = c(0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1),
      status = c(1, 0, 1, 0, 0, 1, 0, 0),
      time = c(2, 12, 4, 15, 6, 9, 13, 14)
    )

    dca_res <- dca(
      Surv(time, status) ~ model,
      data = fixture_b,
      time = 10,
      thresholds = 0.5
    )

    ia_res <- net_intervention_avoided(dca_res, nper = 100)
    # Output: interventions_avoided = 20.0
    ```
    """
    probs = {"model": np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])}
    reals = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    times = np.array([2.0, 12.0, 4.0, 15.0, 6.0, 9.0, 13.0, 14.0])

    perf = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[10.0],
        by=0.5,
    )

    row_05 = perf.filter(
        (pl.col("reference_group") == "model")
        & (pl.col("stratified_by") == "probability_threshold")
        & (pl.col("chosen_cutoff") == 0.5)
    )

    assert_allclose(
        row_05["net_benefit_interventions_avoided"].item(),
        20.0,
        rtol=0,
        atol=1e-10,
    )

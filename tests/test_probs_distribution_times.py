"""Tests for internal time-dependent prediction distribution data producer."""

import numpy as np
import polars as pl
import pytest

from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.performance_data.probs_distribution import (
    _prepare_probs_distribution_data,
    _prepare_probs_distribution_data_times,
)


def test_static_probs_distribution_unchanged():
    """Verify that static _prepare_probs_distribution_data behavior is completely unchanged."""
    probs = {"m1": np.array([0.0, 0.2, 0.5, 0.8, 1.0])}
    reals = np.array([0, 1, 0, 1, 0])

    res = _prepare_probs_distribution_data(probs, reals, by=0.1)

    assert set(res.keys()) == {"bins", "operating_points", "rank_bins"}
    bins = res["bins"]
    assert bins["n_positive"].sum() == 2
    assert bins["n_negative"].sum() == 3


def test_raw_histogram_counts_conserve_n():
    """Verify that raw prediction histogram counts conserve raw input N and remain unadjusted."""
    probs = {"m1": np.array([0.1, 0.4, 0.6, 0.9, 0.95])}
    reals = np.array([0, 1, 2, 0, 0])  # 2 neg, 1 pos, 1 comp, 1 neg = 5 obs total
    times = np.array([1.0, 2.0, 1.5, 3.0, 0.5])
    horizons = [2.0]

    res = _prepare_probs_distribution_data_times(
        probs=probs, reals=reals, times=times, fixed_time_horizons=horizons, by=0.1
    )

    bins = res["bins"]
    rank_bins = res["rank_bins"]

    # Raw counts must sum to raw N = 5
    assert bins["n_observations"].sum() == 5
    assert bins["n_real_positive"].sum() == 1
    assert bins["n_real_negative"].sum() == 3
    assert bins["n_real_competing"].sum() == 1

    assert rank_bins["n_observations"].sum() == 5
    assert rank_bins["n_real_positive"].sum() == 1
    assert rank_bins["n_real_negative"].sum() == 3
    assert rank_bins["n_real_competing"].sum() == 1


def test_ppcr_raw_bins_probability_boundaries_regression():
    """Regression test ensuring PPCR mode generates raw bins on genuine probability boundaries (e.g. 0.1 step) rather than ppcr cutpoints."""
    probs = {
        "m1": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    }
    reals = np.array([0, 1, 0, 2, 1, 0, 1, 2, 0, 1])
    times = np.array([0.5, 1.0, 2.5, 1.5, 3.0, 2.0, 0.8, 4.0, 1.2, 2.2])
    horizons = [2.0]

    res_ppcr = _prepare_probs_distribution_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        stratified_by=("ppcr",),
        by=0.1,
    )

    bins_ppcr = res_ppcr["bins"]

    # In PPCR mode, bins must retain standard raw probability boundaries [0.0, 0.1, ..., 1.0]
    expected_lowers = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    actual_lowers = bins_ppcr["lower"].to_list()
    assert np.allclose(actual_lowers, expected_lowers)


def test_cutoff_region_aj_structure_and_two_rows_per_cutoff():
    """Verify cutoff_region_aj structure and that each cutoff has exactly two rows."""
    probs = {"m1": np.array([0.1, 0.4, 0.6, 0.9])}
    reals = np.array([0, 1, 2, 0])
    times = np.array([1.0, 2.0, 1.5, 3.0])
    horizons = [2.0]

    res = _prepare_probs_distribution_data_times(
        probs=probs, reals=reals, times=times, fixed_time_horizons=horizons, by=0.2
    )

    aj_df = res["cutoff_region_aj"]
    ops_df = res["operating_points"]

    assert set(res.keys()) == {
        "bins",
        "operating_points",
        "rank_bins",
        "cutoff_region_aj",
    }

    # Operating points count
    num_cutoffs = len(ops_df)
    # cutoff_region_aj must have exactly 2 * num_cutoffs rows
    assert len(aj_df) == 2 * num_cutoffs

    # Group by cutoff and check that prediction_label has both predicted_positives and predicted_negatives
    for (cutoff,), group in aj_df.group_by(["chosen_cutoff"]):
        assert len(group) == 2
        labels = set(group["prediction_label"].to_list())
        assert labels == {"predicted_positives", "predicted_negatives"}


def test_probability_threshold_cutoff_semantics():
    """Verify threshold semantics: cutoff 0 puts everyone in predicted positives, cutoff > 0 strict >."""
    probs = {"m1": np.array([0.0, 0.2, 0.5, 0.8, 1.0])}
    reals = np.array([0, 1, 1, 0, 1])
    times = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    horizons = [2.0]

    res = _prepare_probs_distribution_data_times(
        probs=probs, reals=reals, times=times, fixed_time_horizons=horizons, by=0.2
    )

    aj_df = res["cutoff_region_aj"]

    # Cutoff 0.0: predicted_positives region should contain all observations
    aj_c0 = aj_df.filter(pl.col("chosen_cutoff") == 0.0)
    pos_region_c0 = aj_c0.filter(pl.col("prediction_label") == "predicted_positives")
    neg_region_c0 = aj_c0.filter(pl.col("prediction_label") == "predicted_negatives")

    # In cutoff 0, all predicted positives: true_positives=3, false_positives=2
    assert pos_region_c0["true_positives"][0] == pytest.approx(3.0)
    assert pos_region_c0["false_positives"][0] == pytest.approx(2.0)
    assert neg_region_c0["true_negatives"][0] == pytest.approx(0.0)
    assert neg_region_c0["false_negatives"][0] == pytest.approx(0.0)

    # Cutoff 0.4: observations > 0.4 (0.5, 0.8, 1.0) are predicted positive
    aj_c04 = aj_df.filter(pl.col("chosen_cutoff").round(2) == 0.4)
    pos_region_c04 = aj_c04.filter(pl.col("prediction_label") == "predicted_positives")
    neg_region_c04 = aj_c04.filter(pl.col("prediction_label") == "predicted_negatives")

    # Positives > 0.4: 0.5 (pos outcome, TP), 0.8 (neg outcome, FP), 1.0 (pos outcome, TP) -> TP=2, FP=1
    assert pos_region_c04["true_positives"][0] == pytest.approx(2.0)
    assert pos_region_c04["false_positives"][0] == pytest.approx(1.0)
    # Negatives <= 0.4: 0.0 (neg outcome, TN), 0.2 (pos outcome, FN) -> TN=1, FN=1
    assert neg_region_c04["true_negatives"][0] == pytest.approx(1.0)
    assert neg_region_c04["false_negatives"][0] == pytest.approx(1.0)

    # Cutoff 0.6: observations > 0.6 (0.8, 1.0) are predicted positive
    # Observation 0.5 is predicted negative (since 0.5 <= 0.6)
    aj_c06 = aj_df.filter(pl.col("chosen_cutoff").round(2) == 0.6)
    pos_region_c06 = aj_c06.filter(pl.col("prediction_label") == "predicted_positives")
    neg_region_c06 = aj_c06.filter(pl.col("prediction_label") == "predicted_negatives")

    # Positives > 0.6: 0.8 (neg outcome, FP), 1.0 (pos outcome, TP) -> TP=1, FP=1
    assert pos_region_c06["true_positives"][0] == pytest.approx(1.0)
    assert pos_region_c06["false_positives"][0] == pytest.approx(1.0)
    # Negatives <= 0.6: 0.0 (neg outcome, TN), 0.2 (pos outcome, FN), 0.5 (pos outcome, FN) -> TN=1, FN=2
    assert neg_region_c06["true_negatives"][0] == pytest.approx(1.0)
    assert neg_region_c06["false_negatives"][0] == pytest.approx(2.0)


def test_ppcr_semantics():
    """Verify PPCR stratification support and exact reconstruction of prepare_performance_data_times in PPCR mode."""
    probs = {
        "m1": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]),
        "m2": np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
    }
    reals = np.array([0, 1, 0, 2, 1, 0, 1, 2, 0, 1])
    times = np.array([0.5, 1.0, 2.5, 1.5, 3.0, 2.0, 0.8, 4.0, 1.2, 2.2])
    horizons = [2.0]

    perf_df = prepare_performance_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        stratified_by=("ppcr",),
        by=0.1,
    )

    dist_res = _prepare_probs_distribution_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        stratified_by=("ppcr",),
        by=0.1,
    )

    aj_df = dist_res["cutoff_region_aj"]
    ops_df = dist_res["operating_points"]

    assert len(ops_df) > 0
    assert (ops_df["stratified_by"] == "ppcr").all()
    assert (aj_df["stratified_by"] == "ppcr").all()

    # Reconstruct confusion matrix per PPCR cutoff from aj_df
    reconstructed_cm = aj_df.group_by(
        [
            "evaluation",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "stratified_by",
            "chosen_cutoff",
        ]
    ).agg(
        pl.col("true_positives").sum().alias("tp_est"),
        pl.col("false_positives").sum().alias("fp_est"),
        pl.col("true_negatives").sum().alias("tn_est"),
        pl.col("false_negatives").sum().alias("fn_est"),
    )

    for perf_row in perf_df.iter_rows(named=True):
        eval_id = perf_row["reference_group"]
        cutoff = perf_row["chosen_cutoff"]
        horizon = perf_row["fixed_time_horizon"]
        c_heur = perf_row["censoring_heuristic"]
        comp_heur = perf_row["competing_heuristic"]

        recon_row = reconstructed_cm.filter(
            (pl.col("evaluation") == eval_id)
            & (pl.col("chosen_cutoff") == cutoff)
            & (pl.col("fixed_time_horizon") == horizon)
            & (pl.col("censoring_heuristic") == c_heur)
            & (pl.col("competing_heuristic") == comp_heur)
        )
        assert len(recon_row) == 1

        op_row = ops_df.filter(
            (pl.col("evaluation") == eval_id)
            & (pl.col("chosen_cutoff") == cutoff)
            & (pl.col("fixed_time_horizon") == horizon)
            & (pl.col("censoring_heuristic") == c_heur)
            & (pl.col("competing_heuristic") == comp_heur)
        )
        assert len(op_row) == 1

        # Assert exact reconstruction of TP, FP, TN, FN in PPCR mode
        assert recon_row["tp_est"][0] == pytest.approx(perf_row["true_positives"])
        assert recon_row["fp_est"][0] == pytest.approx(perf_row["false_positives"])
        assert recon_row["tn_est"][0] == pytest.approx(perf_row["true_negatives"])
        assert recon_row["fn_est"][0] == pytest.approx(perf_row["false_negatives"])

        # Compare operating_points metrics against performance data in PPCR mode
        for metric in [
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "false_positive_rate",
            "lift",
        ]:
            val_op = op_row[metric][0]
            val_perf = perf_row[metric]
            if val_op is None or np.isnan(val_op):
                assert val_perf is None or np.isnan(val_perf)
            else:
                assert val_op == pytest.approx(val_perf)


def test_equivalence_to_prepare_performance_data_times():
    """Verify that derived TP/FP/TN/FN and metrics from cutoff_region_aj match prepare_performance_data_times."""
    probs = {
        "m1": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]),
        "m2": np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
    }
    reals = np.array([0, 1, 0, 2, 1, 0, 1, 2, 0, 1])
    times = np.array([0.5, 1.0, 2.5, 1.5, 3.0, 2.0, 0.8, 4.0, 1.2, 2.2])
    horizons = [2.0]

    perf_df = prepare_performance_data_times(
        probs=probs, reals=reals, times=times, fixed_time_horizons=horizons, by=0.1
    )

    dist_res = _prepare_probs_distribution_data_times(
        probs=probs, reals=reals, times=times, fixed_time_horizons=horizons, by=0.1
    )

    aj_df = dist_res["cutoff_region_aj"]
    ops_df = dist_res["operating_points"]

    # Sum TP, FP, TN, FN per cutoff from aj_df
    reconstructed_cm = aj_df.group_by(
        [
            "evaluation",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "stratified_by",
            "chosen_cutoff",
        ]
    ).agg(
        pl.col("true_positives").sum().alias("tp_est"),
        pl.col("false_positives").sum().alias("fp_est"),
        pl.col("true_negatives").sum().alias("tn_est"),
        pl.col("false_negatives").sum().alias("fn_est"),
    )

    for perf_row in perf_df.iter_rows(named=True):
        eval_id = perf_row["reference_group"]
        cutoff = perf_row["chosen_cutoff"]
        horizon = perf_row["fixed_time_horizon"]
        c_heur = perf_row["censoring_heuristic"]
        comp_heur = perf_row["competing_heuristic"]

        recon_row = reconstructed_cm.filter(
            (pl.col("evaluation") == eval_id)
            & (pl.col("chosen_cutoff") == cutoff)
            & (pl.col("fixed_time_horizon") == horizon)
            & (pl.col("censoring_heuristic") == c_heur)
            & (pl.col("competing_heuristic") == comp_heur)
        )
        assert len(recon_row) == 1

        op_row = ops_df.filter(
            (pl.col("evaluation") == eval_id)
            & (pl.col("chosen_cutoff") == cutoff)
            & (pl.col("fixed_time_horizon") == horizon)
            & (pl.col("censoring_heuristic") == c_heur)
            & (pl.col("competing_heuristic") == comp_heur)
        )
        assert len(op_row) == 1

        # Compare reconstructed confusion matrix against performance data
        assert recon_row["tp_est"][0] == pytest.approx(perf_row["true_positives"])
        assert recon_row["fp_est"][0] == pytest.approx(perf_row["false_positives"])
        assert recon_row["tn_est"][0] == pytest.approx(perf_row["true_negatives"])
        assert recon_row["fn_est"][0] == pytest.approx(perf_row["false_negatives"])

        # Compare operating_points metrics against performance data
        for metric in [
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "false_positive_rate",
            "lift",
            "net_benefit",
        ]:
            val_op = op_row[metric][0]
            val_perf = perf_row[metric]
            if val_op is None or np.isnan(val_op):
                assert val_perf is None or np.isnan(val_perf)
            else:
                assert val_op == pytest.approx(val_perf)


def test_exact_horizon_fixture():
    """Deterministic fixture asserting exact numeric behavior for events/censoring before, at, and after horizon."""
    # Horizon = 2.0
    # Obs 0: target transition at t=1.0 (before horizon)
    # Obs 1: target transition at t=2.0 (exact horizon) -> included in target state occupancy at horizon
    # Obs 2: competing transition at t=1.0 (before horizon)
    # Obs 3: censoring at t=1.0 (before horizon) -> affects risk set
    # Obs 4: censoring at t=2.0 (exact horizon) -> follows polarstate/AJ behavior
    # Obs 5: censoring at t=3.0 (after horizon) -> does not affect occupancy at horizon
    # Obs 6: baseline/censoring at t=4.0 (after horizon) -> does not affect occupancy at horizon
    probs = {"m1": np.array([0.8, 0.85, 0.2, 0.0, 0.0, 0.0, 0.0])}
    reals = np.array([1, 1, 2, 0, 0, 0, 0])
    times = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    horizon = 2.0

    res = _prepare_probs_distribution_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=[horizon],
        by=0.1,
    )

    aj_df = res["cutoff_region_aj"]

    # In cutoff 0.0, predicted_positives region includes all 7 observations
    c0_pos = aj_df.filter(
        (pl.col("chosen_cutoff") == 0.0)
        & (pl.col("prediction_label") == "predicted_positives")
    )
    assert len(c0_pos) == 1

    # Assert exact AJ state mass estimates at horizon=2.0:
    # Target state occupancy includes both target transitions at t=1.0 and t=2.0 (real_positives_est = 2.25)
    # Competing state occupancy includes competing transition at t=1.0 (real_competing_est = 1.0)
    # Baseline/negative state occupancy includes non-events surviving to horizon (real_negatives_est = 3.75)
    # Censored before horizon (t=1.0) affects risk set calculation without adding censored mass at horizon (real_censored_est = 0.0)
    assert c0_pos["real_positives_est"][0] == pytest.approx(2.25)
    assert c0_pos["real_competing_est"][0] == pytest.approx(1.0)
    assert c0_pos["real_negatives_est"][0] == pytest.approx(3.75)
    assert c0_pos["real_censored_est"][0] == pytest.approx(0.0)

    # Derived classification outcomes for adjusted_as_negative heuristic:
    # true_positives = target mass = 2.25
    # false_positives = baseline mass + competing mass = 3.75 + 1.0 = 4.75
    assert c0_pos["true_positives"][0] == pytest.approx(2.25)
    assert c0_pos["false_positives"][0] == pytest.approx(4.75)


def test_multiple_heuristic_combinations():
    """Verify behavior across all supported censoring/competing heuristic combinations."""
    probs = {"m1": np.array([0.1, 0.4, 0.6, 0.9])}
    reals = np.array([0, 1, 2, 0])
    times = np.array([1.0, 2.0, 1.5, 3.0])
    horizons = [2.0]

    heuristics_sets = [
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_censored",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_composite",
        },
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "excluded",
        },
    ]

    res = _prepare_probs_distribution_data_times(
        probs=probs,
        reals=reals,
        times=times,
        fixed_time_horizons=horizons,
        heuristics_sets=heuristics_sets,
        by=0.2,
    )

    aj_df = res["cutoff_region_aj"]
    ops_df = res["operating_points"]

    # Check that all 4 heuristic combinations are present in the output
    unique_heuristics_aj = aj_df.select(
        ["censoring_heuristic", "competing_heuristic"]
    ).unique()
    assert len(unique_heuristics_aj) == 4

    unique_heuristics_ops = ops_df.select(
        ["censoring_heuristic", "competing_heuristic"]
    ).unique()
    assert len(unique_heuristics_ops) == 4


def test_invalid_inputs_time_dependent():
    """Verify ValueError on invalid inputs."""
    probs = {"m1": np.array([0.2, 0.5])}
    reals = np.array([0, 1])
    times = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="`stratified_by` must be a sequence"):
        _prepare_probs_distribution_data_times(
            probs, reals, times, [1.0], stratified_by=("probability_threshold", "ppcr")
        )

    with pytest.raises(ValueError, match="Unsupported stratification key"):
        _prepare_probs_distribution_data_times(
            probs, reals, times, [1.0], stratified_by=("invalid",)
        )

"""Tests for internal prediction distribution data producer."""

import numpy as np
import polars as pl
import pytest

from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.probs_distribution import (
    _prepare_probs_distribution_data,
)


def _reconstruct_confusion_matrix_from_bins(
    bins: pl.DataFrame,
    cutoff: float,
    strat_type: str = "probability_threshold",
    req_val: float = 0.0,
) -> tuple[int, int, int, int]:
    """Reconstruct TP, FP, TN, FN from bins for a given cutoff.

    For probability_threshold:
      - cutoff == 0.0: everyone predicted positive (TP=total_pos, FP=total_neg, TN=0, FN=0).
      - strict nonzero cutoff: bins with upper <= cutoff are predicted negative;
        bins with upper > cutoff are predicted positive.

    For ppcr:
      - req_val == 0.0: 0 predicted positive (TP=0, FP=0, TN=total_neg, FN=total_pos).
      - req_val == 1.0: everyone predicted positive (TP=total_pos, FP=total_neg, TN=0, FN=0).
      - intermediate PPCR: observations with prob > cutoff are predicted positive.
    """
    if strat_type == "probability_threshold" and cutoff == 0.0:
        tp = int(bins["n_positive"].sum())
        fp = int(bins["n_negative"].sum())
        tn = 0
        fn = 0
    elif strat_type == "ppcr" and req_val == 0.0:
        tp = 0
        fp = 0
        tn = int(bins["n_negative"].sum())
        fn = int(bins["n_positive"].sum())
    elif strat_type == "ppcr" and req_val == 1.0:
        tp = int(bins["n_positive"].sum())
        fp = int(bins["n_negative"].sum())
        tn = 0
        fn = 0
    else:
        neg_bins = bins.filter(pl.col("upper") <= cutoff)
        pos_bins = bins.filter(pl.col("upper") > cutoff)
        tp = int(pos_bins["n_positive"].sum())
        fp = int(pos_bins["n_negative"].sum())
        tn = int(neg_bins["n_negative"].sum())
        fn = int(neg_bins["n_positive"].sum())

    return tp, fp, tn, fn


def assert_reconstruction_invariant(
    probs: dict[str, np.ndarray],
    reals: np.ndarray | dict[str, np.ndarray],
    stratified_by: tuple[str, ...] = ("probability_threshold",),
    by: float = 0.01,
) -> None:
    """Verify that reconstructed confusion matrices match prepare_performance_data output."""
    perf_df = prepare_performance_data(
        probs=probs, reals=reals, stratified_by=stratified_by, by=by
    )
    dist_data = _prepare_probs_distribution_data(
        probs=probs, reals=reals, stratified_by=stratified_by, by=by
    )

    bins_df = dist_data["bins"]
    op_df = dist_data["operating_points"]

    strat_type = stratified_by[0]

    for op_row in op_df.iter_rows(named=True):
        eval_id = op_row["evaluation"]
        cutoff = op_row["cutoff"]
        req_val = op_row["value"]

        eval_bins = bins_df.filter(pl.col("evaluation") == eval_id)
        tp, fp, tn, fn = _reconstruct_confusion_matrix_from_bins(
            eval_bins, cutoff, strat_type=strat_type, req_val=req_val
        )

        if strat_type == "ppcr":
            perf_row = perf_df.filter(
                (pl.col("reference_group") == eval_id) & (pl.col("ppcr") == req_val)
            )
        else:
            perf_row = perf_df.filter(
                (pl.col("reference_group") == eval_id)
                & (pl.col("chosen_cutoff") == cutoff)
            )
        assert len(perf_row) == 1

        p_tp = int(perf_row["true_positives"][0])
        p_fp = int(perf_row["false_positives"][0])
        p_tn = int(perf_row["true_negatives"][0])
        p_fn = int(perf_row["false_negatives"][0])

        assert (tp, fp, tn, fn) == (p_tp, p_fp, p_tn, p_fn), (
            f"Mismatch at evaluation '{eval_id}', strat={strat_type}, req_val={req_val}, cutoff {cutoff}: "
            f"Reconstructed (TP, FP, TN, FN)=({tp}, {fp}, {tn}, {fn}) vs "
            f"PerfData ({p_tp}, {p_fp}, {p_tn}, {p_fn})"
        )


def test_structure_and_schemas():
    probs = {"m1": np.array([0.0, 0.2, 0.5, 0.8, 1.0])}
    reals = np.array([0, 1, 0, 1, 0])

    res = _prepare_probs_distribution_data(probs, reals, by=0.1)

    assert set(res.keys()) == {"bins", "operating_points", "rank_bins"}
    bins = res["bins"]
    ops = res["operating_points"]
    rank_bins = res["rank_bins"]

    assert isinstance(bins, pl.DataFrame)
    assert isinstance(ops, pl.DataFrame)
    assert isinstance(rank_bins, pl.DataFrame)

    assert bins.schema["evaluation"] == pl.String
    assert bins.schema["model"] == pl.String
    assert bins.schema["population"] == pl.String
    assert bins.schema["lower"] == pl.Float64
    assert bins.schema["upper"] == pl.Float64
    assert bins.schema["include_lower"] == pl.Boolean
    assert bins.schema["include_upper"] == pl.Boolean
    assert bins.schema["n_positive"] == pl.UInt32
    assert bins.schema["n_negative"] == pl.UInt32

    assert ops.schema["evaluation"] == pl.String
    assert ops.schema["model"] == pl.String
    assert ops.schema["population"] == pl.String
    assert ops.schema["type"] == pl.String
    assert ops.schema["value"] == pl.Float64
    assert ops.schema["cutoff"] == pl.Float64
    assert ops.schema["realized_ppcr"] == pl.Float64

    assert rank_bins.schema["evaluation"] == pl.String
    assert rank_bins.schema["model"] == pl.String
    assert rank_bins.schema["population"] == pl.String
    assert rank_bins.schema["rank_lower"] == pl.Float64
    assert rank_bins.schema["rank_upper"] == pl.Float64
    assert rank_bins.schema["n_positive"] == pl.Int64
    assert rank_bins.schema["n_negative"] == pl.Int64


def test_nullable_model_single_keyed_population():
    probs = {"validation_population": np.array([0.2, 0.8])}
    reals = {"validation_population": np.array([0, 1])}

    res = _prepare_probs_distribution_data(probs, reals, by=0.5)

    bins = res["bins"]
    ops = res["operating_points"]

    assert bins.schema["model"] == pl.String
    assert ops.schema["model"] == pl.String

    assert (bins["evaluation"] == "validation_population").all()
    assert (bins["population"] == "validation_population").all()
    assert bins["model"].is_null().all()

    assert (ops["evaluation"] == "validation_population").all()
    assert (ops["population"] == "validation_population").all()
    assert ops["model"].is_null().all()


def test_nullable_model_multiple_keyed_populations():
    probs = {"pop1": np.array([0.2, 0.8]), "pop2": np.array([0.3, 0.7])}
    reals = {"pop1": np.array([0, 1]), "pop2": np.array([1, 0])}

    res = _prepare_probs_distribution_data(probs, reals, by=0.5)

    bins = res["bins"]
    ops = res["operating_points"]

    assert bins.schema["model"] == pl.String
    assert ops.schema["model"] == pl.String

    assert bins["model"].is_null().all()
    assert ops["model"].is_null().all()


def test_shared_outcomes_model_identity():
    probs = {"m1": np.array([0.2, 0.8]), "m2": np.array([0.3, 0.7])}
    reals = np.array([0, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.5)

    bins = res["bins"]
    ops = res["operating_points"]

    assert (bins["model"] == bins["evaluation"]).all()
    assert (ops["model"] == ops["evaluation"]).all()


def test_threshold_golden_fixture():
    scores = [0.0, 0.2, 0.5, 0.5, 0.8, 1.0]
    outcomes = [0, 1, 1, 0, 1, 0]

    probs = {"m1": np.array(scores)}
    reals = np.array(outcomes)

    res = _prepare_probs_distribution_data(probs, reals, by=0.1)
    bins = res["bins"]

    # Golden fixture assertions for specific cutoffs
    expected_cutoffs = {
        0.0: (3, 3, 0, 0, 1.0),
        0.2: (2, 2, 1, 1, 4 / 6),
        0.5: (1, 1, 2, 2, 2 / 6),
        1.0: (0, 0, 3, 3, 0.0),
    }

    for cutoff, (e_tp, e_fp, e_tn, e_fn, e_ppcr) in expected_cutoffs.items():
        tp, fp, tn, fn = _reconstruct_confusion_matrix_from_bins(
            bins, cutoff, strat_type="probability_threshold"
        )
        assert (tp, fp, tn, fn) == (e_tp, e_fp, e_tn, e_fn)

        op_row = res["operating_points"].filter(pl.col("cutoff") == cutoff)
        assert pytest.approx(op_row["realized_ppcr"][0]) == e_ppcr


def test_ppcr_tie_fixture():
    scores = [0.1, 0.2, 0.5, 0.5, 0.8, 0.9]
    outcomes = [0, 0, 1, 0, 1, 1]

    probs = {"m1": np.array(scores)}
    reals = np.array(outcomes)

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.5
    )
    ops = res["operating_points"]

    op_05 = ops.filter(pl.col("value") == 0.5)
    assert len(op_05) == 1
    assert op_05["type"][0] == "ppcr"
    assert op_05["value"][0] == 0.5
    assert op_05["cutoff"][0] == 0.5
    assert pytest.approx(op_05["realized_ppcr"][0]) == 2 / 6  # 1/3


def test_reconstruction_invariant_scenarios():
    # 1. Distinct scores mixed outcomes
    assert_reconstruction_invariant(
        {"m1": np.array([0.1, 0.4, 0.6, 0.9])}, np.array([0, 1, 0, 1]), by=0.1
    )

    # 2. Cutoff equal to observed score
    assert_reconstruction_invariant(
        {"m1": np.array([0.0, 0.2, 0.5, 0.8, 1.0])},
        np.array([0, 1, 1, 0, 1]),
        by=0.1,
    )

    # 3. Partially tied scores containing both outcomes
    assert_reconstruction_invariant(
        {"m1": np.array([0.2, 0.5, 0.5, 0.8])}, np.array([0, 1, 0, 1]), by=0.1
    )

    # 4. All scores tied
    assert_reconstruction_invariant(
        {"m1": np.array([0.5, 0.5, 0.5, 0.5])}, np.array([1, 0, 1, 0]), by=0.1
    )

    # 5. Scores equal to zero
    assert_reconstruction_invariant(
        {"m1": np.array([0.0, 0.0, 0.3, 0.7])}, np.array([0, 1, 0, 1]), by=0.1
    )

    # 6. Scores equal to one
    assert_reconstruction_invariant(
        {"m1": np.array([0.2, 0.8, 1.0, 1.0])}, np.array([0, 1, 0, 1]), by=0.1
    )

    # 7. All positive outcomes
    assert_reconstruction_invariant(
        {"m1": np.array([0.1, 0.5, 0.9])}, np.array([1, 1, 1]), by=0.2
    )

    # 8. All negative outcomes
    assert_reconstruction_invariant(
        {"m1": np.array([0.1, 0.5, 0.9])}, np.array([0, 0, 0]), by=0.2
    )

    # 9. Multiple models
    assert_reconstruction_invariant(
        {
            "m1": np.array([0.1, 0.5, 0.9]),
            "m2": np.array([0.2, 0.6, 0.8]),
        },
        np.array([0, 1, 1]),
        by=0.2,
    )

    # 10. Multiple populations
    assert_reconstruction_invariant(
        {
            "p1": np.array([0.1, 0.5, 0.9]),
            "p2": np.array([0.2, 0.6, 0.8]),
        },
        {
            "p1": np.array([0, 1, 1]),
            "p2": np.array([1, 0, 1]),
        },
        by=0.2,
    )

    # 11. Stratified by PPCR
    assert_reconstruction_invariant(
        {"m1": np.array([0.1, 0.2, 0.5, 0.5, 0.8, 0.9])},
        np.array([0, 0, 1, 0, 1, 1]),
        stratified_by=("ppcr",),
        by=0.5,
    )

    # 12. Non-exact divisor step `by`
    assert_reconstruction_invariant(
        {"m1": np.array([0.1, 0.3, 0.7, 0.9])}, np.array([0, 1, 0, 1]), by=0.15
    )


def test_zero_score_interval_present():
    probs = {"m1": np.array([0.2, 0.5, 0.8])}
    reals = np.array([0, 1, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.1)
    bins = res["bins"]

    zero_bin = bins.filter(
        (pl.col("lower") == 0.0)
        & (pl.col("upper") == 0.0)
        & pl.col("include_lower")
        & pl.col("include_upper")
    )
    assert len(zero_bin) == 1
    assert zero_bin["n_positive"][0] == 0
    assert zero_bin["n_negative"][0] == 0


def test_totals_match():
    probs = {"m1": np.array([0.0, 0.2, 0.5, 0.8, 1.0])}
    reals = np.array([0, 1, 0, 1, 0])

    res = _prepare_probs_distribution_data(probs, reals, by=0.1)
    bins = res["bins"]

    assert bins["n_positive"].sum() == 2
    assert bins["n_negative"].sum() == 3


def test_large_observation_vector_performance_regression():
    rng = np.random.default_rng(42)
    n_obs = 10_000
    p_vec = rng.uniform(0.0, 1.0, size=n_obs)
    p_vec[0] = 0.0
    p_vec[1] = 1.0
    r_vec = rng.integers(0, 2, size=n_obs)

    probs = {"m1": p_vec}
    reals = r_vec

    res = _prepare_probs_distribution_data(probs, reals, by=0.01)
    bins = res["bins"]
    ops = res["operating_points"]

    assert len(ops) == 101
    assert bins["n_positive"].sum() == int((r_vec == 1).sum())
    assert bins["n_negative"].sum() == int((r_vec == 0).sum())

    assert_reconstruction_invariant(probs, reals, by=0.01)


def test_invalid_inputs():
    probs = {"m1": np.array([0.2, 0.5])}
    reals = np.array([0, 1])

    with pytest.raises(ValueError, match="`stratified_by` must be a sequence"):
        _prepare_probs_distribution_data(
            probs, reals, stratified_by=("probability_threshold", "ppcr")
        )

    with pytest.raises(ValueError, match="Unsupported stratification key"):
        _prepare_probs_distribution_data(probs, reals, stratified_by=("invalid",))

    with pytest.raises(ValueError, match="Binary outcomes must contain only 0 and 1"):
        _prepare_probs_distribution_data(probs, np.array([0, 2]))

    with pytest.raises(
        ValueError, match="Estimated probabilities must be between 0 and 1"
    ):
        _prepare_probs_distribution_data({"m1": np.array([-0.1, 0.5])}, reals)


def test_rank_bins_stratification_invariance():
    probs = {"m1": np.array([0.0, 0.1, 0.2, 0.5, 0.5, 0.8, 1.0])}
    reals = np.array([0, 0, 1, 1, 0, 1, 0])

    res_thresh = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("probability_threshold",), by=0.2
    )
    res_ppcr = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.2
    )

    rb_thresh = res_thresh["rank_bins"]
    rb_ppcr = res_ppcr["rank_bins"]

    assert rb_thresh.equals(rb_ppcr)


def test_rank_bins_properties():
    # 1. Distinct scores mixed outcomes
    probs = {
        "m1": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    }
    reals = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.2)
    rb = res["rank_bins"]

    # Retains complete requested grid (q = 1 / 0.2 = 5)
    assert len(rb) == 5
    assert np.allclose(rb["rank_lower"].to_list(), [0.0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(rb["rank_upper"].to_list(), [0.2, 0.4, 0.6, 0.8, 1.0])

    # Total mass conservation
    assert rb["n_positive"].sum() == (reals == 1).sum()
    assert rb["n_negative"].sum() == (reals == 0).sum()


def test_rank_bins_ties_never_split():
    # All scores tied
    probs = {"m1": np.array([0.5, 0.5, 0.5, 0.5, 0.5])}
    reals = np.array([1, 0, 1, 0, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.2)
    rb = res["rank_bins"]

    # Retains complete q=5 grid
    assert len(rb) == 5
    # All mass assigned to a single bin, empty strata retained explicitly
    non_zero_strata = rb.filter((pl.col("n_positive") > 0) | (pl.col("n_negative") > 0))
    assert len(non_zero_strata) == 1
    assert non_zero_strata["n_positive"][0] == 3
    assert non_zero_strata["n_negative"][0] == 2

    assert rb["n_positive"].sum() == 3
    assert rb["n_negative"].sum() == 2


def test_primary_golden_fixture():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.20)
    rb = res["rank_bins"]

    expected_df = pl.DataFrame(
        {
            "evaluation": ["m1"] * 5,
            "model": ["m1"] * 5,
            "population": ["__shared_population__"] * 5,
            "rank_lower": [0.00, 0.20, 0.40, 0.60, 0.80],
            "rank_upper": [0.20, 0.40, 0.60, 0.80, 1.00],
            "n_positive": [1, 2, 0, 1, 1],
            "n_negative": [1, 2, 0, 0, 1],
        }
    )

    assert rb.equals(expected_df)


def test_secondary_golden_fixture_n_less_than_q():
    probs = {"m1": np.array([0.10, 0.50, 0.90])}
    reals = np.array([0, 1, 1])

    res = _prepare_probs_distribution_data(probs, reals, by=0.20)
    rb = res["rank_bins"]

    expected_df = pl.DataFrame(
        {
            "evaluation": ["m1"] * 5,
            "model": ["m1"] * 5,
            "population": ["__shared_population__"] * 5,
            "rank_lower": [0.00, 0.20, 0.40, 0.60, 0.80],
            "rank_upper": [0.20, 0.40, 0.60, 0.80, 1.00],
            "n_positive": [0, 0, 1, 0, 1],
            "n_negative": [1, 0, 0, 0, 0],
        }
    )

    assert rb.equals(expected_df)


def test_primary_golden_fixture_order_invariance():
    probs_orig = np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])
    reals_orig = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    # Permuted order including within the tied 0.50 group
    perm_idx = np.array([4, 0, 5, 2, 3, 8, 1, 7, 6])
    probs_perm = probs_orig[perm_idx]
    reals_perm = reals_orig[perm_idx]

    res_orig = _prepare_probs_distribution_data({"m1": probs_orig}, reals_orig, by=0.20)
    res_perm = _prepare_probs_distribution_data({"m1": probs_perm}, reals_perm, by=0.20)

    assert res_orig["rank_bins"].equals(res_perm["rank_bins"])


def test_frozen_r_oracle_ppcr_tied_fixture():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    perf_df = prepare_performance_data(probs, reals, stratified_by=("ppcr",), by=0.20)
    dist = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.20
    )
    ops = dist["operating_points"]

    # Independently hardcoded frozen R oracle expected values
    expected = [
        # req_ppcr, eff_cutoff, realized_ppcr, tp, fp, tn, fn, sens, spec, ppv, npv, lift
        (0.00, 1.00, 0.00 / 9.0, 0, 0, 4, 5, 0.0, 1.0, np.nan, 4 / 9, np.nan),
        (0.20, 0.71, 2.00 / 9.0, 1, 1, 3, 4, 0.2, 0.75, 0.5, 3 / 7, 0.9),
        (0.40, 0.50, 3.00 / 9.0, 2, 1, 3, 3, 0.4, 0.75, 2 / 3, 0.5, 1.2),
        (0.60, 0.50, 3.00 / 9.0, 2, 1, 3, 3, 0.4, 0.75, 2 / 3, 0.5, 1.2),
        (0.80, 0.24, 9.00 / 9.0, 5, 4, 0, 0, 1.0, 0.0, 5 / 9, np.nan, 1.0),
        (1.00, 0.00, 9.00 / 9.0, 5, 4, 0, 0, 1.0, 0.0, 5 / 9, np.nan, 1.0),
    ]

    for (
        req_ppcr,
        eff_cutoff,
        real_ppcr,
        tp,
        fp,
        tn,
        fn,
        sens,
        spec,
        ppv,
        npv,
        lift,
    ) in expected:
        p_row = perf_df.filter(pl.col("chosen_cutoff") == req_ppcr).row(0, named=True)
        op_row = ops.filter(pl.col("value") == req_ppcr).row(0, named=True)

        assert p_row["ppcr"] == pytest.approx(req_ppcr)
        assert p_row["chosen_cutoff"] == pytest.approx(req_ppcr)
        assert p_row["probability_threshold"] == pytest.approx(eff_cutoff, abs=1e-4)

        assert op_row["type"] == "ppcr"
        assert op_row["value"] == pytest.approx(req_ppcr)
        assert op_row["cutoff"] == pytest.approx(eff_cutoff, abs=1e-4)
        assert op_row["realized_ppcr"] == pytest.approx(real_ppcr)

        assert p_row["true_positives"] == tp
        assert p_row["false_positives"] == fp
        assert p_row["true_negatives"] == tn
        assert p_row["false_negatives"] == fn

        assert p_row["sensitivity"] == pytest.approx(sens)
        assert p_row["specificity"] == pytest.approx(spec)

        if np.isnan(ppv):
            assert np.isnan(p_row["ppv"])
        else:
            assert p_row["ppv"] == pytest.approx(ppv)

        if np.isnan(npv):
            assert np.isnan(p_row["npv"])
        else:
            assert p_row["npv"] == pytest.approx(npv)

        if np.isnan(lift):
            assert np.isnan(p_row["lift"])
        else:
            assert p_row["lift"] == pytest.approx(lift)

    # Concrete assertion that operating_point.cutoff != operating_point.value for tied PPCR point
    op_02 = ops.filter(pl.col("value") == 0.20).row(0, named=True)
    assert op_02["cutoff"] != op_02["value"]
    assert op_02["cutoff"] == pytest.approx(0.71)
    assert op_02["value"] == pytest.approx(0.20)


def test_ppcr_repeated_effective_cutoffs_under_ties():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    dist = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.20
    )
    ops = dist["operating_points"]

    op_04 = ops.filter(pl.col("value") == 0.40).row(0, named=True)
    op_06 = ops.filter(pl.col("value") == 0.60).row(0, named=True)

    # Different requested value
    assert op_04["value"] == pytest.approx(0.40)
    assert op_06["value"] == pytest.approx(0.60)
    assert op_04["value"] != op_06["value"]

    # Same effective cutoff, realized PPCR, and confusion matrix
    assert op_04["cutoff"] == pytest.approx(op_06["cutoff"]) == pytest.approx(0.50)
    assert (
        op_04["realized_ppcr"]
        == pytest.approx(op_06["realized_ppcr"])
        == pytest.approx(3 / 9)
    )


def test_ppcr_n_less_than_q():
    probs = {"m1": np.array([0.10, 0.50, 0.90])}
    reals = np.array([0, 1, 1])

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.20
    )
    rb = res["rank_bins"]
    ops = res["operating_points"]

    # Explicit empty rank strata preserved (q = 5)
    assert len(rb) == 5
    assert rb["n_positive"].to_list() == [0, 0, 1, 0, 1]
    assert rb["n_negative"].to_list() == [1, 0, 0, 0, 0]

    # Verify operating point cutoffs for N < q
    cutoffs = ops["cutoff"].to_list()
    assert len(cutoffs) == 6
    assert np.allclose(cutoffs, [0.9, 0.74, 0.58, 0.42, 0.26, 0.1])


def test_ppcr_distinct_predictions_type7_boundaries():
    probs = {"m1": np.array([0.10, 0.30, 0.70, 0.90])}
    reals = np.array([0, 1, 0, 1])

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.25
    )
    ops = res["operating_points"]

    # Expected Type-7 linear quantiles at 1.0, 0.75, 0.50, 0.25, 0.0
    expected_cutoffs = [0.90, 0.75, 0.50, 0.25, 0.10]
    for row, exp_c in zip(ops.iter_rows(named=True), expected_cutoffs):
        assert row["cutoff"] == pytest.approx(exp_c)


def test_ppcr_exact_scores_zero_and_one_endpoints():
    probs = {"m1": np.array([0.0, 0.0, 0.5, 1.0, 1.0])}
    reals = np.array([0, 1, 0, 1, 0])

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.20
    )
    ops = res["operating_points"]

    op_0 = ops.filter(pl.col("value") == 0.0).row(0, named=True)
    assert op_0["cutoff"] == pytest.approx(1.0)
    assert op_0["realized_ppcr"] == pytest.approx(0.0)

    op_1 = ops.filter(pl.col("value") == 1.0).row(0, named=True)
    assert op_1["cutoff"] == pytest.approx(0.0)
    assert op_1["realized_ppcr"] == pytest.approx(1.0)


def test_ppcr_multiple_models_shared_outcomes():
    probs = {
        "m1": np.array([0.1, 0.4, 0.6, 0.9]),
        "m2": np.array([0.2, 0.5, 0.8, 1.0]),
    }
    reals = np.array([0, 1, 0, 1])

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.5
    )
    ops = res["operating_points"]

    ops_m1 = ops.filter(pl.col("model") == "m1")
    ops_m2 = ops.filter(pl.col("model") == "m2")

    assert len(ops_m1) == 3
    assert len(ops_m2) == 3

    # Per-model effective cutoffs computed independently
    m1_c05 = ops_m1.filter(pl.col("value") == 0.5).row(0, named=True)["cutoff"]
    m2_c05 = ops_m2.filter(pl.col("value") == 0.5).row(0, named=True)["cutoff"]

    assert m1_c05 == pytest.approx(0.5)
    assert m2_c05 == pytest.approx(0.65)


def test_ppcr_multiple_populations_unequal_sample_sizes():
    probs = {
        "pop1": np.array([0.1, 0.5, 0.9]),
        "pop2": np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
    }
    reals = {
        "pop1": np.array([0, 1, 1]),
        "pop2": np.array([0, 1, 0, 1, 0]),
    }

    res = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.5
    )
    ops = res["operating_points"]

    p1_ops = ops.filter(pl.col("population") == "pop1")
    p2_ops = ops.filter(pl.col("population") == "pop2")

    assert len(p1_ops) == 3
    assert len(p2_ops) == 3


def test_ppcr_pre_post_non_regression():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    perf_df = prepare_performance_data(probs, reals, stratified_by=("ppcr",), by=0.20)
    dist = _prepare_probs_distribution_data(
        probs, reals, stratified_by=("ppcr",), by=0.20
    )

    # Rank bins schema and masses frozen
    rb = dist["rank_bins"]
    assert rb["n_positive"].to_list() == [1, 2, 0, 1, 1]
    assert rb["n_negative"].to_list() == [1, 2, 0, 0, 1]

    # Metrics frozen
    assert perf_df["true_positives"].to_list() == [0, 1, 2, 2, 5, 5]
    assert perf_df["false_positives"].to_list() == [0, 1, 1, 1, 4, 4]
    assert perf_df["true_negatives"].to_list() == [4, 3, 3, 3, 0, 0]
    assert perf_df["false_negatives"].to_list() == [5, 4, 3, 3, 0, 0]

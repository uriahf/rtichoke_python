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

    assert set(res.keys()) == {"bins", "operating_points"}
    bins = res["bins"]
    ops = res["operating_points"]

    assert isinstance(bins, pl.DataFrame)
    assert isinstance(ops, pl.DataFrame)

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

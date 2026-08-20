"""Regression tests for dcurves compatibility (issue #127)."""

from importlib import resources

import numpy as np
import pandas as pd
import polars as pl
from dcurves import dca
from numpy.testing import assert_allclose

from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)


def _load_dcurves_survival_data() -> pd.DataFrame:
    data_resource = resources.files("dcurves").joinpath("data/df_surv.csv")
    with resources.as_file(data_resource) as path:
        return pd.read_csv(path)


def _rtichoke_survival_dca(data: pd.DataFrame, by: float) -> pl.DataFrame:
    return (
        prepare_performance_data_times(
            probs={"cancerpredmarker": data["cancerpredmarker"].to_numpy()},
            reals=data["cancer"].astype(int).to_numpy(),
            times=data["ttcancer"].to_numpy(),
            fixed_time_horizons=[1.5],
            by=by,
        )
        .filter(
            (pl.col("reference_group") == "cancerpredmarker")
            & (pl.col("stratified_by") == "probability_threshold")
        )
        .sort("chosen_cutoff")
    )


def test_survival_decision_curve_matches_dcurves_issue_127() -> None:
    """Match dcurves TP/FP rates and net benefit on its 1.5-year example."""
    data = _load_dcurves_survival_data()
    thresholds = np.array([0.00, 0.01, 0.05, 0.10, 0.20, 0.50])

    dcurves_result = dca(
        data=data,
        outcome="cancer",
        modelnames=["cancerpredmarker"],
        thresholds=thresholds,
        time=1.5,
        time_to_outcome_col="ttcancer",
    )
    dcurves_model = (
        dcurves_result[dcurves_result["model"] == "cancerpredmarker"]
        .sort_values("threshold")
        .reset_index(drop=True)
    )

    rtichoke_result = _rtichoke_survival_dca(data, by=0.01).filter(
        pl.col("chosen_cutoff").is_in(thresholds.tolist())
    )

    assert data.shape[0] == 750
    assert rtichoke_result.height == len(thresholds)

    rtichoke_prevalence = (
        rtichoke_result.filter(pl.col("chosen_cutoff") == 0.0)
        .select(pl.col("real_positives") / pl.col("n"))
        .item()
    )
    dcurves_prevalence = dcurves_model["prevalence"].iloc[0]

    assert round(dcurves_prevalence, 2) == 0.22
    assert_allclose(rtichoke_prevalence, dcurves_prevalence, rtol=0, atol=1e-10)

    assert_allclose(
        rtichoke_result["true_positives"].to_numpy()
        / rtichoke_result["n"].to_numpy(),
        dcurves_model["tp_rate"].to_numpy(),
        rtol=0,
        atol=1e-10,
    )
    assert_allclose(
        rtichoke_result["false_positives"].to_numpy()
        / rtichoke_result["n"].to_numpy(),
        dcurves_model["fp_rate"].to_numpy(),
        rtol=0,
        atol=1e-10,
    )
    assert_allclose(
        rtichoke_result["net_benefit"].to_numpy(),
        dcurves_model["net_benefit"].to_numpy(),
        rtol=0,
        atol=1e-10,
    )


def test_survival_decision_curve_includes_prediction_equal_to_threshold() -> None:
    """Keep dcurves' prediction >= threshold convention at an exact boundary."""
    data = pd.DataFrame(
        {
            "cancer": [True, False, True, False, False, False],
            "ttcancer": [0.5, 2.0, 0.7, 2.0, 2.0, 2.0],
            "cancerpredmarker": [0.20, 0.20, 0.30, 0.10, 0.40, 0.05],
        }
    )

    dcurves_result = dca(
        data=data,
        outcome="cancer",
        modelnames=["cancerpredmarker"],
        thresholds=[0.20],
        time=1.5,
        time_to_outcome_col="ttcancer",
    )
    dcurves_model = dcurves_result[
        dcurves_result["model"] == "cancerpredmarker"
    ].iloc[0]

    rtichoke_result = _rtichoke_survival_dca(data, by=0.10).filter(
        pl.col("chosen_cutoff") == 0.20
    )

    assert rtichoke_result.height == 1
    assert_allclose(
        rtichoke_result["true_positives"].item() / rtichoke_result["n"].item(),
        dcurves_model["tp_rate"],
        rtol=0,
        atol=1e-10,
    )
    assert_allclose(
        rtichoke_result["false_positives"].item() / rtichoke_result["n"].item(),
        dcurves_model["fp_rate"],
        rtol=0,
        atol=1e-10,
    )
    assert_allclose(
        rtichoke_result["net_benefit"].item(),
        dcurves_model["net_benefit"],
        rtol=0,
        atol=1e-10,
    )

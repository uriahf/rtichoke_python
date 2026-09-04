"""Tests for calibration bins (n_bins) cross-language parity, validation, and contracts."""

import numpy as np
import polars as pl
import pytest
from plotly.graph_objs._figure import Figure

from rtichoke import (
    create_calibration_curve,
    create_calibration_curve_times,
)
from rtichoke.calibration.calibration import (
    _create_calibration_curve_list,
    _create_calibration_curve_list_times,
    _make_calibration_bins_dat_binary,
    _make_adjusted_calibration_bins_data,
)
from rtichoke._calibration_viz_spec_v2 import _calibration_v2_spec_from_curve_list
from rtichoke._viz_browser import _calibration_spec_from_curve_list
from rtichoke.processing.evaluation_semantics import _EvaluationMetadata


def test_n_bins_validation_invalid_values():
    probs = {"m1": np.array([0.1, 0.4, 0.7, 0.9])}
    reals = np.array([0, 0, 1, 1])

    invalid_values = [0, -1, -10, 2.5, None, float("nan"), "10", True, False]
    for val in invalid_values:
        with pytest.raises(ValueError, match="n_bins must be a positive integer >= 1."):
            create_calibration_curve(probs, reals, n_bins=val)

        with pytest.raises(ValueError, match="n_bins must be a positive integer >= 1."):
            create_calibration_curve_times(
                probs,
                reals,
                times=np.array([1, 2, 3, 4]),
                fixed_time_horizons=[2.0],
                heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
                n_bins=val,
            )


def test_n_bins_validation_accepts_numpy_integers():
    probs = {"m1": np.array([0.1, 0.4, 0.7, 0.9])}
    reals = np.array([0, 0, 1, 1])
    fig = create_calibration_curve(probs, reals, n_bins=np.int64(5))
    assert isinstance(fig, Figure)


def test_r_parity_n12_b10():
    # N=12, B=10 -> sizes must be 2, 2, 1, 1, 1, 1, 1, 1, 1, 1
    p = np.linspace(0.01, 0.99, 12)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    probs = {"m1": p}
    reals = y

    bins_dat = _make_calibration_bins_dat_binary(probs, reals, n_bins=10)
    assert bins_dat.height == 10
    counts = bins_dat["n"].to_list()
    assert counts == [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    assert bins_dat["bin"].to_list() == list(range(1, 11))


def test_r_parity_n11_b10():
    # N=11, B=10 -> sizes must be 2, 1, 1, 1, 1, 1, 1, 1, 1, 1
    p = np.linspace(0.01, 0.99, 11)
    y = np.array([0, 1] * 5 + [0])
    probs = {"m1": p}
    reals = y

    bins_dat = _make_calibration_bins_dat_binary(probs, reals, n_bins=10)
    assert bins_dat.height == 10
    counts = bins_dat["n"].to_list()
    assert counts == [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_r_parity_n5_b10():
    # N=5, B=10 -> occupied labels 1, 2, 3, 4, 5, one observation each
    p = np.linspace(0.1, 0.9, 5)
    y = np.array([0, 0, 1, 0, 1])
    probs = {"m1": p}
    reals = y

    bins_dat = _make_calibration_bins_dat_binary(probs, reals, n_bins=10)
    assert bins_dat.height == 5
    assert bins_dat["bin"].to_list() == [1, 2, 3, 4, 5]
    assert bins_dat["n"].to_list() == [1, 1, 1, 1, 1]


def test_r_parity_b_greater_than_n():
    p = np.array([0.1, 0.3, 0.8])
    y = np.array([0, 1, 1])
    bins_dat = _make_calibration_bins_dat_binary({"m1": p}, y, n_bins=5)
    assert bins_dat.height == 3
    assert bins_dat["bin"].to_list() == [1, 2, 3]
    assert bins_dat["n"].to_list() == [1, 1, 1]


def test_r_parity_all_predictions_identical():
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    y = np.array([0, 1, 0, 1, 0])
    bins_dat = _make_calibration_bins_dat_binary({"m1": p}, y, n_bins=10)
    assert bins_dat.height == 1
    assert bins_dat["bin"].to_list() == [1]
    assert bins_dat["n"].to_list() == [5]
    assert bins_dat["x"].to_list() == [0.5]
    assert bins_dat["y"].to_list() == [0.4]


def test_r_parity_partial_ties_stable_ordinal_order():
    p = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0,   1,   0,   1,   0,   1])
    # N=6, B=3 -> q=2, rem=0 -> 3 bins of size 2
    bins_dat = _make_calibration_bins_dat_binary({"m1": p}, y, n_bins=3)
    assert bins_dat.height == 3
    assert bins_dat["n"].to_list() == [2, 2, 2]
    assert bins_dat["bin"].to_list() == [1, 2, 3]


def test_static_default_and_explicit_n_bins():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.5).astype(int)
    probs = {"m1": p}

    # Default n_bins = 10
    cl_default = _create_calibration_curve_list(probs, y)
    assert cl_default["calibration_bins_dat"].height == 10

    # Explicit n_bins = 8
    cl_8 = _create_calibration_curve_list(probs, y, n_bins=8)
    assert cl_8["calibration_bins_dat"].height == 8

    # n_bins = 1
    cl_1 = _create_calibration_curve_list(probs, y, n_bins=1)
    assert cl_1["calibration_bins_dat"].height == 1
    assert cl_1["calibration_bins_dat"]["n"].to_list() == [100]


def test_multiple_models_and_populations_preserve_identities():
    p1 = np.linspace(0.1, 0.9, 20)
    p2 = np.linspace(0.2, 0.8, 20)
    y = np.tile([0, 1], 10)

    cl_multi = _create_calibration_curve_list({"Model A": p1, "Model B": p2}, y, n_bins=5)
    df_bins = cl_multi["calibration_bins_dat"]
    assert set(df_bins["reference_group"].unique().to_list()) == {"Model A", "Model B"}
    assert set(df_bins["model"].unique().to_list()) == {"Model A", "Model B"}
    assert df_bins.filter(pl.col("reference_group") == "Model A").height == 5
    assert df_bins.filter(pl.col("reference_group") == "Model B").height == 5


def test_non_adjusted_time_inherits_static_bin_contract():
    probs = {"m1": np.linspace(0.01, 0.99, 12)}
    reals = np.tile([0, 1], 6)
    times = np.full(12, 3.0)  # follow-up time 3.0 > horizon 2.0 -> no censoring

    cl_time = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[2.0],
        heuristics_sets=[{"censoring_heuristic": "excluded", "competing_heuristic": "excluded"}],
        n_bins=10,
    )
    df_bins = cl_time["calibration_bins_dat"]
    assert df_bins.height == 10
    assert df_bins["n"].to_list() == [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]


def test_adjusted_time_calibration_preserved_semantics():
    # Adjusted time calibration retains rank("average") + floor formula
    df = pl.DataFrame(
        {
            "reference_group": ["m1"] * 10,
            "prob": [0.1, 0.1, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9],
            "real": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "time": [1.0] * 10,
        }
    )
    res_10 = _make_adjusted_calibration_bins_data(df, horizon=2.0, n_bins=10)
    assert "bin" in res_10.columns
    assert "decile" not in res_10.columns

    res_5 = _make_adjusted_calibration_bins_data(df, horizon=2.0, n_bins=5)
    assert "bin" in res_5.columns
    assert res_5.height <= 5


def test_smooth_static_n_bins_no_effect():
    p = np.linspace(0.01, 0.99, 50)
    y = np.tile([0, 1], 25)
    probs = {"m1": p}

    cl_smooth_10 = _create_calibration_curve_list(probs, y, calibration_type="smooth", n_bins=10)
    cl_smooth_5 = _create_calibration_curve_list(probs, y, calibration_type="smooth", n_bins=5)

    assert cl_smooth_10["smooth_dat"].equals(cl_smooth_5["smooth_dat"])
    assert cl_smooth_10["axes_ranges"] == cl_smooth_5["axes_ranges"]


def test_smooth_time_n_bins_no_effect():
    probs = {"m1": np.linspace(0.01, 0.99, 50)}
    reals = np.tile([0, 1], 25)
    times = np.ones(50)

    cl_time_10 = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[2.0],
        heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
        calibration_type="smooth",
        n_bins=10,
    )
    cl_time_5 = _create_calibration_curve_list_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[2.0],
        heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
        calibration_type="smooth",
        n_bins=5,
    )

    assert cl_time_10["smooth_dat"].equals(cl_time_5["smooth_dat"])
    assert cl_time_10["axes_ranges"] == cl_time_5["axes_ranges"]


def test_v1_and_v2_canonical_adapters_consume_calibration_bins_dat():
    p = np.linspace(0.1, 0.9, 10)
    y = np.tile([0, 1], 5)
    probs = {"m1": p}
    reals = y

    curve_list = _create_calibration_curve_list(probs, reals, n_bins=5)
    assert "calibration_bins_dat" in curve_list
    assert "deciles_dat" not in curve_list

    # v1 adapter
    v1_spec = _calibration_spec_from_curve_list(curve_list)
    assert v1_spec["type"] == "calibration"
    assert len(v1_spec["data"]) == 5

    # v2 adapter
    meta = {"m1": _EvaluationMetadata("m1", "eval1", "m1", "Pop 1")}
    v2_spec = _calibration_v2_spec_from_curve_list(curve_list, meta)
    assert v2_spec["type"] == "calibration"
    assert len(v2_spec["data"]) == 5

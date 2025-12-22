import polars as pl
from polars.testing import assert_frame_equal
from rtichoke.calibration.calibration import (
    _make_deciles_dat,
    _create_calibration_curve_list,
)


def test_make_deciles_dat():
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reals = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    result = _make_deciles_dat(probs, reals)
    expected = pl.DataFrame(
        {
            "quintile": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "sum_reals": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "total_obs": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    # The quintile calculation is not exactly the same as R's ntile,
    # so we will only check the other columns
    assert_frame_equal(
        result.drop("quintile"), expected.drop("quintile"), check_row_order=False
    )


def test_create_calibration_curve_list_single_population():
    probs = {"model_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    reals = {"pop_1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]}
    result = _create_calibration_curve_list(probs, reals, [], 500)

    assert result["performance_type"][0] == "one model"
    assert len(result["deciles_dat"]) > 0
    assert len(result["smooth_dat"]) > 0
    assert len(result["histogram_for_calibration"]) > 0


def test_create_calibration_curve_list_multiple_populations():
    probs = {
        "model_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "model_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    reals = {
        "pop_1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "pop_2": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    }
    result = _create_calibration_curve_list(probs, reals, [], 500)

    assert result["performance_type"][0] == "several populations"
    assert len(result["deciles_dat"]) > 0
    assert len(result["smooth_dat"]) > 0
    assert len(result["histogram_for_calibration"]) > 0
    # Check that the data is correctly grouped
    assert (
        len(
            pl.DataFrame(result["deciles_dat"])
            .filter(pl.col("reference_group") == "model_1")
        )
        > 0
    )
    assert (
        len(
            pl.DataFrame(result["deciles_dat"])
            .filter(pl.col("reference_group") == "model_2")
        )
        > 0
    )

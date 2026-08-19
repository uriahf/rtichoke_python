import polars as pl

from rtichoke.discrimination.gains import (
    _get_gains_aj_estimates_times,
    _replace_gains_reference_data_times,
)


def test_gains_reference_uses_cutoff_zero_event_probability():
    performance_data = pl.DataFrame(
        {
            "reference_group": ["model", "model"],
            "fixed_time_horizon": [5.0, 5.0],
            "chosen_cutoff": [0.0, 1.0],
            "real_positives": [20.0, 5.0],
            "n": [100.0, 100.0],
        }
    )

    aj = _get_gains_aj_estimates_times(performance_data)

    assert aj.height == 1
    assert aj["aj_estimate"].item() == 0.2

    curve_list = {
        "fixed_time_horizons": [5.0],
        "reference_data": pl.DataFrame(),
    }
    fixed = _replace_gains_reference_data_times(curve_list, performance_data)
    perfect = fixed["reference_data"].filter(
        pl.col("reference_group") == "perfect_model"
    )

    # For p=0.2, the perfect gains curve has y=x/p, so at x=0.1 y=0.5.
    y_at_point_one = perfect.filter(pl.col("x") == 0.1)["y"].item()
    assert y_at_point_one == 0.5

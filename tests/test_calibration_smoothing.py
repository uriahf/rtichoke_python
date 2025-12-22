import numpy as np
import polars as pl
from rtichoke.calibration.calibration import _calculate_smooth_curve

def test_calculate_smooth_curve_basic():
    # Create a sample DataFrame
    deciles_dat = pl.DataFrame({
        "reference_group": ["A", "A", "B", "B"],
        "x": [0.1, 0.9, 0.2, 0.8],
        "y": [0.2, 0.8, 0.3, 0.7]
    })
    performance_type = "multiple models"

    # Call the function
    smooth_dat = _calculate_smooth_curve(deciles_dat, performance_type)

    # Assert that the output is a DataFrame
    assert isinstance(smooth_dat, pl.DataFrame)

    # Assert that the DataFrame has the expected columns
    assert "x" in smooth_dat.columns
    assert "y" in smooth_dat.columns
    assert "reference_group" in smooth_dat.columns
    assert "text" in smooth_dat.columns

def test_calculate_smooth_curve_single_unique_x():
    # Create a sample DataFrame with a single unique x value for one group
    deciles_dat = pl.DataFrame({
        "reference_group": ["A", "A", "B", "B"],
        "x": [0.5, 0.5, 0.2, 0.8],
        "y": [0.2, 0.8, 0.3, 0.7]
    })
    performance_type = "multiple models"

    # Call the function
    smooth_dat = _calculate_smooth_curve(deciles_dat, performance_type)

    # Assert that the output is a DataFrame
    assert isinstance(smooth_dat, pl.DataFrame)

    # Check the smoothed output for group A
    group_a_smooth = smooth_dat.filter(pl.col("reference_group") == "A")
    assert group_a_smooth.shape[0] == 1
    assert group_a_smooth["x"][0] == 0.5
    assert group_a_smooth["y"][0] == (0.2 + 0.8) / 2

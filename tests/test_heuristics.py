import pytest
import polars as pl
from polars.testing import assert_frame_equal
from rtichoke.calibration.calibration import _apply_heuristics_and_censoring

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "real": [1, 0, 2, 1, 2, 0, 1],
        "time": [1, 2, 3, 8, 9, 10, 12],
    })

def test_competing_as_negative_logic(sample_data):
    # Heuristics that shouldn't change data before horizon
    result = _apply_heuristics_and_censoring(sample_data, 15, "adjusted", "adjusted_as_negative")
    # Competing events at times 3 and 9 should become 0.
    expected = pl.DataFrame({
        "real": [1, 0, 0, 1, 0, 0, 1],
        "time": [1, 2, 3, 8, 9, 10, 12],
    })
    assert_frame_equal(result, expected)

def test_admin_censoring(sample_data):
    result = _apply_heuristics_and_censoring(sample_data, 7, "adjusted", "adjusted_as_negative")
    # Admin censoring for times > 7. Competing event at time=3 becomes 0.
    expected = pl.DataFrame({
        "real": [1, 0, 0, 0, 0, 0, 0],
        "time": [1, 2, 3, 8, 9, 10, 12],
    })
    assert_frame_equal(result, expected)

def test_censoring_excluded(sample_data):
    result = _apply_heuristics_and_censoring(sample_data, 10, "excluded", "adjusted_as_negative")
    # Excludes censored at times 2, 10. Admin censors time > 10. Competing at 3,9 -> 0.
    expected = pl.DataFrame({
        "real": [1, 0, 1, 0, 0],
        "time": [1, 3, 8, 9, 12],
    })
    assert_frame_equal(result.sort("time"), expected.sort("time"))

def test_competing_excluded(sample_data):
    result = _apply_heuristics_and_censoring(sample_data, 10, "adjusted", "excluded")
    # Excludes competing at 3, 9. Admin censors time > 10.
    expected = pl.DataFrame({
        "real": [1, 0, 1, 0, 0],
        "time": [1, 2, 8, 10, 12],
    })
    assert_frame_equal(result.sort("time"), expected.sort("time"))

def test_competing_as_negative(sample_data):
    result = _apply_heuristics_and_censoring(sample_data, 10, "adjusted", "adjusted_as_negative")
    # Competing at 3,9 -> 0. Admin censors time > 10.
    expected = pl.DataFrame({
        "real": [1, 0, 0, 1, 0, 0, 0],
        "time": [1, 2, 3, 8, 9, 10, 12],
    })
    assert_frame_equal(result, expected)

def test_competing_as_composite(sample_data):
    result = _apply_heuristics_and_censoring(sample_data, 10, "adjusted", "adjusted_as_composite")
    # Competing at 3,9 -> 1. Admin censors time > 10.
    expected = pl.DataFrame({
        "real": [1, 0, 1, 1, 1, 0, 0],
        "time": [1, 2, 3, 8, 9, 10, 12],
    })
    assert_frame_equal(result, expected)

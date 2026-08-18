"""
Subpackage for Calibration
"""

from . import calibration as _calibration
from ._interactive_aspect import enforce_square_calibration_panel

_original_create_calibration_curve = _calibration.create_calibration_curve
_original_create_calibration_curve_times = _calibration.create_calibration_curve_times


def create_calibration_curve(*args, **kwargs):
    """Create an interactive calibration plot with a square main panel."""
    return enforce_square_calibration_panel(
        _original_create_calibration_curve(*args, **kwargs)
    )


def create_calibration_curve_times(*args, **kwargs):
    """Create an interactive time-dependent calibration plot with a square main panel."""
    return enforce_square_calibration_panel(
        _original_create_calibration_curve_times(*args, **kwargs)
    )


# Keep direct imports from rtichoke.calibration.calibration aligned with the
# public package entry points.
_calibration.create_calibration_curve = create_calibration_curve
_calibration.create_calibration_curve_times = create_calibration_curve_times

__all__ = ["create_calibration_curve", "create_calibration_curve_times"]

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


def _extract_heuristics_sets(args, kwargs):
    if "heuristics_sets" in kwargs:
        return kwargs["heuristics_sets"]
    if len(args) > 4:
        return args[4]
    return None


def create_calibration_curve_times(*args, **kwargs):
    """Create an interactive time-dependent calibration plot with a square main panel."""
    heuristics_sets = _extract_heuristics_sets(args, kwargs)
    if heuristics_sets is not None and len(heuristics_sets) != 1:
        raise ValueError(
            "create_calibration_curve_times() currently supports exactly one "
            "heuristics set. Multiple heuristic sets would be combined in the "
            "same plotted calibration curve. Call the function separately for "
            "each heuristic set."
        )

    return enforce_square_calibration_panel(
        _original_create_calibration_curve_times(*args, **kwargs)
    )


# Keep direct imports from rtichoke.calibration.calibration aligned with the
# public package entry points.
_calibration.create_calibration_curve = create_calibration_curve
_calibration.create_calibration_curve_times = create_calibration_curve_times

__all__ = ["create_calibration_curve", "create_calibration_curve_times"]

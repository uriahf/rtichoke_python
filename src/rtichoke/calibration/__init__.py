"""
Subpackage for Calibration
"""

from functools import wraps
from inspect import signature

from .calibration import create_calibration_curve
from .calibration import create_calibration_curve_times as _create_calibration_curve_times


@wraps(_create_calibration_curve_times)
def create_calibration_curve_times(*args, **kwargs):
    bound = signature(_create_calibration_curve_times).bind_partial(*args, **kwargs)
    heuristics_sets = bound.arguments.get("heuristics_sets")

    if heuristics_sets is not None:
        unsupported = [
            heuristics
            for heuristics in heuristics_sets
            if heuristics.get("censoring_heuristic") == "adjusted"
            or heuristics.get("competing_heuristic") == "adjusted_as_censored"
        ]
        if unsupported:
            raise ValueError(
                "Unsupported calibration heuristics: "
                "create_calibration_curve_times() does not support "
                "censoring_heuristic='adjusted' or "
                "competing_heuristic='adjusted_as_censored'. "
                "Use a supported heuristic combination such as "
                "censoring_heuristic='excluded' with "
                "competing_heuristic='adjusted_as_negative'."
            )

    return _create_calibration_curve_times(*args, **kwargs)


__all__ = ["create_calibration_curve", "create_calibration_curve_times"]

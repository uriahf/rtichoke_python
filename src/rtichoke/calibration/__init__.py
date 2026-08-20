"""
Subpackage for Calibration
"""

import numpy as np

from . import calibration as _calibration
from ._interactive_aspect import enforce_square_calibration_panel

_original_create_calibration_curve = _calibration.create_calibration_curve
_original_create_calibration_curve_times = _calibration.create_calibration_curve_times

_DEFAULT_TIME_CALIBRATION_HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]


def _argument(args, kwargs, name, position):
    return kwargs[name] if name in kwargs else args[position]


def _validate_probability_values(probs):
    for values in probs.values():
        values = np.asarray(values)
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("Estimated probabilities must be between 0 and 1.")


def _validate_outcome_values(reals, allowed_values):
    values = reals.values() if isinstance(reals, dict) else [reals]
    for outcome_values in values:
        if not np.all(np.isin(np.asarray(outcome_values), allowed_values)):
            if allowed_values == (0, 1):
                raise ValueError("Binary outcomes must contain only 0 and 1.")
            raise ValueError(
                "Time-dependent outcomes must contain only 0, 1, and 2."
            )


def create_calibration_curve(*args, **kwargs):
    """Create an interactive calibration plot with a square main panel."""
    probs = _argument(args, kwargs, "probs", 0)
    reals = _argument(args, kwargs, "reals", 1)
    _validate_probability_values(probs)
    _validate_outcome_values(reals, (0, 1))

    return enforce_square_calibration_panel(
        _original_create_calibration_curve(*args, **kwargs)
    )


def create_calibration_curve_times(*args, **kwargs):
    """Create an interactive time-dependent calibration plot with a square main panel."""
    probs = _argument(args, kwargs, "probs", 0)
    reals = _argument(args, kwargs, "reals", 1)
    _validate_probability_values(probs)
    _validate_outcome_values(reals, (0, 1, 2))

    heuristics_sets = kwargs.get("heuristics_sets")
    if heuristics_sets is None:
        heuristics_sets = [dict(_DEFAULT_TIME_CALIBRATION_HEURISTICS[0])]
        kwargs["heuristics_sets"] = heuristics_sets

    if len(heuristics_sets) != 1:
        raise ValueError(
            "create_calibration_curve_times() currently supports exactly one "
            "heuristic set. Multiple heuristic sets would be combined in the "
            "same calibration trace because the plot has no heuristic selector."
        )

    return enforce_square_calibration_panel(
        _original_create_calibration_curve_times(*args, **kwargs)
    )


# Keep direct imports from rtichoke.calibration.calibration aligned with the
# public package entry points.
_calibration.create_calibration_curve = create_calibration_curve
_calibration.create_calibration_curve_times = create_calibration_curve_times

__all__ = ["create_calibration_curve", "create_calibration_curve_times"]

"""Public API for standalone prediction distribution visualization."""

from __future__ import annotations

from typing import Any, Sequence

from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._viz_spec_v2 import (
    _prediction_distribution_v2_spec_from_performance_data,
)


def create_probs_histogram(
    probs: dict[str, Any],
    reals: Any,
    by: float = 0.01,
    stratified_by: Sequence[str] = ("probability_threshold",),
) -> RtichokeBrowserChart:
    """Create a standalone Prediction Distribution browser chart.

    This function generates a canonical prediction distribution chart displaying
    predicted probability histograms or prediction percentile rank distributions,
    decomposed by observed binary outcome, with interactive threshold and PPCR
    operating-point controls.

    In Probability Threshold mode, the x-axis represents predicted probability
    intervals. In PPCR (Predicted Positives Condition Rate) mode, the x-axis
    displays the producer-owned rank distribution (percentile strata).
    Bars in each bin are decomposed by observed binary outcome (positives and
    negatives). Moving the operating-point control changes the classification
    partition and updates performance metrics attached to each operating point.
    Because tied predictions are never split across boundaries, ties may cause
    the realized classified-positive proportion to differ from the requested PPCR.
    This visualization displays model distribution and performance across operating
    points and does not identify an optimal threshold.

    Parameters
    ----------
    probs : dict[str, numpy.ndarray]
        Nonempty dictionary mapping model or evaluation names to predicted
        probability arrays.
    reals : numpy.ndarray or dict[str, numpy.ndarray]
        True binary labels (0 or 1), as a single shared array or a dictionary of
        arrays matching keys in ``probs``.
    by : float, optional
        Grid resolution step size for evaluation cutoffs and percentiles.
        Defaults to ``0.01``.
    stratified_by : Sequence[str], optional
        Sequence containing exactly one stratification key, either
        ``("probability_threshold",)`` or ``("ppcr",)``. Defaults to
        ``("probability_threshold",)``.

    Returns
    -------
    RtichokeBrowserChart
        A browser chart instance that renders the canonical prediction distribution.

    Examples
    --------
    Create a prediction distribution chart in Probability Threshold mode:

    >>> import numpy as np
    >>> import rtichoke
    >>> probs = {"model_1": np.array([0.1, 0.4, 0.7, 0.8])}
    >>> reals = np.array([0, 0, 1, 1])
    >>> chart = rtichoke.create_probs_histogram(probs=probs, reals=reals)

    Create a prediction distribution chart in PPCR mode:

    >>> chart = rtichoke.create_probs_histogram(
    ...     probs=probs,
    ...     reals=reals,
    ...     stratified_by=("ppcr",),
    ... )

    Compare multiple models sharing one outcome population:

    >>> probs = {
    ...     "model_1": np.array([0.1, 0.4, 0.7, 0.8]),
    ...     "model_2": np.array([0.2, 0.3, 0.6, 0.9]),
    ... }
    >>> reals = np.array([0, 0, 1, 1])
    >>> chart = rtichoke.create_probs_histogram(probs=probs, reals=reals)

    Compare multiple populations with population-keyed outcome arrays:

    >>> probs = {
    ...     "pop_1": np.array([0.1, 0.4, 0.7, 0.8]),
    ...     "pop_2": np.array([0.2, 0.5, 0.9]),
    ... }
    >>> reals = {
    ...     "pop_1": np.array([0, 0, 1, 1]),
    ...     "pop_2": np.array([0, 1, 1]),
    ... }
    >>> chart = rtichoke.create_probs_histogram(probs=probs, reals=reals)
    """
    if isinstance(stratified_by, str):
        raise ValueError(
            f"`stratified_by` must be a sequence of strings (e.g. ({stratified_by!r},)), got plain string {stratified_by!r}."
        )

    if not isinstance(stratified_by, (list, tuple)):
        try:
            strat_tuple = tuple(stratified_by)
        except TypeError as err:
            raise ValueError("`stratified_by` must be a sequence of strings.") from err
    else:
        strat_tuple = tuple(stratified_by)

    if len(strat_tuple) != 1:
        raise ValueError(
            f"`stratified_by` must contain exactly one element, got {len(strat_tuple)} elements: {strat_tuple!r}."
        )

    dimension = strat_tuple[0]
    if dimension not in ("probability_threshold", "ppcr"):
        raise ValueError(
            f"Unsupported stratification key {dimension!r}. Must be 'probability_threshold' or 'ppcr'."
        )

    spec = _prediction_distribution_v2_spec_from_performance_data(
        probs=probs,
        reals=reals,
        by=by,
        stratified_by=strat_tuple,
    )
    return RtichokeBrowserChart(spec=spec)

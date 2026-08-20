"""Validation helpers for time-dependent performance inputs."""

from typing import Dict, Union

import numpy as np


def _validate_probability_values(probs: Dict[str, np.ndarray]) -> None:
    for values in probs.values():
        probs_values = np.asarray(values)
        if not np.all(np.isfinite(probs_values)) or np.any(
            (probs_values < 0) | (probs_values > 1)
        ):
            raise ValueError("Estimated probabilities must be between 0 and 1.")


def _validate_time_outcome_values(
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
) -> None:
    values = reals.values() if isinstance(reals, dict) else [reals]
    for outcome_values in values:
        if not np.all(np.isin(np.asarray(outcome_values), [0, 1, 2])):
            raise ValueError("Time-dependent outcomes must contain only 0, 1, and 2.")


def _validate_time_input_alignment(
    probs: Dict[str, np.ndarray],
    reals: Union[np.ndarray, Dict[str, np.ndarray]],
    times: Union[np.ndarray, Dict[str, np.ndarray]],
) -> None:
    """Validate supported array/dict layouts before time-dependent processing."""
    if not isinstance(probs, dict) or not probs:
        raise ValueError(
            "`probs` must be a non-empty dictionary of probability arrays."
        )

    _validate_probability_values(probs)
    _validate_time_outcome_values(reals)

    groups = list(probs)
    multiple_groups = len(groups) > 1
    reals_is_dict = isinstance(reals, dict)
    times_is_dict = isinstance(times, dict)

    if multiple_groups and reals_is_dict != times_is_dict:
        raise ValueError(
            "For multiple groups, `reals` and `times` must both be arrays or both be dictionaries."
        )

    if multiple_groups and reals_is_dict:
        expected_keys = set(groups)
        if set(reals) != expected_keys or set(times) != expected_keys:
            raise ValueError(
                "For multiple populations, `reals` and `times` dictionary keys must exactly match `probs`."
            )
        for group in groups:
            n_probs = len(np.asarray(probs[group]))
            n_reals = len(np.asarray(reals[group]))
            n_times = len(np.asarray(times[group]))
            if n_probs != n_reals or n_probs != n_times:
                raise ValueError(
                    f"Input lengths must match within group {group!r}: "
                    f"len(probs)={n_probs}, len(reals)={n_reals}, len(times)={n_times}."
                )
        return

    if multiple_groups:
        n_reals = len(np.asarray(reals))
        n_times = len(np.asarray(times))
        if n_reals != n_times:
            raise ValueError(
                "For multiple models sharing outcomes, `reals` and `times` must have the same length."
            )
        for group in groups:
            n_probs = len(np.asarray(probs[group]))
            if n_probs != n_reals:
                raise ValueError(
                    f"Shared outcome length must match probabilities for group {group!r}: "
                    f"len(probs)={n_probs}, len(reals)={n_reals}, len(times)={n_times}."
                )
        return

    group = groups[0]
    if reals_is_dict:
        if group not in reals:
            raise ValueError(
                f"`reals` is missing the key {group!r} required by `probs`."
            )
        reals_values = reals[group]
    else:
        reals_values = reals

    if times_is_dict:
        if group not in times:
            raise ValueError(
                f"`times` is missing the key {group!r} required by `probs`."
            )
        times_values = times[group]
    else:
        times_values = times

    n_probs = len(np.asarray(probs[group]))
    n_reals = len(np.asarray(reals_values))
    n_times = len(np.asarray(times_values))
    if n_probs != n_reals or n_probs != n_times:
        raise ValueError(
            f"Input lengths must match for group {group!r}: "
            f"len(probs)={n_probs}, len(reals)={n_reals}, len(times)={n_times}."
        )

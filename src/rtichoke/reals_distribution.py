"""Public API for standalone reals/outcome distribution visualization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from rtichoke._renderers import RtichokeBrowserChart

_DEFAULT_STATE_LABELS = {
    "real_positive": "Target event",
    "real_competing": "Competing outcome",
    "real_negative": "No target event",
    "real_censored": "Unknown / excluded",
}

_STATE_ORDER = [
    "real_positive",
    "real_competing",
    "real_negative",
    "real_censored",
]

_SUPPORTED_REALS_DISTRIBUTION_RENDERERS = ("browser", "rtichoke_viz")


def _outcome_distribution_v2_spec(
    reals: Union[np.ndarray, List[int], Dict[str, Any]],
    times: Union[np.ndarray, List[float], Dict[str, Any]],
    fixed_time_horizons: List[float],
    *,
    title: str = "Outcome Distribution",
    state_labels: Optional[Dict[str, str]] = None,
) -> dict[str, Any]:
    """Build a canonical v2 outcome_distribution spec using raw counts."""
    labels = dict(_DEFAULT_STATE_LABELS)
    if state_labels is not None:
        labels.update(state_labels)

    eval_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if isinstance(reals, dict) or isinstance(times, dict):
        if not isinstance(reals, dict) or not isinstance(times, dict):
            raise ValueError(
                "When 'reals' or 'times' is a dict, both must be dicts with matching keys."
            )
        if set(reals.keys()) != set(times.keys()):
            raise ValueError(
                "'reals' and 'times' dictionaries must have matching keys."
            )
        for key in reals:
            eval_data[key] = (
                np.asarray(reals[key], dtype=int),
                np.asarray(times[key], dtype=float),
            )
    else:
        eval_data["__shared_population__"] = (
            np.asarray(reals, dtype=int),
            np.asarray(times, dtype=float),
        )

    evaluations: list[dict[str, Any]] = []
    state_distributions: list[dict[str, Any]] = []

    for index, (group_name, (reals_arr, times_arr)) in enumerate(
        eval_data.items(), start=1
    ):
        eval_id = f"evaluation-{index}"

        eval_item: dict[str, Any] = {
            "id": eval_id,
            "population": group_name,
        }
        evaluations.append(eval_item)

        n_obs = len(reals_arr)
        if len(times_arr) != n_obs:
            raise ValueError(
                f"Length mismatch between reals ({n_obs}) and times ({len(times_arr)}) for {group_name!r}."
            )

        # 1. fixed_time_horizon stateDistributions
        horizons_float = [float(h) for h in fixed_time_horizons]
        if 0.0 not in horizons_float:
            fixed_horizons = [0.0] + horizons_float
        else:
            fixed_horizons = horizons_float

        for h in fixed_horizons:
            pos_count = int(np.sum((reals_arr == 1) & (times_arr <= h)))
            comp_count = int(np.sum((reals_arr == 2) & (times_arr <= h)))
            cens_count = int(np.sum((reals_arr == 0) & (times_arr < h)))
            neg_count = n_obs - (pos_count + comp_count + cens_count)

            counts_map = {
                "real_positive": pos_count,
                "real_competing": comp_count,
                "real_negative": neg_count,
                "real_censored": cens_count,
            }

            states_list = [
                {
                    "stateId": sid,
                    "label": labels[sid],
                    "count": counts_map[sid],
                }
                for sid in _STATE_ORDER
            ]

            state_distributions.append(
                {
                    "evaluationId": eval_id,
                    "horizon": float(h),
                    "estimator": "raw",
                    "estimateOrigin": "fixed_time_horizon",
                    "states": states_list,
                }
            )

        # 2. event_table stateDistributions
        unique_event_times = sorted(set([0.0] + [float(t_val) for t_val in times_arr]))
        for t_e in unique_event_times:
            pos_count = int(np.sum((reals_arr == 1) & (times_arr <= t_e)))
            comp_count = int(np.sum((reals_arr == 2) & (times_arr <= t_e)))
            cens_count = int(np.sum((reals_arr == 0) & (times_arr < t_e)))
            neg_count = n_obs - (pos_count + comp_count + cens_count)

            counts_map = {
                "real_positive": pos_count,
                "real_competing": comp_count,
                "real_negative": neg_count,
                "real_censored": cens_count,
            }

            states_list = [
                {
                    "stateId": sid,
                    "label": labels[sid],
                    "count": counts_map[sid],
                }
                for sid in _STATE_ORDER
            ]

            state_distributions.append(
                {
                    "evaluationId": eval_id,
                    "horizon": float(t_e),
                    "estimator": "raw",
                    "estimateOrigin": "event_table",
                    "states": states_list,
                }
            )

    spec: dict[str, Any] = {
        "schemaVersion": "2.0",
        "type": "outcome_distribution",
        "title": title,
        "evaluations": evaluations,
        "stateDistributions": state_distributions,
    }

    return spec


def create_reals_distribution_times(
    reals: Union[np.ndarray, List[int], Dict[str, Any]],
    times: Union[np.ndarray, List[float], Dict[str, Any]],
    fixed_time_horizons: List[float],
    *,
    renderer: str = "browser",
    state_labels: Optional[Dict[str, str]] = None,
) -> RtichokeBrowserChart:
    """Create a time-dependent reals/outcome distribution browser chart."""
    if renderer not in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS:
        supported = ", ".join(
            repr(r) for h, r in enumerate(_SUPPORTED_REALS_DISTRIBUTION_RENDERERS)
        )
        raise ValueError(
            f"Unsupported renderer {renderer!r}. 'create_reals_distribution_times' supports {supported}."
        )

    spec = _outcome_distribution_v2_spec(
        reals=reals,
        times=times,
        fixed_time_horizons=fixed_time_horizons,
        title="Outcome Distribution",
        state_labels=state_labels,
    )
    return RtichokeBrowserChart(spec=spec)

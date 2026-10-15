"""Public API for standalone reals/outcome distribution visualization."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import polarstate
import polars as pl

from rtichoke._renderers import RtichokeBrowserChart

RealsDistributionRenderer = Literal["browser", "rtichoke_viz"]
StateScale = Literal["auto", "nominal", "ordinal"]
CensoringHeuristic = Literal["excluded", "adjusted"]

_DEFAULT_STATE_LABELS = {
    "real_positive": "Event",
    "real_competing": "Competing event",
    "real_negative": "Event-free",
    "real_censored": "Censored",
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

    for h in fixed_time_horizons:
        h_float = float(h)
        if not math.isfinite(h_float) or h_float < 0.0:
            raise ValueError(
                f"Invalid fixed_time_horizons value {h!r}: horizons must be finite and non-negative."
            )

    if isinstance(reals, dict) or isinstance(times, dict):
        if not isinstance(reals, dict) or not isinstance(times, dict):
            raise ValueError(
                "When 'reals' or 'times' is a dict, both must be dicts with matching keys."
            )
        if list(reals.keys()) != list(times.keys()):
            raise ValueError(
                "'reals' and 'times' dictionaries must have matching keys and key order."
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

    fixed_horizons = sorted(set([0.0] + [float(h) for h in fixed_time_horizons]))

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
    renderer: RealsDistributionRenderer = "browser",
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
    )
    return RtichokeBrowserChart(spec=spec)


def _deterministic_state_sort(states: list[Any]) -> list[Any]:
    if all(
        isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(s, bool)
        for s in states
    ):
        return sorted(states)
    return sorted(states, key=lambda s: str(s))


def create_reals_distribution_steps(
    data: pl.DataFrame,
    *,
    id_col: str = "id",
    step_col: str = "step",
    state_col: str = "reals",
    state_order: Optional[Sequence[Any]] = None,
    state_scale: StateScale = "auto",
    censoring_heuristic: CensoringHeuristic = "excluded",
    title: str = "Outcome Distribution",
    renderer: RealsDistributionRenderer = "browser",
) -> RtichokeBrowserChart:
    """Create a discrete-step outcome distribution browser chart."""
    if renderer not in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS:
        supported = ", ".join(repr(r) for r in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS)
        raise ValueError(
            f"Unsupported renderer {renderer!r}. 'create_reals_distribution_steps' supports {supported}."
        )

    if censoring_heuristic not in ("excluded", "adjusted"):
        raise ValueError(
            f"Unsupported censoring_heuristic {censoring_heuristic!r}. Expected 'excluded' or 'adjusted'."
        )

    if state_scale not in ("auto", "nominal", "ordinal"):
        raise ValueError(
            f"Unsupported state_scale {state_scale!r}. Expected 'auto', 'nominal', or 'ordinal'."
        )

    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a Polars DataFrame (pl.DataFrame).")

    if data.is_empty():
        raise ValueError("data cannot be empty.")

    missing_cols = [
        col for col in (id_col, step_col, state_col) if col not in data.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in data: {missing_cols}")

    if data[id_col].null_count() > 0:
        raise ValueError(f"Column {id_col!r} contains null values.")

    if data[step_col].null_count() > 0:
        raise ValueError(f"Column {step_col!r} contains null values.")

    if data[state_col].null_count() > 0:
        raise ValueError(f"Column {state_col!r} contains null values.")

    step_series = data[step_col]
    if not step_series.dtype.is_integer():
        if step_series.dtype.is_float():
            if not (step_series == step_series.round(0)).all():
                raise ValueError(f"Column {step_col!r} must contain integer values.")
        else:
            raise TypeError(f"Column {step_col!r} must be integer-valued.")

    if (step_series < 0).any():
        raise ValueError(f"Steps in column {step_col!r} must be non-negative.")

    if data.select([id_col, step_col]).is_duplicated().any():
        raise ValueError(
            f"Duplicate records found for subject-step pairs in ({id_col!r}, {step_col!r})."
        )

    state_series = data[state_col]
    if state_series.dtype == pl.Object:
        unique_observed_states = list(dict.fromkeys(state_series.to_list()))
    else:
        unique_observed_states = state_series.unique().to_list()
    if len(unique_observed_states) == 0:
        raise ValueError("data contains no observed states.")

    string_ids = [str(s) for s in unique_observed_states]
    if len(set(string_ids)) < len(unique_observed_states):
        raise ValueError(
            "State identity collision detected: distinct state values map to identical string identifiers."
        )

    if state_order is not None:
        state_order_list = list(state_order)
        if set(state_order_list) != set(unique_observed_states) or len(
            state_order_list
        ) != len(set(state_order_list)):
            raise ValueError(
                "state_order must contain every observed state exactly once."
            )
        resolved_state_order = state_order_list
    elif isinstance(data[state_col].dtype, (pl.Enum, pl.Categorical)):
        dtype = data[state_col].dtype
        categories = None
        if isinstance(dtype, pl.Enum):
            categories = dtype.categories.to_list()
        elif isinstance(dtype, pl.Categorical):
            try:
                categories = data[state_col].cat.get_categories().to_list()
            except Exception:
                categories = None

        if categories:
            obs_set = set(unique_observed_states)
            resolved_state_order = [c for c in categories if c in obs_set]
            for s in unique_observed_states:
                if s not in resolved_state_order:
                    resolved_state_order.append(s)
        else:
            resolved_state_order = _deterministic_state_sort(unique_observed_states)
    else:
        resolved_state_order = _deterministic_state_sort(unique_observed_states)

    if state_scale == "nominal":
        resolved_state_scale = "nominal"
    elif state_scale == "ordinal":
        resolved_state_scale = "ordinal"
    else:
        is_ordered_cat = False
        if isinstance(data[state_col].dtype, pl.Enum):
            is_ordered_cat = True
        elif isinstance(data[state_col].dtype, pl.Categorical):
            try:
                is_ordered_cat = (
                    data[state_col].dtype.is_ordered
                    if hasattr(data[state_col].dtype, "is_ordered")
                    else False
                )
            except Exception:
                is_ordered_cat = False

        all_numeric = all(
            isinstance(s, (int, float, np.integer, np.floating))
            and not isinstance(s, bool)
            for s in unique_observed_states
        )

        if is_ordered_cat or all_numeric:
            resolved_state_scale = "ordinal"
        else:
            resolved_state_scale = "nominal"

    # Note: resolved_state_scale is resolved internally for future palette selection contract.
    _ = resolved_state_scale

    observed_steps = sorted([int(s) for s in data[step_col].unique().to_list()])
    if len(observed_steps) == 0:
        raise ValueError("No steps found in data.")

    state_distributions: list[dict[str, Any]] = []

    if censoring_heuristic == "excluded":
        for step_val in observed_steps:
            step_df = data.filter(pl.col(step_col) == step_val)
            if step_df.is_empty():
                raise ValueError(
                    f"At least one observed state required for step {step_val}."
                )

            counts = step_df[state_col].value_counts()
            count_map = {
                row[state_col]: row["count"] for row in counts.iter_rows(named=True)
            }

            states_list = [
                {
                    "stateId": str(s),
                    "label": str(s),
                    "count": int(count_map.get(s, 0)),
                }
                for s in resolved_state_order
            ]

            state_distributions.append(
                {
                    "evaluationId": "evaluation-1",
                    "horizon": int(step_val),
                    "estimator": "raw",
                    "estimateOrigin": "observed_step",
                    "states": states_list,
                }
            )
    else:
        if not hasattr(polarstate, "prepare_transition_table_steps"):
            raise NotImplementedError(
                "Discrete-step Aalen-Johansen estimation ('censoring_heuristic=\"adjusted\"') "
                "requires polarstate >= 0.1.9."
            )
        tt = polarstate.prepare_transition_table_steps(
            data,
            id_col=id_col,
            step_col=step_col,
            state_col=state_col,
            state_order=resolved_state_order,
        )
        if hasattr(polarstate, "_predict_transition_steps"):
            aj_df = polarstate._predict_transition_steps(tt)
        else:
            aj_df = polarstate.predict_aj_estimates(tt)

        for step_val in observed_steps:
            step_aj = aj_df.filter(pl.col("step") == step_val)
            prob_map = {
                row["state"]: row["state_occupancy_probability"]
                for row in step_aj.iter_rows(named=True)
            }

            states_list = [
                {
                    "stateId": str(s),
                    "label": str(s),
                    "estimate": float(prob_map.get(s, 0.0)),
                }
                for s in resolved_state_order
            ]

            state_distributions.append(
                {
                    "evaluationId": "evaluation-1",
                    "horizon": int(step_val),
                    "estimator": "aalen_johansen",
                    "estimateOrigin": "observed_step",
                    "states": states_list,
                }
            )

    spec: dict[str, Any] = {
        "schemaVersion": "2.0",
        "type": "outcome_distribution",
        "title": title,
        "evaluations": [
            {
                "id": "evaluation-1",
                "population": "__shared_population__",
            }
        ],
        "stateDistributions": state_distributions,
    }

    return RtichokeBrowserChart(spec=spec)

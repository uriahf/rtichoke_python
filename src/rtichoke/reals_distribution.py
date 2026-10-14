"""Public API for standalone reals/outcome distribution visualization."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from rtichoke._renderers import RtichokeBrowserChart

RealsDistributionRenderer = Literal["browser", "rtichoke_viz"]

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


def _outcome_distribution_steps_v2_spec(
    data: pl.DataFrame,
    *,
    subject_col: str = "subject",
    step_col: str = "step",
    state_col: str = "state",
    state_order: Sequence[str | int | Any] | None = None,
    title: str = "Outcome Distribution",
) -> dict[str, Any]:
    """Build a canonical v2 outcome_distribution spec for observed step distributions."""
    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"Expected 'data' to be a Polars DataFrame (pl.DataFrame), got {type(data).__name__}."
        )

    for col_name in (subject_col, step_col, state_col):
        if col_name not in data.columns:
            raise ValueError(
                f"Column {col_name!r} not found in DataFrame columns: {data.columns}."
            )

    df = data.select([subject_col, step_col, state_col])

    # 1. Reject missing subject identifiers or step coordinates
    if df[subject_col].null_count() > 0:
        raise ValueError(f"Column {subject_col!r} contains null values.")
    if df[step_col].null_count() > 0:
        raise ValueError(f"Column {step_col!r} contains null values.")

    # 2. Reject duplicate subject-step pairs
    if df.select([subject_col, step_col]).is_duplicated().any():
        raise ValueError("Duplicate subject-step pairs found in input data.")

    # 3. Validate step coordinates
    steps_raw = df[step_col].to_list()
    validated_steps = []
    for step_val in steps_raw:
        if isinstance(step_val, bool):
            raise ValueError(
                f"Invalid step coordinate {step_val!r}: step coordinates must be integer-valued non-negative numbers."
            )
        if isinstance(step_val, (int, np.integer)):
            step_int = int(step_val)
        elif isinstance(step_val, (float, np.floating)):
            if not math.isfinite(step_val) or not step_val.is_integer():
                raise ValueError(
                    f"Invalid step coordinate {step_val!r}: step coordinates must be integer-valued non-negative numbers."
                )
            step_int = int(step_val)
        else:
            try:
                f_val = float(step_val)
                if not math.isfinite(f_val) or not f_val.is_integer():
                    raise ValueError
                step_int = int(f_val)
            except Exception as err:
                raise ValueError(
                    f"Invalid step coordinate {step_val!r}: step coordinates must be integer-valued non-negative numbers."
                ) from err

        if step_int < 0:
            raise ValueError(
                f"Invalid step coordinate {step_val!r}: step coordinates must be non-negative."
            )
        validated_steps.append(step_int)

    df = df.with_columns(pl.Series(step_col, validated_steps, dtype=pl.Int64))

    # 4. Filter out rows with null state
    valid_df = df.filter(pl.col(state_col).is_not_null())

    if valid_df.is_empty():
        raise ValueError(
            "No valid observed steps remain after excluding missing states."
        )

    # Extract list of observed non-null state values directly
    raw_state_list = valid_df[state_col].to_list()
    observed_state_vals = []
    seen = set()
    for val in raw_state_list:
        if val not in seen:
            seen.add(val)
            observed_state_vals.append(val)

    # Determine state ordering and map to string stateId / label
    # Rules:
    # State values mapped to str stateId. If collision, raise ValueError.
    state_id_map: dict[Any, str] = {}
    for val in observed_state_vals:
        sid = str(val)
        if sid in state_id_map.values():
            existing_val = [k for k, v in state_id_map.items() if v == sid][0]
            if existing_val != val:
                raise ValueError(
                    f"State value collision: {val!r} and {existing_val!r} both map to stateId {sid!r}."
                )
        state_id_map[val] = sid

    if state_order is not None:
        state_order_list = list(state_order)
        # Validate state_order
        # Must contain no duplicates
        if len(state_order_list) != len(set(state_order_list)):
            raise ValueError("`state_order` contains duplicate values.")

        order_ids = [str(x) for x in state_order_list]
        observed_ids = set(state_id_map.values())

        if set(order_ids) != observed_ids:
            raise ValueError(
                f"`state_order` does not match observed states exactly. "
                f"Expected exact set {sorted(observed_ids)}, got {sorted(order_ids)}."
            )

        canonical_state_ids = order_ids
    else:
        state_series = valid_df[state_col]
        if isinstance(state_series.dtype, pl.Enum):
            # Enum has declared categories order
            categories = state_series.dtype.categories.to_list()
            canonical_state_ids = [
                str(c) for c in categories if str(c) in state_id_map.values()
            ]
        else:
            # Deterministic sorting
            # Sort by string stateId
            sorted_vals = sorted(
                observed_state_vals,
                key=lambda x: (str(x), x if isinstance(x, (int, float)) else 0),
            )
            canonical_state_ids = [str(v) for v in sorted_vals]

    # Map stateId -> label (default label is string representation, which is stateId)
    state_objects_order = [
        {"stateId": sid, "label": sid} for sid in canonical_state_ids
    ]

    # Group by step and state to get counts
    unique_steps = sorted(valid_df[step_col].unique().to_list())

    state_distributions: list[dict[str, Any]] = []

    eval_id = "evaluation-1"
    evaluations = [
        {
            "id": eval_id,
            "population": "__shared_population__",
        }
    ]

    for step in unique_steps:
        step_df = valid_df.filter(pl.col(step_col) == step)
        state_counts_df = step_df.group_by(state_col).len()
        counts_map = {
            str(row[state_col]): row["len"]
            for row in state_counts_df.iter_rows(named=True)
        }

        states_list = [
            {
                "stateId": sid,
                "label": sid,
                "count": counts_map.get(sid, 0),
            }
            for sid in canonical_state_ids
        ]

        state_distributions.append(
            {
                "evaluationId": eval_id,
                "horizon": int(step),
                "estimator": "raw",
                "estimateOrigin": "observed_step",
                "states": states_list,
            }
        )

    spec: dict[str, Any] = {
        "schemaVersion": "2.0",
        "type": "outcome_distribution",
        "xAxisKind": "step",
        "title": title,
        "stateOrder": [st["stateId"] for st in state_objects_order],
        "evaluations": evaluations,
        "stateDistributions": state_distributions,
    }

    return spec


def create_reals_distribution_steps(
    data: pl.DataFrame,
    *,
    subject_col: str = "subject",
    step_col: str = "step",
    state_col: str = "state",
    state_order: Sequence[str | int | Any] | None = None,
    title: str = "Outcome Distribution",
    renderer: RealsDistributionRenderer = "browser",
) -> RtichokeBrowserChart:
    """Calculate empirical distributions of observed longitudinal states at discrete steps."""
    if renderer not in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS:
        supported = ", ".join(repr(r) for r in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS)
        raise ValueError(
            f"Unsupported renderer {renderer!r}. 'create_reals_distribution_steps' supports {supported}."
        )

    spec = _outcome_distribution_steps_v2_spec(
        data=data,
        subject_col=subject_col,
        step_col=step_col,
        state_col=state_col,
        state_order=state_order,
        title=title,
    )
    return RtichokeBrowserChart(spec=spec)


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

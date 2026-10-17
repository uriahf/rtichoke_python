"""Public API for standalone reals/outcome distribution visualization."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl
import polarstate

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


def create_reals_distribution_steps(
    data: pl.DataFrame,
    *,
    id_col: str = "id",
    step_col: str = "step",
    reals_col: str = "reals",
    reals_order: Sequence[Any] | None = None,
    reals_scale: Literal["auto", "nominal", "ordinal"] = "auto",
    censoring_heuristic: Literal["excluded", "adjusted"] = "excluded",
    title: str = "Outcome Distribution",
    renderer: RealsDistributionRenderer = "browser",
) -> RtichokeBrowserChart:
    """Create a discrete-step reals/outcome distribution browser chart."""
    if not isinstance(data, pl.DataFrame):
        raise TypeError(f"data must be a Polars DataFrame, got {type(data).__name__}")

    if renderer not in _SUPPORTED_REALS_DISTRIBUTION_RENDERERS:
        supported = ", ".join(
            repr(r) for h, r in enumerate(_SUPPORTED_REALS_DISTRIBUTION_RENDERERS)
        )
        raise ValueError(
            f"Unsupported renderer {renderer!r}. 'create_reals_distribution_steps' supports {supported}."
        )

    if censoring_heuristic not in ("excluded", "adjusted"):
        raise ValueError(
            f"Invalid censoring_heuristic {censoring_heuristic!r}. Expected 'excluded' or 'adjusted'."
        )

    if reals_scale not in ("auto", "nominal", "ordinal"):
        raise ValueError(
            f"Invalid reals_scale {reals_scale!r}. Expected 'auto', 'nominal', or 'ordinal'."
        )

    # Validate column presence
    missing_cols = [c for c in (id_col, step_col, reals_col) if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    # Validate non-null IDs and steps
    if data[id_col].null_count() > 0:
        raise ValueError(f"Column {id_col!r} contains missing/null values.")

    if data[step_col].null_count() > 0:
        raise ValueError(f"Column {step_col!r} contains missing/null values.")

    # Validate steps are non-negative integers
    step_series = data[step_col]
    if not step_series.dtype.is_integer():
        # Check if values can be losslessly represented as non-negative integers
        try:
            step_floats = step_series.to_numpy().astype(float)
            if not np.all(np.equal(np.mod(step_floats, 1), 0)):
                raise ValueError
        except Exception:
            raise ValueError(
                f"Column {step_col!r} must contain non-negative integer values."
            )

    if (step_series < 0).any():
        raise ValueError(f"Column {step_col!r} contains negative step values.")

    # Reject duplicate subject-step records
    dup_count = data.select([id_col, step_col]).is_duplicated().sum()
    if dup_count > 0:
        raise ValueError(
            f"Duplicate records found for subject-step pairs ({id_col!r}, {step_col!r})."
        )

    # Determine resolved outcome categories (reals_order)
    reals_dtype = data[reals_col].dtype

    if reals_order is not None:
        resolved_reals_order = [str(x) for x in reals_order]
        # Verify no identity collisions in string conversions
        if len(set(resolved_reals_order)) != len(reals_order):
            raise ValueError(
                "Distinct values in 'reals_order' collapsed into identical string representations."
            )
    elif isinstance(reals_dtype, pl.Enum):
        resolved_reals_order = [str(x) for x in reals_dtype.categories]
    else:
        # Unique non-null observed values, sorted deterministically
        non_null_vals = data[reals_col].drop_nulls().unique()
        if non_null_vals.len() == 0:
            resolved_reals_order = []
        else:
            sorted_vals = non_null_vals.sort()
            resolved_reals_order = [str(x) for x in sorted_vals.to_list()]

    # Validate string identity uniqueness
    if len(set(resolved_reals_order)) != len(resolved_reals_order):
        raise ValueError(
            "Distinct category values collapsed into duplicate string representations in reals_order."
        )

    # Resolve reals_scale
    if reals_scale == "auto":
        if (
            reals_order is not None
            or isinstance(reals_dtype, pl.Enum)
            or reals_dtype.is_numeric()
        ):
            resolved_scale = "ordinal"
        else:
            resolved_scale = "nominal"
    else:
        resolved_scale = reals_scale

    evaluations = [
        {
            "id": "evaluation-1",
            "population": "__shared_population__",
        }
    ]

    state_distributions: list[dict[str, Any]] = []

    if censoring_heuristic == "excluded":
        # Raw empirical distribution per step
        steps_present = sorted(data[step_col].unique().to_list())

        for step_val in steps_present:
            step_sub = data.filter(pl.col(step_col) == step_val)
            observed_reals = step_sub[reals_col].drop_nulls()

            if observed_reals.len() == 0:
                continue

            # Count occurrences of each category
            counts_df = observed_reals.cast(pl.String).value_counts()
            counts_dict = dict(
                zip(counts_df[reals_col].to_list(), counts_df["count"].to_list())
            )

            states_list = [
                {
                    "stateId": cat_str,
                    "label": cat_str,
                    "count": int(counts_dict.get(cat_str, 0)),
                }
                for cat_str in resolved_reals_order
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
        # AJ-adjusted distribution using polarstate
        tt = polarstate.prepare_transition_table_steps(
            data,
            id_col=id_col,
            step_col=step_col,
            state_col=reals_col,
            state_order=resolved_reals_order,
        )
        aj_df = polarstate.predict_aj_estimates(tt)

        # Build stateDistributions from aj_df
        # aj_df has columns: 'step', 'state', 'state_occupancy_probability'
        steps_present = sorted(aj_df["step"].unique().to_list())

        for step_val in steps_present:
            step_sub = aj_df.filter(pl.col("step") == step_val)
            prob_dict = dict(
                zip(
                    step_sub["state"].cast(pl.String).to_list(),
                    step_sub["state_occupancy_probability"].to_list(),
                )
            )

            states_list = [
                {
                    "stateId": cat_str,
                    "label": cat_str,
                    "estimate": float(prob_dict.get(cat_str, 0.0)),
                }
                for cat_str in resolved_reals_order
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
        "stateScale": resolved_scale,
        "evaluations": evaluations,
        "stateDistributions": state_distributions,
    }

    return RtichokeBrowserChart(spec=spec)

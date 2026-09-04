"""Canonical calibration-v2 adapter for existing production calibration outputs.

The builder in this module deliberately consumes the already-computed calibration
curve list produced by :mod:`rtichoke.calibration.calibration`. It does not group
predictions, fit smoothers, estimate event risks, or construct histogram bins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import polars as pl

from rtichoke.processing.evaluation_semantics import _EvaluationMetadata

_CALIBRATION_METHODS = {"discrete", "smooth"}


def _calibration_v2_spec_from_curve_list(
    calibration_curve_list: Mapping[str, Any],
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    calibration_type: str = "discrete",
) -> dict[str, object]:
    """Map production calibration results to a complete canonical v2 spec.

    ``reference_group`` is only a join key between already-computed production
    rows and semantic evaluation metadata. Evaluation and series identities are
    deterministic ordinals and therefore do not encode compatibility labels.
    """
    if calibration_type not in _CALIBRATION_METHODS:
        supported = ", ".join(sorted(_CALIBRATION_METHODS))
        raise ValueError(
            f"Unsupported calibration type {calibration_type!r}; expected {supported}."
        )

    point_key = "calibration_bins_dat" if calibration_type == "discrete" else "smooth_dat"
    point_frame = _require_frame(calibration_curve_list, point_key)
    distribution_frame = _require_frame(
        calibration_curve_list, "histogram_for_calibration"
    )

    required_point_columns = {"reference_group", "x", "y"}
    if calibration_type == "discrete":
        required_point_columns |= {"n_reals", "n"}
    _require_columns(point_frame, required_point_columns, point_key)
    _require_columns(
        distribution_frame,
        {"reference_group", "mids", "counts"},
        "histogram_for_calibration",
    )

    has_horizon = "fixed_time_horizon" in point_frame.columns
    if has_horizon != ("fixed_time_horizon" in distribution_frame.columns):
        raise ValueError(
            "Calibration points and distribution must either both include "
            "fixed_time_horizon or both omit it."
        )

    point_rows = point_frame.to_dicts()
    distribution_rows = distribution_frame.to_dicts()
    row_groups = {str(row["reference_group"]) for row in point_rows + distribution_rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        groups = ", ".join(sorted(missing_metadata))
        raise ValueError("Calibration rows are missing evaluation metadata: " + groups)

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }

    series_keys: list[tuple[str, float | None]] = []
    for group in ordered_groups:
        group_rows = [row for row in point_rows if str(row["reference_group"]) == group]
        horizons = (
            sorted(
                {float(cast(float, row["fixed_time_horizon"])) for row in group_rows}
            )
            if has_horizon
            else [None]
        )
        series_keys.extend((group, horizon) for horizon in horizons)

    series_ids = {
        key: f"series-{index}" for index, key in enumerate(series_keys, start=1)
    }

    evaluations: list[dict[str, object]] = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: dict[str, object] = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

    series: list[dict[str, object]] = []
    for group, horizon in series_keys:
        metadata = evaluation_metadata[group]
        if metadata.model is not None:
            display_value = metadata.model
            display_role = "model"
        else:
            display_value = metadata.population
            display_role = "population"
        item: dict[str, object] = {
            "id": series_ids[(group, horizon)],
            "evaluationId": evaluation_ids[group],
            "display": {
                "label": display_value,
                "group": display_value,
                "role": display_role,
            },
        }
        if horizon is not None:
            item["horizon"] = horizon
        series.append(item)

    data: list[dict[str, object]] = []
    for row in point_rows:
        group = str(row["reference_group"])
        horizon = float(cast(float, row["fixed_time_horizon"])) if has_horizon else None
        datum: dict[str, object] = {
            "seriesId": series_ids[(group, horizon)],
            "predicted": row["x"],
            "observed": row["y"],
            "method": calibration_type,
        }
        if calibration_type == "discrete":
            datum["events"] = row["n_reals"]
            datum["total"] = row["n"]
        data.append(datum)

    distribution: list[dict[str, object]] = []
    for row in distribution_rows:
        group = str(row["reference_group"])
        horizon = float(cast(float, row["fixed_time_horizon"])) if has_horizon else None
        distribution.append(
            {
                "seriesId": series_ids[(group, horizon)],
                "midpoint": row["mids"],
                "count": row["counts"],
                # Production histogram construction uses fixed 0.01-wide bins;
                # this records that existing geometry rather than rebuilding it.
                "binWidth": 0.01,
            }
        )

    y_axis: dict[str, object] = {"label": "Observed probability"}
    if calibration_type == "discrete":
        y_axis["domain"] = [0, 1]

    return {
        "schemaVersion": "2.0",
        "type": "calibration",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "distribution": distribution,
        "x": "predicted",
        "y": "observed",
        "xAxis": {"label": "Predicted probability", "domain": [0, 1]},
        "yAxis": y_axis,
        "references": [{"type": "identity", "scope": "global"}],
    }


def _require_frame(calibration_curve_list: Mapping[str, Any], key: str) -> pl.DataFrame:
    value = calibration_curve_list.get(key)
    if not isinstance(value, pl.DataFrame):
        raise ValueError(f"Calibration production output is missing DataFrame {key!r}.")
    return value


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Calibration {label} is missing columns: {columns}")

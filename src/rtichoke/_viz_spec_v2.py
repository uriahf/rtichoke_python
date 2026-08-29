"""Internal builders for canonical ``rtichoke_viz`` v2 specifications.

These helpers are deliberately not wired into production rendering yet. They
translate already-computed performance data plus semantic evaluation metadata
into the canonical visualization contract.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from rtichoke.processing.evaluation_semantics import _EvaluationMetadata

_REQUIRED_ROC_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "sensitivity",
    "specificity",
}
_REQUIRED_PRECISION_RECALL_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "sensitivity",
    "ppv",
    "real_positives",
    "n",
}
_REQUIRED_GAINS_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "sensitivity",
    "ppcr",
    "real_positives",
    "n",
}
_REQUIRED_LIFT_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "lift",
    "ppcr",
    "real_positives",
    "n",
}


def _roc_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build a canonical ROC-v2 spec without recalculating statistics."""
    return _curve_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        chart_type="roc",
    )


def _precision_recall_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build canonical static Precision-Recall from production quantities."""
    finite_rows = performance_data.filter(
        pl.col("chosen_cutoff").is_finite()
        & pl.col("sensitivity").is_finite()
        & pl.col("ppv").is_finite()
    )
    spec = _curve_v2_spec_from_performance_data(
        finite_rows,
        evaluation_metadata,
        chart_type="precision_recall",
    )
    prevalence = _gains_population_prevalence(performance_data, evaluation_metadata)
    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    spec["references"] = [
        {
            "type": "horizontal",
            "scope": "population",
            "population": population,
            "value": prevalence[population],
            "label": "Prevalence",
        }
        for population in populations
    ]
    return spec


def _precision_recall_times_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build canonical time-dependent precision-recall from calculated production data."""
    required = _REQUIRED_PRECISION_RECALL_COLUMNS | {
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
    }
    missing = required.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Time-dependent precision-recall performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    finite_rows = performance_data.filter(
        pl.col("chosen_cutoff").is_finite()
        & pl.col("sensitivity").is_finite()
        & pl.col("ppv").is_finite()
    )

    rows = finite_rows.select(
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "chosen_cutoff",
        "sensitivity",
        "ppv",
    ).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Time-dependent precision-recall rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    evaluations = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: dict[str, object] = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

    series_keys = list(
        dict.fromkeys(
            (
                str(row["reference_group"]),
                float(row["fixed_time_horizon"]),
                str(row["censoring_heuristic"]),
                str(row["competing_heuristic"]),
            )
            for row in rows
        )
    )
    series_ids = {
        key: f"series-{index}" for index, key in enumerate(series_keys, start=1)
    }
    series = []
    for key in series_keys:
        group, horizon, _, _ = key
        metadata = evaluation_metadata[group]
        display_value = metadata.model or metadata.population
        series.append(
            {
                "id": series_ids[key],
                "evaluationId": evaluation_ids[group],
                "horizon": horizon,
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": "model" if metadata.model is not None else "population",
                },
            }
        )

    data = []
    for row in rows:
        key = (
            str(row["reference_group"]),
            float(row["fixed_time_horizon"]),
            str(row["censoring_heuristic"]),
            str(row["competing_heuristic"]),
        )
        data.append(
            {
                "seriesId": series_ids[key],
                "cutoff": row["chosen_cutoff"],
                "sensitivity": row["sensitivity"],
                "ppv": row["ppv"],
            }
        )

    risks = _gains_population_horizon_risk(performance_data, evaluation_metadata)
    references = []
    for (population, horizon), risk in risks.items():
        references.append(
            {
                "type": "horizontal",
                "scope": "population_horizon",
                "population": population,
                "horizon": horizon,
                "value": risk,
                "label": "Prevalence",
            }
        )

    return {
        "schemaVersion": "2.0",
        "type": "precision_recall",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "sensitivity",
        "y": "ppv",
        "xAxis": {"label": "Sensitivity", "domain": [0, 1]},
        "yAxis": {"label": "Positive Predictive Value", "domain": [0, 1]},
        "references": references,
    }


def _gains_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build a canonical gains-v2 spec from production performance quantities."""
    spec = _curve_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        chart_type="gains",
    )
    prevalence = _gains_population_prevalence(performance_data, evaluation_metadata)

    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    spec["references"] = [
        {"type": "identity", "scope": "global", "label": "Random"},
        *[
            {
                "type": "path",
                "scope": "population",
                "population": population,
                "label": "Perfect Model",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": prevalence[population], "y": 1},
                    {"x": 1, "y": 1},
                ],
            }
            for population in populations
        ],
    ]
    return spec


def _lift_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build a canonical lift-v2 spec from production performance quantities."""
    spec = _curve_v2_spec_from_performance_data(
        performance_data,
        evaluation_metadata,
        chart_type="lift",
    )
    prevalence = _gains_population_prevalence(performance_data, evaluation_metadata)

    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    references: list[dict[str, object]] = [
        {"type": "horizontal", "value": 1.0, "scope": "global", "label": "Random"}
    ]
    perfect_heights: list[float] = []
    for population in populations:
        p = prevalence[population]
        if p > 0:
            perfect_height = 1.0 / p
            perfect_heights.append(perfect_height)
            references.append(
                {
                    "type": "path",
                    "scope": "population",
                    "population": population,
                    "label": "Perfect Model",
                    "points": [
                        {"x": 0.0, "y": perfect_height},
                        {"x": p, "y": perfect_height},
                        {"x": 1.0, "y": 1.0},
                    ],
                }
            )
    spec["yAxis"] = {
        "label": "Lift",
        "domain": [0, _lift_y_axis_upper_bound(performance_data, perfect_heights)],
    }
    spec["references"] = references
    return spec


def _gains_times_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build canonical time-dependent gains from calculated production data."""
    required = _REQUIRED_GAINS_COLUMNS | {
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
    }
    missing = required.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Time-dependent gains performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = performance_data.select(
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "chosen_cutoff",
        "sensitivity",
        "ppcr",
    ).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Time-dependent gains rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    evaluations = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: dict[str, object] = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

    series_keys = list(
        dict.fromkeys(
            (
                str(row["reference_group"]),
                float(row["fixed_time_horizon"]),
                str(row["censoring_heuristic"]),
                str(row["competing_heuristic"]),
            )
            for row in rows
        )
    )
    series_ids = {
        key: f"series-{index}" for index, key in enumerate(series_keys, start=1)
    }
    series = []
    for key in series_keys:
        group, horizon, _, _ = key
        metadata = evaluation_metadata[group]
        display_value = metadata.model or metadata.population
        series.append(
            {
                "id": series_ids[key],
                "evaluationId": evaluation_ids[group],
                "horizon": horizon,
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": "model" if metadata.model is not None else "population",
                },
            }
        )

    data = []
    for row in rows:
        key = (
            str(row["reference_group"]),
            float(row["fixed_time_horizon"]),
            str(row["censoring_heuristic"]),
            str(row["competing_heuristic"]),
        )
        data.append(
            {
                "seriesId": series_ids[key],
                "cutoff": row["chosen_cutoff"],
                "ppcr": row["ppcr"],
                "sensitivity": row["sensitivity"],
            }
        )

    risks = _gains_population_horizon_risk(performance_data, evaluation_metadata)
    references = [{"type": "identity", "scope": "global", "label": "Random"}]
    for (population, horizon), risk in risks.items():
        references.append(
            {
                "type": "path",
                "scope": "population_horizon",
                "population": population,
                "horizon": horizon,
                "label": "Perfect Model",
                "points": [
                    {"x": 0, "y": 0},
                    {"x": risk, "y": 1},
                    {"x": 1, "y": 1},
                ],
            }
        )

    return {
        "schemaVersion": "2.0",
        "type": "gains",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "ppcr",
        "y": "sensitivity",
        "xAxis": {"label": "Predicted Positives (Rate)", "domain": [0, 1]},
        "yAxis": {"label": "Sensitivity", "domain": [0, 1]},
        "references": references,
    }


def _lift_times_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build canonical time-dependent lift from calculated production data."""
    required = _REQUIRED_LIFT_COLUMNS | {
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
    }
    missing = required.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Time-dependent lift performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = performance_data.select(
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "chosen_cutoff",
        "lift",
        "ppcr",
    ).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Time-dependent lift rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    evaluations = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: dict[str, object] = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

    series_keys = list(
        dict.fromkeys(
            (
                str(row["reference_group"]),
                float(row["fixed_time_horizon"]),
                str(row["censoring_heuristic"]),
                str(row["competing_heuristic"]),
            )
            for row in rows
        )
    )
    series_ids = {
        key: f"series-{index}" for index, key in enumerate(series_keys, start=1)
    }
    series = []
    for key in series_keys:
        group, horizon, _, _ = key
        metadata = evaluation_metadata[group]
        display_value = metadata.model or metadata.population
        series.append(
            {
                "id": series_ids[key],
                "evaluationId": evaluation_ids[group],
                "horizon": horizon,
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": "model" if metadata.model is not None else "population",
                },
            }
        )

    data = []
    for row in rows:
        key = (
            str(row["reference_group"]),
            float(row["fixed_time_horizon"]),
            str(row["censoring_heuristic"]),
            str(row["competing_heuristic"]),
        )
        data.append(
            {
                "seriesId": series_ids[key],
                "cutoff": row["chosen_cutoff"],
                "ppcr": row["ppcr"],
                "lift": row["lift"],
            }
        )

    risks = _gains_population_horizon_risk(performance_data, evaluation_metadata)
    references: list[dict[str, object]] = [
        {"type": "horizontal", "value": 1.0, "scope": "global", "label": "Random"}
    ]
    perfect_heights: list[float] = []
    for (population, horizon), risk in risks.items():
        if risk > 0:
            perfect_height = 1.0 / risk
            perfect_heights.append(perfect_height)
            references.append(
                {
                    "type": "path",
                    "scope": "population_horizon",
                    "population": population,
                    "horizon": horizon,
                    "label": "Perfect Model",
                    "points": [
                        {"x": 0.0, "y": perfect_height},
                        {"x": risk, "y": perfect_height},
                        {"x": 1.0, "y": 1.0},
                    ],
                }
            )

    return {
        "schemaVersion": "2.0",
        "type": "lift",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "ppcr",
        "y": "lift",
        "xAxis": {"label": "Predicted Positives (Rate)", "domain": [0, 1]},
        "yAxis": {
            "label": "Lift",
            "domain": [0, _lift_y_axis_upper_bound(performance_data, perfect_heights)],
        },
        "references": references,
    }


def _gains_population_horizon_risk(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[tuple[str, float], float]:
    """Map calculated cutoff-0 AJ event risk to semantic population/horizon."""
    group_risks = (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select(
            "reference_group",
            "fixed_time_horizon",
            (pl.col("real_positives") / pl.col("n")).alias("event_risk"),
        )
        .unique()
        .to_dicts()
    )
    values: dict[tuple[str, float], set[float]] = {}
    for row in group_risks:
        group = str(row["reference_group"])
        metadata = evaluation_metadata.get(group)
        if metadata is None:
            continue
        key = (metadata.population, float(row["fixed_time_horizon"]))
        values.setdefault(key, set()).add(float(row["event_risk"]))

    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    horizons = sorted(
        float(value)
        for value in performance_data["fixed_time_horizon"].unique().to_list()
    )
    risks: dict[tuple[str, float], float] = {}
    for key in (
        (population, horizon)
        for horizon in horizons
        for population in populations
        if (population, horizon) in values
    ):
        candidates = values[key]
        if len(candidates) != 1:
            raise ValueError(
                "Time-dependent gains must have one calculated event risk per "
                f"population and horizon: {key[0]} at {key[1]}"
            )
        risks[key] = next(iter(candidates))
    return risks


def _gains_population_prevalence(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, float]:
    """Map production event prevalence from compatibility groups to populations."""
    group_prevalence = {
        str(row["reference_group"]): float(row["prevalence"])
        for row in (
            performance_data.select(
                "reference_group",
                (pl.col("real_positives") / pl.col("n")).alias("prevalence"),
            )
            .unique()
            .to_dicts()
        )
    }
    population_values: dict[str, set[float]] = {}
    for group, metadata in evaluation_metadata.items():
        if group not in group_prevalence:
            continue
        population_values.setdefault(metadata.population, set()).add(
            group_prevalence[group]
        )

    prevalence: dict[str, float] = {}
    for population in dict.fromkeys(
        metadata.population for metadata in evaluation_metadata.values()
    ):
        values = population_values.get(population, set())
        if len(values) != 1:
            raise ValueError(
                "Gains performance data must have one prevalence per population: "
                f"{population}"
            )
        prevalence[population] = next(iter(values))
    return prevalence


def _lift_y_axis_upper_bound(
    performance_data: pl.DataFrame,
    perfect_heights: list[float],
) -> float:
    """Return the finite numeric Lift bound implied by existing plot quantities."""
    finite_lifts = performance_data["lift"].filter(performance_data["lift"].is_finite())
    observed_lift = finite_lifts.max() if len(finite_lifts) > 0 else 1.0
    if isinstance(observed_lift, (int, float)):
        observed_lift_val = float(observed_lift)
    else:
        observed_lift_val = 1.0
    return max(1.0, observed_lift_val, *perfect_heights)


def _curve_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    chart_type: str,
) -> dict[str, object]:
    """Build common canonical curve semantics without recalculating statistics.

    ``reference_group`` is used only to join existing performance rows to the
    semantic metadata established by the high-level production inputs. Stable
    evaluation/series IDs are ordinal over that semantic metadata, so they do
    not encode compatibility grouping names or presentation values.
    """
    if chart_type == "roc":
        required = _REQUIRED_ROC_COLUMNS
        selected = ["reference_group", "chosen_cutoff", "sensitivity", "specificity"]
    elif chart_type == "precision_recall":
        required = _REQUIRED_PRECISION_RECALL_COLUMNS
        selected = ["reference_group", "chosen_cutoff", "sensitivity", "ppv"]
    elif chart_type == "gains":
        required = _REQUIRED_GAINS_COLUMNS
        selected = ["reference_group", "chosen_cutoff", "sensitivity", "ppcr"]
    elif chart_type == "lift":
        required = _REQUIRED_LIFT_COLUMNS
        selected = ["reference_group", "chosen_cutoff", "lift", "ppcr"]
    else:
        raise ValueError(f"Unsupported v2 curve type: {chart_type}")

    missing = required.difference(performance_data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(
            f"{chart_type.upper()} performance data is missing columns: {missing_columns}"
        )

    rows = performance_data.select(*selected).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    metadata_groups = set(evaluation_metadata)
    missing_metadata = row_groups.difference(metadata_groups)
    if missing_metadata:
        groups = ", ".join(sorted(missing_metadata))
        raise ValueError(
            f"{chart_type.upper()} performance rows are missing evaluation metadata: "
            f"{groups}"
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    series_ids = {
        group: f"series-{index}" for index, group in enumerate(ordered_groups, start=1)
    }

    evaluations: list[dict[str, object]] = []
    series: list[dict[str, object]] = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: dict[str, object] = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

        if metadata.model is not None:
            display_value = metadata.model
            display_role = "model"
        else:
            display_value = metadata.population
            display_role = "population"
        series.append(
            {
                "id": series_ids[group],
                "evaluationId": evaluation_ids[group],
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": display_role,
                },
            }
        )

    data = []
    for row in rows:
        datum = {
            "seriesId": series_ids[str(row["reference_group"])],
            "cutoff": row["chosen_cutoff"],
        }
        if chart_type == "roc":
            datum["sensitivity"] = row["sensitivity"]
            datum["specificity"] = row["specificity"]
        elif chart_type == "precision_recall":
            datum["sensitivity"] = row["sensitivity"]
            datum["ppv"] = row["ppv"]
        elif chart_type == "gains":
            datum["sensitivity"] = row["sensitivity"]
            datum["ppcr"] = row["ppcr"]
        elif chart_type == "lift":
            datum["ppcr"] = row["ppcr"]
            datum["lift"] = row["lift"]
        data.append(datum)

    if chart_type == "roc":
        return {
            "schemaVersion": "2.0",
            "type": "roc",
            "evaluations": evaluations,
            "series": series,
            "data": data,
            "x": "false_positive_rate",
            "y": "sensitivity",
            "xAxis": {"label": "1 - Specificity", "domain": [0, 1]},
            "yAxis": {"label": "Sensitivity", "domain": [0, 1]},
            "references": [{"type": "identity", "scope": "global"}],
        }

    if chart_type == "precision_recall":
        return {
            "schemaVersion": "2.0",
            "type": "precision_recall",
            "evaluations": evaluations,
            "series": series,
            "data": data,
            "x": "sensitivity",
            "y": "ppv",
            "xAxis": {"label": "Sensitivity", "domain": [0, 1]},
            "yAxis": {"label": "Positive Predictive Value", "domain": [0, 1]},
            "references": [],
        }

    if chart_type == "gains":
        return {
            "schemaVersion": "2.0",
            "type": "gains",
            "evaluations": evaluations,
            "series": series,
            "data": data,
            "x": "ppcr",
            "y": "sensitivity",
            "xAxis": {"label": "Predicted Positives (Rate)", "domain": [0, 1]},
            "yAxis": {"label": "Sensitivity", "domain": [0, 1]},
            "references": [],
        }

    return {
        "schemaVersion": "2.0",
        "type": "lift",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "ppcr",
        "y": "lift",
        "xAxis": {"label": "Predicted Positives (Rate)", "domain": [0, 1]},
        "yAxis": {
            "label": "Lift",
            "domain": [0, _lift_y_axis_upper_bound(performance_data, [])],
        },
        "references": [],
    }

"""Internal canonical PerformanceTableSpec builders.

These helpers translate already-calculated production performance data plus
semantic evaluation metadata. They do not calculate or render statistics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import polars as pl

from rtichoke.processing.evaluation_semantics import _EvaluationMetadata

_METRICS: tuple[tuple[str, str], ...] = (
    ("true_positives", "True Positives"),
    ("true_negatives", "True Negatives"),
    ("false_positives", "False Positives"),
    ("false_negatives", "False Negatives"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("false_positive_rate", "False Positive Rate"),
    ("ppv", "PPV"),
    ("npv", "NPV"),
    ("lift", "Lift"),
    ("predicted_positives", "Predicted Positives"),
    ("ppcr", "PPCR"),
    ("net_benefit", "Net Benefit"),
    ("net_benefit_interventions_avoided", "Interventions Avoided"),
)


class _EvaluationSpec(TypedDict, total=False):
    id: str
    model: str
    population: str


class _MetricDefinition(TypedDict):
    id: str
    label: str


class _OperatingPoint(TypedDict):
    type: str
    value: float


class _MetricValue(TypedDict):
    metricId: str
    estimate: float | int | None


class _EvaluationContext(TypedDict):
    censoringHeuristic: str
    competingEventHeuristic: str


class _PerformanceTableRow(TypedDict, total=False):
    evaluationId: str
    operatingPoint: _OperatingPoint
    values: list[_MetricValue]
    horizon: float
    context: _EvaluationContext


class _PerformanceTableSpec(TypedDict):
    schemaVersion: str
    type: str
    evaluations: list[_EvaluationSpec]
    metrics: list[_MetricDefinition]
    rows: list[_PerformanceTableRow]


def _performance_table_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> _PerformanceTableSpec:
    """Build the canonical static PerformanceTableSpec."""
    return _build_performance_table_spec(
        performance_data, evaluation_metadata, time_dependent=False
    )


def _performance_table_times_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> _PerformanceTableSpec:
    """Build the canonical time-dependent PerformanceTableSpec."""
    return _build_performance_table_spec(
        performance_data, evaluation_metadata, time_dependent=True
    )


def _build_performance_table_spec(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    time_dependent: bool,
) -> _PerformanceTableSpec:
    required = {"reference_group", "stratified_by", "chosen_cutoff"}
    if time_dependent:
        required |= {
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
        }
    missing = required.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Performance-table data is missing columns: " + ", ".join(sorted(missing))
        )

    metric_definitions: list[_MetricDefinition] = [
        cast(_MetricDefinition, {"id": metric_id, "label": label})
        for metric_id, label in _METRICS
        if metric_id in performance_data.columns
    ]
    metric_ids = [definition["id"] for definition in metric_definitions]

    rows = performance_data.select(
        [
            column
            for column in (
                "reference_group",
                "fixed_time_horizon",
                "censoring_heuristic",
                "competing_heuristic",
                "stratified_by",
                "chosen_cutoff",
                *metric_ids,
            )
            if column in performance_data.columns
        ]
    ).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Performance-table rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    evaluations: list[_EvaluationSpec] = []
    for group in ordered_groups:
        metadata = evaluation_metadata[group]
        evaluation: _EvaluationSpec = {
            "id": evaluation_ids[group],
            "population": metadata.population,
        }
        if metadata.model is not None:
            evaluation["model"] = metadata.model
        evaluations.append(evaluation)

    canonical_rows: list[_PerformanceTableRow] = []
    for row in rows:
        group = str(row["reference_group"])
        stratified_by = str(row["stratified_by"])
        operating_point: _OperatingPoint
        if stratified_by == "probability_threshold":
            operating_point = {
                "type": "probability_threshold",
                "value": _number(row["chosen_cutoff"]),
            }
        elif stratified_by == "ppcr":
            operating_point = {"type": "ppcr", "value": _number(row["ppcr"])}
        else:
            raise ValueError(
                "Canonical PerformanceTableSpec supports probability_threshold "
                f"or ppcr operating points, not {stratified_by!r}"
            )

        canonical_row: _PerformanceTableRow = {
            "evaluationId": evaluation_ids[group],
            "operatingPoint": operating_point,
            "values": cast(
                list[_MetricValue],
                [
                    {
                        "metricId": metric_id,
                        "estimate": _nullable_number(row[metric_id]),
                    }
                    for metric_id in metric_ids
                ],
            ),
        }
        if time_dependent:
            canonical_row["horizon"] = _number(row["fixed_time_horizon"])
            canonical_row["context"] = {
                "censoringHeuristic": str(row["censoring_heuristic"]),
                "competingEventHeuristic": str(row["competing_heuristic"]),
            }
        canonical_rows.append(canonical_row)

    return {
        "schemaVersion": "2.0",
        "type": "performance_table",
        "evaluations": evaluations,
        "metrics": metric_definitions,
        "rows": canonical_rows,
    }


def _nullable_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _number(value: Any) -> float:
    if value is None:
        raise ValueError("Operating point and horizon values must not be null")
    return float(value)

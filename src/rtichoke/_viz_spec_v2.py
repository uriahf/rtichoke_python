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
_REQUIRED_GAINS_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "sensitivity",
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
    elif chart_type == "gains":
        required = _REQUIRED_GAINS_COLUMNS
        selected = ["reference_group", "chosen_cutoff", "sensitivity", "ppcr"]
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
            "sensitivity": row["sensitivity"],
        }
        if chart_type == "roc":
            datum["specificity"] = row["specificity"]
        else:
            datum["ppcr"] = row["ppcr"]
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


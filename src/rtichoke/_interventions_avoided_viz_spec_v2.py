"""Canonical static Interventions Avoided v2 adapter.

This module translates already-computed production Interventions Avoided
quantities into the shared rtichoke_viz contract. It deliberately does not
recompute model statistics or threshold membership.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from rtichoke.processing.evaluation_semantics import _EvaluationMetadata

_REQUIRED_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "net_benefit_interventions_avoided",
    "real_positives",
    "n",
}

_REQUIRED_TIMES_COLUMNS = _REQUIRED_COLUMNS | {
    "fixed_time_horizon",
    "censoring_heuristic",
    "competing_heuristic",
}


def _interventions_avoided_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> dict[str, object]:
    """Build canonical static Interventions Avoided from production quantities."""
    missing = _REQUIRED_COLUMNS.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Interventions Avoided performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = (
        performance_data.filter(
            pl.col("chosen_cutoff").is_finite()
            & pl.col("net_benefit_interventions_avoided").is_finite()
        )
        .select(
            "reference_group",
            "chosen_cutoff",
            "net_benefit_interventions_avoided",
            "real_positives",
            "n",
        )
        .to_dicts()
    )

    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Interventions Avoided rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
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

        display_value = metadata.model or metadata.population
        series.append(
            {
                "id": series_ids[group],
                "evaluationId": evaluation_ids[group],
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": "model" if metadata.model is not None else "population",
                },
            }
        )

    data = [
        {
            "seriesId": series_ids[str(row["reference_group"])],
            "threshold": float(row["chosen_cutoff"]),
            "interventionsAvoided": float(row["net_benefit_interventions_avoided"]),
        }
        for row in rows
    ]

    prevalence_values: dict[str, set[float]] = {}
    population_thresholds: dict[str, list[float]] = {}
    for row in rows:
        group = str(row["reference_group"])
        population = evaluation_metadata[group].population
        n = float(row["n"])
        if n <= 0:
            raise ValueError("Interventions Avoided population size must be positive.")
        prevalence_values.setdefault(population, set()).add(
            float(row["real_positives"]) / n
        )
        threshold = float(row["chosen_cutoff"])
        if 0.0 < threshold <= 1.0:
            population_thresholds.setdefault(population, []).append(threshold)

    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    references: list[dict[str, object]] = [
        {
            "type": "horizontal",
            "scope": "global",
            "value": 0.0,
            "label": "Treat All",
            "benchmark": "treat_all",
        }
    ]
    for population in populations:
        values = prevalence_values.get(population, set())
        if not values:
            continue
        if len(values) != 1:
            raise ValueError(
                f"Population {population!r} has inconsistent prevalence values."
            )
        prevalence = next(iter(values))
        thresholds = sorted(set(population_thresholds.get(population, [])))
        references.append(
            {
                "type": "path",
                "scope": "population",
                "population": population,
                "label": f"Treat None — {population}",
                "benchmark": "treat_none",
                "points": [
                    {
                        "x": threshold,
                        "y": 100.0
                        * (
                            1.0
                            - prevalence
                            - prevalence * (1.0 - threshold) / threshold
                        ),
                    }
                    for threshold in thresholds
                ],
            }
        )

    return {
        "schemaVersion": "2.0",
        "type": "interventions_avoided",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "threshold",
        "y": "interventionsAvoided",
        "xAxis": {
            "label": "Probability Threshold",
            "domain": [min_p_threshold, max_p_threshold],
        },
        "yAxis": {"label": "Interventions Avoided (per 100)"},
        "references": references,
    }


def _interventions_avoided_times_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> dict[str, object]:
    """Build canonical time-dependent Interventions Avoided v2.

    Model values are copied from the existing production calculation. Only
    Treat None reference geometry is derived from the existing AJ event risk.
    """
    missing = _REQUIRED_TIMES_COLUMNS.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Time-dependent Interventions Avoided performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = (
        performance_data.filter(
            pl.col("chosen_cutoff").is_finite()
            & pl.col("net_benefit_interventions_avoided").is_finite()
        )
        .select(
            "reference_group",
            "fixed_time_horizon",
            "censoring_heuristic",
            "competing_heuristic",
            "chosen_cutoff",
            "net_benefit_interventions_avoided",
            "real_positives",
            "n",
        )
        .to_dicts()
    )
    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Time-dependent Interventions Avoided rows are missing evaluation metadata: "
            + ", ".join(sorted(missing_metadata))
        )

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
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

    series_keys = list(
        dict.fromkeys(
            (str(row["reference_group"]), float(row["fixed_time_horizon"]))
            for row in rows
        )
    )
    series_ids = {
        key: f"series-{index}" for index, key in enumerate(series_keys, start=1)
    }
    series: list[dict[str, object]] = []
    for group, horizon in series_keys:
        metadata = evaluation_metadata[group]
        display_value = metadata.model or metadata.population
        series.append(
            {
                "id": series_ids[(group, horizon)],
                "evaluationId": evaluation_ids[group],
                "horizon": horizon,
                "display": {
                    "label": display_value,
                    "group": display_value,
                    "role": "model" if metadata.model is not None else "population",
                },
            }
        )

    data = [
        {
            "seriesId": series_ids[
                (str(row["reference_group"]), float(row["fixed_time_horizon"]))
            ],
            "threshold": float(row["chosen_cutoff"]),
            "interventionsAvoided": float(row["net_benefit_interventions_avoided"]),
        }
        for row in rows
    ]

    risk_rows = (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select(
            "reference_group",
            "fixed_time_horizon",
            (pl.col("real_positives") / pl.col("n")).alias("event_risk"),
        )
        .unique()
        .to_dicts()
    )
    risk_values: dict[tuple[str, float], set[float]] = {}
    for row in risk_rows:
        group = str(row["reference_group"])
        metadata = evaluation_metadata.get(group)
        if metadata is None:
            continue
        key = (metadata.population, float(row["fixed_time_horizon"]))
        risk_values.setdefault(key, set()).add(float(row["event_risk"]))

    thresholds: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        group = str(row["reference_group"])
        key = (
            evaluation_metadata[group].population,
            float(row["fixed_time_horizon"]),
        )
        threshold = float(row["chosen_cutoff"])
        if 0.0 < threshold <= 1.0:
            thresholds.setdefault(key, []).append(threshold)

    references: list[dict[str, object]] = [
        {
            "type": "horizontal",
            "scope": "global",
            "value": 0.0,
            "label": "Treat All",
            "benchmark": "treat_all",
        }
    ]
    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    horizons = sorted(
        float(value)
        for value in performance_data["fixed_time_horizon"].unique().to_list()
    )
    for population in populations:
        for horizon in horizons:
            key = (population, horizon)
            candidates = risk_values.get(key, set())
            if not candidates:
                continue
            if len(candidates) != 1:
                raise ValueError(
                    "Time-dependent Interventions Avoided must have one calculated "
                    f"event risk per population and horizon: {population} at {horizon}"
                )
            event_risk = next(iter(candidates))
            references.append(
                {
                    "type": "path",
                    "scope": "population_horizon",
                    "population": population,
                    "horizon": horizon,
                    "label": f"Treat None — {population}",
                    "benchmark": "treat_none",
                    "points": [
                        {
                            "x": threshold,
                            "y": 100.0
                            * (
                                1.0
                                - event_risk
                                - event_risk * (1.0 - threshold) / threshold
                            ),
                        }
                        for threshold in sorted(set(thresholds.get(key, [])))
                    ],
                }
            )

    return {
        "schemaVersion": "2.0",
        "type": "interventions_avoided",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "threshold",
        "y": "interventionsAvoided",
        "xAxis": {
            "label": "Probability Threshold",
            "domain": [min_p_threshold, max_p_threshold],
        },
        "yAxis": {"label": "Interventions Avoided (per 100)"},
        "references": references,
    }

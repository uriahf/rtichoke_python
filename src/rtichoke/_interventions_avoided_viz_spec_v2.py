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
            "interventionsAvoided": float(
                row["net_benefit_interventions_avoided"]
            ),
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

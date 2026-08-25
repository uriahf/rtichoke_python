"""Canonical static Decision Curve v2 adapter.

This module translates already-computed production Decision Curve quantities
into the shared rtichoke_viz contract. It deliberately does not recompute model
statistics or threshold membership.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from rtichoke.processing.evaluation_semantics import _EvaluationMetadata

_REQUIRED_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "net_benefit",
    "real_positives",
    "n",
}


def _decision_curve_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    *,
    min_p_threshold: float = 0.0,
    max_p_threshold: float = 1.0,
) -> dict[str, object]:
    """Build canonical static Decision Curve v2 from production quantities."""
    missing = _REQUIRED_COLUMNS.difference(performance_data.columns)
    if missing:
        raise ValueError(
            "Decision Curve performance data is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = (
        performance_data.filter(
            pl.col("chosen_cutoff").is_finite() & pl.col("net_benefit").is_finite()
        )
        .select(
            "reference_group",
            "chosen_cutoff",
            "net_benefit",
            "real_positives",
            "n",
        )
        .to_dicts()
    )

    row_groups = {str(row["reference_group"]) for row in rows}
    missing_metadata = row_groups.difference(evaluation_metadata)
    if missing_metadata:
        raise ValueError(
            "Decision Curve rows are missing evaluation metadata: "
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
            "netBenefit": float(row["net_benefit"]),
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
            raise ValueError("Decision Curve population size must be positive.")
        prevalence_values.setdefault(population, set()).add(
            float(row["real_positives"]) / n
        )
        threshold = float(row["chosen_cutoff"])
        if 0.0 <= threshold < 1.0:
            population_thresholds.setdefault(population, []).append(threshold)

    populations = list(
        dict.fromkeys(metadata.population for metadata in evaluation_metadata.values())
    )
    references: list[dict[str, object]] = [
        {
            "type": "horizontal",
            "scope": "global",
            "value": 0.0,
            "label": "Treat None",
            "benchmark": "treat_none",
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
                "label": f"Treat All — {population}",
                "benchmark": "treat_all",
                "points": [
                    {
                        "x": threshold,
                        "y": prevalence
                        - (1.0 - prevalence) * threshold / (1.0 - threshold),
                    }
                    for threshold in thresholds
                ],
            }
        )

    return {
        "schemaVersion": "2.0",
        "type": "decision_curve",
        "evaluations": evaluations,
        "series": series,
        "data": data,
        "x": "threshold",
        "y": "netBenefit",
        "xAxis": {
            "label": "Probability threshold",
            "domain": [min_p_threshold, max_p_threshold],
        },
        "yAxis": {"label": "Net benefit"},
        "references": references,
    }

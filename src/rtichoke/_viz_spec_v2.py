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


def _roc_v2_spec_from_performance_data(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> dict[str, object]:
    """Build a canonical ROC-v2 spec without recalculating statistics.

    ``reference_group`` is used only to join existing performance rows to the
    semantic metadata established by the high-level production inputs. Stable
    evaluation/series IDs are ordinal over that semantic metadata, so they do
    not encode labels, compatibility grouping names, prevalence, coordinates,
    colors, or other presentation/statistical values.
    """
    missing = _REQUIRED_ROC_COLUMNS.difference(performance_data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"ROC performance data is missing columns: {missing_columns}")

    rows = performance_data.select(
        "reference_group",
        "chosen_cutoff",
        "sensitivity",
        "specificity",
    ).to_dicts()
    row_groups = {str(row["reference_group"]) for row in rows}
    metadata_groups = set(evaluation_metadata)

    missing_metadata = row_groups.difference(metadata_groups)
    if missing_metadata:
        groups = ", ".join(sorted(missing_metadata))
        raise ValueError(f"ROC performance rows are missing evaluation metadata: {groups}")

    ordered_groups = [group for group in evaluation_metadata if group in row_groups]
    evaluation_ids = {
        group: f"evaluation-{index}"
        for index, group in enumerate(ordered_groups, start=1)
    }
    series_ids = {
        group: f"series-{index}"
        for index, group in enumerate(ordered_groups, start=1)
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

    return {
        "schemaVersion": "2.0",
        "type": "roc",
        "evaluations": evaluations,
        "series": series,
        "data": [
            {
                "seriesId": series_ids[str(row["reference_group"])],
                "cutoff": row["chosen_cutoff"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
            }
            for row in rows
        ],
        "x": "false_positive_rate",
        "y": "sensitivity",
        "xAxis": {"label": "1 - Specificity", "domain": [0, 1]},
        "yAxis": {"label": "Sensitivity", "domain": [0, 1]},
        "references": [{"type": "identity", "scope": "global"}],
    }

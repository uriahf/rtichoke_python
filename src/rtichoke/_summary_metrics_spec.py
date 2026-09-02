"""Internal builder for canonical SummaryMetricsSpec v1.0 objects.

Calculates SummaryMetricsSpec components (Prevalence and AUROC) from production
inputs and prepared performance data. Does not perform statistical rendering or
alter global evaluation identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from rtichoke.performance_data.performance_data import _validate_and_align_binary_inputs
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _SHARED_POPULATION,
)


class _PopulationSpec(TypedDict):
    id: str
    label: str


class _EvaluationSpec(TypedDict, total=False):
    id: str
    model: str
    population: str
    label: str


class _PopulationOwner(TypedDict):
    type: str
    populationId: str


class _EvaluationOwner(TypedDict):
    type: str
    evaluationId: str


class _PrevalenceMetric(TypedDict):
    metric: str
    owner: _PopulationOwner
    estimate: float | None


class _EventRiskMetric(TypedDict):
    metric: str
    owner: _PopulationOwner
    horizon: float
    estimate: float | None


class _AurocMetric(TypedDict):
    metric: str
    owner: _EvaluationOwner
    estimate: float | None


class _SummaryMetricsSpec(TypedDict, total=False):
    schemaVersion: str
    type: str
    title: str
    evaluations: list[_EvaluationSpec]
    populations: list[_PopulationSpec]
    metrics: list[dict[str, Any]]


def _compute_auroc_proc_compatible(
    reals: np.ndarray,
    probs: np.ndarray,
) -> float | None:
    """Compute AUROC with historical pROC auto-direction compatibility.

    If outcomes contain only a single class (all 0s or all 1s), returns None.
    Otherwise, compares the median score of controls (y==0) and cases (y==1).
    If median(controls) <= median(cases), computes roc_auc_score with original
    scores. If median(controls) > median(cases), reverses ranking using -scores.
    """
    reals_arr = np.asarray(reals)
    probs_arr = np.asarray(probs)

    unique_reals = np.unique(reals_arr)
    if not set(unique_reals).issubset({0, 1}) or len(unique_reals) < 2:
        return None

    controls = probs_arr[reals_arr == 0]
    cases = probs_arr[reals_arr == 1]
    if len(controls) == 0 or len(cases) == 0:
        return None

    med_controls = np.median(controls)
    med_cases = np.median(cases)

    scores = probs_arr if med_controls <= med_cases else -probs_arr

    score_val = float(roc_auc_score(reals_arr, scores))
    if math.isnan(score_val) or not math.isfinite(score_val):
        return None
    return score_val


def _prevalence_summary_metrics_spec(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> _SummaryMetricsSpec:
    """Build canonical population-owned Prevalence SummaryMetricsSpec v1.0."""
    populations_list: list[_PopulationSpec] = []
    metrics_list: list[dict[str, Any]] = []

    seen_populations: dict[tuple[str, str], str] = {}
    pop_counter = 1

    for group, metadata in evaluation_metadata.items():
        pop_name = metadata.population
        label = "Population" if pop_name == _SHARED_POPULATION else pop_name
        # Keep distinct populations with identical display labels distinct by keying on (group, pop_name) or pop_name
        pop_key = (
            (group, pop_name)
            if pop_name != _SHARED_POPULATION
            and list(evaluation_metadata.values())[0].model is None
            else (pop_name, pop_name)
        )
        if pop_key not in seen_populations:
            pop_id = f"population-{pop_counter}"
            pop_counter += 1
            seen_populations[pop_key] = pop_id
            populations_list.append({"id": pop_id, "label": label})

    # Compute population prevalence from performance data
    group_prevalence: dict[str, float] = {}
    for row in (
        performance_data.select(
            "reference_group",
            (pl.col("real_positives") / pl.col("n")).alias("prevalence"),
        )
        .unique()
        .to_dicts()
    ):
        group = str(row["reference_group"])
        prev_val = row["prevalence"]
        if prev_val is not None and math.isfinite(float(prev_val)):
            group_prevalence[group] = float(prev_val)

    pop_estimates: dict[tuple[str, str], set[float]] = {}
    for group, metadata in evaluation_metadata.items():
        if group in group_prevalence:
            pop_name = metadata.population
            pop_key = (
                (group, pop_name)
                if pop_name != _SHARED_POPULATION and metadata.model is None
                else (pop_name, pop_name)
            )
            pop_estimates.setdefault(pop_key, set()).add(group_prevalence[group])

    for pop_spec in populations_list:
        pop_id = pop_spec["id"]
        pop_key = next(key for key, pid in seen_populations.items() if pid == pop_id)
        estimates = pop_estimates.get(pop_key, set())
        estimate: float | None = next(iter(estimates)) if len(estimates) == 1 else None

        metric_item: _PrevalenceMetric = {
            "metric": "prevalence",
            "owner": {
                "type": "population",
                "populationId": pop_id,
            },
            "estimate": estimate,
        }
        metrics_list.append(cast(dict[str, Any], metric_item))

    return {
        "schemaVersion": "1.0",
        "type": "summary_metrics",
        "title": "Prevalence summary",
        "evaluations": [],
        "populations": populations_list,
        "metrics": metrics_list,
    }


def _event_risk_summary_metrics_spec(
    performance_data: pl.DataFrame,
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
    fixed_time_horizons: list[float],
) -> _SummaryMetricsSpec:
    """Build canonical population-owned Event Risk SummaryMetricsSpec v1.1."""
    populations_list: list[_PopulationSpec] = []
    metrics_list: list[dict[str, Any]] = []

    seen_populations: dict[tuple[str, str], str] = {}
    pop_counter = 1

    for group, metadata in evaluation_metadata.items():
        pop_name = metadata.population
        label = "Population" if pop_name == _SHARED_POPULATION else pop_name
        pop_key = (
            (group, pop_name)
            if pop_name != _SHARED_POPULATION
            and list(evaluation_metadata.values())[0].model is None
            else (pop_name, pop_name)
        )
        if pop_key not in seen_populations:
            pop_id = f"population-{pop_counter}"
            pop_counter += 1
            seen_populations[pop_key] = pop_id
            populations_list.append({"id": pop_id, "label": label})

    pop_horizon_estimates: dict[tuple[tuple[str, str], float], set[float]] = {}
    cutoff_zero_rows = (
        performance_data.filter(pl.col("chosen_cutoff") == 0)
        .select(
            "reference_group",
            "fixed_time_horizon",
            (pl.col("real_positives") / pl.col("n")).alias("event_risk"),
        )
        .to_dicts()
    )
    for row in cutoff_zero_rows:
        group = str(row["reference_group"])
        if group not in evaluation_metadata:
            continue
        metadata = evaluation_metadata[group]
        pop_name = metadata.population
        pop_key = (
            (group, pop_name)
            if pop_name != _SHARED_POPULATION and metadata.model is None
            else (pop_name, pop_name)
        )
        horizon = float(row["fixed_time_horizon"])
        risk_val = row["event_risk"]
        if risk_val is not None and math.isfinite(float(risk_val)):
            pop_horizon_estimates.setdefault((pop_key, horizon), set()).add(
                float(risk_val)
            )

    for horizon in fixed_time_horizons:
        norm_horizon = float(horizon)
        for pop_spec in populations_list:
            pop_id = pop_spec["id"]
            pop_key = next(
                key for key, pid in seen_populations.items() if pid == pop_id
            )
            estimates = pop_horizon_estimates.get((pop_key, norm_horizon), set())
            estimate: float | None = (
                next(iter(estimates)) if len(estimates) == 1 else None
            )

            metric_item: _EventRiskMetric = {
                "metric": "event_risk",
                "owner": {
                    "type": "population",
                    "populationId": pop_id,
                },
                "horizon": norm_horizon,
                "estimate": estimate,
            }
            metrics_list.append(cast(dict[str, Any], metric_item))

    return {
        "schemaVersion": "1.1",
        "type": "summary_metrics",
        "title": "Event Risk",
        "evaluations": [],
        "populations": populations_list,
        "metrics": metrics_list,
    }


def _auroc_summary_metrics_spec(
    probs: dict[str, np.ndarray],
    reals: np.ndarray | dict[str, np.ndarray],
    evaluation_metadata: Mapping[str, _EvaluationMetadata],
) -> _SummaryMetricsSpec:
    """Build canonical evaluation-owned AUROC SummaryMetricsSpec v1.0."""
    reals_normalized = _validate_and_align_binary_inputs(probs, reals)

    evaluations_list: list[_EvaluationSpec] = []
    metrics_list: list[dict[str, Any]] = []

    seen_populations: dict[str, str] = {}
    pop_counter = 1

    for index, (group, metadata) in enumerate(evaluation_metadata.items(), start=1):
        pop_name = metadata.population
        if pop_name not in seen_populations:
            seen_populations[pop_name] = f"population-{pop_counter}"
            pop_counter += 1
        pop_id = seen_populations[pop_name]

        eval_id = f"evaluation-{index}"
        eval_spec: _EvaluationSpec = {
            "id": eval_id,
            "population": pop_id,
        }
        if metadata.model is not None:
            eval_spec["model"] = metadata.model
            eval_spec["label"] = metadata.model
        else:
            label = "Population" if pop_name == _SHARED_POPULATION else pop_name
            eval_spec["label"] = label
        evaluations_list.append(eval_spec)

        # Get raw arrays for this evaluation
        probs_arr = np.asarray(probs[group])
        if isinstance(reals_normalized, dict):
            reals_arr = np.asarray(reals_normalized[group])
        else:
            reals_arr = np.asarray(reals_normalized)

        auroc_est = _compute_auroc_proc_compatible(reals_arr, probs_arr)

        metric_item: _AurocMetric = {
            "metric": "auroc",
            "owner": {
                "type": "evaluation",
                "evaluationId": eval_id,
            },
            "estimate": auroc_est,
        }
        metrics_list.append(cast(dict[str, Any], metric_item))

    return {
        "schemaVersion": "1.0",
        "type": "summary_metrics",
        "title": "AUROC",
        "evaluations": evaluations_list,
        "populations": [],
        "metrics": metrics_list,
    }

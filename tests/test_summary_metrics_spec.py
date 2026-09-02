import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rtichoke._summary_metrics_spec import (
    _auroc_summary_metrics_spec,
    _compute_auroc_proc_compatible,
    _event_risk_summary_metrics_spec,
    _prevalence_summary_metrics_spec,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import (
    _EvaluationMetadata,
    _build_evaluation_metadata,
)


def test_auroc_proc_compatible_direction():
    # Normal orientation: median(controls) <= median(cases)
    reals = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    score = _compute_auroc_proc_compatible(reals, probs)
    assert score == pytest.approx(1.0)
    assert score == pytest.approx(roc_auc_score(reals, probs))

    # Reversed orientation: median(controls) > median(cases)
    # pROC auto-direction flips -probs
    reals_rev = np.array([0, 0, 0, 1, 1, 1])
    probs_rev = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    score_rev = _compute_auroc_proc_compatible(reals_rev, probs_rev)
    assert score_rev == pytest.approx(1.0)
    # Standard sklearn without reversing gives 0.0, but pROC auto-direction gives 1.0!
    assert roc_auc_score(reals_rev, probs_rev) == pytest.approx(0.0)
    assert score_rev == pytest.approx(roc_auc_score(reals_rev, -probs_rev))


def test_auroc_single_class_returns_null():
    reals_all_zeros = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    assert _compute_auroc_proc_compatible(reals_all_zeros, probs) is None

    reals_all_ones = np.array([1, 1, 1, 1])
    assert _compute_auroc_proc_compatible(reals_all_ones, probs) is None


def test_auroc_ties():
    reals = np.array([0, 0, 1, 1])
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    assert _compute_auroc_proc_compatible(reals, probs) == pytest.approx(0.5)


def test_summary_metrics_spec_structure():
    probs = {"Model A": np.array([0.8, 0.7, 0.2, 0.1])}
    reals = np.array([1, 1, 0, 0])
    perf_data = prepare_performance_data(probs, reals)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))

    prev_spec = _prevalence_summary_metrics_spec(perf_data, metadata)
    assert prev_spec["schemaVersion"] == "1.0"
    assert prev_spec["type"] == "summary_metrics"
    assert prev_spec["title"] == "Prevalence summary"
    assert prev_spec["evaluations"] == []
    assert len(prev_spec["populations"]) == 1
    assert prev_spec["populations"][0]["id"] == "population-1"
    assert prev_spec["metrics"][0]["owner"]["type"] == "population"
    assert prev_spec["metrics"][0]["owner"]["populationId"] == "population-1"
    assert prev_spec["metrics"][0]["estimate"] == pytest.approx(0.5)

    auroc_spec = _auroc_summary_metrics_spec(probs, reals, metadata)
    assert auroc_spec["schemaVersion"] == "1.0"
    assert auroc_spec["type"] == "summary_metrics"
    assert auroc_spec["title"] == "AUROC"
    assert auroc_spec["populations"] == []
    assert len(auroc_spec["evaluations"]) == 1
    assert auroc_spec["evaluations"][0]["id"] == "evaluation-1"
    assert auroc_spec["metrics"][0]["owner"]["type"] == "evaluation"
    assert auroc_spec["metrics"][0]["owner"]["evaluationId"] == "evaluation-1"
    assert auroc_spec["metrics"][0]["estimate"] == pytest.approx(1.0)


def test_prevalence_summary_metrics_spec_distinct_populations_same_label():
    probs = {
        "Cohort 1": np.array([0.1, 0.2, 0.8, 0.9]),
        "Cohort 2": np.array([0.2, 0.3, 0.7, 0.8]),
    }
    reals = {
        "Cohort 1": np.array([0, 0, 1, 1]),
        "Cohort 2": np.array([0, 1, 1, 1]),
    }
    perf_data = prepare_performance_data(probs, reals)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))

    prev_spec = _prevalence_summary_metrics_spec(perf_data, metadata)
    assert len(prev_spec["populations"]) == 2
    assert prev_spec["populations"][0]["id"] == "population-1"
    assert prev_spec["populations"][1]["id"] == "population-2"
    assert prev_spec["populations"][0]["label"] == "Cohort 1"
    assert prev_spec["populations"][1]["label"] == "Cohort 2"

    # Distinct evaluation populations sharing identical display string ("Shared Label")
    metadata_shared_label = {
        "group-1": _EvaluationMetadata(
            reference_group="group-1",
            evaluation="group-1",
            model=None,
            population="Shared Label",
        ),
        "group-2": _EvaluationMetadata(
            reference_group="group-2",
            evaluation="group-2",
            model=None,
            population="Shared Label",
        ),
    }

    # Distinct populations remain distinct even if their display labels are identical
    prev_spec_shared = _prevalence_summary_metrics_spec(
        perf_data, metadata_shared_label
    )
    assert len(prev_spec_shared["populations"]) == 2
    assert prev_spec_shared["populations"][0]["id"] == "population-1"
    assert prev_spec_shared["populations"][1]["id"] == "population-2"
    assert prev_spec_shared["populations"][0]["label"] == "Shared Label"
    assert prev_spec_shared["populations"][1]["label"] == "Shared Label"


def test_event_risk_summary_metrics_spec_multiple_models_shared_population():
    probs = {
        "Model 1": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "Model 2": np.array([0.2, 0.4, 0.6, 0.8, 0.95]),
    }
    reals = np.array([0, 1, 0, 1, 1])
    times = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    horizons = [6.0, 2.0]

    perf_data = prepare_performance_data_times(
        probs, reals, times, fixed_time_horizons=horizons
    )
    metadata = _build_evaluation_metadata(probs, reals, times)

    spec = _event_risk_summary_metrics_spec(perf_data, metadata, horizons)
    assert spec["schemaVersion"] == "1.1"
    assert spec["type"] == "summary_metrics"
    assert spec["title"] == "Event Risk"
    assert spec["evaluations"] == []
    assert len(spec["populations"]) == 1
    assert spec["populations"][0]["id"] == "population-1"
    assert spec["populations"][0]["label"] == "Population"

    # Exactly 2 metrics for 1 population x 2 horizons (no duplicates from 2 models)
    assert len(spec["metrics"]) == 2
    assert spec["metrics"][0]["owner"] == {
        "type": "population",
        "populationId": "population-1",
    }
    assert spec["metrics"][0]["horizon"] == 6.0
    assert spec["metrics"][1]["owner"] == {
        "type": "population",
        "populationId": "population-1",
    }
    assert spec["metrics"][1]["horizon"] == 2.0

    # Values match cutoff-0 real_positives / n from performance data
    for metric in spec["metrics"]:
        h = metric["horizon"]
        cutoff_0_row = perf_data.filter(
            (perf_data["fixed_time_horizon"] == h) & (perf_data["chosen_cutoff"] == 0)
        ).to_dicts()[0]
        expected_risk = float(cutoff_0_row["real_positives"]) / float(cutoff_0_row["n"])
        assert metric["estimate"] == pytest.approx(expected_risk)


def test_event_risk_summary_metrics_spec_distinct_populations():
    probs = {
        "Cohort A": np.array([0.1, 0.3, 0.5]),
        "Cohort B": np.array([0.2, 0.4, 0.6]),
    }
    reals = {
        "Cohort A": np.array([0, 1, 0]),
        "Cohort B": np.array([1, 0, 1]),
    }
    times = {
        "Cohort A": np.array([2.0, 4.0, 6.0]),
        "Cohort B": np.array([1.0, 3.0, 5.0]),
    }
    horizons = [3.0, 5.0]

    perf_data = prepare_performance_data_times(
        probs, reals, times, fixed_time_horizons=horizons
    )
    metadata = _build_evaluation_metadata(probs, reals, times)

    spec = _event_risk_summary_metrics_spec(perf_data, metadata, horizons)
    assert len(spec["populations"]) == 2
    assert spec["populations"][0]["id"] == "population-1"
    assert spec["populations"][0]["label"] == "Cohort A"
    assert spec["populations"][1]["id"] == "population-2"
    assert spec["populations"][1]["label"] == "Cohort B"

    # 2 populations x 2 horizons = 4 metrics in requested horizon order
    assert len(spec["metrics"]) == 4
    m0, m1, m2, m3 = spec["metrics"]
    assert m0["horizon"] == 3.0
    assert m0["owner"]["populationId"] == "population-1"
    assert m1["horizon"] == 3.0
    assert m1["owner"]["populationId"] == "population-2"
    assert m2["horizon"] == 5.0
    assert m2["owner"]["populationId"] == "population-1"
    assert m3["horizon"] == 5.0
    assert m3["owner"]["populationId"] == "population-2"

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rtichoke._summary_metrics_spec import (
    _auroc_summary_metrics_spec,
    _compute_auroc_proc_compatible,
    _prevalence_summary_metrics_spec,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


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

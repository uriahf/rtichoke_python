"""Deterministic unit tests for standalone create_probs_histogram() adapter and canonical spec."""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import polars as pl
import pytest

import rtichoke
from rtichoke._viz_spec_v2 import _prediction_distribution_v2_spec_from_performance_data


def test_frozen_tied_fixture_exact_canonical_spec():
    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])
    by = 0.20

    chart = rtichoke.create_probs_histogram(
        probs=probs,
        reals=reals,
        by=by,
        stratified_by=("ppcr",),
    )
    spec = chart.spec

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "prediction_distribution"
    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "__shared_population__", "model": "m1"}
    ]
    assert spec["operatingPoint"] == {"dimension": "ppcr"}

    # Expected rank bins literal assertion
    expected_rank_bins = [
        {
            "evaluationId": "evaluation-1",
            "rankLower": 0.00,
            "rankUpper": 0.20,
            "positiveMass": 1,
            "negativeMass": 1,
        },
        {
            "evaluationId": "evaluation-1",
            "rankLower": 0.20,
            "rankUpper": 0.40,
            "positiveMass": 2,
            "negativeMass": 2,
        },
        {
            "evaluationId": "evaluation-1",
            "rankLower": 0.40,
            "rankUpper": 0.60,
            "positiveMass": 0,
            "negativeMass": 0,
        },
        {
            "evaluationId": "evaluation-1",
            "rankLower": 0.60,
            "rankUpper": 0.80,
            "positiveMass": 1,
            "negativeMass": 0,
        },
        {
            "evaluationId": "evaluation-1",
            "rankLower": 0.80,
            "rankUpper": 1.00,
            "positiveMass": 1,
            "negativeMass": 1,
        },
    ]
    assert spec["rankBins"] == expected_rank_bins

    # Expected PPCR operating points literal assertion
    op_points = spec["operatingPoints"]
    assert len(op_points) == 6

    expected_ops = [
        {
            "value": 0.00,
            "cutoff": 1.00,
            "realizedPpcr": 0.0 / 9.0,
            "TP": 0.0,
            "FP": 0.0,
            "TN": 4.0,
            "FN": 5.0,
        },
        {
            "value": 0.20,
            "cutoff": 0.71,
            "realizedPpcr": 2.0 / 9.0,
            "TP": 1.0,
            "FP": 1.0,
            "TN": 3.0,
            "FN": 4.0,
        },
        {
            "value": 0.40,
            "cutoff": 0.50,
            "realizedPpcr": 3.0 / 9.0,
            "TP": 2.0,
            "FP": 1.0,
            "TN": 3.0,
            "FN": 3.0,
        },
        {
            "value": 0.60,
            "cutoff": 0.50,
            "realizedPpcr": 3.0 / 9.0,
            "TP": 2.0,
            "FP": 1.0,
            "TN": 3.0,
            "FN": 3.0,
        },
        {
            "value": 0.80,
            "cutoff": 0.24,
            "realizedPpcr": 7.0 / 9.0,
            "TP": 4.0,
            "FP": 3.0,
            "TN": 1.0,
            "FN": 1.0,
        },
        {
            "value": 1.00,
            "cutoff": 0.00,
            "realizedPpcr": 9.0 / 9.0,
            "TP": 5.0,
            "FP": 4.0,
            "TN": 0.0,
            "FN": 0.0,
        },
    ]

    for op, exp in zip(op_points, expected_ops):
        assert op["evaluationId"] == "evaluation-1"
        assert op["type"] == "ppcr"
        assert pytest.approx(op["value"]) == exp["value"]
        assert pytest.approx(op["cutoff"]) == exp["cutoff"]
        assert pytest.approx(op["realizedPpcr"]) == exp["realizedPpcr"]

        metrics = {m["metricId"]: m["estimate"] for m in op["performance"]}
        assert pytest.approx(metrics["true_positives"]) == exp["TP"]
        assert pytest.approx(metrics["false_positives"]) == exp["FP"]
        assert pytest.approx(metrics["true_negatives"]) == exp["TN"]
        assert pytest.approx(metrics["false_negatives"]) == exp["FN"]


def test_n_less_than_q_empty_rank_bins_and_mass_conservation():
    probs = {"m1": np.array([0.2, 0.8])}
    reals = np.array([0, 1])
    by = 0.10  # q = 10, N = 2

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=by)
    spec = chart.spec

    rank_bins = spec["rankBins"]
    assert len(rank_bins) == 10

    total_pos = sum(rb["positiveMass"] for rb in rank_bins)
    total_neg = sum(rb["negativeMass"] for rb in rank_bins)

    assert total_pos == 1
    assert total_neg == 1
    # Verify some explicit empty rank bins exist
    empty_bins = [
        rb for rb in rank_bins if rb["positiveMass"] == 0 and rb["negativeMass"] == 0
    ]
    assert len(empty_bins) > 0


def test_threshold_endpoints():
    probs = {"m1": np.array([0.00, 0.25, 0.50, 0.75, 1.00])}
    reals = np.array([0, 1, 0, 1, 1])

    chart = rtichoke.create_probs_histogram(
        probs=probs,
        reals=reals,
        by=0.25,
        stratified_by=("probability_threshold",),
    )
    spec = chart.spec
    ops = {op["value"]: op for op in spec["operatingPoints"]}

    # Cutoff 0.0: everybody predicted positive (including exact-zero score)
    op_0 = ops[0.0]
    metrics_0 = {m["metricId"]: m["estimate"] for m in op_0["performance"]}
    assert metrics_0["true_positives"] == 3
    assert metrics_0["false_positives"] == 2
    assert op_0["realizedPpcr"] == 1.0

    # Cutoff 0.50: predicted positive iff probability > 0.50 (i.e. scores 0.75 and 1.00)
    op_50 = ops[0.50]
    metrics_50 = {m["metricId"]: m["estimate"] for m in op_50["performance"]}
    assert metrics_50["true_positives"] == 2  # 0.75 and 1.00 (both outcome 1)
    assert metrics_50["false_positives"] == 0

    # Cutoff 1.00: nobody predicted positive
    op_1 = ops[1.00]
    metrics_1 = {m["metricId"]: m["estimate"] for m in op_1["performance"]}
    assert metrics_1["true_positives"] == 0
    assert metrics_1["false_positives"] == 0
    assert op_1["realizedPpcr"] == 0.0


def test_multiple_models_shared_population():
    probs = {
        "m1": np.array([0.1, 0.4, 0.7]),
        "m2": np.array([0.2, 0.5, 0.8]),
    }
    reals = np.array([0, 1, 1])

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=0.5)
    spec = chart.spec

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "__shared_population__", "model": "m1"},
        {"id": "evaluation-2", "population": "__shared_population__", "model": "m2"},
    ]

    eval_ids = {bin_item["evaluationId"] for bin_item in spec["bins"]}
    assert eval_ids == {"evaluation-1", "evaluation-2"}


def test_multiple_populations():
    probs = {
        "pop_a": np.array([0.1, 0.6, 0.9]),
        "pop_b": np.array([0.3, 0.7]),
    }
    reals = {
        "pop_a": np.array([0, 1, 1]),
        "pop_b": np.array([0, 1]),
    }

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=0.5)
    spec = chart.spec

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "pop_a"},
        {"id": "evaluation-2", "population": "pop_b"},
    ]
    for ev in spec["evaluations"]:
        assert "model" not in ev

    pop_a_rank_bins = [
        rb for rb in spec["rankBins"] if rb["evaluationId"] == "evaluation-1"
    ]
    pop_b_rank_bins = [
        rb for rb in spec["rankBins"] if rb["evaluationId"] == "evaluation-2"
    ]

    assert sum(rb["positiveMass"] + rb["negativeMass"] for rb in pop_a_rank_bins) == 3
    assert sum(rb["positiveMass"] + rb["negativeMass"] for rb in pop_b_rank_bins) == 2


def test_join_failures():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    with patch("rtichoke._viz_spec_v2.prepare_performance_data") as mock_perf:
        # 1. Missing performance row
        import polars as pl

        mock_perf.return_value = pl.DataFrame(
            schema={
                "reference_group": pl.String,
                "stratified_by": pl.String,
                "chosen_cutoff": pl.Float64,
                "ppcr": pl.Float64,
            }
        )
        with pytest.raises(ValueError, match="Missing performance row"):
            _prediction_distribution_v2_spec_from_performance_data(probs, reals, by=0.5)

        # 2. Duplicated performance row
        dup_row = {
            "reference_group": "m1",
            "stratified_by": "probability_threshold",
            "chosen_cutoff": 0.0,
            "ppcr": 1.0,
            "true_positives": 2,
            "true_negatives": 0,
            "false_positives": 1,
            "false_negatives": 0,
            "sensitivity": 1.0,
            "specificity": 0.0,
            "ppv": 0.66,
            "npv": None,
            "lift": 1.0,
        }
        mock_perf.return_value = pl.DataFrame([dup_row, dup_row])
        with pytest.raises(ValueError, match="Duplicate performance match"):
            _prediction_distribution_v2_spec_from_performance_data(probs, reals, by=0.5)


def test_undefined_metrics_null_serialization():
    probs = {"m1": np.array([0.1, 0.4, 0.7])}
    reals = np.array([0, 0, 0])  # No positives -> sensitivity & PPV undefined

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=0.5)
    spec = chart.spec

    # Check metric estimates
    op = spec["operatingPoints"][0]
    metrics = {m["metricId"]: m["estimate"] for m in op["performance"]}
    assert metrics["sensitivity"] is None

    # Check serialized JSON
    tmp_path = Path("tmp_chart.html")
    try:
        chart.write_html(tmp_path)
        content = tmp_path.read_text(encoding="utf-8")
        assert '{"metricId":"sensitivity","estimate":null}' in content

        # Check that the embedded JSON spec payload has no NaN
        spec_text = content.split(
            '<script id="rtichoke-spec" type="application/json">'
        )[1].split("</script>")[0]
        assert "NaN" not in spec_text
        spec_data = json.loads(spec_text)
        assert spec_data["operatingPoints"][0]["performance"][4]["estimate"] is None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_producer_owned_performance_wins():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    # Patch prepare_performance_data to return a distinct patched value 999.0
    with patch("rtichoke._viz_spec_v2.prepare_performance_data") as mock_perf:
        from rtichoke.performance_data.performance_data import prepare_performance_data

        real_df = prepare_performance_data(
            probs, reals, stratified_by=("probability_threshold",), by=0.5
        )
        patched_df = real_df.with_columns(
            pl.when(pl.col("chosen_cutoff") == 0.0)
            .then(999.0)
            .otherwise(pl.col("true_positives"))
            .alias("true_positives")
        )
        mock_perf.return_value = patched_df

        spec = _prediction_distribution_v2_spec_from_performance_data(
            probs, reals, by=0.5
        )
        op_0 = [op for op in spec["operatingPoints"] if op["value"] == 0.0][0]
        tp_estimate = [
            m["estimate"]
            for m in op_0["performance"]
            if m["metricId"] == "true_positives"
        ][0]
        assert tp_estimate == 999.0

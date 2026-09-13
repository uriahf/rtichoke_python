"""Deterministic unit tests for standalone create_probs_histogram() adapter and canonical spec."""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import polars as pl
import pytest

import rtichoke
from rtichoke._viz_spec_v2 import (
    _prediction_distribution_v2_spec,
    _prediction_distribution_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.probs_distribution import (
    _prepare_probs_distribution_data,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


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
            "TP": 0,
            "FP": 0,
            "TN": 4,
            "FN": 5,
        },
        {
            "value": 0.20,
            "cutoff": 0.71,
            "realizedPpcr": 2.0 / 9.0,
            "TP": 1,
            "FP": 1,
            "TN": 3,
            "FN": 4,
        },
        {
            "value": 0.40,
            "cutoff": 0.50,
            "realizedPpcr": 3.0 / 9.0,
            "TP": 2,
            "FP": 1,
            "TN": 3,
            "FN": 3,
        },
        {
            "value": 0.60,
            "cutoff": 0.50,
            "realizedPpcr": 3.0 / 9.0,
            "TP": 2,
            "FP": 1,
            "TN": 3,
            "FN": 3,
        },
        {
            "value": 0.80,
            "cutoff": 0.24,
            "realizedPpcr": 7.0 / 9.0,
            "TP": 4,
            "FP": 3,
            "TN": 1,
            "FN": 1,
        },
        {
            "value": 1.00,
            "cutoff": 0.00,
            "realizedPpcr": 9.0 / 9.0,
            "TP": 5,
            "FP": 4,
            "TN": 0,
            "FN": 0,
        },
    ]

    for op, exp in zip(op_points, expected_ops):
        assert op["evaluationId"] == "evaluation-1"
        assert op["type"] == "ppcr"
        assert pytest.approx(op["value"]) == exp["value"]
        assert pytest.approx(op["cutoff"]) == exp["cutoff"]
        assert pytest.approx(op["realizedPpcr"]) == exp["realizedPpcr"]

        metrics = {m["metricId"]: m["estimate"] for m in op["performance"]}
        assert metrics["true_positives"] == exp["TP"]
        assert isinstance(metrics["true_positives"], int)
        assert metrics["false_positives"] == exp["FP"]
        assert isinstance(metrics["false_positives"], int)
        assert metrics["true_negatives"] == exp["TN"]
        assert isinstance(metrics["true_negatives"], int)
        assert metrics["false_negatives"] == exp["FN"]
        assert isinstance(metrics["false_negatives"], int)


def test_prepared_data_builder_does_not_call_statistical_producers():
    probs = {"m1": np.array([0.1, 0.4, 0.7])}
    reals = np.array([0, 1, 1])

    dist_data = _prepare_probs_distribution_data(probs, reals, by=0.5)
    perf_data = prepare_performance_data(probs, reals, by=0.5)
    eval_meta = _build_evaluation_metadata(probs, reals, np.array([]))

    with (
        patch("rtichoke._viz_spec_v2.prepare_performance_data") as mock_perf,
        patch("rtichoke._viz_spec_v2._prepare_probs_distribution_data") as mock_dist,
    ):
        spec = _prediction_distribution_v2_spec(
            distribution_data=dist_data,
            performance_data=perf_data,
            evaluation_metadata=eval_meta,
            stratified_by=("probability_threshold",),
        )
        mock_perf.assert_not_called()
        mock_dist.assert_not_called()
        assert spec["type"] == "prediction_distribution"


def test_public_wrapper_and_prepared_builder_produce_identical_spec():
    probs = {"m1": np.array([0.0, 0.25, 0.5, 0.75, 1.0])}
    reals = np.array([0, 0, 1, 1, 1])
    by = 0.25

    spec_wrapper = _prediction_distribution_v2_spec_from_performance_data(
        probs=probs, reals=reals, by=by, stratified_by=("probability_threshold",)
    )

    dist_data = _prepare_probs_distribution_data(probs, reals, by=by)
    perf_data = prepare_performance_data(probs, reals, by=by)
    eval_meta = _build_evaluation_metadata(probs, reals, np.array([]))

    spec_direct = _prediction_distribution_v2_spec(
        distribution_data=dist_data,
        performance_data=perf_data,
        evaluation_metadata=eval_meta,
        stratified_by=("probability_threshold",),
    )

    assert spec_wrapper == spec_direct


def test_confusion_matrix_estimates_are_integers():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=0.5)
    spec = chart.spec

    for op in spec["operatingPoints"]:
        metrics = {m["metricId"]: m["estimate"] for m in op["performance"]}
        for metric_id in (
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ):
            val = metrics[metric_id]
            assert isinstance(val, int), f"{metric_id} should be int, got {type(val)}"


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


def test_authoritative_schema_validation_probability_threshold_and_ppcr():
    from jsonschema import Draft202012Validator

    schema_json = {
        "type": "object",
        "required": [
            "schemaVersion",
            "type",
            "evaluations",
            "bins",
            "operatingPoints",
        ],
        "properties": {
            "schemaVersion": {"const": "2.0", "type": "string"},
            "type": {"const": "prediction_distribution", "type": "string"},
            "evaluations": {
                "minItems": 1,
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "population"],
                    "properties": {
                        "id": {"type": "string"},
                        "model": {"type": "string"},
                        "population": {"type": "string"},
                    },
                },
            },
            "operatingPoint": {
                "type": "object",
                "required": ["dimension"],
                "properties": {
                    "dimension": {"enum": ["probability_threshold", "ppcr"]}
                },
            },
            "bins": {
                "minItems": 1,
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "evaluationId",
                        "lower",
                        "upper",
                        "includeLower",
                        "includeUpper",
                        "nPositive",
                        "nNegative",
                    ],
                    "properties": {
                        "evaluationId": {"type": "string"},
                        "lower": {"minimum": 0, "maximum": 1, "type": "number"},
                        "upper": {"minimum": 0, "maximum": 1, "type": "number"},
                        "includeLower": {"type": "boolean"},
                        "includeUpper": {"type": "boolean"},
                        "nPositive": {"minimum": 0, "type": "integer"},
                        "nNegative": {"minimum": 0, "type": "integer"},
                    },
                },
            },
            "rankBins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "evaluationId",
                        "rankLower",
                        "rankUpper",
                        "positiveMass",
                        "negativeMass",
                    ],
                    "properties": {
                        "evaluationId": {"type": "string"},
                        "rankLower": {"minimum": 0, "maximum": 1, "type": "number"},
                        "rankUpper": {"minimum": 0, "maximum": 1, "type": "number"},
                        "positiveMass": {"minimum": 0, "type": "number"},
                        "negativeMass": {"minimum": 0, "type": "number"},
                    },
                },
            },
            "operatingPoints": {
                "minItems": 1,
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "evaluationId",
                        "type",
                        "value",
                        "cutoff",
                        "realizedPpcr",
                    ],
                    "properties": {
                        "evaluationId": {"type": "string"},
                        "type": {"enum": ["probability_threshold", "ppcr"]},
                        "value": {"minimum": 0, "maximum": 1, "type": "number"},
                        "cutoff": {"minimum": 0, "maximum": 1, "type": "number"},
                        "realizedPpcr": {"minimum": 0, "maximum": 1, "type": "number"},
                        "performance": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["metricId", "estimate"],
                                "properties": {
                                    "metricId": {"type": "string"},
                                    "estimate": {"type": ["number", "null"]},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    validator = Draft202012Validator(schema_json)

    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    # 1. Probability Threshold mode schema validation
    chart_thresh = rtichoke.create_probs_histogram(
        probs=probs, reals=reals, by=0.20, stratified_by=("probability_threshold",)
    )
    errors_thresh = list(validator.iter_errors(chart_thresh.spec))
    assert not errors_thresh, (
        f"Schema validation errors in probability_threshold mode: {errors_thresh}"
    )

    # 2. PPCR mode schema validation
    chart_ppcr = rtichoke.create_probs_histogram(
        probs=probs, reals=reals, by=0.20, stratified_by=("ppcr",)
    )
    errors_ppcr = list(validator.iter_errors(chart_ppcr.spec))
    assert not errors_ppcr, f"Schema validation errors in PPCR mode: {errors_ppcr}"


def test_stratified_by_boundary_validation():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    # Plain string error
    with pytest.raises(ValueError, match="plain string"):
        rtichoke.create_probs_histogram(
            probs, reals, stratified_by="probability_threshold"
        )

    # Empty sequence error
    with pytest.raises(ValueError, match="contain exactly one element"):
        rtichoke.create_probs_histogram(probs, reals, stratified_by=())

    # Sequence with > 1 elements
    with pytest.raises(ValueError, match="contain exactly one element"):
        rtichoke.create_probs_histogram(
            probs, reals, stratified_by=("probability_threshold", "ppcr")
        )

    # Unsupported dimension key error
    with pytest.raises(ValueError, match="Unsupported stratification key"):
        rtichoke.create_probs_histogram(
            probs, reals, stratified_by=("unsupported_key",)
        )


def test_negative_contract_failures():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    dist_data = _prepare_probs_distribution_data(probs, reals, by=0.5)
    perf_data = prepare_performance_data(probs, reals, by=0.5)
    eval_meta = _build_evaluation_metadata(probs, reals, np.array([]))

    # 1. Unknown reference group in bins
    bad_bins = dist_data["bins"].with_columns(
        pl.lit("unknown_group").alias("evaluation")
    )
    bad_dist = dict(dist_data, bins=bad_bins)
    with pytest.raises(ValueError, match="Unknown reference group in bins"):
        _prediction_distribution_v2_spec(bad_dist, perf_data, eval_meta)

    # 2. Unknown reference group in rank bins
    bad_rank_bins = dist_data["rank_bins"].with_columns(
        pl.lit("unknown_group").alias("evaluation")
    )
    bad_dist_rank = dict(dist_data, rank_bins=bad_rank_bins)
    with pytest.raises(ValueError, match="Unknown reference group in rank bins"):
        _prediction_distribution_v2_spec(bad_dist_rank, perf_data, eval_meta)

    # 3. Unknown reference group in operating points
    bad_ops = dist_data["operating_points"].with_columns(
        pl.lit("unknown_group").alias("evaluation")
    )
    bad_dist_op = dict(dist_data, operating_points=bad_ops)
    with pytest.raises(ValueError, match="Unknown reference group in operating points"):
        _prediction_distribution_v2_spec(bad_dist_op, perf_data, eval_meta)

    # 4. Non-finite requested operating-point value
    bad_op_value = dist_data["operating_points"].with_columns(
        pl.when(pl.col("value") == 0.0)
        .then(float("nan"))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    bad_dist_nan = dict(dist_data, operating_points=bad_op_value)
    with pytest.raises(ValueError, match="Non-finite operating point value"):
        _prediction_distribution_v2_spec(bad_dist_nan, perf_data, eval_meta)

    # 5. Incomplete evaluation coverage
    partial_meta = {"m1": eval_meta["m1"], "m2": eval_meta["m1"]}
    with pytest.raises(
        ValueError, match="Missing performance row|Incomplete evaluation coverage"
    ):
        _prediction_distribution_v2_spec(dist_data, perf_data, partial_meta)


def test_producer_owned_performance_wins():
    probs = {"m1": np.array([0.1, 0.5, 0.9])}
    reals = np.array([0, 1, 1])

    # Patch prepare_performance_data to return a distinct patched value 999
    with patch("rtichoke._viz_spec_v2.prepare_performance_data") as mock_perf:
        from rtichoke.performance_data.performance_data import prepare_performance_data

        real_df = prepare_performance_data(
            probs, reals, stratified_by=("probability_threshold",), by=0.5
        )
        patched_df = real_df.with_columns(
            pl.when(pl.col("chosen_cutoff") == 0.0)
            .then(999)
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
        assert tp_estimate == 999
        assert isinstance(tp_estimate, int)

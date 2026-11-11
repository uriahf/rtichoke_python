"""Population preparation against PR #441's independent survival/cmprsk oracle."""

import copy
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from jsonschema import Draft7Validator

from rtichoke._terminal_outcome_distribution import (
    _prepare_terminal_outcome_distributions,
    _terminal_outcome_distribution_v2_spec,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/terminal_event/reference.json").read_text()
)
VENDOR = Path(__file__).parents[1] / "src/rtichoke/_vendor/rtichoke_viz"


def frame(case):
    return pl.DataFrame(case["data"]).rename({"event": "reals", "time": "times"})


def ids(value):
    return (value,) if isinstance(value, str) else tuple(value)


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_every_reference_row(name):
    case = REFERENCE["cases"][name]
    source = frame(case)
    prepared = _prepare_terminal_outcome_distributions({name: source}, case["horizons"])
    expected = {
        (key, float(r["horizon"])): r
        for key, rows in case["variants"].items()
        for r in rows
    }
    assert len(prepared) == len(expected)
    for row in prepared:
        oracle = expected[(f"{row.censoring}:{row.competing}", row.horizon)]
        assert row.evaluation_id == "evaluation-1"
        assert row.population == name
        assert row.retained_ids == ids(oracle["retained_ids"])
        assert row.censored_excluded_ids == ids(oracle["censored_excluded_ids"])
        assert row.competing_excluded_ids == ids(oracle["competing_excluded_ids"])
        assert row.original_n == oracle["original_n"]
        assert row.retained_n == oracle["retained_n"]
        assert row.censored_excluded_count == oracle["censored_excluded_count"]
        assert row.competing_excluded_count == oracle["competing_excluded_count"]
        assert (
            row.retained_n + row.censored_excluded_count + row.competing_excluded_count
            == row.original_n
        )
        assert (
            row.censored_excluded_fraction
            == row.censored_excluded_count / row.original_n
        )
        assert (
            row.competing_excluded_fraction
            == row.competing_excluded_count / row.original_n
        )
        assert row.status == oracle["status"]
        assert row.probabilities == pytest.approx(oracle["probabilities"], abs=1e-12)
        assert row.estimator == (
            "aalen_johansen"
            if row.competing == "adjusted_as_negative"
            else "kaplan_meier"
        )
    assert source.equals(frame(case))  # Retention/recoding never mutates inputs.


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_serialization_preserves_every_population_horizon_and_status(name):
    case = REFERENCE["cases"][name]
    prepared = _prepare_terminal_outcome_distributions(
        {name: frame(case)}, case["horizons"]
    )
    spec = _terminal_outcome_distribution_v2_spec(prepared)
    assert len(spec["variants"]) == 8
    for row in prepared:
        variant = spec["variants"][f"{row.censoring}:{row.competing}"]
        assert variant["status"] == "available"
        assert len(variant["stateDistributions"]) == len(case["horizons"])
        distribution = next(
            d for d in variant["stateDistributions"] if d["horizon"] == row.horizon
        )
        assert distribution["evaluationId"] == row.evaluation_id
        assert distribution["estimateStatus"] == row.status
        assert {
            state["stateId"]: state["estimate"] for state in distribution["states"]
        } == pytest.approx(row.probabilities, abs=1e-12)
        assert all(
            set(state) == {"stateId", "label", "estimate"}
            for state in distribution["states"]
        )
        if row.status == "undefined_empty_population":
            assert all(state["estimate"] is None for state in distribution["states"])
        exclusion = variant["exclusions"][0]
        assert exclusion["originalSampleSize"] == row.original_n
        counts = next(
            c for c in exclusion["excludedCounts"] if c["horizon"] == row.horizon
        )
        assert counts == {
            "horizon": row.horizon,
            "censoredExcludedCount": row.censored_excluded_count,
            "competingExcludedCount": row.competing_excluded_count,
        }


def test_populations_and_horizons_keep_semantic_identity():
    case = REFERENCE["cases"]["fixture_a"]
    data = frame(case)
    prepared = _prepare_terminal_outcome_distributions(
        {"first": data, "numerically_equal_but_distinct": data}, [10, 5, 10]
    )
    assert len(prepared) == 2 * 2 * 8
    assert {r.evaluation_id for r in prepared if r.population == "first"} == {
        "evaluation-1"
    }
    assert {r.evaluation_id for r in prepared if r.population != "first"} == {
        "evaluation-2"
    }
    assert {r.horizon for r in prepared} == {5, 10}  # No injected horizon zero.
    spec = _terminal_outcome_distribution_v2_spec(prepared)
    for variant in spec["variants"].values():
        assert len(variant["stateDistributions"]) == 4
        assert len(variant["exclusions"]) == 2
        for ex in variant["exclusions"]:
            assert [r["horizon"] for r in ex["excludedCounts"]] == [5, 10]


def test_published_schema_and_referential_integrity(tmp_path):
    node = shutil.which("node")
    if node is None:
        raise pytest.skip.Exception(
            "Node is required for the published renderer integrity checks"
        )
    specs = []
    for name, case in REFERENCE["cases"].items():
        rows = _prepare_terminal_outcome_distributions(
            {name: frame(case)}, case["horizons"]
        )
        specs.append(_terminal_outcome_distribution_v2_spec(rows))
    multi = _prepare_terminal_outcome_distributions(
        {
            "a": frame(REFERENCE["cases"]["fixture_a"]),
            "b": frame(REFERENCE["cases"]["fixture_a"]),
        },
        [0, 5, 10],
    )
    specs.append(_terminal_outcome_distribution_v2_spec(multi))
    # Load the immutable vendored module itself, not rtichoke_viz/main or a local build.
    script = tmp_path / "validate.mjs"
    script.write_text("""
import fs from "node:fs";
const source = fs.readFileSync(process.argv[2]);
const viz = await import("data:text/javascript;base64," + source.toString("base64"));
const specs = JSON.parse(fs.readFileSync(0, "utf8"));
for (const spec of specs) {
  try { viz.assertOutcomeDistributionReferentialIntegrity(spec); }
  catch (error) { console.error(error.message); process.exit(1); }
}
const reports = specs.map(spec => ({
  schemaVersion: "1.1", type: "report",
  heuristics: {active: "excluded:excluded", available: Object.keys(spec.variants)},
  sections: [{id: "outcome", title: "Outcome", items: [{type: "component", id: "od", spec}]}]
}));
for (const report of reports) viz.assertReportReferentialIntegrity(report);
const bad = structuredClone(specs[0]);
bad.variants["excluded:excluded"].stateDistributions[0].evaluationId = "missing";
let rejected = false;
try { viz.assertOutcomeDistributionReferentialIntegrity(bad); } catch { rejected = true; }
if (!rejected) throw Error("integrity negative control was not rejected");
// Status-aware integrity must reject nulls declared as defined.
const invalid = structuredClone(specs[0]);
for (const d of invalid.stateDistributions) {
  d.estimateStatus = "defined";
  for (const state of d.states) state.estimate = null;
}
rejected = false;
try { viz.assertOutcomeDistributionReferentialIntegrity(invalid); } catch { rejected = true; }
if (!rejected) throw Error("status negative control was not rejected");
console.log(JSON.stringify({schema: viz.OutcomeDistributionSpecSchema,
 stateSchema: viz.StateDistributionSchema, reportSchema: viz.ReportSpecV1_1Schema, reports}));
""")
    result = subprocess.run(
        [node, str(script), str(VENDOR / "rtichoke-viz.js")],
        input=json.dumps(specs, allow_nan=False),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    schemas = json.loads(result.stdout)
    validator = Draft7Validator(schemas["schema"])
    for spec in specs:
        validator.validate(spec)
    report_validator = Draft7Validator(schemas["reportSchema"])
    for report in schemas["reports"]:
        report_validator.validate(report)
    assert {
        branch["const"]
        for branch in schemas["stateSchema"]["properties"]["estimateStatus"]["anyOf"]
    } == {"defined", "undefined_empty_population", "beyond_support_plateau"}
    assert "retainedSampleSize" not in schemas["stateSchema"]["properties"]
    invalid = copy.deepcopy(specs[0])
    invalid["variants"]["excluded:excluded"]["exclusions"][0]["excludedCounts"][0][
        "censoredExcludedCount"
    ] = -1
    assert list(validator.iter_errors(invalid))


@pytest.mark.parametrize("horizons", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_horizons(horizons):
    with pytest.raises(ValueError, match="Fixed horizons"):
        _prepare_terminal_outcome_distributions(
            {"a": frame(REFERENCE["cases"]["fixture_a"])}, horizons
        )


@pytest.mark.parametrize(
    "data",
    [
        {"id": ["a"], "reals": [0.5], "times": [1]},
        {"id": ["a"], "reals": [3], "times": [1]},
        {"id": ["a"], "reals": [0], "times": [-1]},
        {"id": ["a"], "reals": [0], "times": [float("nan")]},
        {"id": ["a", "a"], "reals": [0, 1], "times": [1, 2]},
        {"id": [None], "reals": [0], "times": [1]},
    ],
)
def test_invalid_population_data(data):
    with pytest.raises(ValueError):
        _prepare_terminal_outcome_distributions({"a": pl.DataFrame(data)}, [1])


def test_empty_input_is_not_an_empty_retained_population():
    with pytest.raises(ValueError, match="At least one population"):
        _prepare_terminal_outcome_distributions({}, [1])
    with pytest.raises(ValueError, match="nonempty"):
        _prepare_terminal_outcome_distributions({"a": pl.DataFrame()}, [1])


def test_mixed_horizons_are_not_silently_dropped():
    case = REFERENCE["cases"]["all_censored"]
    rows = _prepare_terminal_outcome_distributions({"a": frame(case)}, [0, 3])
    assert {r.horizon for r in rows} == {0, 3}
    assert {r.status for r in rows} == {
        "defined",
        "undefined_empty_population",
        "beyond_support_plateau",
    }
    spec = _terminal_outcome_distribution_v2_spec(rows)
    assert {
        d["estimateStatus"]
        for variant in spec["variants"].values()
        for d in variant["stateDistributions"]
    } == {r.status for r in rows}
    assert all(
        [d["horizon"] for d in variant["stateDistributions"]] == [0, 3]
        for variant in spec["variants"].values()
    )


def test_adapter_only_bounds_machine_scale_endpoint_roundoff():
    case = REFERENCE["cases"]["fixture_a"]
    rows = _prepare_terminal_outcome_distributions({"a": frame(case)}, [14])
    spec = _terminal_outcome_distribution_v2_spec(rows)
    for variant in spec["variants"].values():
        assert all(
            0 <= s["estimate"] <= 1
            for d in variant["stateDistributions"]
            for s in d["states"]
        )
    invalid = replace(rows[0], probabilities={"primary": -1e-8, "event_free": 1 + 1e-8})
    with pytest.raises(ValueError, match="finite probability"):
        _terminal_outcome_distribution_v2_spec((invalid, *rows[1:]))
    with pytest.raises(ValueError, match="exactly eight pairs"):
        _terminal_outcome_distribution_v2_spec(rows[:-1])

"""Public outcome-only reports against independent terminal-event references."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft7Validator

from rtichoke import create_reals_summary_report_times
from rtichoke.reals_distribution import _outcome_distribution_v2_spec

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/terminal_event/reference.json").read_text()
)
VENDOR = Path(__file__).parents[1] / "src/rtichoke/_vendor/rtichoke_viz"


def embedded_report(path):
    html = path.read_text(encoding="utf-8")
    return json.loads(
        re.search(
            r'<script id="rtichoke-report-spec" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        ).group(1)
    )


@pytest.mark.parametrize(
    "name", [name for name in REFERENCE["cases"] if name != "zero_ties"]
)
def test_public_report_preserves_every_independent_reference_row(name, tmp_path):
    case = REFERENCE["cases"][name]
    reals = [row["event"] for row in case["data"]]
    times = [row["time"] for row in case["data"]]
    result = create_reals_summary_report_times(
        {name: reals, "same_numbers_distinct_population": reals},
        {name: times, "same_numbers_distinct_population": times},
        list(reversed(case["horizons"][1:])) + case["horizons"][1:2],
        output_file=tmp_path / "subdir/report.html",
        heuristics=True,
    )
    assert result == tmp_path / "subdir/report.html"
    report = embedded_report(result)
    schema = json.loads((VENDOR / "rtichoke-viz-report.schema.json").read_text())
    Draft7Validator(schema).validate(report)
    assert report["heuristics"]["active"] == "adjusted:adjusted_as_negative"
    spec = report["sections"][0]["items"][0]["spec"]
    assert report["heuristics"]["available"] == list(case["variants"])
    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": name},
        {"id": "evaluation-2", "population": "same_numbers_distinct_population"},
    ]
    for key, expected in case["variants"].items():
        variant = spec["variants"][key]
        for evaluation in spec["evaluations"]:
            actual = [
                row
                for row in variant["stateDistributions"]
                if row["evaluationId"] == evaluation["id"]
            ]
            assert [row["horizon"] for row in actual] == case["horizons"]
            exclusion = next(
                row
                for row in variant["exclusions"]
                if row["evaluationId"] == evaluation["id"]
            )
            for row, oracle, counts in zip(
                actual, expected, exclusion["excludedCounts"], strict=True
            ):
                assert row["estimateStatus"] == oracle["status"]
                assert row["estimateOrigin"] == "fixed_time_horizon"
                estimates = {
                    state["stateId"]: state["estimate"] for state in row["states"]
                }
                assert estimates == pytest.approx(oracle["probabilities"], abs=1e-12)
                assert (
                    counts["censoredExcludedCount"] == oracle["censored_excluded_count"]
                )
                assert (
                    counts["competingExcludedCount"]
                    == oracle["competing_excluded_count"]
                )
                assert exclusion["originalSampleSize"] == oracle["original_n"]
                assert all("count" not in state for state in row["states"])

    node = shutil.which("node")
    if node:
        payload = tmp_path / "report.json"
        payload.write_text(json.dumps(report, allow_nan=False))
        subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                f"import * as viz from {json.dumps((VENDOR / 'rtichoke-viz.js').as_uri())};"
                'import fs from "node:fs";'
                'const r=JSON.parse(fs.readFileSync(process.argv[1], "utf8"));'
                "viz.assertReportReferentialIntegrity(r);"
                "viz.assertOutcomeDistributionReferentialIntegrity(r.sections[0].items[0].spec);",
                str(payload),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_default_is_raw_and_has_no_controls_or_performance_sections(tmp_path):
    reals, times, horizons = [1, 2, 0], [1, 2, 0], [4, 2, 4]
    result = create_reals_summary_report_times(
        reals, times, horizons, output_file=tmp_path / "raw.html"
    )
    report = embedded_report(result)
    assert "heuristics" not in report
    assert len(report["sections"]) == 1
    assert report["sections"][0]["items"][0]["spec"] == _outcome_distribution_v2_spec(
        reals, times, horizons
    )
    assert "raw observed outcome status over time" in result.read_text()
    assert "Performance Metrics Cheat Sheet" not in result.read_text()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "reals,times,horizons",
    [
        ([1, 0], [2], [1]),
        ([1], [2, 3], [1]),
        ([], [], [1]),
        ([[1]], [[2]], [1]),
        ([0.5], [2], [1]),
        ([3], [2], [1]),
        ([np.nan], [2], [1]),
        ([1], [np.inf], [1]),
        ([1], [-1], [1]),
        ([1], ["2"], [1]),
        ([1], [2], []),
        ([1], [2], [-1]),
        ([1], [2], [np.nan]),
        ([1], [2], [np.inf]),
        ([1], [2], [[1]]),
        ([1], [2], ["1"]),
        ({"a": [1]}, [2], [1]),
        ({"a": [1]}, {"b": [2]}, [1]),
        ({"a": [1], "b": [0]}, {"b": [2], "a": [3]}, [1]),
        ({}, {}, [1]),
        ({"": [1]}, {"": [2]}, [1]),
    ],
)
def test_invalid_inputs_fail_without_writing(tmp_path, enabled, reals, times, horizons):
    destination = tmp_path / "report.html"
    with pytest.raises(ValueError):
        create_reals_summary_report_times(
            reals, times, horizons, heuristics=enabled, output_file=destination
        )
    assert not destination.exists()


@pytest.mark.parametrize("invalid", [None, "yes", 1, np.bool_(True), [], [True]])
def test_heuristics_requires_python_bool(invalid, tmp_path):
    with pytest.raises(ValueError, match="heuristics must be a bool"):
        create_reals_summary_report_times(
            [1], [2], [1], heuristics=invalid, output_file=tmp_path / "bad.html"
        )


def test_renderer_alias_and_unsupported_renderer(tmp_path):
    create_reals_summary_report_times(
        [1], [2], [1], renderer="rtichoke_viz", output_file=tmp_path / "alias.html"
    )
    with pytest.raises(ValueError, match="Unsupported renderer"):
        create_reals_summary_report_times([1], [2], [1], renderer="plotly")


@pytest.mark.parametrize("n", [600, 1200])
def test_large_complete_cohort_endpoint_roundoff_does_not_abort_public_report(
    n, tmp_path
):
    # With one event per time, the true terminal risk is exactly one. Summing
    # many AJ increments can overshoot by more than eight machine epsilons.
    result = create_reals_summary_report_times(
        [1] * n,
        list(range(1, n + 1)),
        [n // 2, n, n + 1],
        output_file=tmp_path / "complete.html",
        heuristics=True,
    )
    report = embedded_report(result)
    spec = report["sections"][0]["items"][0]["spec"]
    for variant in spec["variants"].values():
        for row in variant["stateDistributions"]:
            values = {s["stateId"]: s["estimate"] for s in row["states"]}
            assert all(0 <= p <= 1 for p in values.values())
            assert sum(values.values()) == pytest.approx(1, abs=1e-12)
            if row["horizon"] >= n:
                assert values["event_free"] == pytest.approx(0, abs=1e-12)
                assert values.get("primary", values.get("composite")) == pytest.approx(
                    1, abs=1e-12
                )
            elif row["horizon"] == n // 2:
                assert values["event_free"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("heuristics", [False, True])
@pytest.mark.parametrize("event", [1, 2])
def test_report_rejects_events_at_time_zero(tmp_path, heuristics, event):
    with pytest.raises(ValueError, match="events must occur after time zero"):
        create_reals_summary_report_times(
            [event, 0],
            [0, 0],
            [1],
            output_file=tmp_path / "report.html",
            heuristics=heuristics,
        )

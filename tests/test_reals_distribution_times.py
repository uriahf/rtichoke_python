import json
from pathlib import Path

import pytest

from rtichoke.reals_distribution import (
    _outcome_distribution_v2_spec,
    create_reals_distribution_times,
)
from rtichoke.summary_report.summary_report import (
    _create_reals_summary_report_times,
    create_summary_report_times,
)


def test_create_reals_distribution_times_returns_browser_chart_with_outcome_distribution_spec():
    times = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3]
    reals = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
    fixed_time_horizons = [10, 20, 30, 40, 50]

    chart = create_reals_distribution_times(reals, times, fixed_time_horizons)
    assert hasattr(chart, "spec")
    spec = chart.spec

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "outcome_distribution"
    assert spec["title"] == "Outcome Distribution"
    assert "evaluations" in spec
    assert "stateDistributions" in spec


def test_canonical_example_exact_fixed_horizon_counts():
    times = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3]
    reals = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
    fixed_time_horizons = [10, 20, 30, 40, 50]

    spec = _outcome_distribution_v2_spec(reals, times, fixed_time_horizons)

    fh_distributions = [
        sd
        for sd in spec["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon"
    ]

    expected = {
        0.0: {
            "Event": 0,
            "Competing event": 0,
            "Event-free": 10,
            "Censored": 0,
        },
        10.0: {
            "Event": 2,
            "Competing event": 0,
            "Event-free": 8,
            "Censored": 0,
        },
        20.0: {
            "Event": 3,
            "Competing event": 1,
            "Event-free": 6,
            "Censored": 0,
        },
        30.0: {
            "Event": 4,
            "Competing event": 1,
            "Event-free": 5,
            "Censored": 0,
        },
        40.0: {
            "Event": 5,
            "Competing event": 1,
            "Event-free": 2,
            "Censored": 2,
        },
        50.0: {
            "Event": 6,
            "Competing event": 2,
            "Event-free": 0,
            "Censored": 2,
        },
    }

    found_horizons = {}
    for sd in fh_distributions:
        h = sd["horizon"]
        counts = {st["label"]: st["count"] for st in sd["states"]}
        found_horizons[h] = counts

    for h, exp_counts in expected.items():
        assert h in found_horizons, (
            f"Missing horizon {h} in fixed_time_horizon stateDistributions"
        )
        assert found_horizons[h] == exp_counts, (
            f"Counts mismatch at horizon {h}: {found_horizons[h]} != {exp_counts}"
        )


def test_state_labels_exact_canonical():
    times = [10.0]
    reals = [1]
    horizons = [10.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)
    labels = [st["label"] for st in spec["stateDistributions"][0]["states"]]
    assert labels == ["Event", "Competing event", "Event-free", "Censored"]


def test_time_zero_included_and_all_observations_event_free():
    times = [10.0, 20.0, 30.0]
    reals = [1, 0, 2]
    horizons = [15.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)

    h0_dist = next(
        sd
        for sd in spec["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon" and sd["horizon"] == 0.0
    )

    counts = {st["label"]: st["count"] for st in h0_dist["states"]}
    assert counts["Event"] == 0
    assert counts["Competing event"] == 0
    assert counts["Censored"] == 0
    assert counts["Event-free"] == len(reals)


def test_censoring_strictly_before_versus_at_horizon():
    # Censored observation at time 10.0, event at time 15.0
    times = [10.0, 15.0]
    reals = [0, 1]

    # At horizon 10.0: censoring at 10.0 is NOT strictly before 10.0 (times < 10.0 is False),
    # so count for Censored is 0, and it remains Event-free.
    spec_h10 = _outcome_distribution_v2_spec(reals, times, [10.0])
    h10_dist = next(
        sd
        for sd in spec_h10["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon" and sd["horizon"] == 10.0
    )
    counts_10 = {st["label"]: st["count"] for st in h10_dist["states"]}
    assert counts_10["Censored"] == 0
    assert counts_10["Event-free"] == 2
    assert counts_10["Event"] == 0

    # At horizon 12.0: censoring at 10.0 IS strictly before 12.0 (times < 12.0 is True),
    # so count for Censored is 1, and Event-free is 1.
    spec_h12 = _outcome_distribution_v2_spec(reals, times, [12.0])
    h12_dist = next(
        sd
        for sd in spec_h12["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon" and sd["horizon"] == 12.0
    )
    counts_12 = {st["label"]: st["count"] for st in h12_dist["states"]}
    assert counts_12["Censored"] == 1
    assert counts_12["Event-free"] == 1
    assert counts_12["Event"] == 0


def test_spec_uses_estimator_raw_and_count_values_only():
    times = [10.0, 20.0]
    reals = [1, 0]
    horizons = [15.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)
    spec_json = json.dumps(spec)

    assert '"estimate"' not in spec_json
    assert '"mass"' not in spec_json

    for sd in spec["stateDistributions"]:
        assert sd["estimator"] == "raw"
        for st in sd["states"]:
            assert "count" in st
            assert isinstance(st["count"], int)


def test_spec_includes_event_table_and_fixed_time_horizon_rows():
    times = [10.0, 20.0]
    reals = [1, 0]
    horizons = [15.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)

    origins = {sd["estimateOrigin"] for sd in spec["stateDistributions"]}
    assert origins == {"event_table", "fixed_time_horizon"}


def test_multi_population_dict_preserves_insertion_order():
    reals = {
        "Pop B": [1, 0],
        "Pop A": [0, 1],
    }
    times = {
        "Pop B": [10.0, 20.0],
        "Pop A": [5.0, 15.0],
    }
    horizons = [10.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)
    evaluations = spec["evaluations"]

    assert len(evaluations) == 2
    assert evaluations[0]["population"] == "Pop B"
    assert evaluations[0]["id"] == "evaluation-1"
    assert evaluations[1]["population"] == "Pop A"
    assert evaluations[1]["id"] == "evaluation-2"


def test_horizon_sorting_deduplication_and_zero_prepending():
    times = [10.0, 20.0]
    reals = [1, 0]
    horizons = [30.0, 10.0, 0.0, 20.0, 10.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)
    fh_horizons = [
        sd["horizon"]
        for sd in spec["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon"
    ]
    assert fh_horizons == [0.0, 10.0, 20.0, 30.0]


def test_unsupported_renderer_fails_clearly():
    times = [10.0, 20.0]
    reals = [1, 0]
    horizons = [15.0]

    with pytest.raises(ValueError, match="Unsupported renderer"):
        create_reals_distribution_times(
            reals,
            times,
            horizons,
            renderer="invalid_renderer",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Unsupported renderer"):
        _create_reals_summary_report_times(
            reals,
            times,
            horizons,
            renderer="invalid_renderer",  # type: ignore[arg-type]
        )


def test_invalid_horizon_values_raise_value_error():
    times = [10.0, 20.0]
    reals = [1, 0]

    for invalid_h in [-5.0, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="fixed_time_horizons"):
            _outcome_distribution_v2_spec(reals, times, [invalid_h])


def test_internal_reals_summary_report_times_returns_valid_minimal_report_spec(
    tmp_path: Path,
):
    times = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3]
    reals = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
    fixed_time_horizons = [10, 20, 30, 40, 50]

    out_file = tmp_path / "minimal_report.html"
    result = _create_reals_summary_report_times(
        reals, times, fixed_time_horizons, output_file=out_file
    )

    assert result == out_file
    assert out_file.exists()

    html = out_file.read_text(encoding="utf-8")
    start = html.index('<script id="rtichoke-report-spec" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    report = json.loads(html[start:end])

    assert report["schemaVersion"] == "1.1"
    assert report["type"] == "report"
    assert report["title"] == "Time-Dependent Summary Report"

    sections = report["sections"]
    assert len(sections) == 1
    section = sections[0]
    assert section["id"] == "outcome-distribution"
    assert section["title"] == "Outcome Distribution"

    items = section["items"]
    assert len(items) == 1
    comp = items[0]
    assert comp["id"] == "outcome-distribution"
    assert comp["title"] == "Outcome Distribution"

    comp_spec = comp["spec"]
    assert comp_spec["schemaVersion"] == "2.0"
    assert comp_spec["type"] == "outcome_distribution"


def test_existing_create_summary_report_times_remains_unchanged(tmp_path: Path):
    probs = {"Model A": [0.1, 0.4, 0.7]}
    reals = [0, 1, 0]
    times = [5.0, 10.0, 15.0]
    horizons = [10.0]

    out_file = tmp_path / "full_report.html"
    result = create_summary_report_times(
        probs, reals, times, horizons, output_file=out_file
    )

    assert result == out_file
    assert out_file.exists()

    html = out_file.read_text(encoding="utf-8")
    start = html.index('<script id="rtichoke-report-spec" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    report = json.loads(html[start:end])

    section_ids = [s["id"] for s in report["sections"]]
    assert section_ids == [
        "event-risk",
        "calibration",
        "discrimination",
        "utility",
        "performance-table",
    ]

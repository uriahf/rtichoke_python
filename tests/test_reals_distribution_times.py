import json


from rtichoke.reals_distribution import (
    _outcome_distribution_v2_spec,
    create_reals_distribution_times,
)
from rtichoke.summary_report.summary_report import create_summary_report_times


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
            "Target event": 0,
            "Competing outcome": 0,
            "No target event": 10,
            "Unknown / excluded": 0,
        },
        10.0: {
            "Target event": 2,
            "Competing outcome": 0,
            "No target event": 8,
            "Unknown / excluded": 0,
        },
        20.0: {
            "Target event": 3,
            "Competing outcome": 1,
            "No target event": 6,
            "Unknown / excluded": 0,
        },
        30.0: {
            "Target event": 4,
            "Competing outcome": 1,
            "No target event": 5,
            "Unknown / excluded": 0,
        },
        40.0: {
            "Target event": 5,
            "Competing outcome": 1,
            "No target event": 2,
            "Unknown / excluded": 2,
        },
        50.0: {
            "Target event": 6,
            "Competing outcome": 2,
            "No target event": 0,
            "Unknown / excluded": 2,
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


def test_time_zero_included_and_all_observations_no_target_event():
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
    assert counts["Target event"] == 0
    assert counts["Competing outcome"] == 0
    assert counts["Unknown / excluded"] == 0
    assert counts["No target event"] == len(reals)


def test_censored_before_horizon_counted_as_unknown_excluded():
    times = [5.0, 15.0]
    reals = [0, 1]
    horizons = [10.0]

    spec = _outcome_distribution_v2_spec(reals, times, horizons)

    h10_dist = next(
        sd
        for sd in spec["stateDistributions"]
        if sd["estimateOrigin"] == "fixed_time_horizon" and sd["horizon"] == 10.0
    )

    counts = {st["label"]: st["count"] for st in h10_dist["states"]}
    assert counts["Unknown / excluded"] == 1
    assert counts["No target event"] == 1
    assert counts["Target event"] == 0
    assert counts["Competing outcome"] == 0


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


def test_create_summary_report_times_returns_valid_minimal_report_spec(tmp_path):
    times = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3]
    reals = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
    fixed_time_horizons = [10, 20, 30, 40, 50]

    out_file = tmp_path / "minimal_report.html"
    result = create_summary_report_times(
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

import json
from typing import Any, cast

import numpy as np

from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
    _performance_table_times_spec_from_performance_data,
)
from rtichoke._report_browser import RtichokeBrowserReport
from rtichoke._report_spec import _report_spec_from_components
from rtichoke._viz_spec_v2 import (
    _gains_v2_spec_from_performance_data,
    _roc_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


def _embedded_report(html: str) -> dict[str, Any]:
    start = html.index('<script id="rtichoke-report-spec" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return cast(dict[str, Any], json.loads(html[start:end]))


def test_browser_report_uses_real_producers_and_only_shared_render_report(tmp_path):
    probs = {"Model A": np.array([0.05, 0.2, 0.7, 0.95])}
    reals = np.array([0, 0, 1, 1])
    performance_data = prepare_performance_data(probs, reals, by=0.25)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))

    table = _performance_table_spec_from_performance_data(performance_data, metadata)
    roc = _roc_v2_spec_from_performance_data(performance_data, metadata)
    gains = _gains_v2_spec_from_performance_data(performance_data, metadata)
    report = _report_spec_from_components(
        [
            {"title": "Performance", "spec": table},
            {"title": "ROC", "spec": roc},
            {"title": "Gains", "spec": gains},
        ],
        title="Canonical report",
    )

    output = RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        tmp_path / "report.html"
    )
    html = output.read_text(encoding="utf-8")

    assert 'import { renderReport } from "./rtichoke-viz.js";' in html
    assert "append(renderReport(spec))" in html
    assert "renderRocV2" not in html
    assert "renderPerformanceTable" not in html
    assert "renderGainsV2" not in html
    assert _embedded_report(html) == report
    assert [component["id"] for component in report["components"]] == [
        "performance-table",
        "roc",
        "gains",
    ]
    assert (tmp_path / "rtichoke-viz.js").exists()
    assert (tmp_path / "rtichoke-viz.css").exists()


def test_browser_report_preserves_component_local_evaluation_ids_and_time_context(
    tmp_path,
):
    probs = {"Model A": np.array([0.1, 0.3, 0.7, 0.9])}
    reals = np.array([0, 1, 0, 1])
    times = np.array([2.0, 3.0, 7.0, 8.0])
    time_data = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        heuristics_sets=[
            {
                "censoring_heuristic": "adjusted",
                "competing_heuristic": "adjusted_as_negative",
            }
        ],
        by=0.5,
    )
    time_table = _performance_table_times_spec_from_performance_data(
        time_data,
        _build_evaluation_metadata(probs, reals, times),
    )

    static_data = prepare_performance_data(probs, reals, by=0.5)
    roc = _roc_v2_spec_from_performance_data(
        static_data,
        _build_evaluation_metadata(probs, reals, np.array([])),
    )
    report = _report_spec_from_components(
        [{"spec": time_table}, {"spec": roc}, {"spec": roc}]
    )

    assert "horizon" not in report
    assert [component["id"] for component in report["components"]] == [
        "performance-table",
        "roc",
        "roc-2",
    ]
    assert {row["horizon"] for row in time_table["rows"]} == {5.0, 10.0}
    assert time_table["evaluations"][0]["id"] == "evaluation-1"
    assert cast(dict[str, Any], roc)["evaluations"][0]["id"] == "evaluation-1"

    output = RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        tmp_path / "time-report.html"
    )
    assert _embedded_report(output.read_text(encoding="utf-8")) == report

import json
from importlib.resources import files
from typing import Any, cast

import numpy as np
import pytest

from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
    _performance_table_times_spec_from_performance_data,
)
from rtichoke._report_browser import (
    RtichokeBrowserReport,
    _resolve_render_report_symbol,
)
from rtichoke._report_spec import _build_report_spec_v11
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
    report = _build_report_spec_v11(
        [
            {
                "id": "sec-1",
                "components": [
                    {"id": "performance-table", "title": "Performance", "spec": table},
                    {"id": "roc", "title": "ROC", "spec": roc},
                    {"id": "gains", "title": "Gains", "spec": gains},
                ],
            }
        ],
        title="Canonical report",
    )

    output = RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        tmp_path / "report.html"
    )
    html = output.read_text(encoding="utf-8")
    vendor = files("rtichoke").joinpath("_vendor", "rtichoke_viz")
    viz_js = vendor.joinpath("rtichoke-viz.js").read_text(encoding="utf-8")
    viz_css = vendor.joinpath("rtichoke-viz.css").read_text(encoding="utf-8")

    assert 'import { renderReport } from "./rtichoke-viz.js";' not in html
    assert viz_js in html
    assert viz_css in html
    assert '<link rel="stylesheet" href="./rtichoke-viz.css">' not in html
    assert 'sectionGroupPresentation: "tabs"' in html
    assert 'groupPresentation: "stacked"' in html
    assert _embedded_report(html) == report
    assert [item["id"] for item in report["sections"][0]["items"]] == [
        "performance-table",
        "roc",
        "gains",
    ]
    assert [f.name for f in sorted(tmp_path.iterdir())] == ["report.html"]
    assert not (tmp_path / "rtichoke-viz.js").exists()
    assert not (tmp_path / "rtichoke-viz.css").exists()


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
    report = _build_report_spec_v11(
        [
            {
                "id": "sec-1",
                "components": [
                    {"id": "performance-table", "spec": time_table},
                    {"id": "roc", "spec": roc},
                    {"id": "roc-2", "spec": roc},
                ],
            }
        ]
    )

    assert "horizon" not in report
    assert [item["id"] for item in report["sections"][0]["items"]] == [
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


def test_browser_report_does_not_delete_or_overwrite_preexisting_assets(tmp_path):
    existing_js = tmp_path / "rtichoke-viz.js"
    existing_css = tmp_path / "rtichoke-viz.css"
    existing_js.write_text("// custom preexisting js", encoding="utf-8")
    existing_css.write_text("/* custom preexisting css */", encoding="utf-8")

    probs = {"Model A": np.array([0.05, 0.2, 0.7, 0.95])}
    reals = np.array([0, 0, 1, 1])
    performance_data = prepare_performance_data(probs, reals, by=0.25)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    table = _performance_table_spec_from_performance_data(performance_data, metadata)
    report = _build_report_spec_v11(
        [{"id": "sec-1", "components": [{"id": "performance-table", "spec": table}]}]
    )

    RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        tmp_path / "report.html"
    )

    assert existing_js.read_text(encoding="utf-8") == "// custom preexisting js"
    assert existing_css.read_text(encoding="utf-8") == "/* custom preexisting css */"


def test_resolve_render_report_symbol_export_clauses():
    # Unminified / shorthand export
    shorthand_js = "export { foo, renderReport, bar };"
    assert _resolve_render_report_symbol(shorthand_js) == "renderReport"

    # Direct function export
    fn_export_js = "export function renderReport(spec, options) { return null; }"
    assert _resolve_render_report_symbol(fn_export_js) == "renderReport"

    # Minified / aliased export with arbitrary identifier
    aliased_js = "export { foo, abc as renderReport, bar };"
    assert _resolve_render_report_symbol(aliased_js) == "abc"

    # Compact minified export syntax
    compact_aliased_js = "export{x1,y2 as renderReport,z3};"
    assert _resolve_render_report_symbol(compact_aliased_js) == "y2"

    # Missing renderReport export raises a clear error
    missing_js = "export { foo, bar };"
    with pytest.raises(ValueError, match="Could not resolve 'renderReport' export"):
        _resolve_render_report_symbol(missing_js)

import json
from typing import Any, cast

import numpy as np
from plotly.graph_objs._figure import Figure

from rtichoke._calibration_viz_spec_v2 import _calibration_v2_spec_from_curve_list
from rtichoke._performance_table_spec import (
    _performance_table_spec_from_performance_data,
)
from rtichoke._report_browser import RtichokeBrowserReport
from rtichoke._report_spec import _report_spec_from_components
from rtichoke._viz_browser import _calibration_spec_from_curve_list
from rtichoke._viz_spec_v2 import _roc_v2_spec_from_performance_data
from rtichoke.calibration.calibration import (
    _create_calibration_curve_list,
    create_calibration_curve,
)
from rtichoke.performance_data.performance_data import prepare_performance_data
from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)


def _metadata(probs, reals):
    return _build_evaluation_metadata(probs, reals, np.array([]))


def _one_model_inputs():
    probs = {
        "Model A": np.array(
            [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
        )
    }
    reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    return probs, reals


def _embedded_report(html: str) -> dict[str, Any]:
    start = html.index('<script id="rtichoke-report-spec" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return cast(dict[str, Any], json.loads(html[start:end]))


def test_calibration_v2_contract_shape_from_real_discrete_production_data():
    probs, reals = _one_model_inputs()
    curve_list = _create_calibration_curve_list(probs, reals)

    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(curve_list, _metadata(probs, reals)),
    )

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "calibration"
    assert spec["x"] == "predicted"
    assert spec["y"] == "observed"
    assert spec["xAxis"] == {"label": "Predicted probability", "domain": [0, 1]}
    assert spec["yAxis"] == {"label": "Observed probability", "domain": [0, 1]}
    assert spec["references"] == [{"type": "identity", "scope": "global"}]
    assert spec["evaluations"] == [
        {
            "id": "evaluation-1",
            "model": "Model A",
            "population": _SHARED_POPULATION,
        }
    ]
    assert spec["series"] == [
        {
            "id": "series-1",
            "evaluationId": "evaluation-1",
            "display": {"label": "Model A", "group": "Model A", "role": "model"},
        }
    ]


def test_calibration_v2_discrete_values_pass_through_without_recalculation():
    probs, reals = _one_model_inputs()
    curve_list = _create_calibration_curve_list(probs, reals)
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(curve_list, _metadata(probs, reals)),
    )

    production_rows = (
        curve_list["deciles_dat"]
        .select("reference_group", "x", "y", "n_reals", "n")
        .to_dicts()
    )
    assert [
        (row["predicted"], row["observed"], row["events"], row["total"])
        for row in spec["data"]
    ] == [(row["x"], row["y"], row["n_reals"], row["n"]) for row in production_rows]
    assert {row["method"] for row in spec["data"]} == {"discrete"}


def test_calibration_v2_smooth_values_pass_through_without_refitting():
    probs, reals = _one_model_inputs()
    curve_list = _create_calibration_curve_list(probs, reals)
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(
            curve_list,
            _metadata(probs, reals),
            calibration_type="smooth",
        ),
    )

    production_rows = (
        curve_list["smooth_dat"].select("reference_group", "x", "y").to_dicts()
    )
    assert [(row["predicted"], row["observed"]) for row in spec["data"]] == [
        (row["x"], row["y"]) for row in production_rows
    ]
    assert {row["method"] for row in spec["data"]} == {"smooth"}
    assert all("events" not in row and "total" not in row for row in spec["data"])


def test_calibration_v2_distribution_values_and_series_ownership_pass_through():
    probs, reals = _one_model_inputs()
    curve_list = _create_calibration_curve_list(probs, reals)
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(curve_list, _metadata(probs, reals)),
    )

    production_rows = (
        curve_list["histogram_for_calibration"]
        .select("reference_group", "mids", "counts")
        .to_dicts()
    )
    assert [(row["midpoint"], row["count"]) for row in spec["distribution"]] == [
        (row["mids"], row["counts"]) for row in production_rows
    ]
    assert {row["seriesId"] for row in spec["distribution"]} == {"series-1"}
    assert {row["binWidth"] for row in spec["distribution"]} == {0.01}


def test_calibration_v2_ids_are_deterministic_and_do_not_encode_labels():
    probs = {
        "Model A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Model B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = np.array([0, 0, 1, 1])
    curve_list = _create_calibration_curve_list(probs, reals)

    first = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(curve_list, _metadata(probs, reals)),
    )
    second = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(curve_list, _metadata(probs, reals)),
    )

    assert [item["id"] for item in first["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [item["id"] for item in first["series"]] == ["series-1", "series-2"]
    assert first["evaluations"] == second["evaluations"]
    assert first["series"] == second["series"]
    assert all("Model" not in item["id"] for item in first["evaluations"])
    assert all("Model" not in item["id"] for item in first["series"])


def test_calibration_v2_multiple_models_share_semantic_population():
    probs = {
        "Model A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Model B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = np.array([0, 0, 1, 1])
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(
            _create_calibration_curve_list(probs, reals), _metadata(probs, reals)
        ),
    )

    assert [item["model"] for item in spec["evaluations"]] == ["Model A", "Model B"]
    assert {item["population"] for item in spec["evaluations"]} == {_SHARED_POPULATION}
    assert {item["display"]["role"] for item in spec["series"]} == {"model"}
    assert {row["seriesId"] for row in spec["data"]} == {"series-1", "series-2"}
    assert {row["seriesId"] for row in spec["distribution"]} == {
        "series-1",
        "series-2",
    }


def test_calibration_v2_multiple_populations_keep_model_unknown():
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([0, 0, 1, 1]),
        "Population B": np.array([0, 1, 0, 1]),
    }
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(
            _create_calibration_curve_list(probs, reals), _metadata(probs, reals)
        ),
    )

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "Population A"},
        {"id": "evaluation-2", "population": "Population B"},
    ]
    assert [item["display"] for item in spec["series"]] == [
        {"label": "Population A", "group": "Population A", "role": "population"},
        {"label": "Population B", "group": "Population B", "role": "population"},
    ]


def test_calibration_v2_equal_valued_populations_are_not_collapsed():
    values = np.array([0.05, 0.2, 0.7, 0.95])
    outcomes = np.array([0, 0, 1, 1])
    probs = {"Population A": values.copy(), "Population B": values.copy()}
    reals = {"Population A": outcomes.copy(), "Population B": outcomes.copy()}
    spec = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(
            _create_calibration_curve_list(probs, reals), _metadata(probs, reals)
        ),
    )

    assert [item["population"] for item in spec["evaluations"]] == [
        "Population A",
        "Population B",
    ]
    assert [item["id"] for item in spec["series"]] == ["series-1", "series-2"]
    assert {row["seriesId"] for row in spec["data"]} == {"series-1", "series-2"}
    assert {row["seriesId"] for row in spec["distribution"]} == {
        "series-1",
        "series-2",
    }


def test_calibration_v2_complete_spec_embeds_unchanged_in_report():
    probs, reals = _one_model_inputs()
    calibration = _calibration_v2_spec_from_curve_list(
        _create_calibration_curve_list(probs, reals), _metadata(probs, reals)
    )
    report = _report_spec_from_components([{"spec": calibration}])

    assert report["components"][0]["id"] == "calibration"
    assert report["components"][0]["spec"] is calibration
    assert report["components"][0]["spec"] == calibration


def test_real_performance_table_roc_calibration_report_uses_shared_renderer(tmp_path):
    probs, reals = _one_model_inputs()
    metadata = _metadata(probs, reals)
    performance_data = prepare_performance_data(probs, reals, by=0.25)

    table = _performance_table_spec_from_performance_data(performance_data, metadata)
    roc = _roc_v2_spec_from_performance_data(performance_data, metadata)
    calibration = _calibration_v2_spec_from_curve_list(
        _create_calibration_curve_list(probs, reals), metadata
    )
    report = _report_spec_from_components(
        [
            {"title": "Performance", "spec": table},
            {"title": "ROC", "spec": roc},
            {"title": "Calibration", "spec": calibration},
        ]
    )

    output = RtichokeBrowserReport(cast(dict[str, Any], report)).write_html(
        tmp_path / "report.html"
    )
    html = output.read_text(encoding="utf-8")
    viz_js = (tmp_path / "rtichoke-viz.js").read_text(encoding="utf-8")

    assert [component["id"] for component in report["components"]] == [
        "performance-table",
        "roc",
        "calibration",
    ]
    assert report["components"][2]["spec"] is calibration
    assert _embedded_report(html) == report
    assert 'import { renderReport } from "./rtichoke-viz.js";' not in html
    assert viz_js in html
    assert "append(renderReport(spec))" in html


def test_duplicate_evaluation_ids_remain_component_local_in_real_report():
    probs, reals = _one_model_inputs()
    metadata = _metadata(probs, reals)
    performance_data = prepare_performance_data(probs, reals, by=0.25)
    table = _performance_table_spec_from_performance_data(performance_data, metadata)
    roc = cast(
        dict[str, Any], _roc_v2_spec_from_performance_data(performance_data, metadata)
    )
    calibration = cast(
        dict[str, Any],
        _calibration_v2_spec_from_curve_list(
            _create_calibration_curve_list(probs, reals), metadata
        ),
    )
    report = _report_spec_from_components(
        [{"spec": table}, {"spec": roc}, {"spec": calibration}]
    )

    assert table["evaluations"][0]["id"] == "evaluation-1"
    assert roc["evaluations"][0]["id"] == "evaluation-1"
    assert calibration["evaluations"][0]["id"] == "evaluation-1"
    assert "evaluations" not in report
    assert report["components"][0]["spec"] is table
    assert report["components"][1]["spec"] is roc
    assert report["components"][2]["spec"] is calibration


def test_existing_v1_calibration_adapter_and_public_plotly_behavior_are_unchanged():
    probs, reals = _one_model_inputs()
    curve_list = _create_calibration_curve_list(probs, reals)
    old_spec = cast(dict[str, Any], _calibration_spec_from_curve_list(curve_list))

    assert old_spec["schemaVersion"] == "1.0"
    assert old_spec["type"] == "calibration"
    assert {row["model"] for row in old_spec["data"]} == {"Model A"}
    assert isinstance(create_calibration_curve(probs, reals), Figure)

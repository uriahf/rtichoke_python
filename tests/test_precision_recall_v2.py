import inspect
import shutil
import subprocess
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from rtichoke import (
    create_precision_recall_curve,
    create_precision_recall_curve_times,
    plot_precision_recall_curve,
    prepare_performance_data,
)
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._viz_spec_v2 import _precision_recall_v2_spec_from_performance_data
from rtichoke.processing.evaluation_semantics import _build_evaluation_metadata


PROBS_A = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95, 0.25, 0.85])
PROBS_B = np.array([0.10, 0.20, 0.40, 0.60, 0.70, 0.90, 0.30, 0.80])
REALS_EQUAL = np.array([0, 0, 0, 0, 1, 1, 1, 1])
REALS_LOW = np.array([0, 0, 0, 0, 0, 0, 1, 1])
REALS_HIGH = np.array([0, 0, 1, 1, 1, 1, 1, 1])


def _spec(probs, reals, *, by=0.25):
    performance_data = prepare_performance_data(probs=probs, reals=reals, by=by)
    metadata = _build_evaluation_metadata(probs, reals, np.array([]))
    return (
        performance_data,
        _precision_recall_v2_spec_from_performance_data(performance_data, metadata),
    )


def test_one_model_spec_is_pure_pass_through_with_deterministic_ids():
    performance_data, spec = _spec({"Model A": PROBS_A}, REALS_EQUAL)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "precision_recall"
    assert spec["x"] == "sensitivity"
    assert spec["y"] == "ppv"
    assert spec["evaluations"] == [
        {
            "id": "evaluation-1",
            "model": "Model A",
            "population": "__shared_population__",
        }
    ]
    assert spec["series"] == [
        {
            "id": "series-1",
            "evaluationId": "evaluation-1",
            "display": {
                "label": "Model A",
                "group": "Model A",
                "role": "model",
            },
        }
    ]

    source_rows = performance_data.select(
        "chosen_cutoff", "ppcr", "sensitivity", "ppv"
    ).to_dicts()
    assert any(np.isnan(row["ppv"]) for row in source_rows)
    expected_rows = [
        row
        for row in source_rows
        if np.isfinite(row["chosen_cutoff"])
        and np.isfinite(row["sensitivity"])
        and np.isfinite(row["ppv"])
    ]
    assert spec["data"] == [
        {
            "seriesId": "series-1",
            "cutoff": row["chosen_cutoff"],
            "ppcr": row["ppcr"],
            "sensitivity": row["sensitivity"],
            "ppv": row["ppv"],
        }
        for row in expected_rows
    ]
    assert spec["references"] == [
        {
            "type": "horizontal",
            "scope": "population",
            "population": "__shared_population__",
            "value": 0.5,
            "label": "Prevalence",
        }
    ]


def test_multiple_models_share_one_population_and_one_prevalence_reference():
    _, spec = _spec(
        {"Model A": PROBS_A, "Model B": PROBS_B},
        REALS_EQUAL,
    )

    assert [evaluation["id"] for evaluation in spec["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [evaluation["model"] for evaluation in spec["evaluations"]] == [
        "Model A",
        "Model B",
    ]
    assert {evaluation["population"] for evaluation in spec["evaluations"]} == {
        "__shared_population__"
    }
    assert [series["id"] for series in spec["series"]] == ["series-1", "series-2"]
    assert [series["display"] for series in spec["series"]] == [
        {"label": "Model A", "group": "Model A", "role": "model"},
        {"label": "Model B", "group": "Model B", "role": "model"},
    ]
    assert spec["references"] == [
        {
            "type": "horizontal",
            "scope": "population",
            "population": "__shared_population__",
            "value": 0.5,
            "label": "Prevalence",
        }
    ]


def test_multiple_populations_keep_model_unknown_and_own_references():
    _, spec = _spec(
        {"Population low": PROBS_A, "Population high": PROBS_B},
        {"Population low": REALS_LOW, "Population high": REALS_HIGH},
    )

    assert spec["evaluations"] == [
        {"id": "evaluation-1", "population": "Population low"},
        {"id": "evaluation-2", "population": "Population high"},
    ]
    assert [series["display"] for series in spec["series"]] == [
        {"label": "Population low", "group": "Population low", "role": "population"},
        {
            "label": "Population high",
            "group": "Population high",
            "role": "population",
        },
    ]
    assert spec["references"] == [
        {
            "type": "horizontal",
            "scope": "population",
            "population": "Population low",
            "value": 0.25,
            "label": "Prevalence",
        },
        {
            "type": "horizontal",
            "scope": "population",
            "population": "Population high",
            "value": 0.75,
            "label": "Prevalence",
        },
    ]


def test_equal_prevalence_populations_remain_distinct_reference_owners():
    _, spec = _spec(
        {"Population A": PROBS_A, "Population B": PROBS_B},
        {"Population A": REALS_EQUAL, "Population B": REALS_EQUAL.copy()},
    )

    assert len(spec["references"]) == 2
    assert [reference["population"] for reference in spec["references"]] == [
        "Population A",
        "Population B",
    ]
    assert [reference["value"] for reference in spec["references"]] == [0.5, 0.5]


def test_ordinal_ids_do_not_depend_on_labels():
    _, first = _spec({"Alpha": PROBS_A, "Beta": PROBS_B}, REALS_EQUAL)
    _, second = _spec({"Renamed 1": PROBS_A, "Renamed 2": PROBS_B}, REALS_EQUAL)

    assert [evaluation["id"] for evaluation in first["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [evaluation["id"] for evaluation in second["evaluations"]] == [
        "evaluation-1",
        "evaluation-2",
    ]
    assert [series["id"] for series in first["series"]] == ["series-1", "series-2"]
    assert [series["id"] for series in second["series"]] == ["series-1", "series-2"]


def test_public_browser_renderer_returns_existing_browser_chart_and_dispatch(
    tmp_path: Path,
):
    chart = create_precision_recall_curve(
        {"Model A": PROBS_A},
        REALS_EQUAL,
        by=0.25,
        renderer="browser",
    )

    assert isinstance(chart, RtichokeBrowserChart)
    assert chart.spec["type"] == "precision_recall"
    output = chart.write_html(tmp_path / "precision-recall.html")
    html = output.read_text(encoding="utf-8")
    assert 'import { renderPrecisionRecallV2 } from "./rtichoke-viz.js"' in html
    assert '"type":"precision_recall"' in html
    assert (tmp_path / "rtichoke-viz.js").is_file()
    assert (tmp_path / "rtichoke-viz.css").is_file()


@pytest.mark.parametrize("renderer", ["browser", "rtichoke_viz"])
def test_browser_aliases_use_same_static_canonical_contract(renderer: str):
    chart = create_precision_recall_curve(
        {"Model A": PROBS_A},
        REALS_EQUAL,
        by=0.25,
        renderer=renderer,
    )
    assert isinstance(chart, RtichokeBrowserChart)
    assert chart.spec["schemaVersion"] == "2.0"
    assert chart.spec["type"] == "precision_recall"


def test_precomputed_browser_api_uses_population_semantics_when_model_is_unknown():
    performance_data = prepare_performance_data(
        probs={"Group A": PROBS_A, "Group B": PROBS_B},
        reals=REALS_EQUAL,
        by=0.25,
    )

    chart = plot_precision_recall_curve(performance_data, renderer="browser")

    assert isinstance(chart, RtichokeBrowserChart)
    assert chart.spec["evaluations"] == [
        {"id": "evaluation-1", "population": "Group A"},
        {"id": "evaluation-2", "population": "Group B"},
    ]
    assert [series["display"]["role"] for series in chart.spec["series"]] == [
        "population",
        "population",
    ]


def test_default_and_explicit_plotly_behavior_are_unchanged():
    probs = {"Model A": PROBS_A, "Model B": PROBS_B}

    default = create_precision_recall_curve(probs, REALS_EQUAL, by=0.25)
    explicit = create_precision_recall_curve(
        probs,
        REALS_EQUAL,
        by=0.25,
        renderer="plotly",
    )

    assert isinstance(default, go.Figure)
    assert pio.to_json(default) == pio.to_json(explicit)

    performance_data = prepare_performance_data(probs=probs, reals=REALS_EQUAL, by=0.25)
    default_plot = plot_precision_recall_curve(performance_data)
    explicit_plot = plot_precision_recall_curve(performance_data, renderer="plotly")
    assert pio.to_json(default_plot) == pio.to_json(explicit_plot)


def test_time_dependent_precision_recall_api_has_renderer_selection():
    parameters = inspect.signature(create_precision_recall_curve_times).parameters
    assert "renderer" in parameters
    assert parameters["renderer"].default == "plotly"


def test_vendored_v050_contains_static_precision_recall_contract_and_export():
    vendor = Path(__file__).parents[1] / "src" / "rtichoke" / "_vendor" / "rtichoke_viz"
    bundle = (vendor / "rtichoke-viz.js").read_text(encoding="utf-8")
    schema = (vendor / "rtichoke-viz-v2.schema.json").read_text(encoding="utf-8")

    assert "PrecisionRecallV2SpecSchema" in bundle
    assert "renderPrecisionRecallV2" in bundle
    assert '"const": "precision_recall"' in schema
    assert '"const": "population"' in schema
    assert '"const": "horizontal"' in schema


def _chrome_executable() -> str:
    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        executable = shutil.which(candidate)
        if executable is not None:
            return executable
    pytest.skip("headless Chrome/Chromium is not available")


@contextmanager
def _serve(directory: Path) -> Iterator[str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_public_browser_chart_executes_to_svg_when_chrome_is_available(tmp_path: Path):
    chart = create_precision_recall_curve(
        {"Model A": PROBS_A},
        REALS_EQUAL,
        by=0.25,
        renderer="browser",
    )
    chart.write_html(tmp_path / "precision-recall.html")

    with _serve(tmp_path) as base_url:
        result = subprocess.run(
            [
                _chrome_executable(),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--enable-logging=stderr",
                "--log-level=0",
                "--dump-dom",
                f"{base_url}/precision-recall.html",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stderr
    assert "INFO:CONSOLE" not in result.stderr, result.stderr
    assert "<svg" in result.stdout
    assert "rtichoke-viz-chart" in result.stdout

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
import pytest

from rtichoke import create_roc_curve_times
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._viz_spec_v2 import (
    _roc_times_v2_spec_from_performance_data,
)
from rtichoke.performance_data.performance_data_times import (
    prepare_performance_data_times,
)
from rtichoke.processing.evaluation_semantics import (
    _SHARED_POPULATION,
    _build_evaluation_metadata,
)

HORIZONS = [5.0, 10.0]
HEURISTICS = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]


def _shared_inputs():
    return (
        {
            "Model A": np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95]),
            "Model B": np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.90]),
        },
        np.array([1, 0, 1, 0, 1, 0]),
        np.array([3.0, 12.0, 8.0, 13.0, 14.0, 15.0]),
    )


def _spec(probs, reals, times, operating_point_dimension="probability_threshold"):
    performance = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )
    return _roc_times_v2_spec_from_performance_data(
        performance,
        _build_evaluation_metadata(probs, reals, times),
        operating_point_dimension=operating_point_dimension,
    )


def test_time_roc_uses_one_evaluation_and_series_per_model_horizon():
    probs, reals, times = _shared_inputs()
    spec = _spec(probs, reals, times)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "roc"
    assert spec["x"] == "false_positive_rate"
    assert spec["y"] == "sensitivity"
    assert spec["xAxis"] == {"label": "False Positive Rate", "domain": [0, 1]}
    assert spec["yAxis"] == {"label": "Sensitivity", "domain": [0, 1]}
    assert len(spec["evaluations"]) == 2
    assert {evaluation["population"] for evaluation in spec["evaluations"]} == {
        _SHARED_POPULATION
    }
    assert len(spec["series"]) == 4
    assert {
        (series["display"]["group"], series["horizon"]) for series in spec["series"]
    } == {
        ("Model A", 5.0),
        ("Model A", 10.0),
        ("Model B", 5.0),
        ("Model B", 10.0),
    }

    # Reference line
    assert spec["references"] == [{"type": "identity", "scope": "global"}]

    # Check data payload fields
    first_datum = spec["data"][0]
    assert "cutoff" in first_datum
    assert "sensitivity" in first_datum
    assert "specificity" in first_datum
    assert "false_positive_rate" in first_datum
    assert "ppcr" in first_datum
    assert first_datum["false_positive_rate"] == 1.0 - first_datum["specificity"]


def test_time_roc_operating_point_metadata():
    probs, reals, times = _shared_inputs()

    spec_thresh = _spec(
        probs, reals, times, operating_point_dimension="probability_threshold"
    )
    assert spec_thresh["operatingPoint"] == {"dimension": "probability_threshold"}

    spec_ppcr = _spec(probs, reals, times, operating_point_dimension="ppcr")
    assert spec_ppcr["operatingPoint"] == {"dimension": "ppcr"}


def test_time_roc_renderers_preserve_plotly_default_and_browser(tmp_path: Path):
    probs, reals, times = _shared_inputs()
    default = create_roc_curve_times(
        probs, reals, times, HORIZONS, heuristics_sets=HEURISTICS, by=0.25
    )
    assert isinstance(default, go.Figure)

    browser = create_roc_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="browser",
    )
    assert isinstance(browser, RtichokeBrowserChart)
    html = browser.write_html(tmp_path / "roc-times.html").read_text(encoding="utf-8")
    assert "renderRocV2" in html
    assert {series["horizon"] for series in browser.spec["series"]} == set(HORIZONS)

    rtichoke_viz_browser = create_roc_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="rtichoke_viz",
    )
    assert isinstance(rtichoke_viz_browser, RtichokeBrowserChart)

    with pytest.raises(ValueError, match="ROC supports"):
        create_roc_curve_times(
            probs,
            reals,
            times,
            HORIZONS,
            heuristics_sets=HEURISTICS,
            by=0.25,
            renderer="matplotlib",
        )


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
    raise RuntimeError("unreachable")


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


def test_time_roc_browser_chart_executes_and_switches_horizons_in_chrome(
    tmp_path: Path,
):
    probs, reals, times = _shared_inputs()
    chart = create_roc_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="browser",
    )
    chart.write_html(tmp_path / "roc-times.html")

    script_path = tmp_path / "test_switch.js"
    script_path.write_text(
        """
        import { renderRocV2 } from "./rtichoke-viz.js";
        const spec = JSON.parse(document.querySelector("#rtichoke-spec").textContent);
        const chart = renderRocV2(spec, { width: 600, height: 600 });
        document.querySelector("#rtichoke-chart").replaceChildren(chart);

        const select = chart.querySelector("select");
        if (!select) {
            console.error("Horizon selector missing!");
        } else {
            console.log("INITIAL_HORIZON:" + select.value);
            select.value = "10";
            select.dispatchEvent(new Event("change", { bubbles: true }));
            console.log("SWITCHED_HORIZON:" + select.value);
        }
        """,
        encoding="utf-8",
    )

    html_file = tmp_path / "roc-times.html"
    content = html_file.read_text(encoding="utf-8")
    content = content.replace(
        "</head>",
        '<script type="module" src="./test_switch.js"></script></head>',
    )
    html_file.write_text(content, encoding="utf-8")

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
                f"{base_url}/roc-times.html",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stderr
    assert "INITIAL_HORIZON:5" in result.stderr, result.stderr
    assert "SWITCHED_HORIZON:10" in result.stderr, result.stderr
    assert "<svg" in result.stdout
    assert "rtichoke-viz-chart" in result.stdout

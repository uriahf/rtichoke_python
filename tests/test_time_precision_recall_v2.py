import shutil
import subprocess
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterator, cast

import numpy as np
import plotly.graph_objects as go
import pytest

from rtichoke import create_precision_recall_curve_times
from rtichoke._renderers import RtichokeBrowserChart
from rtichoke._viz_spec_v2 import (
    _precision_recall_times_v2_spec_from_performance_data,
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


def _spec(probs, reals, times):
    performance = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )
    return _precision_recall_times_v2_spec_from_performance_data(
        performance, _build_evaluation_metadata(probs, reals, times)
    )


def test_time_precision_recall_uses_one_evaluation_and_series_per_model_horizon():
    probs, reals, times = _shared_inputs()
    spec = _spec(probs, reals, times)

    assert spec["schemaVersion"] == "2.0"
    assert spec["type"] == "precision_recall"
    assert spec["x"] == "sensitivity"
    assert spec["y"] == "ppv"
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

    # Prevalence references (scope = population_horizon)
    references = spec["references"]
    assert len(references) == 2
    assert {
        (reference["population"], reference["horizon"]) for reference in references
    } == {
        (_SHARED_POPULATION, 5.0),
        (_SHARED_POPULATION, 10.0),
    }


def test_equal_risk_population_horizons_remain_distinct_pr_reference_owners():
    probs = {
        "Population A": np.array([0.05, 0.2, 0.7, 0.95]),
        "Population B": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    reals = {
        "Population A": np.array([1, 0, 0, 0]),
        "Population B": np.array([1, 0, 0, 0]),
    }
    times = {
        "Population A": np.array([3.0, 12.0, 13.0, 14.0]),
        "Population B": np.array([3.0, 12.0, 13.0, 14.0]),
    }
    references = _spec(probs, reals, times)["references"]

    assert len(references) == 4
    by_owner = {
        (reference["population"], reference["horizon"]): reference["value"]
        for reference in references
    }
    assert len(by_owner) == 4
    assert by_owner[("Population A", 5.0)] == by_owner[("Population B", 5.0)]


def test_time_precision_recall_censoring_and_competing_risk_reference_comes_from_performance_layer():
    probs = {"Model A": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])}
    reals = np.array([1, 0, 2, 1, 0, 2])
    times = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 14.0])
    performance = prepare_performance_data_times(
        probs,
        reals,
        times,
        fixed_time_horizons=HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
    )
    spec = _precision_recall_times_v2_spec_from_performance_data(
        performance, _build_evaluation_metadata(probs, reals, times)
    )
    references = cast(list[dict[str, Any]], spec["references"])
    calculated_risks = {
        float(row["fixed_time_horizon"]): float(row["real_positives"] / row["n"])
        for row in performance.filter(performance["chosen_cutoff"] == 0)
        .select("fixed_time_horizon", "real_positives", "n")
        .unique()
        .to_dicts()
    }

    for reference in references:
        horizon = reference["horizon"]
        risk = calculated_risks[horizon]
        assert reference["value"] == risk
        assert reference["scope"] == "population_horizon"
        assert reference["type"] == "horizontal"


def test_time_precision_recall_renderers_preserve_plotly_default_and_browser(
    tmp_path: Path,
):
    probs, reals, times = _shared_inputs()
    default = create_precision_recall_curve_times(
        probs, reals, times, HORIZONS, heuristics_sets=HEURISTICS, by=0.25
    )
    assert isinstance(default, go.Figure)

    browser = create_precision_recall_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="browser",
    )
    assert isinstance(browser, RtichokeBrowserChart)
    html = browser.write_html(tmp_path / "pr-times.html").read_text(encoding="utf-8")
    assert "renderPrecisionRecallV2" in html
    assert {series["horizon"] for series in browser.spec["series"]} == set(HORIZONS)

    rtichoke_viz_browser = create_precision_recall_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="rtichoke_viz",
    )
    assert isinstance(rtichoke_viz_browser, RtichokeBrowserChart)

    with pytest.raises(ValueError, match="Precision-Recall supports"):
        create_precision_recall_curve_times(
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


def test_time_precision_recall_browser_chart_executes_and_switches_horizons_in_chrome(
    tmp_path: Path,
):
    probs, reals, times = _shared_inputs()
    chart = create_precision_recall_curve_times(
        probs,
        reals,
        times,
        HORIZONS,
        heuristics_sets=HEURISTICS,
        by=0.25,
        renderer="browser",
    )
    chart.write_html(tmp_path / "precision-recall-times.html")

    script_path = tmp_path / "test_switch.js"
    script_path.write_text(
        """
        import { renderPrecisionRecallV2 } from "./rtichoke-viz.js";
        const spec = JSON.parse(document.querySelector("#rtichoke-spec").textContent);
        const chart = renderPrecisionRecallV2(spec, { width: 600, height: 600 });
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

    html_file = tmp_path / "precision-recall-times.html"
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
                f"{base_url}/precision-recall-times.html",
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

"""Chromium acceptance for the installed continuous renderer and step histogram."""

from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import polars as pl
import pytest

from rtichoke import create_reals_distribution_steps


@contextmanager
def _serve(directory: Path):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(SimpleHTTPRequestHandler, directory=str(directory)),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_continuous_heatmap_and_selected_histogram(tmp_path: Path):
    playwright = pytest.importorskip("playwright.sync_api")
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3],
            "step": [0, 1, 3, 0, 1, 3, 0, 3],
            "reals": pl.Series([10, 20, 30, 15, None, 35, 20, 25], dtype=pl.Float64),
        }
    )
    chart = create_reals_distribution_steps(
        df, reals_scale="continuous", bins=[10, 20, 30, 40]
    )
    html = chart.write_html(tmp_path / "continuous.html")
    assert html.exists()

    with _serve(tmp_path) as url, playwright.sync_playwright() as manager:
        browser = manager.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(f"{url}/continuous.html")
        root = page.locator(".rtichoke-continuous-outcome-distribution")
        root.wait_for()
        heatmap = root.locator(".rtichoke-continuous-heatmap-wrapper svg")
        histogram = root.locator(".rtichoke-continuous-histogram-wrapper svg")
        assert heatmap.count() >= 1
        assert histogram.count() == 0  # collapsed by default
        heatmap.first.focus()
        heatmap.first.press("Enter")  # supported keyboard action opens histogram
        histogram.first.wait_for()
        assert histogram.count() >= 1
        slider = root.locator("input[type='range']")
        assert slider.count() == 1
        assert slider.get_attribute("aria-valuetext") == "0"
        page.evaluate(
            """() => {
              const slider = document.querySelector('.rtichoke-continuous-outcome-distribution input[type="range"]');
              slider.value = "2";
              slider.dispatchEvent(new Event("input", { bubbles: true }));
              slider.dispatchEvent(new Event("change", { bubbles: true }));
            }"""
        )
        assert slider.get_attribute("aria-valuetext") == "3"
        assert histogram.count() >= 1
        assert len(errors) == 0, errors
        browser.close()

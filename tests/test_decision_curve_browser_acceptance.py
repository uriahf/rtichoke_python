from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import polars as pl
import pytest

from rtichoke.utility.decision import (
    create_decision_curve,
    create_decision_curve_times,
    plot_decision_curve,
)


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


def test_static_decision_curve_renders_model_and_references_in_real_browser(tmp_path):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("playwright is not available")

    probs = {
        "Model A": np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]),
        "Model B": np.array([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]),
    }
    reals = np.array([0, 0, 0, 1, 0, 1, 1])
    chart = create_decision_curve(
        probs,
        reals,
        by=0.1,
        min_p_threshold=0.1,
        max_p_threshold=0.8,
        renderer="browser",
    )
    chart.write_html(tmp_path / "decision.html")

    with _serve(tmp_path) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors: list[str] = []
            page.on(
                "console",
                lambda msg: (
                    errors.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None
                ),
            )
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(f"{base_url}/decision.html")
            page.wait_for_selector("svg")

            content = page.content()
            assert "Model A" in content
            assert "Model B" in content
            assert "Treat None" in content
            assert "Treat All" in content
            assert page.locator("svg").count() >= 1
            assert len(errors) == 0, f"Console errors found: {errors}"
            browser.close()


def test_time_dependent_decision_curve_renders_in_real_browser(tmp_path: Path):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("playwright is not available")

    probs = {
        "Model A": np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]),
        "Model B": np.array([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]),
    }
    reals = np.array([0, 0, 0, 1, 0, 1, 1])
    times = np.array([1.0, 3.0, 5.0, 2.0, 8.0, 4.0, 10.0])

    chart = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        by=0.1,
        min_p_threshold=0.1,
        max_p_threshold=0.8,
        renderer="browser",
    )
    chart.write_html(tmp_path / "time_decision.html")

    with _serve(tmp_path) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors: list[str] = []
            page.on(
                "console",
                lambda msg: (
                    errors.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None
                ),
            )
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(f"{base_url}/time_decision.html")
            page.wait_for_selector("svg")

            content = page.content()
            assert "Model A" in content
            assert "Model B" in content
            assert "Treat None" in content
            assert "Treat All" in content
            assert page.locator("svg").count() >= 1
            assert len(errors) == 0, f"Console errors found: {errors}"
            browser.close()


def test_static_interventions_avoided_renders_geometry_references_and_axes_in_real_browser(
    tmp_path: Path,
):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("playwright is not available")

    performance_data = pl.DataFrame(
        {
            "reference_group": [
                "Population A",
                "Population A",
                "Population B",
                "Population B",
            ],
            "chosen_cutoff": [0.2, 0.5, 0.2, 0.5],
            "net_benefit_interventions_avoided": [-25.0, 50.0, -100.0, 0.0],
            "real_positives": [2, 2, 4, 4],
            "n": [8, 8, 8, 8],
        }
    )
    chart = plot_decision_curve(
        performance_data,
        decision_type="interventions avoided",
        min_p_threshold=0.2,
        max_p_threshold=0.5,
        renderer="browser",
    )
    treat_none = [
        reference
        for reference in chart.spec["references"]
        if reference["benchmark"] == "treat_none"
    ]
    assert [reference["population"] for reference in treat_none] == [
        "Population A",
        "Population B",
    ]
    chart.write_html(tmp_path / "interventions-avoided.html")

    with _serve(tmp_path) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors: list[str] = []
            page.on(
                "console",
                lambda msg: (
                    errors.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None
                ),
            )
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(f"{base_url}/interventions-avoided.html")
            page.wait_for_selector("svg")

            content = page.content()
            assert "Population A" in content
            assert "Population B" in content
            assert "Treat All" in content
            assert "Treat None" in content
            assert "Interventions Avoided (per 100)" in content
            assert "Probability Threshold" in content
            assert page.locator("svg").count() >= 1
            assert len(errors) == 0, f"Console errors found: {errors}"
            browser.close()


def test_time_interventions_avoided_horizon_switch_replaces_geometry_in_real_browser(
    tmp_path: Path,
):
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
    except ImportError:
        pytest.skip("playwright is not available")  # ty: ignore[too-many-positional-arguments]

    probs = {
        "Model A": np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]),
        "Model B": np.array([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]),
    }
    reals = np.array([0, 0, 0, 1, 0, 1, 1])
    times = np.array([1.0, 3.0, 5.0, 2.0, 8.0, 4.0, 10.0])
    chart = create_decision_curve_times(
        probs,
        reals,
        times,
        fixed_time_horizons=[5.0, 10.0],
        decision_type="interventions avoided",
        by=0.1,
        min_p_threshold=0.1,
        max_p_threshold=0.8,
        renderer="browser",
    )
    chart.write_html(tmp_path / "time-interventions-avoided.html")

    with _serve(tmp_path) as base_url, sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{base_url}/time-interventions-avoided.html")
        selector = page.locator("select[aria-label='Fixed Time Horizon']")
        selector.wait_for()
        page.wait_for_selector("svg")

        assert selector.locator("option").all_text_contents() == ["5", "10"]
        initial_paths = page.locator("svg path").evaluate_all(
            "nodes => nodes.map(node => node.getAttribute('d'))"
        )
        initial_svg_count = page.locator("svg").count()
        assert "Treat All" in page.content()
        assert "Treat None" in page.content()

        selector.select_option("10")
        page.wait_for_function(
            "document.querySelector(\"select[aria-label='Fixed Time Horizon']\").value === '10'"
        )
        switched_paths = page.locator("svg path").evaluate_all(
            "nodes => nodes.map(node => node.getAttribute('d'))"
        )

        assert switched_paths != initial_paths
        assert page.locator("svg").count() == initial_svg_count
        assert "Treat All" in page.content()
        assert "Treat None" in page.content()
        assert not errors
        browser.close()

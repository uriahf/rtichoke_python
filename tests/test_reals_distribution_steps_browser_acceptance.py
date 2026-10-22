from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import polars as pl
import pytest
from rtichoke.reals_distribution import create_reals_distribution_steps


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


def test_reals_distribution_steps_browser_acceptance(tmp_path: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not available")

    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "step": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "reals": [
                "None",
                "Mild",
                "Moderate",
                "None",
                "Mild",
                "Mild",
                "None",
                "None",
                "Mild",
            ],
        }
    )

    # 1. Raw / Excluded outcome distribution chart
    chart_excl = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild", "Moderate"],
        reals_scale="ordinal",
        censoring_heuristic="excluded",
    )
    html_file_excl = tmp_path / "reals_dist_excl.html"
    chart_excl.write_html(html_file_excl)

    # 2. AJ / Adjusted outcome distribution chart
    chart_adj = create_reals_distribution_steps(
        df,
        reals_order=["None", "Mild", "Moderate"],
        reals_scale="ordinal",
        censoring_heuristic="adjusted",
    )
    html_file_adj = tmp_path / "reals_dist_adj.html"
    chart_adj.write_html(html_file_adj)

    # Verify rendering over local HTTP server
    with _serve(tmp_path) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            # Test Excluded Mode HTML
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
            page.goto(f"{base_url}/reals_dist_excl.html")
            page.wait_for_selector("svg")

            assert page.locator("svg").count() >= 1
            assert len(errors) == 0, f"Console errors in excluded mode: {errors}"
            page.close()

            # Test Adjusted Mode HTML
            page_adj = browser.new_page()
            errors_adj: list[str] = []
            page_adj.on(
                "console",
                lambda msg: (
                    errors_adj.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None
                ),
            )
            page_adj.on("pageerror", lambda err: errors_adj.append(str(err)))
            page_adj.goto(f"{base_url}/reals_dist_adj.html")
            page_adj.wait_for_selector("svg")

            assert page_adj.locator("svg").count() >= 1
            assert len(errors_adj) == 0, (
                f"Console errors in adjusted mode: {errors_adj}"
            )
            page_adj.close()

            browser.close()

    # Verify direct file:// loading
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        page = browser.new_page()
        errors = []
        page.on(
            "console",
            lambda msg: (
                errors.append(msg.text) if msg.type in ["error", "warning"] else None
            ),
        )
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(html_file_excl.as_uri())
        page.wait_for_selector("svg")

        assert page.locator("svg").count() >= 1
        assert len(errors) == 0, f"Console errors found over file://: {errors}"
        browser.close()


def test_reals_distribution_nominal_browser_acceptance(tmp_path: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not available")

    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [0, 1, 0, 1],
            "reals": ["Red", "Blue", "Green", "Red"],
        }
    )

    chart = create_reals_distribution_steps(
        df,
        reals_order=["Red", "Green", "Blue"],
        reals_scale="nominal",
        censoring_heuristic="excluded",
    )
    html_file = tmp_path / "reals_dist_nominal.html"
    chart.write_html(html_file)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        page = browser.new_page()
        errors: list[str] = []
        page.on(
            "console",
            lambda msg: (
                errors.append(msg.text) if msg.type in ["error", "warning"] else None
            ),
        )
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(html_file.as_uri())
        page.wait_for_selector("svg")

        assert page.locator("svg").count() >= 1
        assert len(errors) == 0, f"Console errors in nominal mode: {errors}"
        browser.close()

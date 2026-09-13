from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import pytest

import rtichoke


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


def test_probs_histogram_browser_acceptance_http_and_file_uri(tmp_path: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not available")

    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    chart = rtichoke.create_probs_histogram(probs=probs, reals=reals, by=0.20)
    html_file = tmp_path / "probs_histogram.html"
    chart.write_html(html_file)

    # 1. Test via HTTP server
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
            page.goto(f"{base_url}/probs_histogram.html")
            page.wait_for_selector("svg")

            content = page.content()
            assert "m1" in content
            assert page.locator("svg").count() >= 1
            assert len(errors) == 0, f"Console errors found over HTTP: {errors}"
            browser.close()

    # 2. Test direct file:// access (with --allow-file-access-from-files)
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
        page.goto(html_file.as_uri())
        page.wait_for_selector("svg")

        content = page.content()
        assert "m1" in content
        assert page.locator("svg").count() >= 1
        assert len(errors) == 0, f"Console errors found over file://: {errors}"

        # 3. Test changing operating point / controls
        op_slider = page.locator("input[type='range']")
        if op_slider.count() > 0:
            page.evaluate("""
                const el = document.querySelector("input[type='range']");
                el.value = 2;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            """)
            assert op_slider.input_value() == "2"

        browser.close()


def test_probs_histogram_ppcr_mode_browser_acceptance(tmp_path: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not available")

    probs = {"m1": np.array([0.00, 0.15, 0.30, 0.50, 0.50, 0.50, 0.65, 0.80, 1.00])}
    reals = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])

    chart = rtichoke.create_probs_histogram(
        probs=probs, reals=reals, by=0.20, stratified_by=("ppcr",)
    )
    html_file = tmp_path / "probs_histogram_ppcr.html"
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

        assert len(errors) == 0, f"Console errors found in PPCR mode: {errors}"

        # Capture initial rect heights representing rank bin bars
        initial_rects = page.locator("svg rect").evaluate_all(
            "nodes => nodes.map(n => n.getAttribute('height'))"
        )

        # Change PPCR slider position
        page.evaluate("""
            const el = document.querySelector("input[type='range']");
            if (el) {
                el.value = 3;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """)

        switched_rects = page.locator("svg rect").evaluate_all(
            "nodes => nodes.map(n => n.getAttribute('height'))"
        )
        # Verify rank distribution bars remain unchanged while PPCR operating point moves
        assert switched_rects == initial_rects
        browser.close()

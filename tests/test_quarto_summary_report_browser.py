import shutil
import subprocess
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import pytest

from rtichoke.summary_report.summary_report import create_summary_report


def _quarto_executable() -> str:
    executable = shutil.which("quarto")
    if executable is None:
        pytest.skip("quarto executable is not available")
    return executable


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


def _dump_dom(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _chrome_executable(),
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--enable-logging=stderr",
            "--log-level=0",
            "--dump-dom",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_quarto_single_browser_summary_report(tmp_path):
    quarto_bin = _quarto_executable()

    probs = {
        "Model A": np.array(
            [0.03, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.62, 0.75, 0.88, 0.96]
        )
    }
    reals = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

    report_path = tmp_path / "report.html"
    create_summary_report(probs, reals, renderer="browser", output_file=report_path)

    qmd = tmp_path / "doc.qmd"
    qmd.write_text(
        """---
title: "Single Report Host"
format: html
---

<iframe src="report.html" style="width: 100%; height: 1200px; border: 0;"></iframe>
""",
        encoding="utf-8",
    )

    res = subprocess.run(
        [quarto_bin, "render", str(qmd), "--to", "html"], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr

    rendered_html = tmp_path / "doc.html"
    assert rendered_html.exists()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        with _serve(tmp_path) as base_url:
            browser = _dump_dom(f"{base_url}/doc.html")
        assert browser.returncode == 0, browser.stderr
        assert "INFO:CONSOLE" not in browser.stderr, browser.stderr
        assert '<iframe src="report.html"' in browser.stdout
    else:
        with _serve(tmp_path) as base_url:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                errors: list[str] = []
                page.on(
                    "console",
                    lambda msg: errors.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None,
                )
                page.on("pageerror", lambda err: errors.append(str(err)))

                page.goto(f"{base_url}/doc.html")
                page.wait_for_selector("iframe")

                frame_el = page.query_selector("iframe")
                assert frame_el is not None
                frame = frame_el.content_frame()
                assert frame is not None

                frame.wait_for_selector("#rtichoke-report")
                frame.wait_for_selector("table")
                frame.wait_for_selector("svg")

                assert "Performance" in frame.content()
                assert "ROC" in frame.content()
                assert "Calibration" in frame.content()

                tbl_text = frame.locator("table").first.inner_text()
                assert "Model" in tbl_text or "True Positives" in tbl_text
                assert frame.locator("svg").count() >= 2
                assert len(errors) == 0, f"Console errors found: {errors}"
                browser.close()


def test_quarto_two_browser_summary_reports(tmp_path):
    quarto_bin = _quarto_executable()

    probs1 = {"Model A": np.array([0.03, 0.12, 0.25, 0.40, 0.62, 0.75, 0.88, 0.96])}
    reals1 = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    probs2 = {
        "Model X": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "Model Y": np.array([0.2, 0.4, 0.6, 0.8, 0.95]),
    }
    reals2 = np.array([0, 0, 1, 1, 1])

    create_summary_report(
        probs1, reals1, renderer="browser", output_file=tmp_path / "report1.html"
    )
    create_summary_report(
        probs2, reals2, renderer="browser", output_file=tmp_path / "report2.html"
    )

    qmd = tmp_path / "multi_doc.qmd"
    qmd.write_text(
        """---
title: "Two Reports Host"
format: html
---

## First Report

<iframe src="report1.html" style="width: 100%; height: 1200px; border: 0;"></iframe>

## Second Report

<iframe src="report2.html" style="width: 100%; height: 1200px; border: 0;"></iframe>
""",
        encoding="utf-8",
    )

    res = subprocess.run(
        [quarto_bin, "render", str(qmd), "--to", "html"], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr

    rendered_html = tmp_path / "multi_doc.html"
    assert rendered_html.exists()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        with _serve(tmp_path) as base_url:
            browser = _dump_dom(f"{base_url}/multi_doc.html")
        assert browser.returncode == 0, browser.stderr
        assert "INFO:CONSOLE" not in browser.stderr, browser.stderr
        assert '<iframe src="report1.html"' in browser.stdout
        assert '<iframe src="report2.html"' in browser.stdout
    else:
        with _serve(tmp_path) as base_url:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                errors: list[str] = []
                page.on(
                    "console",
                    lambda msg: errors.append(msg.text)
                    if msg.type in ["error", "warning"]
                    else None,
                )
                page.on("pageerror", lambda err: errors.append(str(err)))

                page.goto(f"{base_url}/multi_doc.html")
                page.wait_for_selector("iframe")

                iframes = page.query_selector_all("iframe")
                assert len(iframes) == 2

                for iframe_el in iframes:
                    frame = iframe_el.content_frame()
                    assert frame is not None

                    frame.wait_for_selector("#rtichoke-report")
                    frame.wait_for_selector("table")
                    frame.wait_for_selector("svg")

                    assert "Performance" in frame.content()
                    assert "ROC" in frame.content()
                    assert "Calibration" in frame.content()

                    tbl_text = frame.locator("table").first.inner_text()
                    assert "Model" in tbl_text or "True Positives" in tbl_text
                    assert frame.locator("svg").count() >= 2

                assert len(errors) == 0, f"Console errors found: {errors}"
                browser.close()

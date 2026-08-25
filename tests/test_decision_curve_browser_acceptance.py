from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import pytest

from rtichoke.utility.decision import create_decision_curve


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
                lambda msg: errors.append(msg.text)
                if msg.type in ["error", "warning"]
                else None,
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

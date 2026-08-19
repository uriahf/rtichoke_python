"""Performance/decision curve renderer asset for the lightweight summary report."""
from pathlib import Path


def curve_renderer_source() -> str:
    """Return the D3 performance/decision curve renderer source."""
    return Path(__file__).with_name("curve_renderer.js").read_text(encoding="utf-8")

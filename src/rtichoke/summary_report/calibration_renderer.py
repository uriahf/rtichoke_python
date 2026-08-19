"""Renderer assets for the lightweight summary report."""
from pathlib import Path


def calibration_renderer_source() -> str:
    """Return the lightweight D3 renderer sources embedded in the report."""
    root = Path(__file__).parent
    assets = (
        root / "calibration_renderer.js",
        root / "performance_table_renderer.js",
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in assets)

"""Calibration renderer asset for the lightweight summary report."""
from pathlib import Path


def calibration_renderer_source() -> str:
    """Return the calibration-only D3 renderer source."""
    return Path(__file__).with_name("calibration_renderer.js").read_text(encoding="utf-8")

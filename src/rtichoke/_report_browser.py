"""Internal offline browser rendering for canonical ReportSpec values."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def _sanitize_nan_values(obj: Any) -> Any:
    """Recursively replace NaN and Inf float values with None for valid JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan_values(v) for v in obj]
    if isinstance(obj, float) and not (
        obj == obj and obj != float("inf") and obj != float("-inf")
    ):
        return None
    return obj


class RtichokeBrowserReport:
    """A complete canonical ReportSpec rendered by shared ``rtichoke_viz``."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def write_html(self, path: str | Path) -> Path:
        """Write an offline HTML page that delegates composition to renderReport()."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        vendor = files("rtichoke").joinpath("_vendor", "rtichoke_viz")

        sanitized_spec = _sanitize_nan_values(self.spec)
        spec_json = json.dumps(sanitized_spec, separators=(",", ":")).replace(
            "</", "<\\/"
        )
        viz_js = vendor.joinpath("rtichoke-viz.js").read_text(encoding="utf-8")
        viz_css = vendor.joinpath("rtichoke-viz.css").read_text(encoding="utf-8")
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
{viz_css}
  </style>
  <title>rtichoke report</title>
</head>
<body>
  <div id="rtichoke-report"></div>
  <script id="rtichoke-report-spec" type="application/json">{spec_json}</script>
  <script type="module">
{viz_js}
    const spec = JSON.parse(
      document.querySelector("#rtichoke-report-spec").textContent
    );
        document.querySelector("#rtichoke-report").append(renderReport(spec, {{
      sectionGroupPresentation: "tabs",
      groupPresentation: "stacked"
        }}));
  </script>
</body>
</html>
"""
        output.write_text(html, encoding="utf-8")
        return output

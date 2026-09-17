"""Internal offline browser rendering for canonical ReportSpec values."""

from __future__ import annotations

import html
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any


def _resolve_render_report_symbol(
    viz_js: str, symbol_name: str = "renderReport"
) -> str:
    """Resolve the local callable identifier exported as symbol_name from an ESM bundle."""
    export_pattern = re.compile(r"export\s*\{([^}]+)\}", re.DOTALL)
    for match in export_pattern.finditer(viz_js):
        clause = match.group(1)
        for item in clause.split(","):
            parts = item.strip().split()
            if not parts:
                continue
            if len(parts) == 3 and parts[1] == "as" and parts[2] == symbol_name:
                return parts[0]
            if len(parts) == 1 and parts[0] == symbol_name:
                return symbol_name

    if re.search(
        rf"export\s+(?:async\s+)?function\s+{symbol_name}\b|export\s+(?:const|let|var)\s+{symbol_name}\b",
        viz_js,
    ):
        return symbol_name

    raise ValueError(
        f"Could not resolve {symbol_name!r} export in provided JavaScript bundle."
    )


def _performance_metrics_cheat_sheet_html() -> str:
    """Generate Performance Metrics Cheat Sheet HTML for browser summary reports."""
    return (
        '<details class="rtichoke-cheat-sheet">\n'
        "  <summary>Performance Metrics Cheat Sheet</summary>\n"
        '  <div class="rtichoke-cheat-sheet__content">\n'
        '    <section class="rtichoke-cheat-sheet__section">\n'
        "      <h4>Confusion Matrix</h4>\n"
        '      <table class="rtichoke-cheat-sheet__table">\n'
        "        <thead>\n"
        "          <tr>\n"
        "            <th></th>\n"
        "            <th>Predicted +</th>\n"
        "            <th>Predicted -</th>\n"
        "          </tr>\n"
        "        </thead>\n"
        "        <tbody>\n"
        "          <tr>\n"
        "            <th>Real Positive</th>\n"
        "            <td>TP</td>\n"
        "            <td>FN</td>\n"
        "          </tr>\n"
        "          <tr>\n"
        "            <th>Real Negative</th>\n"
        "            <td>FP</td>\n"
        "            <td>TN</td>\n"
        "          </tr>\n"
        "        </tbody>\n"
        "      </table>\n"
        "    </section>\n"
        '    <section class="rtichoke-cheat-sheet__section">\n'
        "      <h4>Metrics &amp; Formulas</h4>\n"
        '      <dl class="rtichoke-cheat-sheet__metrics">\n'
        "        <dt>Prevalence</dt>\n"
        "        <dd><code>(TP + FN) / (TP + FP + TN + FN)</code></dd>\n"
        "        <dt>PPCR</dt>\n"
        "        <dd><code>(TP + FP) / (TP + FP + TN + FN)</code></dd>\n"
        "        <dt>Sensitivity / Recall / TPR</dt>\n"
        "        <dd>\n"
        "          <code>TP / (TP + FN)</code><br />\n"
        "          <code>TP / Real Positives</code><br />\n"
        "          <code>P(Predicted Positive | Real Positive)</code>\n"
        "        </dd>\n"
        "        <dt>Specificity / TNR</dt>\n"
        "        <dd>\n"
        "          <code>TN / (TN + FP)</code><br />\n"
        "          <code>TN / Real Negatives</code><br />\n"
        "          <code>P(Predicted Negative | Real Negative)</code>\n"
        "        </dd>\n"
        "        <dt>PPV / Precision</dt>\n"
        "        <dd>\n"
        "          <code>TP / (TP + FP)</code><br />\n"
        "          <code>TP / Predicted Positives</code><br />\n"
        "          <code>P(Real Positive | Predicted Positive)</code>\n"
        "        </dd>\n"
        "        <dt>NPV</dt>\n"
        "        <dd>\n"
        "          <code>TN / (TN + FN)</code><br />\n"
        "          <code>TN / Predicted Negatives</code><br />\n"
        "          <code>P(Real Negative | Predicted Negative)</code>\n"
        "        </dd>\n"
        "        <dt>Lift</dt>\n"
        "        <dd><code>PPV / Prevalence</code></dd>\n"
        "        <dt>Net Benefit</dt>\n"
        "        <dd>\n"
        "          <code>TP / N - FP / N * p_t / (1 - p_t)</code><br />\n"
        "          <small>where N = TP + FP + TN + FN</small>\n"
        "        </dd>\n"
        "      </dl>\n"
        "    </section>\n"
        "  </div>\n"
        "</details>"
    )


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


def _summary_report_density_css() -> str:
    """Generate layout density CSS for browser summary reports."""
    return (
        ".rtichoke-report {\n"
        "  max-width: 1040px;\n"
        "  gap: 1.5rem;\n"
        "}\n"
        ".rtichoke-report__section {\n"
        "  gap: 1.25rem;\n"
        "}\n"
        ".rtichoke-report__group {\n"
        "  gap: 0.875rem;\n"
        "}\n"
        ".rtichoke-report__component,\n"
        ".rtichoke-report__tabpanel {\n"
        "  gap: 0.5rem;\n"
        "}\n"
        ".rtichoke-report .rtichoke-viz-chart {\n"
        "  min-height: 500px;\n"
        "  height: 500px;\n"
        "}\n"
        ".rtichoke-report .rtichoke-calibration {\n"
        "  min-height: 550px;\n"
        "  height: 550px;\n"
        "}\n"
        ".rtichoke-report__tabpanel .rtichoke-report__component-title {\n"
        "  display: none;\n"
        "}\n"
        ".rtichoke-report__nav {\n"
        "  background-color: transparent;\n"
        "  border: none;\n"
        "  border-bottom: 1px solid #e5e7eb;\n"
        "  border-radius: 0;\n"
        "  padding: 0.5rem 0;\n"
        "}\n"
        ".rtichoke-report .rtichoke-summary-metrics {\n"
        "  border: none;\n"
        "  background: transparent;\n"
        "  box-shadow: none;\n"
        "}\n"
        ".rtichoke-report .rtichoke-summary-metrics__title {\n"
        "  display: none;\n"
        "}\n"
        ".rtichoke-report .rtichoke-summary-metrics__table th,\n"
        ".rtichoke-report .rtichoke-summary-metrics__table td {\n"
        "  padding: 0.35rem 0.65rem;\n"
        "  border-bottom: 1px solid #e5e7eb;\n"
        "}\n"
    )


class RtichokeBrowserReport:
    """A complete canonical ReportSpec rendered by shared ``rtichoke_viz``."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        include_cheat_sheet: bool = False,
    ) -> None:
        self.spec = spec
        self.include_cheat_sheet = include_cheat_sheet

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
        render_fn = _resolve_render_report_symbol(viz_js)

        raw_title = self.spec.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            doc_title = html.escape(raw_title)
        else:
            doc_title = "rtichoke report"

        if self.include_cheat_sheet:
            cheat_sheet_json = json.dumps(
                _performance_metrics_cheat_sheet_html()
            ).replace("</", "<\\/")
            mount_js = f"""    const reportNode = {render_fn}(spec, {{
      sectionGroupPresentation: "tabs",
      groupPresentation: "tabs",
      sectionComponentPresentation: "tabs"
    }});
    const headerNode = reportNode.querySelector(".rtichoke-report__header");
    const cheatSheetWrapper = document.createElement("div");
    cheatSheetWrapper.innerHTML = {cheat_sheet_json};
    const cheatSheetNode = cheatSheetWrapper.firstElementChild;
    if (headerNode && headerNode.nextSibling) {{
      reportNode.insertBefore(cheatSheetNode, headerNode.nextSibling);
    }} else {{
      reportNode.appendChild(cheatSheetNode);
    }}
    document.querySelector("#rtichoke-report").append(reportNode);"""
        else:
            mount_js = f"""    document.querySelector("#rtichoke-report").append({render_fn}(spec, {{
      sectionGroupPresentation: "tabs",
      groupPresentation: "tabs",
      sectionComponentPresentation: "tabs"
    }}));"""

        html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
{viz_css}
{_summary_report_density_css()}
  </style>
  <title>{doc_title}</title>
</head>
<body>
  <div id="rtichoke-report"></div>
  <script id="rtichoke-report-spec" type="application/json">{spec_json}</script>
  <script type="module">
{viz_js}
    const spec = JSON.parse(
      document.querySelector("#rtichoke-report-spec").textContent
    );
{mount_js}
  </script>
</body>
</html>
"""
        output.write_text(html_content, encoding="utf-8")
        return output

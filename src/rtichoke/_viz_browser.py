"""Internal browser-rendering proof helpers for vendored ``rtichoke_viz`` assets."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import polars as pl

_REQUIRED_ROC_COLUMNS = {
    "reference_group",
    "chosen_cutoff",
    "sensitivity",
    "specificity",
}


def _copy_viz_assets(output_dir: Path) -> None:
    vendor = files("rtichoke").joinpath("_vendor", "rtichoke_viz")
    (output_dir / "rtichoke-viz.js").write_bytes(
        vendor.joinpath("rtichoke-viz.js").read_bytes()
    )
    (output_dir / "rtichoke-viz.css").write_bytes(
        vendor.joinpath("rtichoke-viz.css").read_bytes()
    )


def _roc_spec_from_performance_data(
    performance_data: pl.DataFrame,
) -> dict[str, object]:
    """Map existing rtichoke_python ROC rows to the canonical rtichoke_viz spec."""
    missing = _REQUIRED_ROC_COLUMNS.difference(performance_data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"ROC performance data is missing columns: {missing_columns}")

    rows = performance_data.select(
        "reference_group",
        "chosen_cutoff",
        "sensitivity",
        "specificity",
    ).to_dicts()

    return {
        "schemaVersion": "1.0",
        "type": "roc",
        "data": [
            {
                "model": row["reference_group"],
                "cutoff": row["chosen_cutoff"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
            }
            for row in rows
        ],
        "x": "false_positive_rate",
        "y": "sensitivity",
        "xAxis": {"label": "1 - Specificity", "domain": [0, 1]},
        "yAxis": {"label": "Sensitivity", "domain": [0, 1]},
        "references": [{"type": "identity"}],
    }


def _calibration_spec_from_curve_list(
    calibration_curve_list: dict[str, Any],
) -> dict[str, object]:
    """Map existing discrete calibration output to the canonical spec."""
    rows = calibration_curve_list["deciles_dat"].select(
        "reference_group", "x", "y", "n_reals", "n"
    )
    distribution_rows = calibration_curve_list["histogram_for_calibration"].select(
        "reference_group", "mids", "counts"
    )

    return {
        "schemaVersion": "1.0",
        "type": "calibration",
        "data": [
            {
                "model": row["reference_group"],
                "predicted": row["x"],
                "observed": row["y"],
                "method": "discrete",
                "events": row["n_reals"],
                "total": row["n"],
            }
            for row in rows.to_dicts()
        ],
        "distribution": [
            {
                "model": row["reference_group"],
                "midpoint": row["mids"],
                "count": row["counts"],
                "binWidth": 0.01,
            }
            for row in distribution_rows.to_dicts()
        ],
        "x": "predicted",
        "y": "observed",
        "xAxis": {"label": "Predicted probability", "domain": [0, 1]},
        "yAxis": {"label": "Observed probability", "domain": [0, 1]},
        "references": [{"type": "identity"}],
    }


def _write_roc_browser_html(
    performance_data: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a standalone proof page that uses the vendored browser renderer."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _copy_viz_assets(output.parent)

    spec_json = json.dumps(_roc_spec_from_performance_data(performance_data)).replace(
        "</", "<\\/"
    )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>rtichoke_viz ROC vendoring proof</title>
  <link rel=\"stylesheet\" href=\"./rtichoke-viz.css\">
</head>
<body>
  <div id=\"roc-chart\" class=\"rtichoke-viz-chart\"></div>
  <script id=\"roc-spec\" type=\"application/json\">{spec_json}</script>
  <script type=\"module\">
    import {{ renderRoc }} from \"./rtichoke-viz.js\";
    const spec = JSON.parse(document.querySelector(\"#roc-spec\").textContent);
    document.querySelector(\"#roc-chart\").append(renderRoc(spec));
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def _write_calibration_browser_html(
    calibration_curve_list: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a standalone discrete-calibration proof using vendored assets."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _copy_viz_assets(output.parent)

    spec_json = json.dumps(
        _calibration_spec_from_curve_list(calibration_curve_list)
    ).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>rtichoke_viz calibration vendoring proof</title>
  <link rel=\"stylesheet\" href=\"./rtichoke-viz.css\">
</head>
<body>
  <div id=\"calibration-chart\" class=\"rtichoke-viz-chart\"></div>
  <script id=\"calibration-spec\" type=\"application/json\">{spec_json}</script>
  <script type=\"module\">
    import {{ renderCalibration }} from \"./rtichoke-viz.js\";
    const spec = JSON.parse(document.querySelector(\"#calibration-spec\").textContent);
    document.querySelector(\"#calibration-chart\").append(renderCalibration(spec));
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output

"""Renderer selection for canonical rtichoke visualization specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

Renderer = Literal["plotly", "matplotlib", "browser", "rtichoke_viz"]

_SUPPORTED_RENDERERS = ("plotly", "matplotlib", "browser", "rtichoke_viz")


def _validate_renderer(renderer: str) -> Renderer:
    """Validate and normalize a public renderer name."""
    if renderer not in _SUPPORTED_RENDERERS:
        supported = ", ".join(repr(name) for name in _SUPPORTED_RENDERERS)
        raise ValueError(
            f"Unsupported renderer {renderer!r}. Supported renderers are: {supported}."
        )
    return renderer  # type: ignore[return-value]


@dataclass(frozen=True)
class RtichokeBrowserChart:
    """A canonical v2 chart that can be written for offline browser rendering."""

    spec: dict[str, Any]
    size: int = 600

    def write_html(self, path: str | Path) -> Path:
        """Write an offline HTML page plus its packaged renderer assets.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination for the HTML page.

        Returns
        -------
        pathlib.Path
            The written HTML path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        vendor = files("rtichoke").joinpath("_vendor", "rtichoke_viz")
        for asset in ("rtichoke-viz.js", "rtichoke-viz.css"):
            (output.parent / asset).write_bytes(vendor.joinpath(asset).read_bytes())

        render_export = {
            "roc": "renderRocV2",
            "calibration": "renderCalibrationV2",
            "precision_recall": "renderPrecisionRecallV2",
            "gains": "renderGainsV2",
            "lift": "renderLiftV2",
        }.get(str(self.spec.get("type")))
        if render_export is None:
            raise ValueError(
                f"rtichoke_viz does not support chart type {self.spec.get('type')!r}."
            )

        spec_json = json.dumps(self.spec, separators=(",", ":")).replace("</", "<\\/")
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="./rtichoke-viz.css">
  <title>rtichoke {self.spec.get("type")} chart</title>
</head>
<body>
  <div id="rtichoke-chart" class="rtichoke-viz-chart"></div>
  <script id="rtichoke-spec" type="application/json">{spec_json}</script>
  <script type="module">
    import {{ {render_export} }} from "./rtichoke-viz.js";
    const spec = JSON.parse(document.querySelector("#rtichoke-spec").textContent);
    const chart = {render_export}(spec, {{ width: {self.size}, height: {self.size} }});
    document.querySelector("#rtichoke-chart").append(chart);
  </script>
</body>
</html>
"""
        output.write_text(html, encoding="utf-8")
        return output


def _render_gains_matplotlib(
    spec: dict[str, Any], *, size: int, color_values: list[str]
) -> Any:
    """Render canonical gains quantities with an optional Matplotlib backend."""
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError(
            "The 'matplotlib' renderer requires the optional matplotlib dependency. "
            "Install it with `pip install 'rtichoke[matplotlib]'`."
        ) from error

    series = spec.get("series", [])
    data = spec.get("data", [])
    assert isinstance(series, list) and isinstance(data, list)
    horizons = list(
        dict.fromkeys(
            item["horizon"] for item in series if item.get("horizon") is not None
        )
    )
    panels: list[float | None] = horizons or [None]
    figure = Figure(figsize=(size / 100 * len(panels), size / 100), dpi=100)
    axes_value = figure.subplots(1, len(panels), squeeze=False)
    axes = list(axes_value[0])
    references = spec.get("references", [])
    assert isinstance(references, list)
    display_groups = list(dict.fromkeys(item["display"]["group"] for item in series))
    colors = {
        group: (
            "black"
            if len(display_groups) == 1
            else color_values[index % len(color_values)]
        )
        for index, group in enumerate(display_groups)
    }
    x_axis = spec["xAxis"]
    y_axis = spec["yAxis"]
    for axis, horizon in zip(axes, panels):
        for reference in references:
            if not isinstance(reference, dict):
                continue
            if (
                reference.get("scope") == "population_horizon"
                and reference.get("horizon") != horizon
            ):
                continue
            if reference.get("type") == "identity":
                x_values, y_values = [0, 1], [0, 1]
            elif reference.get("type") == "path":
                points = reference.get("points", [])
                x_values = [point["x"] for point in points]
                y_values = [point["y"] for point in points]
            else:
                continue
            axis.plot(
                x_values,
                y_values,
                color="#BEBEBE",
                linestyle="--",
                linewidth=2,
            )

        panel_series = [
            item
            for item in series
            if item.get("horizon") is None or item.get("horizon") == horizon
        ]
        for item in panel_series:
            rows = [row for row in data if row["seriesId"] == item["id"]]
            display = item["display"]
            axis.plot(
                [row["ppcr"] for row in rows],
                [row["sensitivity"] for row in rows],
                label=display["label"],
                color=colors[display["group"]],
                linewidth=2,
            )

        axis.set_xlabel(x_axis["label"])
        axis.set_ylabel(y_axis["label"])
        axis.set_xlim(*x_axis["domain"])
        axis.set_ylim(*y_axis["domain"])
        if horizon is not None:
            axis.set_title(f"Fixed Time Horizon: {horizon:g}")
        if len(panel_series) > 1:
            axis.legend()
    figure.tight_layout()
    return figure


def _render_gains_v2(
    spec: dict[str, Any],
    *,
    renderer: str,
    size: int,
    color_values: list[str],
) -> Any:
    """Render a canonical gains v2 spec with a non-default backend."""
    selected = _validate_renderer(renderer)
    if selected == "matplotlib":
        return _render_gains_matplotlib(spec, size=size, color_values=color_values)
    if selected in {"browser", "rtichoke_viz"}:
        return RtichokeBrowserChart(spec=spec, size=size)
    raise ValueError("The Plotly renderer must use the existing production path.")


def _render_lift_matplotlib(
    spec: dict[str, Any], *, size: int, color_values: list[str]
) -> Any:
    """Render canonical lift quantities with an optional Matplotlib backend."""
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError(
            "The 'matplotlib' renderer requires the optional matplotlib dependency. "
            "Install it with `pip install 'rtichoke[matplotlib]'`."
        ) from error

    series = spec.get("series", [])
    data = spec.get("data", [])
    assert isinstance(series, list) and isinstance(data, list)
    horizons = list(
        dict.fromkeys(
            item["horizon"] for item in series if item.get("horizon") is not None
        )
    )
    panels: list[float | None] = horizons or [None]
    figure = Figure(figsize=(size / 100 * len(panels), size / 100), dpi=100)
    axes_value = figure.subplots(1, len(panels), squeeze=False)
    axes = list(axes_value[0])
    references = spec.get("references", [])
    assert isinstance(references, list)
    display_groups = list(dict.fromkeys(item["display"]["group"] for item in series))
    colors = {
        group: (
            "black"
            if len(display_groups) == 1
            else color_values[index % len(color_values)]
        )
        for index, group in enumerate(display_groups)
    }
    x_axis = spec["xAxis"]
    y_axis = spec["yAxis"]
    for axis, horizon in zip(axes, panels):
        for reference in references:
            if not isinstance(reference, dict):
                continue
            if (
                reference.get("scope") == "population_horizon"
                and reference.get("horizon") != horizon
            ):
                continue
            if reference.get("type") == "horizontal":
                value = reference.get("value", 1.0)
                axis.axhline(
                    y=value,
                    color="#BEBEBE",
                    linestyle="--",
                    linewidth=2,
                )
            elif reference.get("type") == "path":
                points = reference.get("points", [])
                x_values = [point["x"] for point in points]
                y_values = [point["y"] for point in points]
                axis.plot(
                    x_values,
                    y_values,
                    color="#BEBEBE",
                    linestyle="--",
                    linewidth=2,
                )
            else:
                continue

        panel_series = [
            item
            for item in series
            if item.get("horizon") is None or item.get("horizon") == horizon
        ]
        for item in panel_series:
            rows = [row for row in data if row["seriesId"] == item["id"]]
            display = item["display"]
            axis.plot(
                [row["ppcr"] for row in rows],
                [row["lift"] for row in rows],
                label=display["label"],
                color=colors[display["group"]],
                linewidth=2,
            )

        axis.set_xlabel(x_axis["label"])
        axis.set_ylabel(y_axis["label"])
        axis.set_xlim(*x_axis["domain"])
        if y_axis["domain"][1] is not None:
            axis.set_ylim(*y_axis["domain"])
        else:
            axis.set_ylim(bottom=y_axis["domain"][0])
        if horizon is not None:
            axis.set_title(f"Fixed Time Horizon: {horizon:g}")
        if len(panel_series) > 1:
            axis.legend()
    figure.tight_layout()
    return figure


def _render_lift_v2(
    spec: dict[str, Any],
    *,
    renderer: str,
    size: int,
    color_values: list[str],
) -> Any:
    """Render a canonical lift v2 spec with a non-default backend."""
    selected = _validate_renderer(renderer)
    if selected == "matplotlib":
        return _render_lift_matplotlib(spec, size=size, color_values=color_values)
    if selected in {"browser", "rtichoke_viz"}:
        raise ValueError(
            "Browser rendering for Lift curves requires a newer vendored release of rtichoke_viz containing Lift support."
        )
    raise ValueError("The Plotly renderer must use the existing production path.")

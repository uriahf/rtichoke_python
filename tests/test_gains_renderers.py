from pathlib import Path

import matplotlib.figure
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from rtichoke import create_gains_curve
from rtichoke._renderers import RtichokeBrowserChart


def _inputs():
    return (
        {
            "Model A": np.array([0.05, 0.2, 0.7, 0.95]),
            "Model B": np.array([0.1, 0.4, 0.6, 0.9]),
        },
        np.array([0, 0, 1, 1]),
    )


def test_default_and_explicit_plotly_preserve_existing_renderer():
    probs, reals = _inputs()

    default = create_gains_curve(probs, reals, by=0.25)
    explicit = create_gains_curve(probs, reals, by=0.25, renderer="plotly")

    assert isinstance(default, go.Figure)
    assert pio.to_json(default) == pio.to_json(explicit)


def test_matplotlib_renders_canonical_gains_spec():
    probs, reals = _inputs()

    figure = create_gains_curve(probs, reals, by=0.25, renderer="matplotlib")

    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes[0].lines) == 4  # random, perfect, and two series
    assert figure.axes[0].get_xlabel() == "Predicted Positives (Rate)"


@pytest.mark.parametrize("renderer", ["browser", "rtichoke_viz"])
def test_browser_renderer_writes_offline_v2_chart(renderer: str, tmp_path: Path):
    probs, reals = _inputs()

    chart = create_gains_curve(probs, reals, by=0.25, renderer=renderer)
    assert isinstance(chart, RtichokeBrowserChart)
    assert chart.spec["schemaVersion"] == "2.0"
    assert chart.spec["type"] == "gains"
    assert len(chart.spec["evaluations"]) == 2
    assert {row["seriesId"] for row in chart.spec["data"]} == {
        "series-1",
        "series-2",
    }

    output = chart.write_html(tmp_path / "gains.html")
    html = output.read_text(encoding="utf-8")
    assert 'import { renderGainsV2 } from "./rtichoke-viz.js"' in html
    assert '"schemaVersion":"2.0"' in html
    assert (tmp_path / "rtichoke-viz.js").is_file()
    assert (tmp_path / "rtichoke-viz.css").is_file()
    assert "http://" not in html and "https://" not in html


def test_unsupported_renderer_is_clear():
    probs, reals = _inputs()

    with pytest.raises(ValueError, match="Unsupported renderer 'canvas'"):
        create_gains_curve(probs, reals, renderer="canvas")

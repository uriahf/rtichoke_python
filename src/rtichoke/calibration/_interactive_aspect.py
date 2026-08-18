"""Shared layout constraints for interactive calibration plots."""

from typing import Any

from plotly.graph_objs._figure import Figure


_CALIBRATION_DOMAIN = [0.22, 1.0]
_HISTOGRAM_DOMAIN = [0.0, 0.16]
_HEIGHT_RATIO = 1.30


def enforce_square_calibration_panel(fig: Figure) -> Figure:
    """Keep the upper calibration panel square without squeezing it horizontally.

    Calibration figures include a histogram in a separate lower subplot. The
    requested ``size`` therefore defines the widget width, while the widget is
    made taller so the upper calibration panel can remain square. The histogram
    keeps its own unconstrained rectangular panel below it.
    """
    width = fig.layout.width or 600
    fig.update_layout(height=round(width * _HEIGHT_RATIO))
    fig.update_yaxes(
        domain=_CALIBRATION_DOMAIN,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_yaxes(domain=_HISTOGRAM_DOMAIN, row=2, col=1)
    return fig


def shared_calibration_axis_layout(axis_range: list[float]) -> dict[str, Any]:
    """Return the common zoom settings used by calibration x and y axes."""
    return {"range": axis_range, "fixedrange": False}

"""Shared layout constraints for interactive calibration plots."""

from typing import Any

from plotly.graph_objs._figure import Figure


def enforce_square_calibration_panel(fig: Figure) -> Figure:
    """Keep the upper calibration panel on a 1:1 predicted/observed scale.

    Calibration figures include a histogram in a separate lower subplot.  The
    aspect-ratio constraint therefore belongs only to the upper calibration
    panel, not to the full Plotly widget or to the histogram.
    """
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=1,
    )
    return fig


def shared_calibration_axis_layout(axis_range: list[float]) -> dict[str, Any]:
    """Return the common zoom settings used by calibration x and y axes."""
    return {"range": axis_range, "fixedrange": False}

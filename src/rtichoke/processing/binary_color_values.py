"""Helpers for applying custom palettes to binary Plotly curves."""

from plotly.graph_objs._figure import Figure


def _apply_color_values_binary(fig: Figure, color_values) -> Figure:
    """Apply custom colors to multiple binary reference groups.

    rtichoke's R implementation keeps a single model black and uses
    ``color_values`` only when multiple models or populations are shown.
    """
    if color_values is None:
        return fig

    reference_groups = []
    for trace in fig.data:
        if trace.showlegend is True and trace.name is not None:
            name = str(trace.name)
            if name not in reference_groups:
                reference_groups.append(name)

    # A single model intentionally remains black, matching R.
    if len(reference_groups) <= 1:
        return fig

    if len(color_values) < len(reference_groups):
        raise ValueError(
            "color_values must contain at least one color per reference group"
        )

    group_colors = dict(zip(reference_groups, color_values))

    for trace in fig.data:
        legendgroup = "" if trace.legendgroup is None else str(trace.legendgroup)
        group = next(
            (
                reference_group
                for reference_group in reference_groups
                if legendgroup == reference_group
                or legendgroup.endswith(f"_{reference_group}")
            ),
            None,
        )
        if group is None:
            continue

        color = group_colors[group]
        if trace.line is not None:
            trace.line.color = color
        if trace.marker is not None and trace.mode and "markers" in trace.mode:
            trace.marker.color = color
        if trace.hoverlabel is not None:
            trace.hoverlabel.bgcolor = color
            trace.hoverlabel.bordercolor = color

    return fig

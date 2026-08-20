"""Reactable renderer for rtichoke performance tables."""

from __future__ import annotations

from collections.abc import Sequence

import htmltools as html
import numpy as np
import polars as pl
from reactable import Reactable, Column, ColFormat, ColGroup
from reactable.models import CellInfo, RowInfo


DEFAULT_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#07004D",
    "#E6AB02",
    "#FE5F55",
    "#54494B",
    "#006E90",
    "#BC96E6",
    "#52050A",
    "#1F271B",
    "#BE7C4D",
    "#63768D",
    "#08A045",
    "#320A28",
    "#82FF9E",
    "#2176FF",
    "#D1603D",
    "#585123",
)


def _bar_style(
    value: float | None, maximum: float = 1.0, color: str = "lightgreen"
) -> dict[str, str]:
    if value is None or not np.isfinite(value) or maximum <= 0:
        return {}
    width = min(abs(float(value)) / maximum, 1.0) * 100
    return {
        "background": f"linear-gradient(90deg, {color} {width}%, transparent {width}%)",
        "backgroundSize": "98% 88%",
        "backgroundRepeat": "no-repeat",
        "backgroundPosition": "center",
    }


def _net_benefit_style(value: float | None, maximum: float) -> dict[str, str]:
    if value is None or not np.isfinite(value) or maximum <= 0:
        return {}
    width = max(-1.0, min(float(value) / maximum, 1.0))
    position = (0.5 + width / 2) * 100
    if width >= 0:
        background = (
            "linear-gradient(90deg, transparent 50%, lightgreen 50%, "
            f"lightgreen {position}%, transparent {position}%)"
        )
    else:
        background = (
            f"linear-gradient(90deg, transparent {position}%, pink {position}%, "
            "pink 50%, transparent 50%)"
        )
    return {
        "background": background,
        "backgroundSize": "98% 88%",
        "backgroundRepeat": "no-repeat",
        "backgroundPosition": "center",
    }


def _metric_column(column_id: str, name: str, maximum: float = 1.0) -> Column:
    return Column(
        id=column_id,
        name=name,
        format=ColFormat(digits=2),
        style=lambda info: _bar_style(info.value, maximum),
    )


def render_performance_table_reactable(
    performance_data: pl.DataFrame,
    color_values: Sequence[str] = DEFAULT_COLORS,
) -> Reactable:
    """Render prepared binary or time-dependent performance data."""
    if performance_data.is_empty():
        raise ValueError("performance_data must contain at least one row")

    stratifications = performance_data.get_column("stratified_by").unique().to_list()
    if len(stratifications) != 1:
        raise ValueError("performance_data must contain exactly one stratification")
    stratified_by = stratifications[0]

    display_columns = [
        "reference_group",
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
        "chosen_cutoff",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "lift",
        "predicted_positives",
        "net_benefit",
        "ppcr",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    ]
    data = performance_data.select(
        [c for c in display_columns if c in performance_data.columns]
    )
    rename_map = {
        "reference_group": "Model",
        "fixed_time_horizon": "Time",
        "censoring_heuristic": "Censoring",
        "competing_heuristic": "Competing Event",
    }
    data = data.rename({k: v for k, v in rename_map.items() if k in data.columns})

    context_sort = [
        c for c in ("Time", "Censoring", "Competing Event") if c in data.columns
    ]
    if stratified_by == "probability_threshold":
        data = data.rename({"chosen_cutoff": "Threshold"})
        sort_columns = context_sort + [
            c for c in ("Threshold", "Model") if c in data.columns
        ]
    else:
        if "chosen_cutoff" in data.columns:
            data = data.drop("chosen_cutoff")
        sort_columns = context_sort + [
            c for c in ("ppcr", "Model") if c in data.columns
        ]
    if sort_columns:
        data = data.sort(sort_columns)

    lift_max = data.get_column("lift").drop_nulls().max() or 1.0
    nb_max = 1.0
    if "net_benefit" in data.columns:
        nb_max = data.get_column("net_benefit").drop_nulls().abs().max() or 1.0

    models = (
        data.get_column("Model").unique(maintain_order=True).to_list()
        if "Model" in data.columns
        else []
    )
    colors = {
        model: color_values[i % len(color_values)] for i, model in enumerate(models)
    }

    def model_cell(info: CellInfo):
        value = info.value
        color = colors.get(value, "#aaa")
        return html.span(
            html.span(
                style=(
                    "display:inline-block;margin-right:8px;width:9px;height:9px;"
                    f"background-color:{color};border-radius:50%;"
                )
            ),
            str(value),
        )

    def ppcr_cell(info: CellInfo) -> str:
        if info.value is None:
            return ""
        count = data[info.row_index, "predicted_positives"]
        return f"{count} ({float(info.value) * 100:.2f}%)"

    def confusion_matrix(info: RowInfo):
        i = info.row_index
        tp = data[i, "true_positives"]
        tn = data[i, "true_negatives"]
        fp = data[i, "false_positives"]
        fn = data[i, "false_negatives"]
        predicted_positive = tp + fp
        predicted_negative = fn + tn
        real_positive = tp + fn
        real_negative = fp + tn
        total = tp + fp + fn + tn

        matrix = pl.DataFrame(
            {
                "Outcome": ["Predicted Positive", "Predicted Negative", " "],
                "Real Positive": [tp, fn, real_positive],
                "Real Negative": [fp, tn, real_negative],
                "Total": [predicted_positive, predicted_negative, total],
            }
        )

        def matrix_cell(info: CellInfo) -> str:
            if info.value is None or total == 0:
                return ""
            return f"{info.value} ({float(info.value) / total * 100:.2f}%)"

        def matrix_style(colors: tuple[str, str, str]):
            return lambda info: _bar_style(
                info.value, float(total), colors[info.row_index]
            )

        nested = Reactable(
            matrix,
            columns=[
                Column(id="Outcome", style={"fontWeight": "bold"}),
                Column(
                    id="Real Positive",
                    align="left",
                    cell=matrix_cell,
                    style=matrix_style(("lightgreen", "pink", "lightgrey")),
                ),
                Column(
                    id="Real Negative",
                    align="left",
                    cell=matrix_cell,
                    style=matrix_style(("pink", "lightgreen", "lightgrey")),
                ),
                Column(
                    id="Total",
                    name=" ",
                    align="left",
                    cell=matrix_cell,
                    style=matrix_style(("lightgrey", "lightgrey", "lightgrey")),
                ),
            ],
            full_width=False,
            sortable=False,
            pagination=False,
        )
        return html.div(nested.to_widget(), style="padding:16px;")

    columns = [Column(id="Model", cell=model_cell, min_width=120)]
    if "Time" in data.columns:
        columns.append(
            Column(
                id="Time",
                name="Time Horizon",
                format=ColFormat(digits=2),
                min_width=100,
            )
        )
    if "Censoring" in data.columns:
        columns.append(Column(id="Censoring", name="Censoring", min_width=110))
    if "Competing Event" in data.columns:
        columns.append(
            Column(id="Competing Event", name="Competing Event", min_width=140)
        )
    if stratified_by == "probability_threshold":
        columns.append(
            Column(
                id="Threshold",
                name="Probability Threshold",
                format=ColFormat(digits=2),
                min_width=130,
            )
        )
    columns.extend(
        [
            Column(
                id="ppcr",
                name="Predicted Positives",
                cell=ppcr_cell,
                min_width=150,
                style=lambda info: _bar_style(info.value, 1.0, "#d3d3d3"),
            ),
            Column(id="predicted_positives", show=False),
            _metric_column("sensitivity", "Sens"),
            _metric_column("specificity", "Spec"),
            _metric_column("ppv", "PPV"),
            _metric_column("npv", "NPV"),
            _metric_column("lift", "Lift", float(lift_max)),
            Column(
                id="net_benefit",
                name="Net Benefit",
                format=ColFormat(digits=2),
                style=lambda info: _net_benefit_style(info.value, float(nb_max)),
                show=stratified_by == "probability_threshold",
            ),
            Column(id="true_positives", show=False),
            Column(id="true_negatives", show=False),
            Column(id="false_positives", show=False),
            Column(id="false_negatives", show=False),
        ]
    )

    metric_columns = ["sensitivity", "specificity", "ppv", "npv", "lift"]
    if stratified_by == "probability_threshold":
        metric_columns.append("net_benefit")

    return Reactable(
        data,
        columns=columns,
        column_groups=[ColGroup(name="Performance Metrics", columns=metric_columns)],
        default_col_def=Column(align="left"),
        bordered=True,
        compact=True,
        striped=True,
        highlight=True,
        details=confusion_matrix,
        show_sort_icon=False,
    )

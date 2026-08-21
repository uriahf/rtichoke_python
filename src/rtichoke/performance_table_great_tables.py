"""Great Tables renderer for rtichoke performance tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl
from great_tables import GT, loc, style


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


def _bar_css(
    value: float | None, maximum: float = 1.0, color: str = "lightgreen"
) -> str:
    """CSS matching rtichoke R's reactable metric bars."""
    if value is None or not np.isfinite(value) or maximum <= 0:
        width = 0.0
    else:
        width = min(abs(float(value)) / maximum, 1.0) * 100
    return (
        f"background:linear-gradient(90deg,{color} {width:.4f}%,transparent {width:.4f}%);"
        "background-size:98% 88%;background-repeat:no-repeat;background-position:center;"
    )


def _net_benefit_css(value: float | None, maximum: float) -> str:
    """Diverging bar CSS matching rtichoke R's net-benefit treatment."""
    if value is None or not np.isfinite(value) or maximum <= 0:
        scaled = 0.0
    else:
        scaled = max(-1.0, min(float(value) / maximum, 1.0))
    position = (0.5 + scaled / 2.0) * 100
    if scaled >= 0:
        background = (
            "linear-gradient(90deg,transparent 50%,lightgreen 50%,"
            f"lightgreen {position:.4f}%,transparent {position:.4f}%)"
        )
    else:
        background = (
            f"linear-gradient(90deg,transparent {position:.4f}%,pink {position:.4f}%,"
            "pink 50%,transparent 50%)"
        )
    return (
        f"background:{background};background-size:98% 88%;"
        "background-repeat:no-repeat;background-position:center;"
    )


def render_performance_table_great_tables(
    performance_data: pl.DataFrame,
    color_values: Sequence[str] = DEFAULT_COLORS,
) -> GT:
    """Render prepared binary or time-dependent performance data."""
    if performance_data.is_empty():
        raise ValueError("performance_data must contain at least one row")

    stratifications = performance_data.get_column("stratified_by").unique().to_list()
    if len(stratifications) != 1:
        raise ValueError("performance_data must contain exactly one stratification")
    stratified_by = stratifications[0]

    group_label = "Model"
    if "__group_role" in performance_data.columns:
        group_roles = performance_data.get_column("__group_role").unique().to_list()
        if group_roles == ["population"]:
            group_label = "Population"

    context_columns = [
        "fixed_time_horizon",
        "censoring_heuristic",
        "competing_heuristic",
    ]
    source_columns = [
        "reference_group",
        *context_columns,
        "chosen_cutoff",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "lift",
        "predicted_positives",
        "net_benefit",
        "ppcr",
    ]
    data = performance_data.select(
        [c for c in source_columns if c in performance_data.columns]
    )
    if "reference_group" in data.columns:
        data = data.rename({"reference_group": group_label})
    if "fixed_time_horizon" in data.columns:
        data = data.rename({"fixed_time_horizon": "Time"})
    if "censoring_heuristic" in data.columns:
        data = data.rename({"censoring_heuristic": "Censoring"})
    if "competing_heuristic" in data.columns:
        data = data.rename({"competing_heuristic": "Competing Event"})

    context_sort = [
        c for c in ("Time", "Censoring", "Competing Event") if c in data.columns
    ]
    if stratified_by == "probability_threshold":
        data = data.rename({"chosen_cutoff": "Threshold"})
        sort_columns = context_sort + [
            c for c in ("Threshold", group_label) if c in data.columns
        ]
    else:
        if "chosen_cutoff" in data.columns:
            data = data.drop("chosen_cutoff")
        sort_columns = context_sort + [
            c for c in ("ppcr", group_label) if c in data.columns
        ]
    if sort_columns:
        data = data.sort(sort_columns)

    data = data.with_columns(
        pl.concat_str(
            [
                pl.col("predicted_positives").cast(pl.String),
                pl.lit(" ("),
                (pl.col("ppcr") * 100).round(2).cast(pl.String),
                pl.lit("%)"),
            ]
        ).alias("Predicted Positives")
    )

    if stratified_by != "probability_threshold" and "net_benefit" in data.columns:
        data = data.drop("net_benefit")

    metric_columns = [
        c
        for c in ["sensitivity", "specificity", "ppv", "npv", "lift", "net_benefit"]
        if c in data.columns
    ]
    display_columns = [
        c
        for c in [
            group_label,
            "Time",
            "Censoring",
            "Competing Event",
            "Threshold",
            "Predicted Positives",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "lift",
            "net_benefit",
        ]
        if c in data.columns
    ]
    display = data.select(display_columns)

    labels = {
        group_label: group_label,
        "Time": "Time Horizon",
        "Censoring": "Censoring",
        "Competing Event": "Competing Event",
        "Threshold": "Probability Threshold",
        "Predicted Positives": "Predicted Positives",
        "sensitivity": "Sens",
        "specificity": "Spec",
        "ppv": "PPV",
        "npv": "NPV",
        "lift": "Lift",
        "net_benefit": "Net Benefit",
    }

    table = (
        GT(display)
        .cols_label(cases={c: labels[c] for c in display.columns})
        .tab_spanner(label="Performance Metrics", columns=metric_columns)
        .fmt_number(columns=metric_columns, decimals=2)
        .cols_width({c: "100px" for c in metric_columns + ["Predicted Positives"]})
        .opt_vertical_padding(scale=0.75)
        .opt_horizontal_padding(scale=0.8)
        .opt_css(
            ".gt_table{font-size:14px;}"
            ".gt_col_heading{font-weight:600;}"
            ".gt_row{vertical-align:middle;}"
        )
    )

    table = table.tab_style(
        style=style.css("text-align:left;"),
        locations=loc.body(columns=display.columns),
    )
    if "net_benefit" in display.columns:
        table = table.tab_style(
            style=style.css("text-align:center;"),
            locations=loc.body(columns="net_benefit"),
        )

    if group_label in data.columns:
        groups = data.get_column(group_label).unique(maintain_order=True).to_list()
        for index, group in enumerate(groups):
            color = color_values[index % len(color_values)]
            rows = [
                i
                for i, value in enumerate(data.get_column(group_label).to_list())
                if value == group
            ]
            table = table.tab_style(
                style=style.css(
                    f"color:{color};font-weight:600;text-shadow:0 0 0 currentColor;"
                ),
                locations=loc.body(columns=group_label, rows=rows),
            )

    lift_max = float(
        cast(float | int, data.get_column("lift").drop_nulls().max() or 1.0)
    )
    nb_max = 1.0
    if "net_benefit" in data.columns:
        nb = data.get_column("net_benefit").drop_nulls().abs().max()
        nb_max = float(cast(float | int, nb or 1.0))

    for row_index in range(data.height):
        ppcr_value = data[row_index, "ppcr"]
        table = table.tab_style(
            style=style.css(_bar_css(ppcr_value, color="lightgrey")),
            locations=loc.body(columns="Predicted Positives", rows=[row_index]),
        )
        for column in ["sensitivity", "specificity", "ppv", "npv"]:
            if column in data.columns:
                table = table.tab_style(
                    style=style.css(_bar_css(data[row_index, column])),
                    locations=loc.body(columns=column, rows=[row_index]),
                )
        if "lift" in data.columns:
            table = table.tab_style(
                style=style.css(_bar_css(data[row_index, "lift"], maximum=lift_max)),
                locations=loc.body(columns="lift", rows=[row_index]),
            )
        if "net_benefit" in data.columns:
            table = table.tab_style(
                style=style.css(
                    _net_benefit_css(data[row_index, "net_benefit"], nb_max)
                ),
                locations=loc.body(columns="net_benefit", rows=[row_index]),
            )

    return table

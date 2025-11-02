import marimo

__generated_with = "0.14.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import numpy as np
    import polars as pl
    import plotly.express as px

    from rtichoke.helpers.sandbox_observable_helpers import (
        create_breaks_values,
        create_list_data_to_adjust,
        create_adjusted_data,
        create_aj_data_combinations,
        cast_and_join_adjusted_data,
    )

    return (
        cast_and_join_adjusted_data,
        create_adjusted_data,
        create_aj_data_combinations,
        create_breaks_values,
        create_list_data_to_adjust,
        np,
        pl,
        px,
    )


@app.cell
def _(np, pl):
    probs_test = {
        "small_data_set": np.array(
            [0.9, 0.85, 0.95, 0.88, 0.6, 0.7, 0.51, 0.2, 0.1, 0.33]
        )
    }
    reals_dict_test = [1, 1, 1, 1, 0, 2, 1, 2, 0, 1]
    times_dict_test = [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3]

    data_to_adjust = pl.DataFrame(
        {
            "strata": np.repeat("small_data_test", 10),
            # "probs": probs_test["test_data"],
            "reals": reals_dict_test,
            "times": times_dict_test,
        }
    )

    data_to_adjust
    return probs_test, reals_dict_test, times_dict_test


@app.cell
def _(create_aj_data_combinations, create_breaks_values):
    by = 0.2
    breaks = create_breaks_values(None, "probability_threshold", by)
    stratified_by = ["probability_threshold", "ppcr"]
    # stratified_by = ["probability_threshold"]

    # stratified_by = ["ppcr"]

    heuristics_sets = [
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "adjusted_as_negative",
        },
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "adjusted_as_composite",
        },
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "adjusted_as_censored",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_negative",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_censored",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "adjusted_as_composite",
        },
        {
            "censoring_heuristic": "excluded",
            "competing_heuristic": "excluded",
        },
        {
            "censoring_heuristic": "adjusted",
            "competing_heuristic": "excluded",
        },
    ]

    aj_data_combinations = create_aj_data_combinations(
        ["small_data_set"],
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=[10.0, 20.0, 30.0, 40.0, 50.0],
        stratified_by=stratified_by,
        by=by,
        breaks=breaks,
        risk_set_scope=["pooled_by_cutoff", "within_stratum"],
    )

    # aj_data_combinations

    aj_data_combinations
    return aj_data_combinations, breaks, by, heuristics_sets, stratified_by


@app.cell
def _(
    aj_data_combinations,
    by,
    create_list_data_to_adjust,
    probs_test,
    reals_dict_test,
    stratified_by,
    times_dict_test,
):
    list_data_to_adjust_polars_probability_threshold = create_list_data_to_adjust(
        aj_data_combinations,
        probs_test,
        reals_dict_test,
        times_dict_test,
        stratified_by=stratified_by,
        by=by,
    )

    list_data_to_adjust_polars_probability_threshold
    return (list_data_to_adjust_polars_probability_threshold,)


@app.cell
def _(
    breaks,
    create_adjusted_data,
    heuristics_sets,
    list_data_to_adjust_polars_probability_threshold,
    stratified_by,
):
    adjusted_data = create_adjusted_data(
        list_data_to_adjust_polars_probability_threshold,
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=[10.0, 20.0, 30.0, 40.0, 50.0],
        breaks=breaks,
        stratified_by=stratified_by,
        # risk_set_scope = ["pooled_by_cutoff"]
        risk_set_scope=["pooled_by_cutoff", "within_stratum"],
    )

    adjusted_data
    return (adjusted_data,)


@app.cell
def _(adjusted_data, aj_data_combinations, cast_and_join_adjusted_data):
    final_adjusted_data_polars = cast_and_join_adjusted_data(
        aj_data_combinations, adjusted_data
    )

    final_adjusted_data_polars
    return (final_adjusted_data_polars,)


@app.cell
def _(final_adjusted_data_polars):
    from rtichoke.helpers.sandbox_observable_helpers import (
        _calculate_cumulative_aj_data,
    )

    cumulative_aj_data = _calculate_cumulative_aj_data(final_adjusted_data_polars)

    cumulative_aj_data
    return (cumulative_aj_data,)


@app.cell
def _(cumulative_aj_data):
    from rtichoke.helpers.sandbox_observable_helpers import (
        _turn_cumulative_aj_to_performance_data,
    )

    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    performance_data
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    fill_color_radio = mo.ui.radio(
        options=["classification_outcome", "reals_labels"],
        value="classification_outcome",
        label="Fill Colors",
    )

    fill_color_radio
    return (fill_color_radio,)


@app.cell(hide_code=True)
def _(mo):
    risk_set_scope_radio = mo.ui.radio(
        options=["pooled_by_cutoff", "within_stratum"],
        value="pooled_by_cutoff",
        label="Risk Set Scope",
    )

    risk_set_scope_radio
    return (risk_set_scope_radio,)


@app.cell(hide_code=True)
def _(mo):
    stratified_by_radio = mo.ui.radio(
        options=["probability_threshold", "ppcr"],
        value="probability_threshold",
        label="Stratified By",
    )

    stratified_by_radio
    return (stratified_by_radio,)


@app.cell(hide_code=True)
def _(by):
    import marimo as mo

    slider_cutoff = mo.ui.slider(start=0, stop=1, step=by, label="Cutoff")
    slider_cutoff
    return mo, slider_cutoff


@app.cell(hide_code=True)
def _(mo):
    fixed_time_horizons_slider = mo.ui.slider(
        start=10, stop=50, step=10, label="Fixed Time Horizon"
    )
    fixed_time_horizons_slider
    return (fixed_time_horizons_slider,)


@app.cell(hide_code=True)
def _(mo):
    censoring_heuristic_radio = mo.ui.radio(
        options=["adjusted", "excluded"],
        value="adjusted",
        label="Censoring Heuristic",
    )

    censoring_heuristic_radio
    return (censoring_heuristic_radio,)


@app.cell(hide_code=True)
def _(mo):
    competing_heuristic_radio = mo.ui.radio(
        options=[
            "adjusted_as_negative",
            "adjusted_as_censored",
            "adjusted_as_composite",
            "excluded",
        ],
        value="adjusted_as_negative",
        label="Censoring Heuristic",
    )

    competing_heuristic_radio
    return (competing_heuristic_radio,)


@app.cell(column=2, hide_code=True)
def _(
    by,
    censoring_heuristic_radio,
    competing_heuristic_radio,
    fill_color_radio,
    final_adjusted_data_polars,
    fixed_time_horizons_slider,
    pl,
    px,
    risk_set_scope_radio,
    slider_cutoff,
    stratified_by_radio,
):
    chosen_cutoff_data = final_adjusted_data_polars.filter(
        pl.col("censoring_heuristic") == censoring_heuristic_radio.value,
        pl.col("competing_heuristic") == competing_heuristic_radio.value,
        pl.col("chosen_cutoff") == slider_cutoff.value,
        pl.col("fixed_time_horizon") == fixed_time_horizons_slider.value,
        pl.col("risk_set_scope") == risk_set_scope_radio.value,
        pl.col("stratified_by") == stratified_by_radio.value,
    ).sort(pl.col("strata"))

    color_discrete_map = {
        "real_positives": "#4C5454",
        "real_competing": "#C880B7",
        "real_negatives": "#E0E0E0",
        "real_censored": "#E3F09B",
        "true_negatives": "#009e73",
        "true_positives": "#009e73",
        "false_negatives": "#FAC8CD",
        "false_positives": "#FAC8CD",
    }

    fig_new = px.bar(
        chosen_cutoff_data,
        x="mid_point",
        y="reals_estimate",
        color=fill_color_radio.value,
        color_discrete_map=color_discrete_map,
        # color="reals_labels",
        # color_discrete_map=color_discrete_map,
        category_orders={
            "reals_labels": list(color_discrete_map.keys())
        },  # fixes domain order
        hover_data=chosen_cutoff_data.columns,  # like tip: true
    )

    fig_new.update_layout(
        barmode="stack",  # stacked bars (use "group" for side-by-side)
        plot_bgcolor="rgba(0,0,0,0)",  # transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=""),
    )

    if stratified_by_radio.value == "probability_threshold":
        vertical_line = slider_cutoff.value
    else:
        vertical_line = 1 - slider_cutoff.value + by / 2

    fig_new.add_vline(
        x=vertical_line,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"Cutoff: {slider_cutoff.value}",
        annotation_position="top right",
    )

    fig_new
    return


if __name__ == "__main__":
    app.run()

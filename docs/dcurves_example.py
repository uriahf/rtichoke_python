import marimo

__generated_with = "0.14.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from dcurves import dca
    import pandas as pd
    import numpy as np
    import lifelines
    import plotly.express as px
    import polars as pl
    from rtichoke.helpers.sandbox_observable_helpers import (
        create_list_data_to_adjust,
        create_adjusted_data,
        create_aj_data_combinations,
        cast_and_join_adjusted_data,
        create_breaks_values,
    )

    df_time_to_cancer_dx = pd.read_csv(
        "https://raw.githubusercontent.com/ddsjoberg/dca-tutorial/main/data/df_time_to_cancer_dx.csv"
    )
    return (
        cast_and_join_adjusted_data,
        create_adjusted_data,
        create_aj_data_combinations,
        create_breaks_values,
        create_list_data_to_adjust,
        dca,
        df_time_to_cancer_dx,
        lifelines,
        np,
        pl,
        px,
    )


@app.cell
def _(df_time_to_cancer_dx, lifelines):
    cph = lifelines.CoxPHFitter()
    cph.fit(
        df=df_time_to_cancer_dx,
        duration_col="ttcancer",
        event_col="cancer",
        formula="age + famhistory + marker",
    )

    cph_pred_vals = cph.predict_survival_function(
        df_time_to_cancer_dx[["age", "famhistory", "marker"]], times=[1.5]
    )

    df_time_to_cancer_dx["pr_failure18"] = [1 - val for val in cph_pred_vals.iloc[0, :]]
    return


@app.cell
def _(df_time_to_cancer_dx):
    (df_time_to_cancer_dx["pr_failure18"] >= 0.5).sum()
    return


@app.cell
def _(df_time_to_cancer_dx):
    df_time_to_cancer_dx
    return


@app.cell
def _():
    outcome = "cancer"
    time_to_outcome_col = "ttcancer"
    prevalence = None
    time = 1.5
    return outcome, prevalence, time, time_to_outcome_col


@app.cell
def _(df_time_to_cancer_dx):
    (df_time_to_cancer_dx["pr_failure18"] >= 0.5).sum()
    return


@app.cell
def _(df_time_to_cancer_dx, outcome, time_to_outcome_col):
    from dcurves.risks import _create_risks_df

    risks_df = _create_risks_df(
        data=df_time_to_cancer_dx,
        outcome=outcome,
        time=1.5,
        time_to_outcome_col=time_to_outcome_col,
    )

    risks_df
    return (risks_df,)


@app.cell
def _(risks_df):
    risks_df["pr_failure18"].hist()
    return


@app.cell
def _(risks_df):
    (risks_df["pr_failure18"] >= 0.5).sum()
    return


@app.cell
def _(df_time_to_cancer_dx, risks_df):
    import plotly.graph_objects as go

    x = risks_df["pr_failure18"]
    y = df_time_to_cancer_dx["pr_failure18"]
    cancer = risks_df["cancer"]

    fig_test = go.Figure()

    # Cancer = 0 (circle)
    fig_test.add_trace(
        go.Scatter(
            x=x[cancer == 0],
            y=y[cancer == 0],
            mode="markers",
            marker=dict(symbol="circle", size=8, opacity=0.6),
            name="Cancer = 0",
        )
    )

    # Cancer = 1 (square)
    fig_test.add_trace(
        go.Scatter(
            x=x[cancer == 1],
            y=y[cancer == 1],
            mode="markers",
            marker=dict(symbol="square", size=8, opacity=0.6),
            name="Cancer = 1",
        )
    )

    fig_test.update_layout(
        title="Comparison of pr_failure18 across DataFrames",
        xaxis_title="risks_df['pr_failure18']",
        yaxis_title="df_time_to_cancer_dx['pr_failure18']",
        template="plotly_white",
    )

    fig_test.show()
    return


@app.cell
def _(risks_df):
    from dcurves.risks import _rectify_model_risk_boundaries

    modelnames = ["pr_failure18"]

    rectified_risks_df = _rectify_model_risk_boundaries(
        risks_df=risks_df, modelnames=modelnames
    )

    rectified_risks_df
    return modelnames, rectified_risks_df


@app.cell
def _(outcome, prevalence, rectified_risks_df, time, time_to_outcome_col):
    from dcurves.prevalence import _calc_prevalence

    prevalence_value = _calc_prevalence(
        risks_df=rectified_risks_df,
        outcome=outcome,
        prevalence=prevalence,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    prevalence_value
    return (prevalence_value,)


@app.cell
def _(modelnames, np, prevalence_value, rectified_risks_df):
    from dcurves.dca import _create_initial_df

    thresholds = np.arange(0, 1, 0.5)

    initial_df = _create_initial_df(
        thresholds=thresholds,
        modelnames=modelnames,
        input_df_rownum=len(rectified_risks_df.index),
        prevalence_value=prevalence_value,
    )

    initial_df
    return initial_df, thresholds


@app.cell
def _(outcome, risks_df, thresholds, time, time_to_outcome_col):
    from dcurves.dca import _calc_risk_rate_among_test_pos

    risk_rate_among_test_pos = _calc_risk_rate_among_test_pos(
        risks_df=risks_df,
        outcome=outcome,
        model="pr_failure18",
        thresholds=thresholds,
        time_to_outcome_col=time_to_outcome_col,
        time=time,
    )

    risk_rate_among_test_pos
    return


@app.cell
def _(
    outcome,
    prevalence_value,
    risks_df,
    thresholds,
    time,
    time_to_outcome_col,
):
    from dcurves.dca import _calc_test_pos_rate, _calc_tp_rate

    test_pos_rate = _calc_test_pos_rate(
        risks_df=risks_df, thresholds=thresholds, model="pr_failure18"
    )

    print("test positive rate:", test_pos_rate)

    tp_rate = _calc_tp_rate(
        risks_df=risks_df,
        thresholds=thresholds,
        model="pr_failure18",
        outcome=outcome,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
        test_pos_rate=test_pos_rate,
        prevalence_value=prevalence_value,
    )

    print("true positive rate:", tp_rate)
    return


@app.cell
def _(
    initial_df,
    outcome,
    prevalence_value,
    rectified_risks_df,
    thresholds,
    time,
    time_to_outcome_col,
):
    from dcurves.dca import _calc_initial_stats

    initial_stats_df = _calc_initial_stats(
        initial_df=initial_df,
        risks_df=rectified_risks_df,
        thresholds=thresholds,
        outcome=outcome,
        prevalence_value=prevalence_value,
        time=time,
        time_to_outcome_col=time_to_outcome_col,
    )

    initial_stats_df
    return


@app.cell
def _(rectified_risks_df):
    rectified_risks_df
    return


@app.cell
def _(df_time_to_cancer_dx):
    probs_dict = {"full": df_time_to_cancer_dx["pr_failure18"]}

    reals_mapping = {
        "censor": 0,
        "diagnosed with cancer": 1,
        "dead other causes": 2,
    }

    reals_dict = df_time_to_cancer_dx["cancer_cr"].map(reals_mapping)

    times_dict = df_time_to_cancer_dx["ttcancer"]

    df_time_to_cancer_dx["cancer_enum"] = reals_dict

    df_time_to_cancer_dx
    return probs_dict, reals_dict, times_dict


@app.cell
def _(dca, df_time_to_cancer_dx, np):
    stdca_coxph_results_composite = dca(
        data=df_time_to_cancer_dx,
        outcome="cancer_enum",
        modelnames=["pr_failure18"],
        # thresholds=np.arange(0, 0.51, 0.1),
        # thresholds=np.arange(0.5, 1, 0.1),
        thresholds=np.arange(0, 1, 0.5),
        time=1.5,
        time_to_outcome_col="ttcancer",
    )

    stdca_coxph_results_composite
    return


@app.cell
def _(create_aj_data_combinations, create_breaks_values, probs_dict):
    stratified_by = ["probability_threshold"]
    # stratified_by = ["probability_threshold"]
    # stratified_by = ["ppcr"]
    # stratified_by = ["probability_threshold"]

    by = 0.01
    breaks = create_breaks_values(None, "probability_threshold", by)

    heuristics_sets = [
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
    ]

    aj_data_combinations = create_aj_data_combinations(
        list(probs_dict.keys()),
        heuristics_sets,
        fixed_time_horizons=[1.5],
        stratified_by=stratified_by,
        by=by,
        breaks=breaks,
    )

    aj_data_combinations
    return aj_data_combinations, breaks, by, heuristics_sets, stratified_by


@app.cell
def _(
    aj_data_combinations,
    by,
    create_list_data_to_adjust,
    probs_dict,
    reals_dict,
    stratified_by,
    times_dict,
):
    list_data_to_adjust_polars = create_list_data_to_adjust(
        aj_data_combinations,
        probs_dict,
        reals_dict,
        times_dict,
        stratified_by=stratified_by,
        by=by,
    )
    list_data_to_adjust_polars
    return (list_data_to_adjust_polars,)


@app.cell
def _(
    breaks,
    create_adjusted_data,
    heuristics_sets,
    list_data_to_adjust_polars,
    stratified_by,
):
    adjusted_data = create_adjusted_data(
        list_data_to_adjust_polars,
        heuristics_sets=heuristics_sets,
        fixed_time_horizons=[1.5],
        breaks=breaks,
        stratified_by=stratified_by,
        # risk_set_scope=["within_stratum"]#,  # ,   ,
        # risk_set_scope=["pooled_by_cutoff"],  # ,  # ,   ,
        risk_set_scope=["pooled_by_cutoff", "within_stratum"],  # ,   ,
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
    final_adjusted_data_polars
    return


@app.cell
def _(final_adjusted_data_polars):
    from rtichoke.helpers.sandbox_observable_helpers import (
        _calculate_cumulative_aj_data,
    )

    cumulative_aj_data = _calculate_cumulative_aj_data(final_adjusted_data_polars)

    cumulative_aj_data
    return (cumulative_aj_data,)


@app.cell
def _():
    return


@app.cell
def _(cumulative_aj_data):
    from rtichoke.helpers.sandbox_observable_helpers import (
        _turn_cumulative_aj_to_performance_data,
    )

    performance_data = _turn_cumulative_aj_to_performance_data(cumulative_aj_data)

    performance_data
    return (performance_data,)


@app.cell
def _(performance_data):
    from rtichoke.discrimination.gains import plot_gains_curve

    plot_gains_curve(performance_data)
    return


@app.cell
def _(performance_data, pl):
    performance_data_with_nb_calculated = (
        performance_data.with_columns(
            (
                (pl.col("true_positives") / pl.col("n"))
                - (pl.col("false_positives") / pl.col("n"))
                * pl.col("chosen_cutoff")
                / (1 - pl.col("chosen_cutoff"))
            ).alias("net_benefit")
        )
        .filter(
            pl.col("censoring_heuristic") == "adjusted",
            pl.col("competing_heuristic") == "adjusted_as_censored",
        )
        .sort(pl.col("chosen_cutoff"))
    )

    performance_data_with_nb_calculated
    return


@app.cell
def _(dca, df_time_to_cancer_dx, np):
    stdca_coxph_results = dca(
        data=df_time_to_cancer_dx,
        outcome="cancer",
        modelnames=["pr_failure18"],
        thresholds=np.arange(0, 0.51, 0.01),
        time=1.5,
        time_to_outcome_col="ttcancer",
    )

    stdca_coxph_results
    return (stdca_coxph_results,)


@app.cell
def _(px, stdca_coxph_results):
    # Create plotly express figure
    fig = px.line(
        stdca_coxph_results,
        x="threshold",
        y="net_benefit",
        color="model",
        markers=True,
        title="Decision Curve Analysis",
        labels={
            "threshold": "Threshold Probability",
            "net_benefit": "Net Benefit",
        },
    )

    # Update layout to match rtichoke look
    fig.update_layout(
        template="simple_white",
        title_font_size=20,
        title_x=0.5,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(range=[-0.01, 0.23], showgrid=False, tickmode="linear", dtick=0.05),
        yaxis=dict(
            range=[-0.01, 0.23],
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        ),
    )

    fig.show()
    return


@app.cell(column=1, hide_code=True)
def _():
    import marimo as mo

    fill_color_radio = mo.ui.radio(
        options=["classification_outcome", "reals_labels"],
        value="classification_outcome",
        label="Fill Colors",
    )

    fill_color_radio
    return fill_color_radio, mo


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
        options=["probability_threshold"],
        value="probability_threshold",
        label="Stratified By",
    )

    stratified_by_radio
    return (stratified_by_radio,)


@app.cell(hide_code=True)
def _(mo):
    censoring_heuristic_radio = mo.ui.radio(
        options=["adjusted"],
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
        ],
        value="adjusted_as_negative",
        label="Competing Heuristic",
    )

    competing_heuristic_radio
    return (competing_heuristic_radio,)


@app.cell(hide_code=True)
def _(by, mo):
    slider_cutoff = mo.ui.slider(start=0, stop=1, step=by, label="Cutoff")
    slider_cutoff
    return (slider_cutoff,)


@app.cell(column=2, hide_code=True)
def _(
    by,
    censoring_heuristic_radio,
    competing_heuristic_radio,
    fill_color_radio,
    final_adjusted_data_polars,
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


@app.cell(hide_code=True)
def _(
    censoring_heuristic_radio,
    competing_heuristic_radio,
    performance_data,
    pl,
    px,
    stratified_by_radio,
):
    chosen_performance_data = performance_data.filter(
        pl.col("censoring_heuristic") == censoring_heuristic_radio.value,
        pl.col("competing_heuristic") == competing_heuristic_radio.value,
        pl.col("stratified_by") == stratified_by_radio.value,
    ).sort(pl.col("chosen_cutoff"))

    # Create plotly express figure
    fig_rtichoke = px.line(
        chosen_performance_data,
        x="chosen_cutoff",
        y="net_benefit",
        markers=True,
        title="Decision Curve Analysis",
        labels={
            "threshold": "Threshold Probability",
            "net_benefit": "Net Benefit",
        },
    )

    # Update layout to match rtichoke look
    fig_rtichoke.update_layout(
        template="simple_white",
        title_font_size=20,
        title_x=0.5,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, tickmode="linear", dtick=0.05),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        ),
    )

    fig_rtichoke.show()
    return


if __name__ == "__main__":
    app.run()

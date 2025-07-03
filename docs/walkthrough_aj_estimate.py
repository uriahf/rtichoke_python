import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Import data and Packages""")
    return


@app.cell
def _():
    from lifelines import AalenJohansenFitter
    import numpy as np
    from itertools import product
    import itertools
    from lifelines import CoxPHFitter
    from lifelines import WeibullAFTFitter
    import polars as pl

    print("Polars version:", pl.__version__)

    import pandas as pd
    import pickle

    with open(
        r"C:\Users\I\Documents\GitHub\rtichoke_python\probs_dict.pkl", "rb"
    ) as file:
        probs_dict = pickle.load(file)

    with open(
        r"C:\Users\I\Documents\GitHub\rtichoke_python\reals_dict.pkl", "rb"
    ) as file:
        reals_dict = pickle.load(file)

    with open(
        r"C:\Users\I\Documents\GitHub\rtichoke_python\times_dict.pkl", "rb"
    ) as file:
        times_dict = pickle.load(file)
    return pl, probs_dict, reals_dict, times_dict


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(probs_dict):
    from rtichoke.helpers.sandbox_observable_helpers import (
        create_aj_data_combinations_polars,
        extract_aj_estimate_for_strata,
        create_aj_data_polars,
    )

    fixed_time_horizons = [1.0, 3.0, 5.0]
    stratified_by = ["probability_threshold", "ppcr"]
    by = 0.1

    aj_data_combinations = create_aj_data_combinations_polars(
        list(probs_dict.keys()), fixed_time_horizons, stratified_by, by
    )

    print(aj_data_combinations["strata"])
    return by, create_aj_data_polars, fixed_time_horizons, stratified_by


@app.cell
def _(mo):
    mo.md(r"""## create list data to adjust polars""")
    return


@app.cell
def _(by, probs_dict, reals_dict, stratified_by, times_dict):
    from rtichoke.helpers.sandbox_observable_helpers import (
        create_list_data_to_adjust_polars,
    )

    list_data_to_adjust_polars = create_list_data_to_adjust_polars(
        probs_dict, reals_dict, times_dict, stratified_by=stratified_by, by=by
    )

    list_data_to_adjust_polars
    return (list_data_to_adjust_polars,)


@app.cell
def _(mo):
    mo.md(r"""## create adjusted data list polars""")
    return


@app.cell
def _(list_data_to_adjust_polars, pl):
    example_polars_df = list_data_to_adjust_polars.get("full").select(
        pl.col("strata"), pl.col("reals"), pl.col("times")
    )

    example_polars_df
    return (example_polars_df,)


@app.cell
def _(mo):
    mo.md(r"""## Create AJ estimates Data""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Create aj_data""")
    return


@app.cell
def _(create_aj_data_polars, example_polars_df, fixed_time_horizons, pl):
    aj_estimates_per_strata_adj_adjneg = create_aj_data_polars(
        example_polars_df, "adjusted", "adjusted_as_negative", fixed_time_horizons
    )

    aj_estimates_per_strata_excl_adjneg = create_aj_data_polars(
        example_polars_df, "excluded", "adjusted_as_negative", fixed_time_horizons
    )

    aj_estimates_per_strata_adj_adjcens = create_aj_data_polars(
        example_polars_df, "adjusted", "adjusted_as_censored", fixed_time_horizons
    )

    aj_estimates_per_strata_excl_adjcens = create_aj_data_polars(
        example_polars_df, "excluded", "adjusted_as_censored", fixed_time_horizons
    )

    aj_estimates_per_strata_adj_excl = create_aj_data_polars(
        example_polars_df, "adjusted", "excluded", fixed_time_horizons
    )

    aj_estimates_per_strata_excl_excl = create_aj_data_polars(
        example_polars_df, "excluded", "excluded", fixed_time_horizons
    )

    aj_estimates_data = pl.concat(
        [
            aj_estimates_per_strata_adj_adjneg,
            aj_estimates_per_strata_adj_adjcens,
            aj_estimates_per_strata_adj_excl,
            aj_estimates_per_strata_excl_adjneg,
            aj_estimates_per_strata_excl_adjcens,
            aj_estimates_per_strata_excl_excl,
        ]
    ).unpivot(
        index=[
            "strata",
            "fixed_time_horizon",
            "censoring_assumption",
            "competing_assumption",
        ],
        variable_name="reals_labels",
        value_name="reals_estimate",
    )
    return (aj_estimates_data,)


@app.cell
def _(aj_estimates_data):
    aj_estimates_data
    return


if __name__ == "__main__":
    app.run()

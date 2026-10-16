import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from rtichoke import create_performance_table, create_performance_table_times

    return create_performance_table, create_performance_table_times, mo, np


@app.cell
def _(mo):
    mo.md(
        """
        # rtichoke performance table

        PR preview for the Python port of `rtichoke::create_performance_table()`.
        This Marimo preview uses the **Great Tables** renderer. The same public
        API also supports `renderer="reactable"` for Quarto/Jupyter contexts.
        """
    )
    return


@app.cell
def _(np):
    reals = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    model_a = np.array(
        [0.04, 0.10, 0.20, 0.24, 0.33, 0.42, 0.48, 0.61, 0.70, 0.82, 0.86, 0.94]
    )
    model_b = np.array(
        [0.08, 0.18, 0.14, 0.39, 0.30, 0.50, 0.43, 0.57, 0.65, 0.74, 0.76, 0.88]
    )
    return model_a, model_b, reals


@app.cell
def _(create_performance_table, mo, model_a, reals):
    mo.md("## One model — probability threshold")
    create_performance_table(probs={"Model A": model_a}, reals=reals, by=0.10)
    return


@app.cell
def _(create_performance_table, mo, model_a, model_b, reals):
    mo.md("## Multiple models — probability threshold")
    create_performance_table(
        probs={"Model A": model_a, "Model B": model_b}, reals=reals, by=0.10
    )
    return


@app.cell
def _(create_performance_table, mo, model_a, model_b, reals):
    mo.md("## Multiple models — PPCR")
    create_performance_table(
        probs={"Model A": model_a, "Model B": model_b},
        reals=reals,
        by=0.10,
        stratified_by=("ppcr",),
    )
    return


@app.cell
def _(np):
    # Sufficient observations in every threshold stratum for time-dependent
    # Aalen-Johansen calculations (the former 10-row example had empty risk sets).
    n = 400
    time_probs = {"Model A": np.linspace(0.05, 0.95, n)}
    time_reals = np.tile(np.array([0, 1, 1, 0, 1]), n // 5)
    times = np.resize(np.array([2, 3, 5, 7, 9, 11, 14]), n).astype(float)
    return time_probs, time_reals, times

@app.cell
def _(create_performance_table_times, mo, time_probs, time_reals, times):
    mo.md("## Fixed time horizons — 5 and 10")
    create_performance_table_times(
        probs=time_probs,
        reals=time_reals,
        times=times,
        fixed_time_horizons=[5, 10],
        by=0.10,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        The Great Tables backend is the Marimo-safe renderer. Reactable is kept
        as an optional richer backend because it supports sortable columns and
        expandable confusion-matrix details in environments that support its
        Jupyter widget bridge.
        """
    )
    return


if __name__ == "__main__":
    app.run()

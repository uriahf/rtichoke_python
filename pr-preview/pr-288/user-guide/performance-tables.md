# Performance Tables

Performance tables summarize several model-performance quantities at the same probability threshold. They are useful when you want a compact comparison across models rather than a separate ROC, precision-recall, calibration, or decision curve.

`rtichoke` provides two public constructors:

- `create_performance_table()` for binary outcomes.
- `create_performance_table_times()` for time-to-event outcomes at one or more fixed horizons.

Both use the existing [prepare_performance_data()](../reference/prepare_performance_data.md#rtichoke.prepare_performance_data) / [prepare_performance_data_times()](../reference/prepare_performance_data_times.md#rtichoke.prepare_performance_data_times) pipelines as their numerical source of truth. The table layer is presentation only.


# Basic performance table

A minimal two-model example:

``` python
import numpy as np
import rtichoke as rk

reals = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1])

probs = {
    "Model A": np.array([0.04, 0.10, 0.20, 0.24, 0.33, 0.42, 0.48, 0.61, 0.70, 0.82, 0.86, 0.94]),
    "Model B": np.array([0.08, 0.18, 0.14, 0.39, 0.30, 0.50, 0.43, 0.57, 0.65, 0.74, 0.76, 0.88]),
}

table = rk.create_performance_table(
    probs=probs,
    reals=reals,
    by=0.10,
)

table
```

The default stratification is by `probability_threshold`, so each row corresponds to a threshold for one model. The table collects the performance quantities produced by [prepare_performance_data()](../reference/prepare_performance_data.md#rtichoke.prepare_performance_data) into one view, including discrimination, classification, and decision-analytic quantities where available.

For an alternative view based on the predicted-positive proportion, use:

``` python
rk.create_performance_table(
    probs=probs,
    reals=reals,
    by=0.10,
    stratified_by=("ppcr",),
)
```


# Time-dependent performance tables

`create_performance_table_times()` applies the same idea to time-to-event prediction. You supply observed times and one or more fixed horizons:

``` python
import numpy as np
import rtichoke as rk

probs = {
    "Model A": np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
}

# 0 = censored, 1 = event of interest
reals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

rk.create_performance_table_times(
    probs=probs,
    reals=reals,
    times=times,
    fixed_time_horizons=[5, 10],
    by=0.10,
)
```

The time horizon remains visible in the output, so results from different horizons are not collapsed together.

By default, time-dependent performance tables use:

``` python
heuristics_sets = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]
```

You can pass multiple heuristic sets. The censoring and competing-event heuristic columns remain visible so distinct evaluation scenarios stay distinguishable.

As with the other time-dependent rtichoke functions, a censoring heuristic affects estimates only when censored observations are present, and a competing-event heuristic affects estimates only when competing events are present.


# Renderer choice

The default renderer is **Great Tables**:

``` python
rk.create_performance_table(probs=probs, reals=reals)
```

Great Tables is the recommended renderer for Marimo and ordinary HTML output. It is styled to preserve the visual ideas of the original R performance table, including model labeling, grouped performance columns, compact metric bars, predicted-positive bars, and diverging net-benefit bars.

For Quarto or Jupyter environments, Reactable remains available explicitly:

``` python
rk.create_performance_table(
    probs=probs,
    reals=reals,
    renderer="reactable",
)
```

The Reactable backend adds richer interaction such as sortable columns and expandable confusion-matrix details. It is retained as an option for environments that support its Jupyter widget bridge; it is **not** the Marimo renderer.

The same `renderer=` argument is available on `create_performance_table_times()`.


# Render prepared performance data directly

If you already called [prepare_performance_data()](../reference/prepare_performance_data.md#rtichoke.prepare_performance_data) or [prepare_performance_data_times()](../reference/prepare_performance_data_times.md#rtichoke.prepare_performance_data_times), render the resulting Polars DataFrame without recomputing it:

``` python
performance_data = rk.prepare_performance_data(
    probs=probs,
    reals=reals,
    by=0.10,
)

rk.render_performance_table(performance_data)
```

Use `renderer="reactable"` here as well if you want the Reactable backend.


# Related documentation

For the underlying numerical data, see the [prepare_performance_data()](../reference/prepare_performance_data.md#rtichoke.prepare_performance_data) and [prepare_performance_data_times()](../reference/prepare_performance_data_times.md#rtichoke.prepare_performance_data_times) API reference. For time-dependent censoring and competing-event semantics, see [Curve API Compatibility](curve-api-compatibility.md).

# Curve API Compatibility

`rtichoke` exposes parallel function families for discrimination, calibration, and decision-curve analysis. Their interfaces are similar, but time-dependent defaults and accepted heuristics are not identical.

This page makes those differences explicit so that users -- and coding agents reading `llms-full.txt` -- can choose the right call without experimentally probing each function.


# Capability matrix

| Function family | Multiple named populations | Unequal population sizes with `dict`/`dict` inputs | Time-dependent heuristic default | `fixed_time_horizons` |
|----|----|----|----|----|
| ROC | Supported | Supported | `adjusted` / `adjusted_as_negative` | Use floating-point values |
| Precision-recall | Supported | Supported | `adjusted` / `adjusted_as_negative` | Use floating-point values |
| Decision curve | Supported | Supported | `adjusted` / `adjusted_as_negative` | Use floating-point values |
| Calibration | Supported with matching population keys | Supported | **No default on [create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times)**; pass explicitly | Use floating-point values |


# Calibration with named populations

Matching dictionary keys are paired population-by-population. Each probability vector must match the outcome vector for its own population, but populations may have different sample sizes.

``` python
import numpy as np
import rtichoke as rk

probs = {
    "Train": np.array([0.10, 0.90, 0.20, 0.80, 0.30, 0.70]),
    "Test": np.array([0.15, 0.85, 0.25, 0.75]),
}
reals = {
    "Train": np.array([0, 1, 0, 1, 0, 1]),
    "Test": np.array([0, 1, 0, 0]),
}

fig = rk.create_calibration_curve(probs=probs, reals=reals)
```

In this example, Train has six observations and Test has four. That is supported. What matters is:

``` text
len(probs["Train"]) == len(reals["Train"])
len(probs["Test"])  == len(reals["Test"])
```


# Time-dependent calibration heuristics

[create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times) still differs from its ROC, precision-recall, and decision-curve siblings in two important ways:

1.  `heuristics_sets` is currently required rather than defaulted.
2.  The sibling default `censoring_heuristic="adjusted"` is not a safe value to copy blindly into calibration. In the current calibration path, that choice can remove every requested horizon and end in `No data remaining after applying heuristics and time horizons.`

Pass the calibration heuristic explicitly. For the currently working exclusion-based path:

``` python
heuristics_sets = [
    {
        "censoring_heuristic": "excluded",
        "competing_heuristic": "adjusted_as_negative",
    }
]
```

Then call:

``` python
fig = rk.create_calibration_curve_times(
    probs=probs,
    reals=reals,
    times=times,
    fixed_time_horizons=[3.0, 6.0, 9.0],
    heuristics_sets=heuristics_sets,
)
```


# Time horizons: prefer floats

Use floating-point horizons such as `[3.0, 6.0, 9.0]`, not `[3, 6, 9]`. The current implementation can otherwise expose an internal Polars join-key datatype mismatch (`i64` versus `f64`) rather than a rtichoke-specific validation message.

``` python
# Prefer
fixed_time_horizons = [3.0, 6.0, 9.0]

# Avoid for now
fixed_time_horizons = [3, 6, 9]
```

This is a usability limitation rather than a conceptual requirement; normalizing numeric horizons internally is a good candidate for a small future code fix.


# Related functions

When moving between curve families, compare the API reference for [create_calibration_curve()](../reference/create_calibration_curve.md#rtichoke.create_calibration_curve), [create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times), [create_roc_curve_times()](../reference/create_roc_curve_times.md#rtichoke.create_roc_curve_times), [create_precision_recall_curve_times()](../reference/create_precision_recall_curve_times.md#rtichoke.create_precision_recall_curve_times), and [create_decision_curve_times()](../reference/create_decision_curve_times.md#rtichoke.create_decision_curve_times) rather than assuming their defaults and accepted heuristics are identical.

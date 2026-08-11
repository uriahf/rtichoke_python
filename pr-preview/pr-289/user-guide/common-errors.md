# Common Errors & Fixes

This page is deliberately keyed by **literal error text**. If an rtichoke call fails, search this page for a distinctive part of the exception before tracing into the implementation.


# `probs['...'] length=... does not match sum of population sizes=...`


## Where this commonly appears

Calibration calls using dictionaries for both predictions and outcomes, especially when the named populations have different numbers of observations.


## Why it happens

The current calibration path has stricter alignment requirements than ROC, precision-recall, and decision curves. An input pattern that works for those sibling functions can therefore fail for [create_calibration_curve()](../reference/create_calibration_curve.md#rtichoke.create_calibration_curve) or [create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times).


## Fix

First verify that every prediction vector corresponds to the intended outcome population. If you are deliberately comparing differently sized populations, do not assume calibration supports the same dict/dict overlay pattern as ROC or PR. Create the calibration result for each aligned population separately.

See [Curve API Compatibility](curve-api-compatibility.md) for the family-by-family comparison.


# `No data remaining after applying heuristics and time horizons.`


## Where this commonly appears

[create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times).


## Why it happens

One current failure mode is passing the heuristic combination used as the default by ROC/PR/decision time-dependent functions. In calibration, `censoring_heuristic="adjusted"` can skip every horizon, leaving no data to plot. The final exception does not currently explain which heuristic caused the removal.


## Fix

Pass the calibration heuristic explicitly. For the currently working exclusion-based path:

``` python
heuristics_sets = [
    {
        "censoring_heuristic": "excluded",
        "competing_heuristic": "adjusted_as_negative",
    }
]
```

Do not infer calibration defaults from [create_roc_curve_times()](../reference/create_roc_curve_times.md#rtichoke.create_roc_curve_times), [create_precision_recall_curve_times()](../reference/create_precision_recall_curve_times.md#rtichoke.create_precision_recall_curve_times), or [create_decision_curve_times()](../reference/create_decision_curve_times.md#rtichoke.create_decision_curve_times).


# Polars join-key datatype mismatch: `i64` versus `f64`


## Where this commonly appears

Time-dependent functions when integer values are supplied in `fixed_time_horizons`, for example:

``` python
fixed_time_horizons=[3, 6, 9]
```


## Why it happens

The current internal time-horizon data can be floating point, while integer literals create integer-typed join keys. The resulting Polars error leaks an implementation detail instead of explaining the rtichoke input requirement.


## Fix

Use floating-point horizons:

``` python
fixed_time_horizons=[3.0, 6.0, 9.0]
```

Internal numeric normalization is a candidate for a future defensive code fix.


# Why is `heuristics_sets` missing?

If Python reports that [create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times) is missing the required `heuristics_sets` argument, that is currently expected API behavior. Unlike the ROC, precision-recall, and decision-curve `_times` functions, calibration does not currently provide a default.

Pass it explicitly rather than copying a sibling default:

``` python
heuristics_sets = [
    {
        "censoring_heuristic": "excluded",
        "competing_heuristic": "adjusted_as_negative",
    }
]
```


# Still stuck?

Check [Curve API Compatibility](curve-api-compatibility.md) first. The most important debugging rule is that similarly named rtichoke curve functions can still differ in accepted population shapes, heuristic defaults, and time-horizon handling.

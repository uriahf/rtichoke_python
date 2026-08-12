# Curve API Compatibility

`rtichoke` exposes parallel function families for discrimination, calibration, and decision-curve analysis. Their interfaces are similar, but time-dependent defaults and accepted heuristics are not identical.

This page makes those differences explicit so that users -- and coding agents reading `llms-full.txt` -- can choose the right call without experimentally probing each function.


# Capability matrix

| Function family | Multiple named populations | Unequal population sizes with `dict`/`dict` inputs | Time-dependent heuristic default | `fixed_time_horizons` |
|----|----|----|----|----|
| ROC | Supported | Supported | `adjusted` / `adjusted_as_negative` | Integer and floating-point values supported |
| Precision-recall | Supported | Supported | `adjusted` / `adjusted_as_negative` | Integer and floating-point values supported |
| Gains | Supported | Supported | `adjusted` / `adjusted_as_negative` | Integer and floating-point values supported |
| Lift | Supported | Supported | `adjusted` / `adjusted_as_negative` | Integer and floating-point values supported |
| Decision curve | Supported | Supported | `adjusted` / `adjusted_as_negative` | Integer and floating-point values supported |
| Calibration | Supported | Supported | **No default on [create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times)**; pass explicitly | Integer and floating-point values supported |


# Multiple named populations

The curve families accept named probability arrays, so populations such as Train and Test can be evaluated together. When outcomes are also supplied as a dictionary, matching dictionary keys are paired population-by-population. Each probability vector must match the outcome vector for its own population, but different populations may have different sample sizes.

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

Here Train has six observations and Test has four. That is supported. What matters is the within-population alignment:

``` text
len(probs["Train"]) == len(reals["Train"])
len(probs["Test"])  == len(reals["Test"])
```

The same named-population pattern is used by ROC, precision-recall, Gains, Lift, decision, and calibration curve families. For time-dependent calls, `times` follows the same population alignment when supplied as a dictionary.


# Censoring and competing-event heuristics

Time-dependent functions distinguish censoring from competing events. The heuristic for an outcome type matters only when observations of that type are present:

- If there are no competing events, changing `competing_heuristic` does not change the statistical estimates because there are no competing events for that rule to act on.
- If there are no censored observations, changing `censoring_heuristic` does not change the statistical estimates because there are no censored observations for that rule to act on.
- If neither censoring nor competing events are present, the heuristic choices do not alter the estimates.

These statements describe the effect of the heuristics on the estimates. Function-specific input validation still applies: a function can reject an unsupported heuristic combination even when the corresponding outcome type is absent.


# Time-dependent calibration heuristics

[create_calibration_curve_times()](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times) differs from its ROC, precision-recall, Gains, Lift, and decision-curve siblings in two important ways:

1.  `heuristics_sets` is currently required rather than defaulted.
2.  Calibration explicitly rejects unsupported heuristic combinations, including `censoring_heuristic="adjusted"` and `competing_heuristic="adjusted_as_censored"`, with an `Unsupported calibration heuristics` error instead of silently skipping every requested horizon.

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


# Numeric time horizons

`fixed_time_horizons` accepts integer or floating-point numeric values. Integer horizons are normalized to floats at the shared time-dependent processing boundary, so `[3, 6, 9]` and `[3.0, 6.0, 9.0]` are equivalent.


# Related functions

When moving between curve families, compare the API reference for the relevant `_times()` functions rather than assuming their defaults and accepted heuristics are identical. In particular, calibration has a narrower heuristic contract than the other time-dependent curve families.

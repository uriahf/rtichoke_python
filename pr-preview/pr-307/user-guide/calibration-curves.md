# Calibration Curves

Calibration evaluates how well predicted risks from a classification or time-to-event model align with observed outcome rates.


# Standard Calibration ([create_calibration_curve](../reference/create_calibration_curve.md#rtichoke.create_calibration_curve))

For binary outcomes:

``` python
import rtichoke as rk

fig = rk.create_calibration_curve(
    probs={"Model A": probs_a},
    reals=reals_binary,
    calibration_type="smooth",
)
```

`calibration_type` can be set to `"discrete"` (deciles) or `"smooth"` (lowess curve).


# Time-Dependent Calibration ([create_calibration_curve_times](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times))

When evaluating risk predictions at a specific time horizon t:

``` python
heuristics_sets = [
    {
        "censoring_heuristic": "adjusted",
        "competing_heuristic": "adjusted_as_negative",
    }
]

fig = rk.create_calibration_curve_times(
    probs={"Model A": probs_a},
    reals=reals_time_to_event,
    times=times,
    fixed_time_horizons=[3.0, 5.0],
    heuristics_sets=heuristics_sets,
    calibration_type="smooth",
    smooth_method="local_aj",
)
```


# Smoothing Methods for Time-Dependent Calibration

When `calibration_type="smooth"`, [create_calibration_curve_times](../reference/create_calibration_curve_times.md#rtichoke.create_calibration_curve_times) supports three distinct statistical smoothing methods via the `smooth_method` parameter:


## 1. Local Aalen-Johansen (`smooth_method="local_aj"`, Default)

**Gerds' favoured local neighborhood method** (`riskRegression::plotCalibration(method="nne", cens.method="local")`):

- Computes local Aalen-Johansen / Kaplan-Meier cumulative incidence estimates within nearest-neighborhood risk windows across predicted probabilities.
- Fully non-parametric and handles both standard survival and competing risks (0=\text{censored}, 1=\text{event}, 2=\text{competing event}).
- You can tune the neighborhood window using the optional `bandwidth` parameter (e.g., `bandwidth=0.2`).


## 2. Secondary Cox Model (`smooth_method="secondary_cox"`)

To construct a smooth time-dependent calibration curve at prediction horizon t, `rtichoke` fits a secondary Cox proportional hazards model (Austin, Harrell & van Klaveren 2020; McLernon et al. 2023) on the complementary log-log transformed predicted probabilities:

x_i = \log\left(-\log(1 - \hat{p}\_i)\right)

The secondary Cox model relates observed event times to x_i:

h(t \mid x_i) = h_0(t) \exp(\beta x_i)

The estimated actual risk at horizon t across predicted probabilities \hat{p} is then calculated as:

\hat{F}(t \mid \hat{p}) = 1 - \hat{S}\_0(t)^{\exp(\beta x)}


## 3. Pseudo-Values LOWESS (`smooth_method="pseudo_values"`)

**Jackknife pseudo-observations method**:

- Computes leave-one-out Aalen-Johansen pseudo-values for each subject at horizon t.
- Applies LOWESS smoothing against predicted probabilities.


# Discrete (Binned) Calibration

For binned decile plots at time horizons, pass `calibration_type="discrete"`:

``` python
fig = rk.create_calibration_curve_times(
    probs={"Model A": probs_a},
    reals=reals_time_to_event,
    times=times,
    fixed_time_horizons=[5.0],
    heuristics_sets=heuristics_sets,
    calibration_type="discrete",
)
```

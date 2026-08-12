## create_calibration_curve_times()


Create a time-dependent calibration curve across fixed horizons.


Usage

``` python
create_calibration_curve_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets,
    calibration_type="discrete",
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


## Raises

`ValueError`  
If a heuristic set requests adjusted censoring or treats competing events as censored, which calibration does not support.

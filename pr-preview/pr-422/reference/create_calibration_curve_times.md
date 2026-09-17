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
    smooth_method="local_aj",
    bandwidth=None,
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"],
    *,
    n_bins=10
)
```


This function generates time-dependent calibration curves evaluating predicted probabilities against observed outcomes over specified prediction horizons.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names to 1-D numpy arrays of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
True outcome indicators (0 for censored, 1 for event of interest, 2 for competing risk).

`times: Union[np.ndarray, Dict[str, np.ndarray]]`  
Follow-up times corresponding to `reals`.

`fixed_time_horizons: List[float]`  
List of prediction horizons (times) at which to evaluate calibration.

`heuristics_sets: List[Dict[str, str]]`  
List of heuristic dictionaries defining censoring and competing risk adjustments.

`calibration_type: str = ``"discrete"`  
Type of calibration plot, either `"discrete"` (binned) or `"smooth"`. Defaults to `"discrete"`.

`smooth_method: str = ``"local_aj"`  
Smoothing method when `calibration_type="smooth"`. Supported options are `"local_aj"` (Gerds' local Aalen-Johansen/KM neighborhood estimation), `"secondary_cox"` (Austin, Harrell & McLernon secondary Cox regression with 3-knot restricted cubic splines on complementary log-log predictions), or `"pseudo_values"` (jackknife pseudo-values lowess). Defaults to `"local_aj"`.

`bandwidth: Union[float, None] = None`  
Bandwidth fraction for `"local_aj"` neighborhood smoothing. Defaults to None.

`size: int = ``600`  
Width and height of the Plotly figure in pixels. Defaults to 600.

`color_values: List[str] = [`\
`    `<span class="st">`"#1b9e77",`\
`    ``"#d95f02"``,`\
`    ``"#7570b3"``,`\
`    ``"#e7298a"``,`\
`    ``"#07004D"``,`\
`    ``"#E6AB02"``,`\
`    ``"#FE5F55"``,`\
`    ``"#54494B"``,`\
`    ``"#006E90"``,`\
`    ``"#BC96E6"``,`\
`    ``"#52050A"``,`\
`    ``"#1F271B"``,`\
`    ``"#BE7C4D"``,`\
`    ``"#63768D"``,`\
`    ``"#08A045"``,`\
`    ``"#320A28"``,`\
`    ``"#82FF9E"``,`\
`    ``"#2176FF"``,`\
`    ``"#D1603D"``,`\
`    ``"#585123"``,`\
`]`\
</span>  
List of hex color strings for traces.

`n_bins: int = ``10`  
Number of bins for discrete calibration curves. Defaults to 10.


## Returns


`Figure`  
A Plotly `Figure` object representing the time-dependent calibration curve.


## Raises


`ValueError`  
If a heuristic set requests `competing_heuristic='adjusted_as_censored'`.

## create_decision_curve_times()


Creates a time-dependent Decision Curve.


Usage

``` python
create_decision_curve_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    decision_type="conventional",
    heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
    min_p_threshold=0,
    max_p_threshold=1,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


Generates a Decision Curve for time-to-event models, which is evaluated at specified time horizons and handles censored data and competing risks.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event statuses.

`times: Union[np.ndarray, Dict[str, np.ndarray]]`  
The event or censoring times.

`fixed_time_horizons: list[float]`  
A list of time points for performance evaluation.

`decision_type: str = ``"conventional"`  
Type of decision curve to plot. Defaults to `"conventional"`.

`heuristics_sets: list[Dict] = [{``"censoring_heuristic": `<span class="st">`"adjusted"``, ``"competing_heuristic"``: ``"adjusted_as_negative"``}]`</span>  
Specifies how to handle censored data and competing events.

`min_p_threshold: float = ``0`  
The minimum probability threshold to plot. Defaults to 0.

`max_p_threshold: float = ``1`  
The maximum probability threshold to plot. Defaults to 1.

`by: float = ``0.01`  
The step size for the probability thresholds. Defaults to 0.01.

`stratified_by: Sequence[str] = [``"probability_threshold"]`  
Variables for stratification. Defaults to `["probability_threshold"]`.

`size: int = ``600`  
The width and height of the plot in pixels. Defaults to 600.

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
`]`</span>  
A list of hex color strings for the plot lines.


## Returns


`Figure`  
A Plotly `Figure` object for the time-dependent Decision Curve.

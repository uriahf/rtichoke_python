## create_roc_curve_times()


Creates a time-dependent Receiver Operating Characteristic (ROC) curve.


Usage

``` python
create_roc_curve_times(
    probs,
    reals,
    times,
    fixed_time_horizons,
    heuristics_sets=[{"censoring_heuristic": "adjusted", "competing_heuristic": "adjusted_as_negative"}],
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


This function generates an ROC curve for time-to-event models. It evaluates the model's performance at specified time horizons, handling censored data and competing risks according to the chosen heuristics.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true event statuses (e.g., 0=censored, 1=event, 2=competing).

`times: Union[np.ndarray, Dict[str, np.ndarray]]`  
The event or censoring times.

`fixed_time_horizons: list[float]`  
A list of time points for performance evaluation.

`heuristics_sets: list[Dict] = [{``"censoring_heuristic": `<span class="st">`"adjusted"``, ``"competing_heuristic"``: ``"adjusted_as_negative"``}]`</span>  
Specifies how to handle censored data and competing events.

`by: float = ``0.01`  
The step size for probability thresholds. Defaults to 0.01.

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
A Plotly `Figure` object representing the time-dependent ROC curve.

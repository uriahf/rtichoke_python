## create_gains_curve()


Creates a Gains curve.


Usage

``` python
create_gains_curve(
    probs,
    reals,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


A Gains curve is a marketing and business analytics tool that evaluates the performance of a predictive model. It shows the percentage of positive outcomes (the "gain") that can be captured by targeting a certain percentage of the population, sorted by predicted probability.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names to 1-D numpy arrays of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true binary labels (0 or 1).

`by: float = ``0.01`  
The step size for the probability thresholds. Defaults to 0.01.

`stratified_by: Sequence[str] = [`<span class="st">`"probability_threshold"]`\
</span>  
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
`]`\
</span>  
A list of hex color strings for the plot lines.


## Returns


`Figure`  
A Plotly `Figure` object representing the Gains curve.

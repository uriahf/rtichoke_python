## create_decision_curve()


Creates a Decision Curve.


Usage

``` python
create_decision_curve(
    probs,
    reals,
    decision_type="conventional",
    min_p_threshold=0,
    max_p_threshold=1,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


Decision Curve Analysis is a method for evaluating and comparing prediction models that incorporates the clinical consequences of a decision. The curve plots the net benefit of a model against the probability threshold used to determine positive cases. This helps to assess the real-world utility of a model.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names to 1-D numpy arrays of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true binary labels (0 or 1).

`decision_type: str = ``"conventional"`  
Type of decision curve. `"conventional"` for a standard decision curve or another value for the "interventions avoided" variant. Defaults to `"conventional"`.

`min_p_threshold: float = ``0`  
The minimum probability threshold to plot. Defaults to 0.

`max_p_threshold: float = ``1`  
The maximum probability threshold to plot. Defaults to 1.

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
A Plotly `Figure` object representing the Decision Curve.

## create_lift_curve()


Creates a Lift curve.


Usage

``` python
create_lift_curve(
    probs,
    reals,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"],
    renderer="plotly"
)
```


A Lift curve is a visual tool used to evaluate the performance of a classification model. It shows how much better the model is at identifying positive outcomes compared to a random guess. The "lift" is the ratio of the results obtained with the model to the results from a random selection.


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

`renderer: (plotly, matplotlib, browser, rtichoke_viz) = ``"plotly"`  
Rendering backend. The default, `"plotly"`, preserves the existing return value and behavior. `"matplotlib"` requires the optional Matplotlib dependency. `"browser"` and its `"rtichoke_viz"` alias return an offline browser chart backed by the packaged TypeScript bundle.


## Returns


`Figure or RtichokeBrowserChart`  
A Plotly or Matplotlib figure, or an offline browser chart, depending on `renderer`.

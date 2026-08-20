## create_roc_curve()


Creates a Receiver Operating Characteristic (ROC) curve.


Usage

``` python
create_roc_curve(
    probs,
    reals,
    by=0.01,
    stratified_by=["probability_threshold"],
    size=600,
    color_values=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02", "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D", "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"]
)
```


This function generates an ROC curve, which visualizes the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

It first calculates the performance data using the provided probabilities and true labels, and then generates the plot.


## Parameters


`probs: Dict[str, np.ndarray]`  
A dictionary mapping model or dataset names to 1-D numpy arrays of predicted probabilities.

`reals: Union[np.ndarray, Dict[str, np.ndarray]]`  
The true binary labels (0 or 1). Can be a single array for all probabilities or a dictionary mapping names to label arrays.

`by: float = ``0.01`  
The step size for the probability thresholds, controlling the curve's granularity. Defaults to 0.01.

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
A list of hex color strings for the plot lines. A default palette is used if not provided.


## Returns


`Figure`  
A Plotly `Figure` object representing the ROC curve.
